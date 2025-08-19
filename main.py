# ==================== top-of-file shim (keep this at the very top) ====================
import os, sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras as K
# force any "import keras" to resolve to tf.keras
sys.modules["keras"] = K
sys.modules["keras.models"] = K.models
sys.modules["keras.layers"] = K.layers
sys.modules["keras.activations"] = K.activations
sys.modules["keras.initializers"] = K.initializers
sys.modules["keras.utils"] = K.utils
sys.modules["keras.losses"] = K.losses
sys.modules["keras.backend"] = K.backend
# =====================================================================================

import csv
import time
from datetime import datetime
from importlib import reload
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

# Local imports from your project
import config
import u_8_s_pose_keras_qat as cfg
from loss import distill_loss_pose
from process import (build_dataset, normalize_teacher_pred, rep_data_gen,
                     try_load_keras_model)

# ================================== Helper Functions ==================================

def enable_gpu_mem_growth():
    """設定 GPU 記憶體為動態增長模式。"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ No GPU detected. Running on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        lg = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPUs, {len(lg)} Logical GPUs. Memory growth enabled.")
    except RuntimeError as e:
        print(f"❌ TF GPU config error: {e}")

def setup_mixed_precision():
    """如果 config.USE_AMP 為 True，則啟用混合精度訓練。"""
    if config.USE_AMP:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision (AMP) enabled.")
        except ImportError:
            print("⚠️ Could not import mixed_precision. Skipping AMP setup.")
    else:
        print("ℹ️ Mixed precision (AMP) is disabled.")

def plot_and_save_lr_schedule(schedule, total_steps, save_path):
    """繪製學習率變化曲線並儲存。"""
    steps = np.arange(total_steps)
    lrs = [schedule(step) for step in steps]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Learning rate schedule plot saved to {save_path}")

def plot_and_save_loss_curve(history, save_path):
    """繪製損失曲線並儲存。"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.title('Distillation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve plot saved to {save_path}")

# ================================== Core Logic ==================================

def build_student_qat():
    """建立並量化學生模型。"""
    reload(cfg)
    student = cfg.build_u8s_pose(
        input_shape=(config.IMGSZ, config.IMGSZ, 3),
        num_classes=config.NUM_CLS,
        num_kpt=config.NUM_KPT,
        kpt_vals=config.KPT_VALS
    )
    student = tfmot.quantization.keras.quantize_model(student)
    qlayers = [l for l in student.submodules if "Quantize" in l.__class__.__name__]
    print(f"[CHECK] quantization layers count: {len(qlayers)}")
    return student


def run_qat(student, teacher, ds, steps_per_epoch, output_paths):
    """執行 QAT 訓練，並增加可視化和日誌記錄。"""
    print("\n--- Starting QAT Fine-tuning ---")
    # 1) 凍結 BN 層
    for l in student.submodules:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
    print(" BN layers frozen.")

    # 2) 設定學習率排程
    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, end_lr, warmup_steps, total_steps):
            super().__init__()
            self.base_lr, self.end_lr = float(base_lr), float(end_lr)
            self.warmup_steps, self.total_steps = int(warmup_steps), int(max(total_steps, warmup_steps + 1))
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            ws, ts = tf.cast(self.warmup_steps, tf.float32), tf.cast(self.total_steps, tf.float32)
            warm = self.base_lr * (step + 1.0) / tf.maximum(ws, 1.0)
            t = (step - ws) / tf.maximum(ts - ws, 1.0)
            cos = self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1.0 + tf.cos(np.pi * t))
            return tf.where(step < ws, warm, cos)
        def get_config(self):
            return {"base_lr": self.base_lr, "end_lr": self.end_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    total_steps = max(1, config.EPOCHS * steps_per_epoch)
    warmup_steps = min(1000, max(1, total_steps // 10))
    schedule = WarmupCosine(base_lr=config.base_lr, end_lr=config.end_lr, warmup_steps=warmup_steps, total_steps=total_steps)
    plot_and_save_lr_schedule(schedule, total_steps, output_paths['lr_plot'])

    # 3) 建立優化器
    opt = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=config.momentum, nesterov=True, clipnorm=1.0)
    if config.USE_AMP:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    print(f" Optimizer: SGD with WarmupCosine LR schedule (total steps: {total_steps}).")

    # 4) 定義訓練步驟
    expected_C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    @tf.function
    def train_step(batch_imgs):
        y_t = teacher(batch_imgs, training=False)
        y_t = tf.stop_gradient(normalize_teacher_pred(y_t, expected_C=expected_C))
        with tf.GradientTape() as tape:
            y_s = student(batch_imgs, training=True)
            loss = distill_loss_pose(y_t, y_s, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
            scaled_loss = opt.get_scaled_loss(loss) if config.USE_AMP else loss
        
        scaled_grads = tape.gradient(scaled_loss, student.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads) if config.USE_AMP else scaled_grads
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # 5) 執行訓練迴圈
    loss_history = []
    with open(output_paths['log_csv'], 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'loss', 'learning_rate'])
        
        for e in range(config.EPOCHS):
            epoch_loss_agg = tf.keras.metrics.Mean()
            it = iter(ds)
            
            progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {e+1}/{config.EPOCHS}", unit="step")
            for step in progress_bar:
                imgs = next(it)
                loss = train_step(imgs)
                epoch_loss_agg.update_state(loss)
                progress_bar.set_postfix(loss=f"{loss:.4f}")
            
            avg_loss = epoch_loss_agg.result().numpy()
            current_lr = schedule((e + 1) * steps_per_epoch).numpy()
            loss_history.append(avg_loss)
            
            # 寫入 CSV
            csv_writer.writerow([e + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}"])
            
            print(f"Epoch {e+1}/{config.EPOCHS} - Average Loss: {avg_loss:.4f}, Current LR: {current_lr:.6f}")
            
    print(f"✅ Training finished. Log saved to {output_paths['log_csv']}")
    return loss_history


def choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, expected_C):
    """自動偵測學生模型 P3,P4,P5 輸出塊的最佳排列順序。"""
    print("\n--- Aligning Student/Teacher Output Order ---")
    y_te_nc = normalize_teacher_pred(teacher(sample_one, training=False), expected_C=expected_C)
    y_st = student_infer(sample_one, training=False)
    if y_st.shape.rank != 3: raise ValueError(f"[EXPORT] Student single output must be rank-3, got {y_st.shape}")
    
    y_st_nc = tf.transpose(y_st, [0, 2, 1]) if int(y_st.shape[2]) == int(y_te_nc.shape[1]) else y_st
    
    lens = [N3, N4, N5]
    teacher_orders = {"forward": [N3, N4, N5], "reverse": [N5, N4, N3]}
    y_te_orders = {
        "forward": y_te_nc,
        "reverse": tf.concat([y_te_nc[:, N3+N4:, :], y_te_nc[:, N3:N3+N4, :], y_te_nc[:, :N3, :]], axis=1),
    }

    best_perm, best_order, best_mae = None, None, float("inf")
    for perm in permutations(lens, 3):
        s0, s1, s2 = perm
        split_student = tf.split(y_st_nc, [s0, s1, s2], axis=1)
        for name, to_order in teacher_orders.items():
            # 根據 teacher 的順序重排 student 的塊
            reorder_map = {s0: split_student[0], s1: split_student[1], s2: split_student[2]}
            y_st_nc_aligned = tf.concat([reorder_map[l] for l in to_order], axis=1)
            mae = tf.reduce_mean(tf.abs(y_st_nc_aligned - y_te_orders[name])).numpy()
            if mae < best_mae:
                best_mae, best_perm, best_order = mae, perm, name

    if best_perm is None: raise RuntimeError("[ALIGN] failed to decide student N-order.")
    
    split_index_by_len = {best_perm[0]: 0, best_perm[1]: 1, best_perm[2]: 2}
    reorder_idx = [split_index_by_len[l] for l in teacher_orders[best_order]]
    print(f"✅ Alignment complete: lens_perm={best_perm}, teacher_order={best_order}, MAE={best_mae:.6e}")
    return best_perm, reorder_idx

# ================================== Export Module ==================================
class ExportModule(tf.Module):
    # (此 Class 維持原樣，因為其內部邏輯是為了匹配 C++ code)
    """Export wrapper to:
       - unify to (B,N,C)
       - reorder P3/P4/P5 grid flatten modes (N)
       - apply channel mapping (C)
       - (optional) convert xywh -> ltrb distances in stride units (match Ultralytics decode)
       - final (1,C,N) with name 'output0'
    """
    def __init__(self, model, C, lens_perm, reorder_idx,
                 grid_modes=config.GRID_MODES, porder=config.PORDER, ch_map=config.CHANNEL_MAPPING,
                 xywh_to_ltrb=config.XYWH_TO_LTRB, xywh_is_norm01=config.XYWH_IS_NORMALIZED_01):
        super().__init__()
        self.model = model
        self.C = int(C)
        self.lens_perm = tuple(int(x) for x in lens_perm)
        self.reorder_idx = tuple(int(x) for x in reorder_idx)
        self.grid_modes = tuple(grid_modes)
        self.porder = tuple(porder)
        self.ch_map = tf.constant(list(map(int, ch_map)), tf.int32)
        self.xywh_to_ltrb = bool(xywh_to_ltrb)
        self.xywh_is_norm01 = bool(xywh_is_norm01)

    @tf.function(input_signature=[tf.TensorSpec([1, config.IMGSZ, config.IMGSZ, 3], tf.float32, name="images")])
    def serving_fn(self, x):
        y = self.model(x, training=False)
        if y.shape.rank != 3: raise ValueError(f"export expects rank=3, got {y.shape}")
        out_nc = tf.transpose(y, [0, 2, 1]) if (y.shape[1] == self.C) else y

        segs = tf.split(out_nc, num_or_size_splits=list(self.lens_perm), axis=1)
        segs = [segs[i] for i in self.reorder_idx]
        out_nc = tf.concat(segs, axis=1)

        H3, W3 = config.IMGSZ // 8, config.IMGSZ // 8
        H4, W4 = config.IMGSZ // 16, config.IMGSZ // 16
        H5, W5 = config.IMGSZ // 32, config.IMGSZ // 32
        N3, N4, N5 = H3*W3, H4*W4, H5*W5
        
        def reorder_block_tf(block_BNC, H, W, scan='row', flip_y=False, flip_x=False):
            B, C = tf.shape(block_BNC)[0], tf.shape(block_BNC)[2]
            x_ = tf.reshape(block_BNC, [B, H, W, C])
            if scan == 'col': x_ = tf.transpose(x_, [0, 2, 1, 3])
            if flip_y: x_ = tf.reverse(x_, axis=[1])
            if flip_x: x_ = tf.reverse(x_, axis=[2])
            return tf.reshape(x_, [B, -1, C])

        p3, p4, p5 = tf.split(out_nc, [N3, N4, N5], axis=1)
        m3, m4, m5 = self.grid_modes
        p3r = reorder_block_tf(p3, H3, W3, scan=m3[0], flip_y=bool(m3[1]), flip_x=bool(m3[2]))
        p4r = reorder_block_tf(p4, H4, W4, scan=m4[0], flip_y=bool(m4[1]), flip_x=bool(m4[2]))
        p5r = reorder_block_tf(p5, H5, W5, scan=m5[0], flip_y=bool(m5[1]), flip_x=bool(m5[2]))

        preds_list = [p3r, p4r, p5r]
        out_nc = tf.concat([preds_list[i] for i in self.porder], axis=1)
        out_nc = tf.gather(out_nc, self.ch_map, axis=-1)

        # 輸出是 logits，但 TFLite 需要的是機率，因此在這裡加上 sigmoid
        raw_box, raw_cls, raw_kpt = tf.split(out_nc, [4, config.NUM_CLS, -1], axis=-1)

        box = tf.sigmoid(raw_box)
        cls = tf.sigmoid(raw_cls)
        
        kpt_reshaped = tf.reshape(raw_kpt, [-1, N3+N4+N5, config.NUM_KPT, config.KPT_VALS])
        kpt_xy = kpt_reshaped[..., :2] # xy 是 logits，但在 C++ 端處理
        kpt_v = tf.sigmoid(kpt_reshaped[..., 2:3]) # v 是機率
        kpt = tf.reshape(tf.concat([kpt_xy, kpt_v], axis=-1), [-1, N3+N4+N5, config.NUM_KPT * config.KPT_VALS])
        
        # 產生和 C++ code 預期一致的輸出：[box_prob, cls_prob, kpt_logits_v_prob]
        pred_ultra = tf.concat([box, cls, kpt], axis=-1)
        out_cn = tf.transpose(pred_ultra, [0, 2, 1])
        return {"output0": tf.reshape(out_cn, [1, self.C, -1])}


# ================================== Main Execution ==================================
def main():
    # 0) 初始化設定
    start_time = time.time()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.TFLITE_OUT) / run_timestamp
    output_paths = {
        'logs': output_dir / 'logs',
        'plots': output_dir / 'plots',
        'models': output_dir / 'models',
    }
    for p in output_paths.values():
        p.mkdir(parents=True, exist_ok=True)
    
    output_paths['log_csv'] = output_paths['logs'] / 'training_log.csv'
    output_paths['lr_plot'] = output_paths['plots'] / 'learning_rate_schedule.png'
    output_paths['loss_plot'] = output_paths['plots'] / 'loss_curve.png'
    
    print(f"--- QAT Script Started at {run_timestamp} ---")
    print(f"--- All outputs will be saved in: {output_dir} ---")
    
    enable_gpu_mem_growth()
    setup_mixed_precision()

    # 1) 載入教師模型
    print("\n--- Loading Teacher Model ---")
    teacher, _ = try_load_keras_model(config.EXPORTED_DIR)
    teacher.trainable = False
    print("✅ Teacher model loaded and frozen.")

    # 2) 建立學生模型
    print("\n--- Building Student Model ---")
    student = build_student_qat()
    student.summary(line_length=120)

    # 3) 準備資料集
    print("\n--- Preparing Dataset ---")
    ds, n_files = build_dataset(img_glob=config.REP_DIR, batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    # 4) QAT 微調
    loss_history = run_qat(student, teacher, ds, steps_per_epoch, output_paths)
    plot_and_save_loss_curve(loss_history, output_paths['loss_plot'])

    # 5) 導出前準備
    print("\n--- Preparing for Export ---")
    N3 = (config.IMGSZ // 8)  ** 2
    N4 = (config.IMGSZ // 16) ** 2
    N5 = (config.IMGSZ // 32) ** 2
    C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    print(f"Expected N={N3+N4+N5}, C={C}")

    if hasattr(tfmot.quantization.keras, "strip_quantization"):
        print("[INFO] Stripping quantization wrappers from the model.")
        student_infer = tfmot.quantization.keras.strip_quantization(student)
    else:
        print("[WARN] `strip_quantization` not found; exporting wrapped model.")
        student_infer = student

    # 6) 自動對齊輸出順序
    try:
        sample_one = next(iter(ds))[:1]
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)
    
    lens_perm, reorder_idx = choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, C)

    # 7) 導出 SavedModel
    print("\n--- Exporting SavedModel ---")
    export_mod = ExportModule(
        student_infer, C=C, lens_perm=lens_perm, reorder_idx=reorder_idx,
        grid_modes=config.GRID_MODES, porder=config.PORDER, ch_map=config.CHANNEL_MAPPING,
        xywh_to_ltrb=config.XYWH_TO_LTRB, xywh_is_norm01=config.XYWH_IS_NORMALIZED_01
    )
    
    saved_model_path = str(output_paths['models'] / "qat_saved_model")
    concrete_fn = export_mod.serving_fn.get_concrete_function()
    tf.saved_model.save(export_mod, saved_model_path, signatures=concrete_fn)
    print(f"✅ SavedModel exported to → {saved_model_path}")

    loaded = tf.saved_model.load(saved_model_path)
    sig = loaded.signatures["serving_default"]
    print(" Signature inputs:", sig.structured_input_signature)
    print(" Signature outputs:", sig.structured_outputs)
    
    # 8) 轉換為 TFLite
    print("\n--- Converting to TFLite INT8 ---")
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    conv.experimental_new_converter = True
    try:
        conv.experimental_new_quantizer = True
    except Exception:
        pass

    tfl_bytes = conv.convert()
    tflite_path = str(output_paths['models'] / "best_qat_int8.tflite")
    Path(tflite_path).write_bytes(tfl_bytes)
    print(f"✅ TFLite model written to → {tflite_path}")

    # 9) 檢查 TFLite I/O
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    print(" TFLite inputs:", interp.get_input_details())
    print(" TFLite outputs:", interp.get_output_details())
    
    # 10) 完成
    end_time = time.time()
    print(f"\n--- 🎉 All tasks completed in {((end_time - start_time) / 60):.2f} minutes. ---")


if __name__ == "__main__":
    main()