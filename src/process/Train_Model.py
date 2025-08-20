'''
===================================================
Tensor 版本強制設定
===================================================
'''
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

'''
===================================================
import Depance file
===================================================
'''

import csv
from importlib import reload
from itertools import permutations

import numpy as np 
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

'''
===================================================
Local imports from your project
===================================================
'''
import config
import src.Model_cfg.u_8_s_pose_keras_qat as cfg
from src.Loss_function.loss import distill_loss_pose
from src.process.pred_model import normalize_teacher_pred

if config.PLOT_Switch == True:
    from src.process.Plot_Data import plot_and_save_lr_schedule

'''
==================================================================================
Core Logic
==================================================================================
'''

def build_student_qat():
    """
    ==============================================================================
    建立並量化學生模型。
    ==============================================================================
    """
    
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
    """
    ==============================================================================
    執行 QAT 訓練，並把『學生輸出 N 維（P3/P4/P5）』在 train_step 中重排到與 Teacher 一致。
    ==============================================================================
    """
    
    print("\n--- Starting QAT Fine-tuning ---")
    # 1) 視 config.BNSTOP__ 來凍/解凍 BN
    for l in student.submodules:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = config.BNSTOP__
    print(f" BN layers trainable = {config.BNSTOP__}.")

    # 2) 學習率排程
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

    total_steps  = max(1, config.EPOCHS * steps_per_epoch)
    warmup_steps = min(1000, max(1, total_steps // 10))
    schedule     = WarmupCosine(config.base_lr, config.end_lr, warmup_steps, total_steps)
    opt = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=config.momentum, nesterov=True, clipnorm=1.0)
    if config.USE_AMP:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    print(f" Optimizer: SGD + WarmupCosine (total steps: {total_steps}).")
    
    if config.PLOT_Switch == True:
        plot_and_save_lr_schedule(schedule, total_steps, output_paths['lr_plot'])

    # 3) 『訓練前』先用一張樣本決定學生輸出 N 維的重排方式（與 teacher 對齊）
    expected_C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    H3, W3 = config.IMGSZ // 8,  config.IMGSZ // 8
    H4, W4 = config.IMGSZ // 16, config.IMGSZ // 16
    H5, W5 = config.IMGSZ // 32, config.IMGSZ // 32
    N3, N4, N5 = H3 * W3, H4 * W4, H5 * W5

    try:
        sample_one = next(iter(ds))[:1]  # 取 1 張做對齊（ds 通常有 repeat，不會影響訓練）
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    # 直接用你現有的自動對齊函式
    lens_perm, reorder_idx = choose_student_split_order(student, teacher, sample_one, N3, N4, N5, expected_C)
    # 轉成 Python 基本型別，方便 tf.function 內部使用
    lens_perm  = tuple(int(x) for x in lens_perm)    # e.g. (N3, N4, N5) 的某種排列
    reorder_idx = [int(x) for x in reorder_idx]      # e.g. [0,1,2] 或 [2,1,0] 等

    print(f" [TRAIN ALIGN] lens_perm={lens_perm}, reorder_idx={reorder_idx}")

    # 4) 定義把學生輸出重排到與 Teacher 一致的工具（在 tf.function 中用）
    def align_student_BNC(y_s_raw):
        """y_s_raw: (B,C,N) 或 (B,N,C) 任何一種；回傳對齊後的 (B,N,C)。"""
        # 先確保是 (B,N,C)
        y_s_nc = tf.transpose(y_s_raw, [0, 2, 1]) if (y_s_raw.shape.rank == 3 and int(y_s_raw.shape[1]) == expected_C) else y_s_raw
        # 依 lens_perm 切三段 (對應 P3/P4/P5)，再依 reorder_idx 重排後 concat 回 (B,N,C)
        segs = tf.split(y_s_nc, num_or_size_splits=list(lens_perm), axis=1)
        segs = [segs[i] for i in reorder_idx]
        return tf.concat(segs, axis=1)  # (B,N,C)

    # 5) 訓練步驟：把『學生輸出』先對齊 N 維再算蒸餾 loss
    @tf.function
    def train_step(batch_imgs):
        # Teacher → 正規化到 (B,N,C)，語義 0~1
        y_t = teacher(batch_imgs, training=False)
        y_t = tf.stop_gradient(normalize_teacher_pred(y_t, expected_C=expected_C))  # (B,N,C)

        with tf.GradientTape() as tape:
            y_s_raw = student(batch_imgs, training=True)     # (B,?,?)，可能是 (B,C,N) 或 (B,N,C)
            y_s_aln = align_student_BNC(y_s_raw)             # (B,N,C) 與 teacher 對齊
            loss = distill_loss_pose(y_t, y_s_aln, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
            scaled_loss = opt.get_scaled_loss(loss) if config.USE_AMP else loss

        scaled_grads = tape.gradient(scaled_loss, student.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads) if config.USE_AMP else scaled_grads
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # 6) 評估：epoch 末計算 MAE（Student vs Teacher）與 across-N 變異數
    @tf.function
    def eval_epoch_metrics(x_eval):
        # teacher normalized (B,N,C)
        y_t = teacher(x_eval, training=False)
        y_t = normalize_teacher_pred(y_t, expected_C=expected_C)
        # student aligned (B,N,C)
        y_s_raw = student(x_eval, training=False)
        y_s = align_student_BNC(y_s_raw)
        # MAE by slices
        c0_box, c1_box = 0, 4
        c0_cls, c1_cls = 4, 4 + config.NUM_CLS
        c0_kpt, c1_kpt = c1_cls, expected_C
        mae_box = tf.reduce_mean(tf.abs(y_s[..., c0_box:c1_box] - y_t[..., c0_box:c1_box]))
        mae_cls = tf.reduce_mean(tf.abs(y_s[..., c0_cls:c1_cls] - y_t[..., c0_cls:c1_cls]))
        mae_kpt = tf.reduce_mean(tf.abs(y_s[..., c0_kpt:c1_kpt] - y_t[..., c0_kpt:c1_kpt]))
        # variance across N for a few probe channels (student)
        probe_idx = tf.constant([0, 1, c0_cls, c1_cls - 1, expected_C - 1], dtype=tf.int32)
        # y_s: (1, N, C) -> gather channels then var on axis=1 (N)
        ys_probe = tf.gather(y_s[0], probe_idx, axis=1)  # shape (N, len)
        # 上面那行會得到 (N,5)；我們要每個通道 across-N 的變異數
        vars_vec = tf.math.reduce_variance(ys_probe, axis=0)  # (5,)
        return mae_box, mae_cls, mae_kpt, vars_vec  # tensors

    # 7) 訓練迴圈 + 每 epoch 末評估並寫 CSV
    loss_history = []
    with open(output_paths['log_csv'], 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'loss', 'learning_rate', 'mae_box', 'mae_cls', 'mae_kpt'])

        for e in range(config.EPOCHS):
            epoch_loss_agg = tf.keras.metrics.Mean()
            it = iter(ds)

            progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {e+1}/{config.EPOCHS}", unit="step")
            for _ in progress_bar:

                if config.STOP_REQUESTED:                     # <<< 新增
                    print("[⚠️ Interrupt] Stop requested. Leaving training loop...")   # <<< 新增
                    break                               # <<< 新增

                imgs = next(it)
                loss = train_step(imgs)
                epoch_loss_agg.update_state(loss)
                progress_bar.set_postfix(loss=f"{loss:.4f}")

            # 如果剛剛收到中斷，直接跳出 epoch 迴圈
            if config.STOP_REQUESTED:                         # <<< 新增
                avg_loss = epoch_loss_agg.result().numpy().item() if epoch_loss_agg.count.numpy() > 0 else float('nan')  # <<< 新增
                print(f"[⚠️ Interrupt] Early stop at epoch {e+1}. Avg Loss so far: {avg_loss}")                          # <<< 新增
                break                                   # <<< 新增


            avg_loss = epoch_loss_agg.result().numpy().item()
            current_lr = schedule((e + 1) * steps_per_epoch).numpy().item()
            loss_history.append(avg_loss)

            # --- epoch-end diagnostics (MAE + variance) ---
            mae_box_t, mae_cls_t, mae_kpt_t, vars_vec = eval_epoch_metrics(sample_one)
            mae_box_t = float(mae_box_t.numpy()); mae_cls_t = float(mae_cls_t.numpy()); mae_kpt_t = float(mae_kpt_t.numpy())
            vars_vec  = vars_vec.numpy().tolist()  # [var_x, var_y, var_cls0, var_cls_last, var_last]

            csv_writer.writerow([e + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}",
                                 f"{mae_box_t:.6f}", f"{mae_cls_t:.6f}", f"{mae_kpt_t:.6f}"])

            print(f"Epoch {e+1}/{config.EPOCHS} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f} | "
                  f"MAE(box/cls/kpt): {mae_box_t:.4f}/{mae_cls_t:.4f}/{mae_kpt_t:.4f}")
            print(f"  Var across N (x,y,cls0,clsLast,lastCh): "
                  f"{vars_vec[0]:.3e}, {vars_vec[1]:.3e}, {vars_vec[2]:.3e}, {vars_vec[3]:.3e}, {vars_vec[4]:.3e}")

    print(f"✅ Training finished. Log saved to {output_paths['log_csv']}")
    return loss_history

def choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, expected_C):
    """
    ==============================================================================
    自動偵測學生模型 P3,P4,P5 輸出塊的最佳排列順序。
    ==============================================================================
    """

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

'''
==================================================================================
Export Module 
==================================================================================
'''
class ExportModule(tf.Module):
    # (此 Class 維持原樣，因為其內部邏輯是為了匹配 C++ code)
    """
    ==============================================================================
       Export wrapper to:
       - unify to (B,N,C)
       - reorder P3/P4/P5 grid flatten modes (N)
       - apply channel mapping (C)
       - (optional) convert xywh -> ltrb distances in stride units (match Ultralytics decode)
       - final (1,C,N) with name 'output0'
    ==============================================================================
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
        # kpt_xy = kpt_reshaped[..., :2] # xy 是 logits，但在 C++ 端處理
        kpt_xy = tf.sigmoid(kpt_reshaped[..., :2]) # 新的程式碼，輸出歸一化座標
        kpt_v = tf.sigmoid(kpt_reshaped[..., 2:3]) # v 是機率
        kpt = tf.reshape(tf.concat([kpt_xy, kpt_v], axis=-1), [-1, N3+N4+N5, config.NUM_KPT * config.KPT_VALS])
        
        # 產生和 C++ code 預期一致的輸出：[box_prob, cls_prob, kpt_logits_v_prob]
        pred_ultra = tf.concat([box, cls, kpt], axis=-1)
        out_cn = tf.transpose(pred_ultra, [0, 2, 1])
        return {"output0": tf.reshape(out_cn, [1, self.C, -1])}

'''
==================================================================================
One-shot diagnostics helpers 
==================================================================================
'''

def _run_tflite_infer(tflite_path: str, x_bhwc: tf.Tensor) -> np.ndarray:
    """
    ==============================================================================
    Run TFLite and return (1, C, N) float32.
    ==============================================================================
    """

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    idt = interp.get_input_details()[0]
    odt = interp.get_output_details()[0]
    # set input
    xin = x_bhwc.numpy().astype(np.float32)
    interp.set_tensor(idt["index"], xin)
    interp.invoke()
    y = interp.get_tensor(odt["index"]).astype(np.float32)  # expect (1, C, N) or (1, N, C)
    # robust to layout
    if y.ndim != 3:
        raise RuntimeError(f"Unexpected TFLite output rank: {y.shape}")
    Ccand = [y.shape[1], y.shape[2]]
    # Let caller decide C, so we just return [1,C,N] if possible
    if y.shape[1] < y.shape[2]:
        return y  # (1, C, N)
    else:
        return np.transpose(y, (0, 2, 1))  # -> (1, C, N)


def run_diagnostics_once(
    export_mod: "ExportModule",
    teacher: tf.keras.Model,
    tflite_path: str,
    sample_one: tf.Tensor,
    C: int,
    NUM_CLS: int,
    NUM_KPT: int,
    KPT_VALS: int,
):
    """
    ==============================================================================
    Do three checks on the same image:
      1) Keras(student serving_fn) vs TFLite(student)  -> MAE box/cls/kpt
      2) Teacher(normalized) vs Student(serving_fn)    -> MAE box/cls/kpt
      3) Variance across N for several channels        -> detect plateau
    ==============================================================================
    """
    # ---- 1) Keras student (serving_fn -> [1,C,N]) ----
    out_keras = export_mod.serving_fn(sample_one)["output0"].numpy()  # (1, C, N)

    # ---- 2) TFLite student (-> [1,C,N]) ----
    out_tfl = _run_tflite_infer(tflite_path, sample_one)              # (1, C, N)

    # ---- 3) Teacher normalized (B,N,C) -> transpose to (1,C,N) ----
    expected_C = C
    te_nc = normalize_teacher_pred(teacher(sample_one, training=False), expected_C=expected_C).numpy()  # (1,N,C)
    te_cn = np.transpose(te_nc, (0, 2, 1))  # (1, C, N)

    # sanity shape
    assert out_keras.shape[1] == C and out_tfl.shape[1] == C and te_cn.shape[1] == C, \
        f"Channel mismatch: keras={out_keras.shape}, tflite={out_tfl.shape}, teacher={te_cn.shape}"

    # slices
    c0_box = 0
    c1_box = 4
    c0_cls = 4
    c1_cls = 4 + NUM_CLS
    c0_kpt = c1_cls
    c1_kpt = C

    def mae_slice(a, b, s0, s1):
        return float(np.mean(np.abs(a[:, s0:s1, :] - b[:, s0:s1, :])))

    # ---- MAE: TFLite vs Keras (量化/匯出一致性) ----
    mae_box_tk = mae_slice(out_tfl, out_keras, c0_box, c1_box)
    mae_cls_tk = mae_slice(out_tfl, out_keras, c0_cls, c1_cls)
    mae_kpt_tk = mae_slice(out_tfl, out_keras, c0_kpt, c1_kpt)

    # ---- MAE: Student(Keras) vs Teacher (對齊/訓練程度) ----
    mae_box_st = mae_slice(out_keras, te_cn, c0_box, c1_box)
    mae_cls_st = mae_slice(out_keras, te_cn, c0_cls, c1_cls)
    mae_kpt_st = mae_slice(out_keras, te_cn, c0_kpt, c1_kpt)

    # ---- Variance across N（檢查是否又變成常數） ----
    def var_across_N(y_cn, ch, limit=500):
        N = y_cn.shape[2]
        n = min(N, limit)
        seg = y_cn[0, ch, :n]
        return float(np.var(seg))

    probe_channels = {
        "box_x": 0,
        "box_y": 1,
        "cls_0": c0_cls,
        "cls_last": c1_cls - 1,
        "kpt_last": C - 1,
    }

    print("\n========== One-shot Diagnostics ==========")
    print(f"[Shapes] keras={out_keras.shape}, tflite={out_tfl.shape}, teacher={te_cn.shape}")
    print("---- MAE: Student TFLite  vs Student Keras (serving_fn) ----")
    print(f"  box: {mae_box_tk:.6f} | cls: {mae_cls_tk:.6f} | kpt: {mae_kpt_tk:.6f}")
    print("---- MAE: Student Keras  vs Teacher (normalized) ----")
    print(f"  box: {mae_box_st:.6f} | cls: {mae_cls_st:.6f} | kpt: {mae_kpt_st:.6f}")

    print("---- Variance across N (first 500 anchors) ----")
    for name, ch in probe_channels.items():
        v_k = var_across_N(out_keras, ch)
        v_t = var_across_N(out_tfl, ch)
        print(f"  ch {name:>8s} | var Keras={v_k:.3e} | var TFLite={v_t:.3e}")

    # quick hints
    print("\n[Hints]")
    print("  • TFLite≈Keras: 如果三段 MAE < 0.01~0.02，量化/匯出基本 OK。")
    print("  • Student vs Teacher: 如果 MAE 持續 > 0.05（特別是 box/kpt），優先檢查：")
    print("      (1) 訓練時是否已做與匯出一致的 N 對齊（P3/P4/P5 順序、row/col、flip、通道映射）")
    print("      (2) 蒸餾前景選樣（conf 門檻或 Top-K），避免 8400 背景沖淡梯度")
    print("      (3) BN 是否保持可訓、學習率排程是否合理")
    print("  • Variance: 若某些通道 var ~ 0（例如 < 1e-6），表示該通道在 N 維幾乎常數，")
    print("      需檢查監督是否對齊、或模型是否未學到該通道。")
    print("==========================================\n")
