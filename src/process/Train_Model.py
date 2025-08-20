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
import numpy as np 
import tensorflow_model_optimization as tfmot

from tqdm import tqdm
from importlib import reload
from itertools import permutations

'''
===================================================
Local imports from your project
===================================================
'''
import config
import src.Model_cfg.u_8_s_pose_keras_qat as cfg
from src.Loss_function.loss import (distill_loss_pose, _split_outputs)
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
    def eval_epoch_metrics(x_eval):
        y_t = normalize_teacher_pred(teacher(x_eval, training=False), expected_C=expected_C)  # (B,N,C)
        y_s_raw = align_student_BNC(student(x_eval, training=False))                          # (B,N,C)

        # 分拆
        box_t, obj_t, cls_t, kxy_t, ks_t = _split_outputs(y_t,  config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
        box_s_logits, obj_s, cls_s, kxy_s_logits, ks_s = _split_outputs(y_s_raw, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)

        # 用和蒸餾一致的數域
        box_s = tf.nn.sigmoid(box_s_logits)
        kxy_s = tf.nn.sigmoid(kxy_s_logits)
        p_t_cls = tf.nn.softmax(cls_t)
        p_s_cls = tf.nn.softmax(cls_s)

        mae_box = tf.reduce_mean(tf.abs(box_t - box_s))
        mae_kpt = tf.reduce_mean(tf.abs(kxy_t - kxy_s))
        mae_cls = tf.reduce_mean(tf.abs(p_t_cls - p_s_cls))  # 分佈的 L1 距離（0~2）

        return mae_box, mae_cls, mae_kpt

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
            mae_box_t, mae_cls_t, mae_kpt_t = eval_epoch_metrics(sample_one)
            mae_box_t = float(mae_box_t.numpy()); mae_cls_t = float(mae_cls_t.numpy()); mae_kpt_t = float(mae_kpt_t.numpy())
            
            csv_writer.writerow([e + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}",
                                 f"{mae_box_t:.6f}", f"{mae_cls_t:.6f}", f"{mae_kpt_t:.6f}"])

            print(f"Epoch {e+1}/{config.EPOCHS} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f} | "
                  f"MAE(box/cls/kpt): {mae_box_t:.4f}/{mae_cls_t:.4f}/{mae_kpt_t:.4f}")


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

