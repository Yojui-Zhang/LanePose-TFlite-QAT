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
from src.process.pred_model import (normalize_teacher_pred, split_BNC, 
                                    align_student_to_domain, choose_student_split_order,
                                    ensure_BNC_static)

if config.PLOT_Switch == True:
    from src.process.Plot_Data import plot_and_save_lr_schedule


'''
==================================================================================
Val Model
==================================================================================
'''


def assert_kd_path_not_quantized(model):
    bad = []
    for l in model.layers:
        if "Quantize" in l.__class__.__name__:
            inner = getattr(l, "layer", None)
            if inner is not None and (inner.name or "").startswith("kd_"):
                bad.append(f"{l.name} -> wraps {inner.name}")
        # 保持舊邏輯以防萬一
        elif (l.name or "").startswith("kd_") and "Quantize" in l.__class__.__name__:
            bad.append(l.name)
    if bad:
        raise RuntimeError(f"[KD] 這些 kd_* 層仍被量化包住：{bad}")
    else:
        print("[KD] OK：kd_* 分支未被量化。")


def _ensure_bhwc4(x, imgsz=640):
    """把輸入轉成 (1, imgsz, imgsz, 3) 的 float32，無論你丟進來是單張、已 batch、或 dtype 不對。"""
    x = tf.convert_to_tensor(x)
    if x.shape.rank == 3:
        # (H, W, C) -> (1, H, W, C)
        x = x[tf.newaxis, ...]
    elif x.shape.rank == 4:
        # (B, H, W, C) -> 取前 1 張
        x = x[:1]
    else:
        raise ValueError(f"expect rank 3 or 4 image tensor, got rank={x.shape.rank}, shape={x.shape}")
    # 型別與尺寸
    x = tf.image.resize(x, (imgsz, imgsz))
    x = tf.cast(x, tf.float32)
    # 若你的前處理需要 0~1，這裡一起做（依你訓練的正規化邏輯調整）
    if tf.reduce_max(x) > 1.5:
        x = x / 255.0
    return x

def probe_kd_output_distribution(student_model, dataset, expected_C, imgsz=640):
    """檢查 KD 分支輸出是否為連續浮點值（不是 8-bit 格點）。"""
    # 嘗試從 dataset 取圖片；失敗就用全零假圖
    try:
        batch = next(iter(dataset))
        # dataset 可能回傳 (imgs, labels) 或 dict，做防呆
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        elif isinstance(batch, dict):
            # 依你的 pipeline 調整 key
            imgs = batch.get("image", next(iter(batch.values())))
        else:
            imgs = batch
        imgs = _ensure_bhwc4(imgs, imgsz)
    except Exception as e:
        print(f"[probe] 取 dataset 失敗（{e}），改用假圖。")
        imgs = tf.zeros([1, imgsz, imgsz, 3], dtype=tf.float32)

    # 前向：雙輸出 [deploy_raw, kd_raw]
    out = student_model(imgs, training=False)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        deploy_raw, kd_raw = out
    else:
        raise RuntimeError("student_model 應該回傳 [deploy_preds, kd_preds] 兩個輸出。")

    kd = tf.reshape(kd_raw, [-1, expected_C]).numpy()  # (N, C)
    arr = kd.ravel()
    uniq = np.unique(arr)
    print(f"[KD] sample values={arr.size}, unique={len(uniq)}, min={arr.min():.6f}, max={arr.max():.6f}")
    qsteps = [0.4765625, 0.48828125, 0.5, 0.51171875, 0.5234375]
    hits = {q: (np.isclose(arr, q, atol=1e-6).mean()*100) for q in qsteps}
    print("[KD] 命中常見量化格點(%)：", {f"{k:.6f}": f"{v:.2f}%" for k, v in hits.items()})

'''
==================================================================================
Core Logic
==================================================================================
'''

def build_student_qat():
    """
    建立雙輸出學生 + 選擇性量化：
      - deploy_* 分支 & backbone/neck：量化（QAT）
      - kd_* 分支：保留浮點（未量化輸出，供 KD）
    """
    reload(cfg)
    base = cfg.build_u8s_pose_dual(  # <<== 用雙頭版本
        input_shape=(config.IMGSZ, config.IMGSZ, 3),
        num_classes=config.NUM_CLS,
        num_kpt=config.NUM_KPT,
        kpt_vals=config.KPT_VALS
    )

    from tensorflow.keras.models import clone_model

    def annotate_fn(layer):
        name = layer.name or ""
        # 跳過 kd_* 層（KD 頭保持浮點）
        if name.startswith("kd_") or "/kd_" in name:
            return layer  # 不標註 -> 不量化
        # 其餘標註量化
        try:
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        except Exception:
            return layer

    annotated = clone_model(base, clone_function=annotate_fn)
    student = tfmot.quantization.keras.quantize_apply(annotated)

    # 檢查：兩個輸出仍存在（[deploy_preds, kd_preds]）
    if not isinstance(student.output, (list, tuple)) or len(student.output) != 2:
        raise RuntimeError("Expect dual outputs [deploy_preds, kd_preds] after quantize_apply().")

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
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)  # 取 1 張做對齊（ds 通常有 repeat，不會影響訓練）
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    # 直接用你現有的自動對齊函式
    lens_perm, reorder_idx = choose_student_split_order(student, teacher, sample_one, N3, N4, N5, expected_C, 
                                                        config.NUM_CLS, config.NUM_KPT, config.KPT_VALS, )
    # 轉成 Python 基本型別，方便 tf.function 內部使用
    lens_perm  = tuple(int(x) for x in lens_perm)    # e.g. (N3, N4, N5) 的某種排列
    reorder_idx = [int(x) for x in reorder_idx]      # e.g. [0,1,2] 或 [2,1,0] 等

    print(f" [TRAIN ALIGN] lens_perm={lens_perm}, reorder_idx={reorder_idx}")


    # --- (1) 靜態版保證 (B,N,C) ---
    def _ensure_BNC(y, expected_C):
        return ensure_BNC_static(y, expected_C)

    # --- (2) 依 (lens_perm, reorder_idx) 重排 N 區塊 ---
    def _reorder_N_blocks(y_BNC):
        s0, s1, s2 = lens_perm   # e.g. (N3, N5, N4)
        parts = tf.split(y_BNC, [s0, s1, s2], axis=1)
        return tf.concat([parts[reorder_idx[0]], parts[reorder_idx[1]], parts[reorder_idx[2]]], axis=1)


    @tf.function
    def train_step(batch_imgs):
        NUM_CLS  = config.NUM_CLS
        NUM_KPT  = config.NUM_KPT
        KPT_VALS = config.KPT_VALS
        C = 4 + NUM_CLS + NUM_KPT * KPT_VALS  # 56

        with tf.GradientTape() as tape:
            # fwd
            y_t_raw = teacher(batch_imgs, training=False)
            y_s_out = student(batch_imgs, training=True)
            kd_raw  = y_s_out[1] if isinstance(y_s_out, (list,tuple)) else y_s_out

            # --- Teacher 統一到 0–1（或 pixel） ---
            y_t_unit, t_is_pixel = normalize_teacher_pred(
                y_t_raw, expected_C=C,
                num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
                batch_imgs=batch_imgs, target_domain='unit', return_detected=True
            )
            t_box, t_cls, t_kxy, t_ksc = split_BNC(y_t_unit, NUM_CLS, NUM_KPT, KPT_VALS)

            # ========== ★★ 關鍵：Student KD 先保證 (B,N,C) 再「重排 N」 ★★ ==========
            kd_BNC = _ensure_BNC(kd_raw, C)          # (B,N,C)
            kd_BNC = _reorder_N_blocks(kd_BNC)       # (B,N,C) 與 Teacher 同順序
            s_box, s_cls_logit, s_kxy, s_ksc_logit = align_student_to_domain(
                kd_BNC, NUM_CLS, NUM_KPT, KPT_VALS, batch_imgs=batch_imgs, target_domain_is_pixel=False
            )

            # --- KD Loss ---
            kd_cls = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=t_cls, logits=s_cls_logit)
            )
            kd_ksc = (tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=t_ksc, logits=s_ksc_logit)
            ) if t_ksc is not None else 0.0)
            kd_box = tf.reduce_mean(tf.losses.huber(t_box, s_box))
            kd_kpt = tf.reduce_mean(tf.losses.huber(t_kxy, s_kxy))

            # 注意你用的是 W_KPT_V，請確認名稱是否正確；若你配置檔叫 W_KPT_SC，請改回去
            loss_kd = (config.W_BOX * kd_box
                    + config.W_CLS * kd_cls
                    + config.W_KPT_XY * kd_kpt
                    + (config.W_KPT_V * kd_ksc if t_ksc is not None else 0.0))

            # --- 一致性（deploy 也要「保證 BNC → 重排 N → 數域對齊」） ---
            alpha = 0.05
            if isinstance(y_s_out, (list,tuple)):
                deploy_raw = y_s_out[0]
                d_BNC = _ensure_BNC(deploy_raw, C)
                d_BNC = _reorder_N_blocks(d_BNC)
                d_box, d_cls_logit, d_kxy, d_ksc_logit = align_student_to_domain(
                    d_BNC, NUM_CLS, NUM_KPT, KPT_VALS,
                    batch_imgs=batch_imgs, target_domain_is_pixel=False
                )
                loss_cons = alpha * (
                    tf.reduce_mean(tf.abs(d_box - tf.stop_gradient(s_box))) +
                    tf.reduce_mean(tf.abs(d_kxy - tf.stop_gradient(s_kxy))) +
                    tf.reduce_mean(tf.abs(tf.nn.sigmoid(d_cls_logit) - tf.stop_gradient(tf.nn.sigmoid(s_cls_logit))))
                    + (tf.reduce_mean(tf.abs(tf.nn.sigmoid(d_ksc_logit) - tf.stop_gradient(tf.nn.sigmoid(s_ksc_logit))))
                    if t_ksc is not None else 0.0)
                )
            else:
                loss_cons = 0.0

            loss = loss_kd + loss_cons

            # ========== ★★ AMP 正確用法（自訂 loop 必須手動 scale/unscale） ★★ ==========
            scaled_loss = opt.get_scaled_loss(loss) if config.USE_AMP else loss

        scaled_grads = tape.gradient(scaled_loss, student.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads) if config.USE_AMP else scaled_grads
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # 6) 評估：epoch 末計算 MAE（Student vs Teacher）與 across-N 變異數
    def eval_epoch_metrics(x_eval):
        NUM_CLS  = config.NUM_CLS
        NUM_KPT  = config.NUM_KPT
        KPT_VALS = config.KPT_VALS
        expected_C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

        # Teacher → 0–1
        y_t_unit = normalize_teacher_pred(
            teacher(x_eval, training=False),
            expected_C=expected_C,
            num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
            batch_imgs=x_eval, target_domain='unit', return_detected=False
        )
        t_box, t_cls, t_kxy, t_ksc = split_BNC(y_t_unit, NUM_CLS, NUM_KPT, KPT_VALS)

        # Student KD 分支
        out = student(x_eval, training=False)
        kd_raw = out[1] if isinstance(out, (list, tuple)) and len(out) == 2 else out

        # ★ 保證 BNC + 重排
        kd_BNC = _ensure_BNC(kd_raw, expected_C)
        kd_BNC = _reorder_N_blocks(kd_BNC)
        s_box, s_cls_logit, s_kxy, s_ksc_logit = align_student_to_domain(
            kd_BNC, NUM_CLS, NUM_KPT, KPT_VALS, batch_imgs=x_eval, target_domain_is_pixel=False
        )

        p_t_cls = tf.where((t_cls < 0.) | (t_cls > 1.), tf.sigmoid(t_cls), t_cls)
        p_s_cls = tf.sigmoid(s_cls_logit)

        mae_box = tf.reduce_mean(tf.abs(t_box - s_box))
        mae_kpt = tf.reduce_mean(tf.abs(t_kxy - s_kxy))
        mae_cls = tf.reduce_mean(tf.abs(p_t_cls - p_s_cls))
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

