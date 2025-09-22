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
import os
import csv
import numpy as np 
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import clone_model
from src.Model_cfg.u_8_s_pose_keras_qat import TeacherCompatHead, U8PoseCompatHead,ChannelAttention, SpatialAttention, CBAM  # 確保可 import 到

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

from src.Loss_function.pose_label_loss_tf import (
    PoseLabelLoss, build_targets_from_labels, build_grid_shapes
)
from src.process.pred_reshape import bcn_to_bnc

from src.process.labels_yolo_pose_tf import parse_label_lines

from src.Loss_function.loss_tf import v8PoseLossTF

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


def build_batch_dict_from_targets(targets_per_image, num_kpt, kpt_vals):
    """
    Args:
      targets_per_image: 長度為 B 的 list；每個元素是 (Gi, 5 + num_kpt*kpt_vals) 的 tf.Tensor 或 numpy
                         每列: [cls, cx, cy, w, h, kp1x, kp1y, kp1v, ...]
      num_kpt: 關鍵點數
      kpt_vals: 每個 keypoint 的欄位數（通常 3: x,y,v）
    Returns:
      batch_dict = {
        'batch_idx': (G,1) int32,
        'cls'      : (G,1) int32,
        'bboxes'   : (G,4) float32 (xywh normalized),
        'keypoints': (G, num_kpt, kpt_vals) float32
      }
    """
    batch_idx_list, cls_list, bboxes_list, kpts_list = [], [], [], []
    for b, arr in enumerate(targets_per_image):
        if arr is None:
            continue
        t = tf.convert_to_tensor(arr)
        if tf.shape(t)[0] == 0:
            continue
        t = tf.cast(t, tf.float32)
        cls = tf.cast(t[:, 0:1], tf.int32)              # (Gi,1)
        xywh = t[:, 1:5]                                 # (Gi,4)
        kpts_flat = t[:, 5:5 + num_kpt * kpt_vals]       # (Gi, num_kpt*kpt_vals)
        kpts = tf.reshape(kpts_flat, [-1, num_kpt, kpt_vals])

        gi = tf.shape(t)[0]
        batch_idx_list.append(tf.fill([gi,1], b))
        cls_list.append(cls)
        bboxes_list.append(xywh)
        kpts_list.append(kpts)

    if len(cls_list) == 0:
        # 空 batch（沒有任何 GT）
        z = tf.zeros([0,1], tf.int32)
        return {'batch_idx': z,
                'cls': z,
                'bboxes': tf.zeros([0,4], tf.float32),
                'keypoints': tf.zeros([0, num_kpt, kpt_vals], tf.float32)}

    batch_idx = tf.concat(batch_idx_list, axis=0)
    cls_all   = tf.concat(cls_list, axis=0)
    boxes_all = tf.concat(bboxes_list, axis=0)
    kpts_all  = tf.concat(kpts_list, axis=0)
    return {'batch_idx': batch_idx,
            'cls': cls_all,
            'bboxes': boxes_all,
            'keypoints': kpts_all}


# 放在 Train_Model.py 顶部合適位置
def ul_tuple_to_BNC(kd_raw, num_cls, num_kpt, kpt_vals):
    """kd_raw 來自 U8PoseCompatHead 的 (feats_list, kpts_list)。
       feats_list: [ (B, no, H, W) * 3 ]，no = nc + 4*reg_max
       kpts_list : [ (B, nk*kv, H, W) * 3 ]，nk*kv = num_kpt * kpt_vals
       輸出: (B, N_total, C_total) 其中 C_total = no + nk*kv
    """
    feats_list, kpts_list = kd_raw
    # 逐層 (B, C, H, W) -> (B, H*W, C)
    def bchw_to_bnc(t):
        # t 是 tf.Tensor
        b, c, h, w = t.shape
        x = tf.transpose(t, perm=[0, 2, 3, 1])      # (B,H,W,C)
        x = tf.reshape(x, [b, h*w, c])              # (B,N,C)
        return x
    feats_bnc = [bchw_to_bnc(t) for t in feats_list]
    kpts_bnc  = [bchw_to_bnc(t) for t in kpts_list]
    # 每層把 (no) 和 (nk*kv) 在 C 維度 concat，再沿 N 維拼三層
    bnc_layers = [tf.concat([f, k], axis=-1) for f, k in zip(feats_bnc, kpts_bnc)]
    return tf.concat(bnc_layers, axis=1)           # (B, N3+N4+N5, no+nk*kv)


def build_student_qat():
    """
    建立雙輸出學生 + 選擇性量化：
      - backbone/neck：量化（QAT）
      - 兩個 head（deploy_head, kd_head）：不量化
    """

    reload(cfg)

    if config.TRAIN_SUPERVISION == 'label':
        # 1) 先建雙頭的 base model（head 輸出都是 (B,C,N)）
        base = cfg.build_u8s_pose_dual(
            input_shape=(config.IMGSZ, config.IMGSZ, 3),
            num_classes=config.NUM_CLS,
            num_kpt=config.NUM_KPT,
            kpt_vals=config.KPT_VALS
        )
        print(f"\nBuild the u8s_pose_dual model...")
    else:
        base = cfg.build_u8s_pose_dual_distill(
            input_shape=(config.IMGSZ, config.IMGSZ, 3),
            num_classes=config.NUM_CLS,
            num_kpt=config.NUM_KPT,
            kpt_vals=config.KPT_VALS
        )
        print(f"\nBuild the u8s_pose_dual_distill model...")

    # 2) 只註解要量化的層；凡是 head 相關一律跳過
    QUANTIZABLE = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.Dense,
        tf.keras.layers.Activation,   # 視需要
        tf.keras.layers.ReLU,         # 視需要
        tf.keras.layers.LeakyReLU,    # 視需要
        tf.keras.layers.PReLU,        # 視需要
        tf.keras.layers.SeparableConv2D,  # 若你有用
    )

    def in_heads(layer_name: str) -> bool:
        # 覆蓋 head 本體與其子層（例：deploy_head/p3_out）
        return ("kd_head" in layer_name) or ("deploy_head" in layer_name)

    def annotate_fn(layer):
        name = layer.name or ""
        # a) 自訂 head 本體：不量化
        if isinstance(layer, TeacherCompatHead):
            return layer
        if isinstance(layer, U8PoseCompatHead):
            return layer
        # b) 任一 head 節點（含其中的 Conv）：不量化
        if in_heads(name):
            return layer
        # c) 非 head 的量化白名單才標註
        if isinstance(layer, QUANTIZABLE):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        # 其他層照原樣返回
        return layer

    annotated = clone_model(base, clone_function=annotate_fn)

    # 3) 在 quantize_scope 內套用（讓 TFMOT 認得自訂層）
    with tfmot.quantization.keras.quantize_scope({
        "TeacherCompatHead": TeacherCompatHead,
        "U8PoseCompatHead" : U8PoseCompatHead,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "CBAM": CBAM
    }):
        student = tfmot.quantization.keras.quantize_apply(annotated)

    # 4) 驗證：兩個輸出仍在，且 head 節點沒有被包 Quantize
    # outs = student.outputs
    # if not isinstance(outs, (list, tuple)) or len(outs) != 2:
    #     raise RuntimeError("Expect dual outputs [deploy_preds, kd_preds] after quantize_apply().")

    # 可選：列出被量化的層數 & 檢查 head 是否未量化
    qlayers = [l for l in student.submodules if "Quantize" in l.__class__.__name__]
    print(f"[CHECK] quantization layers count: {len(qlayers)}")
    for l in student.submodules:
        if "kd_head" in l.name or "deploy_head" in l.name:
            assert "Quantize" not in l.__class__.__name__, f"Head was quantized unexpectedly: {l.name}"

    return student


def run_qat(student, teacher, ds, steps_per_epoch, output_paths):
    """
    ==============================================================================
    執行 QAT 訓練，並把『學生輸出 N 維（P3/P4/P5）』在 train_step 中重排到與 Teacher 一致。
    ==============================================================================
    """
    pose_loss = v8PoseLossTF(
        strides=(8,16,32),
        nc=config.NUM_CLS,
        reg_max=16,
        kpt_shape=(config.NUM_KPT, config.KPT_VALS),
        img_size=config.IMGSZ,      # e.g., 640
        topk=10
    )
    
    print("\n--- Starting QAT Fine-tuning ---")

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

    try:
        sample_batch = next(iter(ds))
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            sample_imgs, label_paths = sample_batch[0], sample_batch[1]
        else:
            sample_imgs, label_paths = sample_batch, None

        labels_list = None
        if label_paths is not None:

            # 轉成 1D 並拿到 numpy
            lp_np = tf.reshape(label_paths, [-1]).numpy()

            # 安全轉字串：bytes -> str；str 原樣返回
            def _to_str(x):
                # x 可能是 bytes 或 str
                if isinstance(x, (bytes, bytearray)):
                    return x.decode("utf-8")
                return str(x)

            paths = [_to_str(p) for p in lp_np]

            labels_list = []
            for p in paths:
                # 用 tf.io.gfile.exists 更健壯
                if tf.io.gfile.exists(p):
                    with tf.io.gfile.GFile(p, "r") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                else:
                    lines = []
                arr = parse_label_lines(lines, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS)
                labels_list.append(arr)

            print(f"\nLoaded label files: {len(labels_list)}")
            for i, arr in enumerate(labels_list):
                print(f"  img[{i}] -> {arr.shape[0]} objects (shape={arr.shape})")

        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
        

    except Exception as e:
        print(f"\n[warn] sample inspect failed: {e}")
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)
    
# =====================================================


# =====================================================

    # lens_perm, reorder_idx = choose_student_split_order(student, teacher, sample_one, N3, N4, N5, expected_C, 
    #                                                     config.NUM_CLS, config.NUM_KPT, config.KPT_VALS, )
    # lens_perm  = tuple(int(x) for x in lens_perm)
    # reorder_idx = [int(x) for x in reorder_idx]

    # print(f" [TRAIN ALIGN] lens_perm={lens_perm}, reorder_idx={reorder_idx}")

    # def _reorder_N_blocks(y_BNC):
    #     s0, s1, s2 = lens_perm   # e.g. (N3, N5, N4)
    #     parts = tf.split(y_BNC, [s0, s1, s2], axis=1)
    #     return tf.concat([parts[reorder_idx[0]], parts[reorder_idx[1]], parts[reorder_idx[2]]], axis=1)

    def concat_scales_to_bcn(scales_list):
        outs = []
        for t in scales_list:  # t: (B, C, H, W)
            B, C, H, W = t.shape
            outs.append(tf.reshape(t, [B, C, H*W]))
        return tf.concat(outs, axis=2)  # (B, C, N_total)

    @tf.function
    def train_step_distill(batch_imgs):
        # 這個函式維持你原本蒸餾的 tf.function 版本（從原 train_step 的 else branch 重用）
        NUM_CLS, NUM_KPT, KPT_VALS = config.NUM_CLS, config.NUM_KPT, config.KPT_VALS
        C = 4 + NUM_CLS + NUM_KPT * KPT_VALS
        huber = tf.keras.losses.Huber(delta=1.0, reduction="sum_over_batch_size")
        L_BOX, L_KXY, L_V, L_CLS = 5.0, 9.0, 1.0, (1.0 if NUM_CLS > 0 else 0.0)
        L_DEPLOY = 2.0

        with tf.GradientTape() as tape:
            y_t_raw = teacher(batch_imgs, training=False)
            y_s_out = student(batch_imgs, training=True)

            kd_raw     = y_s_out[1] if isinstance(y_s_out, (list,tuple)) else y_s_out
            deploy_raw = y_s_out[0] if isinstance(y_s_out, (list,tuple)) else y_s_out

            t_BNC = ensure_BNC_static(y_t_raw, C)
            s_BNC = ensure_BNC_static(kd_raw, C)
            d_BNC = ensure_BNC_static(deploy_raw, C)

            t_box, t_cls, t_kxy, t_v = split_BNC(t_BNC, NUM_CLS, NUM_KPT, KPT_VALS)
            s_box, s_cls, s_kxy, s_v = split_BNC(s_BNC, NUM_CLS, NUM_KPT, KPT_VALS)
            d_box, d_cls, d_kxy, d_v = split_BNC(d_BNC, NUM_CLS, NUM_KPT, KPT_VALS)

            loss_box = L_BOX * huber(s_box, t_box)
            loss_kxy = L_KXY * huber(s_kxy, t_kxy) if (s_kxy is not None) else 0.0
            loss_v   = L_V   * huber(s_v,   t_v  ) if (s_v   is not None) else 0.0
            loss_cls = L_CLS * huber(s_cls, t_cls) if (NUM_CLS > 0) else 0.0

            loss_box_d = L_BOX * huber(d_box, t_box)
            loss_kxy_d = L_KXY * huber(d_kxy, t_kxy) if (s_kxy is not None) else 0.0
            loss_v_d   = L_V   * huber(d_v,   t_v  ) if (s_v   is not None) else 0.0
            loss_cls_d = L_CLS * huber(d_cls, t_cls) if (NUM_CLS > 0) else 0.0

            loss_kd = loss_box + loss_kxy + loss_v + loss_cls
            loss_dep = loss_box_d + loss_kxy_d + loss_v_d + loss_cls_d

            loss = loss_kd + L_DEPLOY * loss_dep
            scaled_loss = opt.get_scaled_loss(loss) if config.USE_AMP else loss

        scaled_grads = tape.gradient(scaled_loss, student.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads) if config.USE_AMP else scaled_grads
        opt.apply_gradients(zip(grads, student.trainable_variables))

        # return metrics (你原本使用的那組)
        mae_box_s = tf.reduce_mean(tf.abs(s_box - t_box))
        mae_kxy_s = tf.reduce_mean(tf.abs(s_kxy - t_kxy)) if (s_kxy is not None) else 0.0
        mae_v_s   = tf.reduce_mean(tf.abs(s_v   - t_v  )) if (s_v   is not None) else 0.0
        mae_cls_s = tf.reduce_mean(tf.abs(s_cls - t_cls)) if (NUM_CLS > 0) else 0.0
        mae_all_s = (mae_box_s + mae_kxy_s + mae_v_s + mae_cls_s)

        mae_box_d = tf.reduce_mean(tf.abs(d_box - t_box))
        mae_kxy_d = tf.reduce_mean(tf.abs(d_kxy - t_kxy)) if (d_kxy is not None) else 0.0
        mae_v_d   = tf.reduce_mean(tf.abs(d_v   - t_v  )) if (d_v   is not None) else 0.0
        mae_cls_d = tf.reduce_mean(tf.abs(d_cls - t_cls)) if (NUM_CLS > 0) else 0.0
        mae_all_d = (mae_box_d + mae_kxy_d + mae_v_d + mae_cls_d)

        return loss, mae_all_s, mae_box_s, mae_cls_s, mae_kxy_s, mae_v_s, mae_all_d

    def train_step_label(batch_imgs, targets_per_image):
        # 非 @tf.function：在這裡直接用 targets（Tensor），做 forward/backward
        NUM_CLS, NUM_KPT, KPT_VALS = config.NUM_CLS, config.NUM_KPT, config.KPT_VALS
        C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

        with tf.GradientTape() as tape:
            
            y_s_out = student(batch_imgs, training=True)
            deploy_feats, deploy_kpts = y_s_out[0] if isinstance(y_s_out, (list, tuple)) else y_s_out
            kd_feats, kd_kpts = y_s_out[1] if isinstance(y_s_out, (list, tuple)) else y_s_out

            # t_BNC = ensure_BNC_static(teacher(batch_imgs, training=False), C)

            batch_dict = build_batch_dict_from_targets(
                targets_per_image, num_kpt=NUM_KPT, kpt_vals=KPT_VALS
            )

            total_loss_kd, logs = pose_loss((kd_feats, kd_kpts), batch_dict)
            total_loss_depoly, logs_depoly = pose_loss((deploy_feats, deploy_kpts), batch_dict)
            total_loss = total_loss_kd + total_loss_depoly 
            

        grads = tape.gradient(total_loss, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        '''
        # parts_* 是 dict
        mae_box = parts_dp['box']
        mae_cls = parts_dp.get('cls', 0.0)
        mae_kxy = parts_dp.get('kpt', 0.0)
        mae_v   = parts_dp.get('kobj', 0.0)
        mae_all = mae_box + mae_cls + mae_kxy + mae_v
        '''
        return total_loss, logs

    # 6) 評估：epoch 末計算 MAE（Student vs Teacher）與 across-N 變異數
    def eval_epoch_metrics(x_eval):
        NUM_CLS  = config.NUM_CLS
        NUM_KPT  = config.NUM_KPT
        KPT_VALS = config.KPT_VALS
        expected_C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

        # Teacher 輸出
        out_teacher = teacher(x_eval, training=False)
        y_t_BNC = ensure_BNC_static(out_teacher, expected_C)
        # y_t_BNC = normalize_teacher_pred(
        #     teacher(x_eval, training=False),
        #     expected_C=expected_C,
        #     num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
        #     batch_imgs=x_eval, target_domain='pixel', return_detected=False
        # )
        
        t_box, t_cls, t_kxy, t_ksc = split_BNC(y_t_BNC, NUM_CLS, NUM_KPT, KPT_VALS)

        # Student KD 分支
        out = student(x_eval, training=False)
        kd_raw = out[1] if isinstance(out, (list, tuple)) and len(out) == 2 else out

        # ★ 保證 BNC + 重排
        kd_BNC = ensure_BNC_static(kd_raw, expected_C)
        # kd_BNC = _reorder_N_blocks(kd_BNC)
        s_box, s_cls, s_kxy, s_ksc = split_BNC(kd_BNC, NUM_CLS, NUM_KPT, KPT_VALS)
        # s_box, s_cls, s_kxy, s_ksc = align_student_to_domain(
        #     kd_BNC, NUM_CLS, NUM_KPT, KPT_VALS, batch_imgs=x_eval, target_domain_is_pixel=False
        # )

        mae_box = tf.reduce_mean(tf.abs(t_box - s_box))
        mae_cls = tf.reduce_mean(tf.abs(t_cls - s_cls))
        mae_kpt = tf.reduce_mean(tf.abs(t_kxy - s_kxy))
        mae_ksc = tf.reduce_mean(tf.abs(t_ksc - s_ksc))

        return mae_box, mae_cls, mae_kpt, mae_ksc

    # 7) 訓練迴圈 + 每 epoch 末評估並寫 CSV
    loss_history = []
    with open(output_paths['log_csv'], 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'loss', 'learning_rate', 'mae_box', 'mae_cls', 'mae_kpt', 'mae_ksc'])

        for e in range(config.EPOCHS):
            epoch_loss_agg = tf.keras.metrics.Mean()
            it = iter(ds)

            progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {e+1}/{config.EPOCHS}", unit="step")
            for _ in progress_bar:

                if config.STOP_REQUESTED:                     # <<< 新增
                    print("[⚠️ Interrupt] Stop requested. Leaving training loop...")   # <<< 新增
                    break                               # <<< 新增

                batch = next(it)
                if isinstance(batch, (list, tuple)):
                    batch_imgs, batch_label_paths = batch[0], batch[1]
                else:
                    batch_imgs, batch_label_paths = batch, None

# ==================================================================
                if config.TRAIN_SUPERVISION == 'label' and batch_label_paths is not None:
                    # safe convert tensor -> python list of str
                    lp_np = tf.reshape(batch_label_paths, [-1]).numpy()
                    def _to_str(x):
                        if isinstance(x, (bytes, bytearray)):
                            return x.decode('utf-8')
                        return str(x)
                    paths = [_to_str(p) for p in lp_np]

                    labels_list = []
                    for p in paths:
                        if tf.io.gfile.exists(p):
                            with tf.io.gfile.GFile(p, "r") as f:
                                lines = [ln.strip() for ln in f if ln.strip()]
                        else:
                            lines = []
                        arr = parse_label_lines(lines, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS)
                        labels_list.append(arr)

                    # build targets -> tensors
                    # 用實際 batch 的 H 來推 grid（若未知就退回 config.IMGSZ）
                    _h = batch_imgs.shape[1]
                    try:
                        imgsz_runtime = int(_h) if _h is not None else int(config.IMGSZ)
                    except Exception:
                        imgsz_runtime = int(config.IMGSZ)

                    '''
                    targets, pos_mask, _ = build_targets_from_labels(
                        batch_labels=labels_list,
                        num_classes=config.NUM_CLS,
                        num_kpt=config.NUM_KPT,
                        kpt_vals=config.KPT_VALS,
                        imgsz=imgsz_runtime,
                        strides=(8,16,32)
                    )

                    targets['pos_mask']  = pos_mask
                    '''
                    # call non-tf train step
                    loss, loss_log = train_step_label(batch_imgs, labels_list)

                    epoch_loss_agg.update_state(loss)
                    progress_bar.set_postfix(loss=f"{loss:.4f}")


                else:
                    # distill path (fast, tf.function)
                    # ensure we pass only images tensor, not tuple
                    loss, mae_all, mae_box, mae_cls, mae_kxy, mae_v, mae_dep_all = train_step_distill(batch_imgs)

                    epoch_loss_agg.update_state(loss)
                    progress_bar.set_postfix(loss=f"{loss:.4f}", MAE_ALL_s=f"{mae_all:.4f}",MAE_BOX=f"{mae_box:.4f}", MAE_CLS=f"{mae_cls:.4f}", MAE_kxy=f"{mae_kxy:.4f}", MAE_v=f"{mae_v:.4f}", MAE_depoly=f"{mae_dep_all:.4f}")

# ==================================================================


            # 如果剛剛收到中斷，直接跳出 epoch 迴圈
            if config.STOP_REQUESTED:
                avg_loss = epoch_loss_agg.result().numpy().item() if epoch_loss_agg.count.numpy() > 0 else float('nan')
                print(f"[⚠️ Interrupt] Early stop at epoch {e+1}. Avg Loss so far: {avg_loss}")
                break

            avg_loss = epoch_loss_agg.result().numpy().item()
            current_lr = schedule((e + 1) * steps_per_epoch).numpy().item()
            loss_history.append(avg_loss)
            '''
            # --- epoch-end diagnostics (MAE + variance) ---
            mae_box_t, mae_cls_t, mae_kpt_t, mae_ksc_t = eval_epoch_metrics(sample_one)
            mae_box_t = float(mae_box_t.numpy()); mae_cls_t = float(mae_cls_t.numpy()); mae_kpt_t = float(mae_kpt_t.numpy()); mae_ksc_t = float(mae_ksc_t.numpy())
            
            csv_writer.writerow([e + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}",
                                 f"{mae_box_t:.6f}", f"{mae_cls_t:.6f}", f"{mae_kpt_t:.6f}", f"{mae_ksc_t:.6f}"])

            print(f"Epoch {e+1}/{config.EPOCHS} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f} | "
                  f"MAE(box/cls/kpt/ksc): {mae_box_t:.4f}/{mae_cls_t:.4f}/{mae_kpt_t:.4f}/{mae_ksc_t:.4f}")
                '''
            csv_writer.writerow([e + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}"])

    print(f"✅ Training finished. Log saved to {output_paths['log_csv']}")
    return loss_history

