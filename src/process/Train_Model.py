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
import math
import pandas as pd
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

from src.process.labels_yolo_pose_tf import parse_label_lines, letterbox_adjust_yolo_pose


from src.Loss_function.loss_tf import pose_loss_from_labels

if config.PLOT_Switch == True:
    from src.process.Plot_Data import (plot_and_save_lr_schedule, save_gt_and_plot, save_pred_and_plot)


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

class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        # 建立影子變數 (Shadow Variables)
        self.shadow = [tf.Variable(w, trainable=False) for w in model.weights]

    def update(self, model):
        d = self.decay
        for s, w in zip(self.shadow, model.weights):
            # [Fix] 只對浮點數變數做 EMA 運算
            if w.dtype.is_floating:
                s.assign(d * s + (1.0 - d) * w)
            else:
                # [Fix] 對於整數變數 (如 int32, int64)，直接複製值，不進行加權平均
                s.assign(w)

    def apply_to(self, model):
        self.backup = [tf.identity(w) for w in model.weights]
        for w, s in zip(model.weights, self.shadow):
            w.assign(s)

    def restore(self, model):
        for w, b in zip(model.weights, self.backup):
            w.assign(b)
        self.backup = None


def get_anchors(h, w, dtype=tf.float32):
    """
    根據 Feature Map 的高 (h) 和寬 (w) 產生歸一化網格。
    """
    # 產生 0~1 的網格
    # y 軸有 h 格, x 軸有 w 格
    # stride_h = 1.0 / h
    # stride_w = 1.0 / w
    
    # 轉換 h, w 為 float 以進行除法
    h_f = tf.cast(h, dtype)
    w_f = tf.cast(w, dtype)
    
    # 產生中心點座標
    # grid_y: 0.5, 1.5, ..., h-0.5 -> 除以 h -> 歸一化
    grid_y = (tf.range(h, dtype=dtype) + 0.5) / h_f
    grid_x = (tf.range(w, dtype=dtype) + 0.5) / w_f
    
    # Meshgrid: gy (h, w), gx (h, w)
    gy, gx = tf.meshgrid(grid_y, grid_x, indexing='ij')
    
    # Flatten -> (N, 1)
    cx = tf.reshape(gx, (-1, 1))
    cy = tf.reshape(gy, (-1, 1))
    
    # Anchor 寬高 (設為一個 Grid 大小)
    cw = tf.ones_like(cx) / w_f
    ch = tf.ones_like(cy) / h_f
    
    # 合併成 (N, 4) -> [cx, cy, cw, ch]
    anchors = tf.concat([cx, cy, cw, ch], axis=-1)
    
    return anchors

def generate_anchors_from_output(output, data_format='channels_last'):
    """
    根據模型輸出自動產生對應的 Anchors。
    data_format: 'channels_last' (NHWC) 或 'channels_first' (NCHW)
    """
    # 1. 遞迴處理 List/Tuple (多尺度)
    if isinstance(output, (list, tuple)):
        anchors_list = []
        for feat in output:
            # 遞迴傳遞 data_format
            anchors_list.append(generate_anchors_from_output(feat, data_format=data_format))
        return tf.concat(anchors_list, axis=0)
        
    # 2. 處理 Tensor
    shape = tf.shape(output)
    rank = len(output.shape)
    
    if rank == 4: 
        if data_format == 'channels_last': # NHWC (Standard Keras)
            H, W = shape[1], shape[2]
        else: # channels_first / NCHW (Your KD Head)
            H, W = shape[2], shape[3]
            
        return get_anchors(H, W, dtype=output.dtype)
        
    elif rank == 3: # (B, N, C) - 已經被 Flatten 過的
        N = shape[1]
        side = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)
        return get_anchors(side, side, dtype=output.dtype)
        
    else:
        raise ValueError(f"Unsupported output shape rank: {rank}")

def build_batch_dict_from_targets(targets, num_kpt, kpt_vals):
    """
    將 list of tensors (每個 image 的 targets) 轉換為 batch dictionary (padded)。
    同時計算 'num_objects' 供 Loss function 使用。
    """
    # 1. 取得每張圖的物件數量 (這是 Tensor Scalar 的 List)
    lengths = []
    for t in targets:
        lengths.append(tf.shape(t)[0]) # tf.shape 回傳的是 Tensor
    
    # 2. 計算 Max Length (需先用 stack 轉成 Tensor 向量)
    if len(lengths) > 0:
        lengths_tensor = tf.stack(lengths)
        max_len = tf.reduce_max(lengths_tensor)
        # 防止 max_len 為 0 (若整個 batch 都沒物件)
        max_len = tf.maximum(max_len, 1)
    else:
        # 空 batch 的極端情況
        max_len = tf.constant(1, dtype=tf.int32)

    batch_bboxes = []
    batch_cls = []
    batch_kpts = []
    batch_indices = []
    batch_num_objs = [] 
    batch_valid_mask = []

    for i, t in enumerate(targets):
        num_obj = tf.shape(t)[0]
        batch_num_objs.append(num_obj) # 加入 list (Tensor)

        # 若沒有任何物件 (Empty image)
        if num_obj == 0:
            cls_real = tf.zeros((0, 1), dtype=tf.float32)
            box_real = tf.zeros((0, 4), dtype=tf.float32)
            kpt_real = tf.zeros((0, num_kpt, kpt_vals), dtype=tf.float32)

            mask_real = tf.zeros((0,), dtype=tf.float32)
        else:
            cls_real = t[:, 0:1]       # (M, 1)
            box_real = t[:, 1:5]       # (M, 4)
            kpt_data = t[:, 5:]        # (M, K*V)
            kpt_real = tf.reshape(kpt_data, (num_obj, num_kpt, kpt_vals))

            mask_real = tf.ones((num_obj,), dtype=tf.float32)

        # --- Padding ---
        pad_len = max_len - num_obj
        
        # 定義 padding: [[top, bottom], [left, right]]
        paddings_cls = [[0, pad_len], [0, 0]]
        paddings_box = [[0, pad_len], [0, 0]]
        paddings_kpt = [[0, pad_len], [0, 0], [0, 0]]

        paddings_mask = [[0, pad_len]]

        cls_padded = tf.pad(cls_real, paddings_cls, mode='CONSTANT', constant_values=0)
        box_padded = tf.pad(box_real, paddings_box, mode='CONSTANT', constant_values=0)
        kpt_padded = tf.pad(kpt_real, paddings_kpt, mode='CONSTANT', constant_values=0)

        mask_padded = tf.pad(mask_real, paddings_mask, mode='CONSTANT', constant_values=0)

        # Batch Index
        b_idx = tf.fill([max_len, 1], tf.cast(i, tf.float32))
        
        batch_cls.append(cls_padded)
        batch_bboxes.append(box_padded)
        batch_kpts.append(kpt_padded)
        batch_indices.append(b_idx)
        batch_valid_mask.append(mask_padded)

    # 3. 堆疊成 Batch Tensor
    out_dict = {
        'bboxes': tf.stack(batch_bboxes, axis=0),
        'cls': tf.stack(batch_cls, axis=0),
        'keypoints': tf.stack(batch_kpts, axis=0),

        'valid_mask': tf.stack(batch_valid_mask, axis=0), # [新增] 對應 loss: batch_dict["valid_mask"],

        'batch_idx': tf.stack(batch_indices, axis=0),
        # ✅ 修改處：使用 tf.stack 代替 tf.constant
        # batch_num_objs 是一個 Tensor list，必須用 stack 合併
        'num_objects': tf.expand_dims(tf.stack(batch_num_objs), axis=-1) 
    }
    
    return out_dict

def build_batch_dict_from_padded_labels(batch_labels, num_kpt, kpt_vals):
    """
    batch_labels: (B, M, D) float32
      D = 5 + num_kpt * kpt_vals
      format per row: [cls, cx, cy, w, h, kpts_flat...]
      padding rows are zeros -> invalid
    returns batch_dict compatible with pose_loss_from_labels():
      bboxes:     (B,M,4)
      cls:        (B,M,1)
      keypoints:  (B,M,K,V)
      valid_mask: (B,M) float32 0/1
      num_objects:(B,1) int32
      batch_idx:  (B,M,1) float32 (optional, for your existing debug utilities)
    """
    batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

    B = tf.shape(batch_labels)[0]
    M = tf.shape(batch_labels)[1]

    cls = batch_labels[..., 0:1]      # (B,M,1)
    bboxes = batch_labels[..., 1:5]   # (B,M,4)

    kpt_flat = batch_labels[..., 5:]  # (B,M,K*V)
    keypoints = tf.reshape(kpt_flat, [B, M, num_kpt, kpt_vals])

    # valid if w>0 and h>0 (same logic you used in numpy path)
    valid = tf.logical_and(batch_labels[..., 3] > 0.0, batch_labels[..., 4] > 0.0)
    valid_mask = tf.cast(valid, tf.float32)  # (B,M)

    num_objects = tf.reduce_sum(tf.cast(valid, tf.int32), axis=1, keepdims=True)  # (B,1)

    batch_idx = tf.broadcast_to(
        tf.reshape(tf.range(B, dtype=tf.float32), [B, 1, 1]),
        [B, M, 1]
    )

    return {
        "bboxes": bboxes,
        "cls": cls,
        "keypoints": keypoints,
        "valid_mask": valid_mask,
        "num_objects": num_objects,
        "batch_idx": batch_idx,
    }


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

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

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


    # base = cfg.build_u8s_pose_dual_distill(
    #     input_shape=(config.IMGSZ, config.IMGSZ, 3),
    #     num_classes=config.NUM_CLS,
    #     num_kpt=config.NUM_KPT,
    #     kpt_vals=config.KPT_VALS
    # )
    # print(f"\nBuild the u8s_pose_dual_distill model...")
    
    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

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






def run_qat(student, teacher, ds, steps_per_epoch, output_paths,
            class_weights=None,
            ds_val=None,
            val_steps=0):
    """
    2026 update (label-focused):
    - 在 label 監督下，loss 的 decode 與 export 完全一致：box/cls/kpt(xy,v) 都以 sigmoid 映射到 0..1 domain。
    - 支援可選 val dataset；以 val_total_loss 做 best checkpoint（比 teacher-mae 更合理）。
    - 仍保留 distill 路徑（如果 config.TRAIN_SUPERVISION != 'label'）。
    """
    # ------------------------------------------------------------
    # Common setup
    # ------------------------------------------------------------
    NUM_CLS, NUM_KPT, KPT_VALS = int(config.NUM_CLS), int(config.NUM_KPT), int(config.KPT_VALS)
    C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

    os.makedirs(output_paths['logs'], exist_ok=True)
    log_csv = output_paths.get('log_csv', os.path.join(output_paths['logs'], "training_log.csv"))

    # learning-rate schedule: warmup + cosine
    base_lr = float(getattr(config, "base_lr", 1e-3))
    end_lr  = float(getattr(config, "end_lr", 1e-4))
    total_steps = int(config.EPOCHS) * int(steps_per_epoch)
    alpha = max(0.0, min(1.0, end_lr / base_lr)) if base_lr > 0 else 0.0

    cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=max(1, total_steps),
        alpha=alpha
    )

    warmup_steps = int(getattr(config, "WARMUP_STEPS", max(100, steps_per_epoch // 2)))
    class WarmupThen(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, base_schedule):
            super().__init__()
            self.warmup_steps = tf.constant(int(warmup_steps), tf.float32)
            self.base_schedule = base_schedule

        def __call__(self, step):
            step_f = tf.cast(step, tf.float32)
            lr = self.base_schedule(step_f)
            w = tf.clip_by_value(step_f / tf.maximum(self.warmup_steps, 1.0), 0.0, 1.0)
            return lr * w

    lr_schedule = WarmupThen(warmup_steps, cosine)

    opt = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=float(getattr(config, "momentum", 0.9)),
        nesterov=bool(getattr(config, "NESTEROV", True)),
        clipnorm=float(getattr(config, "CLIPNORM", 1.0)),
    )

    if getattr(config, "USE_AMP", False):
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    # anchors (match student head concat order: P3->P4->P5)
    H3 = config.IMGSZ // 8
    H4 = config.IMGSZ // 16
    H5 = config.IMGSZ // 32
    anchors_all = tf.concat([
        get_anchors(H3, H3, dtype=tf.float32),
        get_anchors(H4, H4, dtype=tf.float32),
        get_anchors(H5, H5, dtype=tf.float32),
    ], axis=0)   # (N,4)

    # CSV init
    is_new = not os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "epoch",
                "train_total", "train_box", "train_cls", "train_kpt",
                "val_total", "val_box", "val_cls", "val_kpt",
                "lr"
            ])

    # EMA (optional)
    ema = ModelEMA(student, decay=getattr(config, "EMA_DECAY", 0.9998))

    # ------------------------------------------------------------
    # Label supervision path (recommended for stable QAT)
    # ------------------------------------------------------------
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            imgs = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            meta = batch[2] if len(batch) > 2 else None
        else:
            imgs, labels, meta = batch, None, None
        return imgs, labels, meta


    IMGSZ = int(config.IMGSZ)
    D = 5 + int(config.NUM_KPT) * int(config.KPT_VALS)
    MAX_M = int(getattr(config, "MAX_OBJS", 64))

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, IMGSZ, IMGSZ, 3], tf.float32),
            tf.TensorSpec([None, MAX_M, D], tf.float32),
        ],
        reduce_retracing=True,
    )

    @tf.function
    def train_step_label(batch_imgs, batch_labels):
        batch_dict = build_batch_dict_from_padded_labels(
            batch_labels, num_kpt=NUM_KPT, kpt_vals=KPT_VALS
        )
        with tf.GradientTape() as tape:
            y_s_out = student(batch_imgs, training=True)
            kd_raw     = y_s_out[1] if isinstance(y_s_out, (list, tuple)) else y_s_out
            deploy_raw = y_s_out[0] if isinstance(y_s_out, (list, tuple)) else y_s_out

            s_BNC = ensure_BNC_static(kd_raw, C)
            d_BNC = ensure_BNC_static(deploy_raw, C)

            # loss on both branches (keep your original design; can re-weight)
            loss_kd, *logs_kd = pose_loss_from_labels(
                s_BNC, batch_dict, anchors_all,
                num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
                class_weights=class_weights
            )
            loss_dep, *logs_dep = pose_loss_from_labels(
                d_BNC, batch_dict, anchors_all,
                num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
                class_weights=class_weights
            )

            total_loss = float(getattr(config, "KD_LOSS_WEIGHT", 1.0)) * loss_kd + \
                         float(getattr(config, "DEPLOY_LOSS_WEIGHT", 1.0)) * loss_dep

            if getattr(config, "USE_AMP", False):
                scaled_loss = opt.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        if getattr(config, "USE_AMP", False):
            scaled_grads = tape.gradient(scaled_loss, student.trainable_variables)
            grads = opt.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(total_loss, student.trainable_variables)

        opt.apply_gradients(zip(grads, student.trainable_variables))
        ema.update(student)

        # deploy-side breakdown is more meaningful (matches deployment head)
        logs = {
            "total": total_loss,
            "box": logs_dep[0] if logs_dep else 0.0,
            "cls": logs_dep[1] if logs_dep else 0.0,
            "kpt": logs_dep[2] if logs_dep else 0.0,
        }
        return logs

    @tf.function
    def val_step_label(batch_imgs, batch_labels):
        batch_dict = build_batch_dict_from_padded_labels(
            batch_labels, num_kpt=NUM_KPT, kpt_vals=KPT_VALS
        )
        y_s_out = student(batch_imgs, training=False)
        deploy_raw = y_s_out[0] if isinstance(y_s_out, (list, tuple)) else y_s_out
        d_BNC = ensure_BNC_static(deploy_raw, C)

        loss_dep, *logs_dep = pose_loss_from_labels(
            d_BNC, batch_dict, anchors_all,
            num_cls=NUM_CLS, num_kpt=NUM_KPT, kpt_vals=KPT_VALS,
            class_weights=class_weights
        )
        logs = {
            "total": loss_dep,
            "box": logs_dep[0] if logs_dep else 0.0,
            "cls": logs_dep[1] if logs_dep else 0.0,
            "kpt": logs_dep[2] if logs_dep else 0.0,
        }
        return logs

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    loss_history = []
    best_metric = float("inf")
    global_step = 0

    # materialize val iterator once (repeat=False)
    for e in range(int(config.EPOCHS)):
        if getattr(config, "STOP_REQUESTED", False):
            print("[⚠️ Interrupt] Stop requested. Stop before starting next epoch.")
            break

        train_total = tf.keras.metrics.Mean()
        train_box   = tf.keras.metrics.Mean()
        train_cls   = tf.keras.metrics.Mean()
        train_kpt   = tf.keras.metrics.Mean()

        it = iter(ds)
        progress = tqdm(range(int(steps_per_epoch)), desc=f"Epoch {e+1}/{int(config.EPOCHS)}", unit="step")
        for _ in progress:
            if getattr(config, "STOP_REQUESTED", False):
                print("[⚠️ Interrupt] Stop requested. Leaving epoch loop...")
                break
            batch = next(it)
            batch_imgs, batch_labels, _ = _unpack_batch(batch)

            if config.TRAIN_SUPERVISION != 'label':
                raise RuntimeError("This run_qat rewrite focuses on TRAIN_SUPERVISION='label'. Please set it in config.py.")

            if batch_labels is None:
                raise ValueError("Label supervision requires dataset with_labels=True (batch_labels is None).")

            logs = train_step_label(batch_imgs, batch_labels)
            train_total.update_state(logs["total"])
            train_box.update_state(logs["box"])
            train_cls.update_state(logs["cls"])
            train_kpt.update_state(logs["kpt"])

            lr_now = float(lr_schedule(global_step).numpy())
            progress.set_postfix(
                loss=f"{float(logs['total']):.4f}",
                box=f"{float(logs['box']):.4f}",
                cls=f"{float(logs['cls']):.4f}",
                kpt=f"{float(logs['kpt']):.4f}",
                lr=f"{lr_now:.2e}",
            )
            global_step += 1

        # ----- validation -----
        val_total = val_box = val_cls = val_kpt = 0.0
        if ds_val is not None and int(val_steps or 0) > 0:
            val_total_m = tf.keras.metrics.Mean()
            val_box_m   = tf.keras.metrics.Mean()
            val_cls_m   = tf.keras.metrics.Mean()
            val_kpt_m   = tf.keras.metrics.Mean()

            for batch in ds_val.take(int(val_steps)):
                batch_imgs, batch_labels, _ = _unpack_batch(batch)
                logs = val_step_label(batch_imgs, batch_labels)
                val_total_m.update_state(logs["total"])
                val_box_m.update_state(logs["box"])
                val_cls_m.update_state(logs["cls"])
                val_kpt_m.update_state(logs["kpt"])

            val_total = float(val_total_m.result().numpy())
            val_box   = float(val_box_m.result().numpy())
            val_cls   = float(val_cls_m.result().numpy())
            val_kpt   = float(val_kpt_m.result().numpy())

        train_total_v = float(train_total.result().numpy())
        train_box_v   = float(train_box.result().numpy())
        train_cls_v   = float(train_cls.result().numpy())
        train_kpt_v   = float(train_kpt.result().numpy())

        lr_epoch = float(lr_schedule(global_step).numpy())

        # log CSV
        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                e + 1,
                train_total_v, train_box_v, train_cls_v, train_kpt_v,
                val_total, val_box, val_cls, val_kpt,
                lr_epoch
            ])

        loss_history.append({
            "epoch": e + 1,
            "train_total": train_total_v,
            "train_box": train_box_v,
            "train_cls": train_cls_v,
            "train_kpt": train_kpt_v,
            "val_total": val_total,
            "val_box": val_box,
            "val_cls": val_cls,
            "val_kpt": val_kpt,
            "lr": lr_epoch,
        })

        # ----- save best -----
        metric = val_total if (ds_val is not None and int(val_steps or 0) > 0) else train_total_v
        if metric < best_metric:
            best_metric = metric
            best_path = output_paths.get("best_weights", str(output_paths["models"] / "student_best.weights.h5"))
            student.save_weights(best_path)
            print(f"[BEST] ✅ Saved best weights to: {best_path} (metric={best_metric:.6f})")

            # also save EMA weights as best_ema if you want
            try:
                ema.apply_to(student)
                best_ema_path = output_paths.get("best_ema_weights", str(output_paths["models"] / "student_best_ema.weights.h5"))
                student.save_weights(best_ema_path)
                print(f"[BEST][EMA] ✅ Saved best EMA weights to: {best_ema_path}")
            finally:
                ema.restore(student)

    print(f"✅ Training finished. Log saved to {log_csv}")
    return loss_history

