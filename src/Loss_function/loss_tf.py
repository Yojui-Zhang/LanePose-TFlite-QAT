# src/Loss_function/loss_tf.py
import tensorflow as tf
import math
import config
from src.process.pred_model import (split_BNC, ensure_BNC_static)

# ==========================
# Helper Functions
# ==========================

def huber_no_reduce(y_true, y_pred, delta=1.0):
    """
    計算 Huber Loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = y_pred - y_true
    abs_diff = tf.abs(diff)
    small = 0.5 * tf.square(diff)
    big   = delta * (abs_diff - 0.5 * delta)
    return tf.where(abs_diff <= delta, small, big)

# ==========================
# Core Loss Function (Direct 0-1 Regression)
# ==========================

def pose_loss_from_labels(
    pred_raw,
    batch_dict,
    num_cls,
    num_kpt,
    kpt_vals,
    img_size,
    lambda_box=5.0,    # Box 權重
    lambda_kxy=9.0,    # Keypoint 權重
    lambda_v=2.0,      # Visibility 權重
    lambda_cls=1.0,    # Class 權重
):
    """
    完全回歸版 Loss:
    1. 移除所有 Sigmoid/Log 轉換。
    2. 強迫模型 Linear 輸出直接擬合 0~1 的目標值。
    3. 適用於不希望修改推論腳本的情況。
    """

    # ------------------------------------------------------
    # 0. 準備預測值 (不做 Activation)
    # ------------------------------------------------------
    C = 4 + num_cls + num_kpt * kpt_vals
    BNC = ensure_BNC_static(pred_raw, C)
    
    # 拆分預測值
    box_pred, cls_pred, kxy_pred, v_pred = split_BNC(BNC, num_cls, num_kpt, kpt_vals)

    # 型別轉換 (保持 Linear，不加 Sigmoid)
    box_pred = tf.cast(box_pred, tf.float32)
    
    # [Box] 直接取值，不做 sigmoid，也不做 exp
    # 預期模型輸出: [x(0~1), y(0~1), w(0~1), h(0~1)]
    pred_xywh = box_pred 

    # [Keypoint] 直接取值
    if kxy_pred is not None:
        kxy_pred = tf.cast(kxy_pred, tf.float32)
    
    # [Class] 直接取值 (將用 Huber Loss 回歸 0/1)
    if cls_pred is not None: 
        cls_pred = tf.cast(cls_pred, tf.float32)

    # [Visibility] 調整形狀
    if v_pred is not None:
        B_dyn = tf.shape(box_pred)[0]
        N_dyn = tf.shape(box_pred)[1]
        v_pred = tf.cast(v_pred, tf.float32)
        v_pred = tf.reshape(v_pred, [B_dyn, N_dyn, num_kpt])

    # ------------------------------------------------------
    # 1. 解析 Ground Truth
    # ------------------------------------------------------
    gt_batch = tf.cast(tf.reshape(batch_dict["batch_idx"], [-1]), tf.int32)
    gt_cls   = tf.cast(tf.reshape(batch_dict["cls"],       [-1]), tf.int32)
    # 這些原始數據已經是 0~1 歸一化的
    gt_box   = tf.cast(batch_dict["bboxes"],    tf.float32) # (G, 4) -> xywh (0~1)
    gt_kpts  = tf.cast(batch_dict["keypoints"], tf.float32) # (G, K, V) (0~1)

    # ------------------------------------------------------
    # 2. 定義 Strides 與建立 Targets (全圖 0~1)
    # ------------------------------------------------------
    strides = [8, 16, 32]
    # 這裡只需要計算網格索引來決定「誰負責預測」，不需要計算偏移量
    if isinstance(img_size, (tuple, list)):
        img_w = float(img_size[0])
    else:
        img_w = float(img_size)
    
    grid_dims = [int(img_w / s) for s in strides]
    offsets = [0]
    for gd in grid_dims[:-1]:
        offsets.append(offsets[-1] + gd * gd)
    
    box_t_list, cls_t_list, kxy_t_list, v_t_list, pos_mask_list = [], [], [], [], []
    B_int = int(tf.shape(box_pred)[0])

    for b in range(B_int):
        # 初始化
        box_t_b = tf.zeros_like(pred_xywh[b])
        pos_mask_b = tf.zeros([tf.shape(box_pred)[1]], tf.float32)
        cls_t_b = tf.zeros_like(cls_pred[b]) if cls_pred is not None else None
        kxy_t_b = tf.zeros_like(kxy_pred[b]) if kxy_pred is not None else None
        v_t_b   = tf.zeros_like(v_pred[b])   if v_pred is not None else None

        # 找出 GT
        b_tensor = tf.cast(b, gt_batch.dtype)
        gt_mask_b = tf.reshape(tf.where(tf.equal(gt_batch, b_tensor)), [-1])
        Mb = tf.shape(gt_mask_b)[0]

        if Mb > 0:
            raw_boxes = tf.gather(gt_box, gt_mask_b) # (Mb, 4) 0~1
            raw_cls   = tf.gather(gt_cls, gt_mask_b) 
            raw_kpts  = tf.gather(gt_kpts, gt_mask_b) # (Mb, K, V) 0~1

            # 絕對座標用於計算落在哪個 Grid
            abs_x = raw_boxes[:, 0] * img_w
            abs_y = raw_boxes[:, 1] * img_w # 假設正方形
            
            all_indices = []
            all_target_boxes = []
            all_target_cls = []
            all_target_kxy = []
            all_target_v = []

            for i, s in enumerate(strides):
                gs = grid_dims[i]
                offset = offsets[i]

                # 計算 Grid Index (僅用於定位)
                gx = abs_x / s
                gy = abs_y / s
                gi = tf.cast(tf.math.floor(gx), tf.int32)
                gj = tf.cast(tf.math.floor(gy), tf.int32)
                gi = tf.clip_by_value(gi, 0, gs - 1)
                gj = tf.clip_by_value(gj, 0, gs - 1)
                anch_idx = offset + gj * gs + gi

                # ★★★ 重點：Target 直接使用歸一化數值 ★★★
                
                # 1. Box Target: 直接用 0~1 的 raw_boxes
                t_box = raw_boxes 

                # 2. Keypoint Target: 直接用 0~1 的 raw_kpts
                t_kxy = raw_kpts[..., 0:2]

                # 3. Class Target: One-hot (0或1)
                t_cls = tf.one_hot(raw_cls, depth=num_cls)

                # 4. Visibility Target
                if kpt_vals > 2:
                    t_v = raw_kpts[..., 2]
                else:
                    t_v = tf.ones_like(raw_kpts[..., 0])

                all_indices.append(anch_idx)
                all_target_boxes.append(t_box)
                all_target_cls.append(t_cls)
                all_target_kxy.append(t_kxy)
                all_target_v.append(t_v)

            # 合併與寫入
            indices_flat = tf.concat(all_indices, axis=0)
            
            # 使用 tensor_scatter_nd_update
            box_t_b = tf.tensor_scatter_nd_update(
                box_t_b, tf.expand_dims(indices_flat, 1), tf.concat(all_target_boxes, axis=0)
            )
            pos_mask_b = tf.tensor_scatter_nd_update(
                pos_mask_b, tf.expand_dims(indices_flat, 1), tf.ones_like(indices_flat, dtype=tf.float32)
            )
            if cls_t_b is not None:
                cls_t_b = tf.tensor_scatter_nd_update(
                    cls_t_b, tf.expand_dims(indices_flat, 1), tf.concat(all_target_cls, axis=0)
                )
            if kxy_t_b is not None:
                kxy_t_b = tf.tensor_scatter_nd_update(
                    kxy_t_b, tf.expand_dims(indices_flat, 1), tf.concat(all_target_kxy, axis=0)
                )
            if v_t_b is not None:
                v_t_b = tf.tensor_scatter_nd_update(
                    v_t_b, tf.expand_dims(indices_flat, 1), tf.concat(all_target_v, axis=0)
                )

        box_t_list.append(box_t_b)
        pos_mask_list.append(pos_mask_b)
        if cls_t_list is not None: cls_t_list.append(cls_t_b)
        if kxy_t_list is not None: kxy_t_list.append(kxy_t_b)
        if v_t_list is not None:   v_t_list.append(v_t_b)

    # ------------------------------------------------------
    # 4. 計算 Loss
    # ------------------------------------------------------
    box_t = tf.stack(box_t_list, axis=0)
    pos_mask_f = tf.cast(tf.stack(pos_mask_list, axis=0), tf.float32)
    cls_t = tf.stack(cls_t_list, axis=0) if cls_t_list else None
    kxy_t = tf.stack(kxy_t_list, axis=0) if kxy_t_list else None
    v_t   = tf.stack(v_t_list, axis=0)   if v_t_list else None

    num_pos = tf.maximum(tf.reduce_sum(pos_mask_f), 1.0)

    # 4.1 Box Loss (使用 Huber 直接回歸)
    # pred_xywh 是 Linear，box_t 是 0~1
    box_diff = tf.reduce_sum(huber_no_reduce(box_t, pred_xywh), axis=-1)
    box_loss = lambda_box * tf.reduce_sum(box_diff * pos_mask_f) / num_pos

    # 4.2 Class Loss (改用 Huber 或 MSE，避免 Logits 問題)
    # 因為沒有 Sigmoid，不能用 CrossEntropyWithLogits
    cls_loss = 0.0
    if cls_pred is not None:
        cls_diff = huber_no_reduce(cls_t, cls_pred)
        cls_loss = lambda_cls * tf.reduce_sum(tf.reduce_mean(cls_diff, axis=-1) * pos_mask_f) / num_pos

    # 4.3 Keypoint Loss (直接回歸)
    kxy_loss = 0.0
    if kxy_pred is not None and v_pred is not None:
        kxy_diff = huber_no_reduce(kxy_t, kxy_pred)
        
        vis_mask = v_t[..., None]
        anchor_mask = pos_mask_f[..., None, None]
        valid_mask = vis_mask * anchor_mask
        valid_mask = tf.broadcast_to(valid_mask, tf.shape(kxy_diff))
        
        num_valid_kpt = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
        kxy_loss = lambda_kxy * tf.reduce_sum(kxy_diff * valid_mask) / num_valid_kpt

    # 4.4 Visibility Loss
    v_loss = 0.0
    if v_pred is not None and v_t is not None:
        v_diff = huber_no_reduce(v_t, v_pred)
        anchor_mask_v = tf.broadcast_to(pos_mask_f[..., None], tf.shape(v_diff))
        v_loss = lambda_v * tf.reduce_sum(v_diff * anchor_mask_v) / num_pos

    total_loss = box_loss + cls_loss + kxy_loss + v_loss

    logs = {
        "loss_box": box_loss,
        "loss_cls": cls_loss,
        "loss_kxy": kxy_loss,
        "loss_v": v_loss,
        "num_pos": num_pos
    }
    return total_loss, logs