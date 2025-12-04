import tensorflow as tf
import numpy as np
import config

# ==============================================================================
# Helper Functions
# ==============================================================================

def iou_batch(bboxes1, bboxes2):
    """
    計算 IoU (Intersection over Union)
    bboxes1: (B, N, 4) - Anchors (xywh)
    bboxes2: (B, M, 4) - GT Boxes (xywh)
    回傳: (B, N, M)
    """
    # 轉換成 x1, y1, x2, y2 格式以便計算
    b1_x1 = bboxes1[..., 0] - bboxes1[..., 2] / 2.0
    b1_y1 = bboxes1[..., 1] - bboxes1[..., 3] / 2.0
    b1_x2 = bboxes1[..., 0] + bboxes1[..., 2] / 2.0
    b1_y2 = bboxes1[..., 1] + bboxes1[..., 3] / 2.0

    b2_x1 = bboxes2[..., 0] - bboxes2[..., 2] / 2.0
    b2_y1 = bboxes2[..., 1] - bboxes2[..., 3] / 2.0
    b2_x2 = bboxes2[..., 0] + bboxes2[..., 2] / 2.0
    b2_y2 = bboxes2[..., 1] + bboxes2[..., 3] / 2.0

    # Expand dims for broadcasting: (B, N, 1) vs (B, 1, M)
    b1_x1, b1_y1, b1_x2, b1_y2 = [tf.expand_dims(x, -1) for x in [b1_x1, b1_y1, b1_x2, b1_y2]]
    b2_x1, b2_y1, b2_x2, b2_y2 = [tf.expand_dims(x, 1) for x in [b2_x1, b2_y1, b2_x2, b2_y2]]

    intersect_x1 = tf.maximum(b1_x1, b2_x1)
    intersect_y1 = tf.maximum(b1_y1, b2_y1)
    intersect_x2 = tf.minimum(b1_x2, b2_x2)
    intersect_y2 = tf.minimum(b1_y2, b2_y2)

    w = tf.maximum(0.0, intersect_x2 - intersect_x1)
    h = tf.maximum(0.0, intersect_y2 - intersect_y1)
    intersection = w * h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)

def huber_no_reduce(y_true, y_pred, delta=1.0):
    """標準 Huber Loss，但不做 reduce_mean，保留形狀以便後續加權"""
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)

def decode_box(pred_rel, anchors):
    """
    【關鍵修正】
    將模型輸出的相對數值 (0~1) 解碼為絕對座標。
    pred_rel: (B, N, 4) range 0~1 (after sigmoid)
    anchors:  (N, 4) range 0~1 [cx, cy, w, h]
    """
    # 擴展 anchors 以匹配 batch size
    anchors = tf.expand_dims(anchors, axis=0) # (1, N, 4)
    
    a_xy = anchors[..., 0:2]
    a_wh = anchors[..., 2:4]

    p_xy = pred_rel[..., 0:2]
    p_wh = pred_rel[..., 2:4]

    # Decode Strategy:
    # XY: 允許中心點在 Anchor 寬高範圍內偏移。
    # 0.5 代表在 Anchor 中心。
    # box_cx = anchor_cx + (pred_cx - 0.5) * anchor_w
    decoded_xy = a_xy + (p_xy - 0.5) * a_wh
    
    # WH: 允許寬高縮放。
    # 0.5 代表跟 Anchor 一樣大。
    # box_w = anchor_w * (2 * pred_w)^2  (使用平方是為了讓梯度更平滑且確保非負)
    decoded_wh = a_wh * tf.square(p_wh * 2.0)

    return tf.concat([decoded_xy, decoded_wh], axis=-1)

def decode_kpt(pred_kpt_rel, anchors, num_kpt, kpt_vals):
    """
    解碼 Keypoints
    pred_kpt_rel: (B, N, num_kpt * kpt_vals)
    """
    anchors = tf.expand_dims(anchors, axis=0) # (1, N, 4)
    a_xy = anchors[..., 0:2]
    a_wh = anchors[..., 2:4]

    # Reshape: (B, N, num_kpt, kpt_vals)
    B = tf.shape(pred_kpt_rel)[0]
    N = tf.shape(pred_kpt_rel)[1]
    
    pred_kpt = tf.reshape(pred_kpt_rel, (B, N, num_kpt, kpt_vals))
    
    # 取出 xy (前兩個數值)
    kp_xy = pred_kpt[..., 0:2] # (B, N, K, 2)
    
    # 擴展 anchors 維度以匹配 K
    a_xy_exp = tf.expand_dims(a_xy, axis=2) # (1, N, 1, 2)
    a_wh_exp = tf.expand_dims(a_wh, axis=2) # (1, N, 1, 2)
    
    # Decode: kpt_x = anchor_cx + (pred_x - 0.5) * anchor_w * scale
    # 這裡假設 keypoints 範圍比較大，給予 4 倍 anchor 大小的搜尋範圍
    decoded_kp_xy = a_xy_exp + (kp_xy - 0.5) * a_wh_exp * 4.0
    
    # 組合回去 (如果有 visibility 屬性則保留原值)
    if kpt_vals > 2:
        kp_rest = pred_kpt[..., 2:] # Visibility or others
        decoded_kpt = tf.concat([decoded_kp_xy, kp_rest], axis=-1)
    else:
        decoded_kpt = decoded_kp_xy

    # Flatten back
    return tf.reshape(decoded_kpt, (B, N, num_kpt * kpt_vals))

# ==============================================================================
# Main Loss Function
# ==============================================================================
def pose_loss_from_labels(y_true, y_pred, anchors=None, 
                          num_cls=1, num_kpt=17, kpt_vals=3, 
                          lambda_box=10.0, lambda_cls=1.0, lambda_kpt=5.0):
    """
    y_true: dict, 包含 'bboxes', 'cls', 'keypoints', 'num_objects'
        bboxes   shape: (B, M, 4),  [cx, cy, w, h]，0~1
        cls      shape: (B, M, 1) or (B, M)
        keypoints:       (B, M, K, V)
        num_objects:     (B, 1)

    y_pred: Tensor, shape (B, N, 4 + num_cls + num_kpt*kpt_vals)，
            格式假設與 pred_save_model 相同：
            [cx, cy, w, h, cls_scores..., kpt0_x, kpt0_y, kpt0_v, ...]

    anchors: 目前不再使用，只保留參數以相容呼叫端。
    """

    # 1. 解析模型輸出（不要再做 sigmoid / anchor decode）
    # -------------------------------------------------------------------------
    C = 4 + num_cls + num_kpt * kpt_vals
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1], C))

    idx_box_end = 4
    idx_cls = 4 + num_cls

    # 直接當成絕對 xywh（0~1）
    pred_box      = y_pred[..., :idx_box_end]   # (B, N, 4)
    pred_cls_prob = y_pred[..., idx_box_end:idx_cls]      # (B, N, num_cls)
    pred_kpt_flat = y_pred[..., idx_cls:]                 # (B, N, K*V)

    # 2. 從 GT 取出 box / cls / kpts，並做匹配
    # -------------------------------------------------------------------------
    gt_boxes   = y_true['bboxes']      # (B, M, 4)
    gt_classes = y_true['cls']         # (B, M, 1) or (B, M)
    gt_kpts    = y_true['keypoints']   # (B, M, K, V)
    num_objs   = y_true['num_objects'] # (B, 1)

    # 用「預測的 box」跟 GT 算 IoU： (B, N, M)
    iou_map = iou_batch(pred_box, gt_boxes)

    # 每個 prediction 找到 IoU 最大的 GT
    best_gt_idx = tf.argmax(iou_map, axis=2, output_type=tf.int32)  # (B, N)
    best_iou    = tf.reduce_max(iou_map, axis=2)                    # (B, N)

    # 只允許匹配到「有效 GT」（index < num_objects）
    num_objs_expand = tf.broadcast_to(num_objs, tf.shape(best_gt_idx))  # (B, N)
    valid_gt_mask   = best_gt_idx < num_objs_expand

    # IoU 閾值，可以先設低一點讓學習較穩
    iou_thr = 0.2
    pos_mask   = (best_iou > iou_thr) & valid_gt_mask  # (B, N)
    pos_mask_f = tf.cast(pos_mask, tf.float32)         # (B, N)

    num_pos      = tf.reduce_sum(pos_mask_f)
    num_pos_safe = tf.maximum(num_pos, 1.0)

    # 3. 根據 best_gt_idx 把對應的 GT 拉出來
    # -------------------------------------------------------------------------
    batch_size   = tf.shape(gt_boxes)[0]
    num_preds    = tf.shape(pred_box)[1]

    batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
    batch_indices = tf.tile(batch_indices, [1, num_preds])    # (B, N)

    gather_idx = tf.stack([batch_indices, best_gt_idx], axis=-1)  # (B, N, 2)

    target_box = tf.gather_nd(gt_boxes,   gather_idx)  # (B, N, 4)
    target_cls = tf.gather_nd(gt_classes, gather_idx)  # (B, N, 1) or (B, N)
    target_kpt = tf.gather_nd(gt_kpts,    gather_idx)  # (B, N, K, V)

    target_kpt_flat = tf.reshape(target_kpt,
                                 (batch_size, num_preds, num_kpt * kpt_vals))

    # 4. Box Loss（直接用 pred_box 和 target_box 比）
    # -------------------------------------------------------------------------
    box_diff = huber_no_reduce(target_box, pred_box)         # (B, N, 4)
    loss_box = tf.reduce_sum(tf.reduce_mean(box_diff, axis=-1) * pos_mask_f)
    loss_box = lambda_box * (loss_box / num_pos_safe)

    # 5. Class Loss
    # -------------------------------------------------------------------------
    if num_cls > 0:
        if len(target_cls.shape) == 3:
            target_cls = tf.squeeze(target_cls, -1)
        target_cls = tf.cast(target_cls, tf.int32)

        # one-hot → (B, N, num_cls)
        t_cls_onehot = tf.one_hot(target_cls, depth=num_cls)

        # 背景處理：非正樣本 → target 全 0
        pos_mask_exp = tf.expand_dims(pos_mask_f, -1)   # (B, N, 1)
        t_cls_onehot = t_cls_onehot * pos_mask_exp

        # Huber 差異
        cls_diff = huber_no_reduce(t_cls_onehot, pred_cls_prob)  # (B, N, C)

        # 正樣本權重 1.0，負樣本 0.1，避免背景淹沒
        neg_weight = 0.1
        weights = pos_mask_exp * 1.0 + (1.0 - pos_mask_exp) * neg_weight  # (B,N,1)

        loss_cls = tf.reduce_sum(
            tf.reduce_mean(cls_diff, axis=-1) * tf.squeeze(weights, -1)
        )
        loss_cls = lambda_cls * (loss_cls / num_pos_safe)
    else:
        loss_cls = tf.constant(0.0, dtype=tf.float32)

    # 6. Keypoint Loss（同樣直接用 pred_kpt_flat）
    # -------------------------------------------------------------------------
    if num_kpt > 0 and kpt_vals > 0:
        kpt_diff = huber_no_reduce(target_kpt_flat, pred_kpt_flat)  # (B, N, K*V)
        loss_kpt = tf.reduce_sum(tf.reduce_mean(kpt_diff, axis=-1) * pos_mask_f)
        loss_kpt = lambda_kpt * (loss_kpt / num_pos_safe)
    else:
        loss_kpt = tf.constant(0.0, dtype=tf.float32)

    # 7. 總 loss & 回傳（保持原本介面）
    # -------------------------------------------------------------------------
    total_loss = loss_box + loss_cls + loss_kpt
    return total_loss, loss_box, loss_cls, loss_kpt
