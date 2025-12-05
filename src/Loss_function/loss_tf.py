import tensorflow as tf
import numpy as np
import config

# ==============================================================================
# Helper Functions
# ==============================================================================

def iou_batch(bboxes1, bboxes2):
    """
    計算 IoU (Intersection over Union)
    bboxes1: (B, N, 4) - Anchors or predictions (xywh, normalized 0~1)
    bboxes2: (B, M, 4) - GT Boxes (xywh, normalized 0~1)
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
    b2_x1, b2_y1, b2_x2, b2_y2 = [tf.expand_dims(x, 1)  for x in [b2_x1, b2_y1, b2_x2, b2_y2]]

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
    將模型輸出的相對數值 (0~1) 解碼為絕對座標 (0~1)。
    pred_rel: (B, N, 4) range 0~1 (after sigmoid)
    anchors:  (N, 4) range 0~1 [cx, cy, w, h]
    """
    if anchors is None:
        raise ValueError("decode_box 需要 anchors (N,4)")

    anchors = tf.expand_dims(anchors, axis=0)  # (1, N, 4)
    a_xy = anchors[..., 0:2]
    a_wh = anchors[..., 2:4]

    p_xy = pred_rel[..., 0:2]
    p_wh = pred_rel[..., 2:4]

    # Decode:
    # XY: 允許中心點在 Anchor 寬高範圍內偏移，0.5 代表在 Anchor 中心。
    decoded_xy = a_xy + (p_xy - 0.5) * a_wh

    # WH: 允許寬高縮放。
    # 0.5 代表跟 Anchor 一樣大。
    # box_w = anchor_w * (2 * pred_w)^2  (平方確保非負且梯度較平滑)
    decoded_wh = a_wh * tf.square(p_wh * 2.0)

    return tf.concat([decoded_xy, decoded_wh], axis=-1)


def decode_kpt(pred_kpt_rel, anchors, num_kpt, kpt_vals):
    """
    解碼 Keypoints（從相對 anchor 空間 → 絕對 normalized 0~1 座標）

    pred_kpt_rel: (B, N, num_kpt * kpt_vals)，0~1 (after sigmoid)
    anchors:      (N, 4) 0~1 [cx, cy, w, h]
    """
    if anchors is None:
        raise ValueError("decode_kpt 需要 anchors (N,4)")

    anchors = tf.expand_dims(anchors, axis=0)  # (1, N, 4)
    a_xy = anchors[..., 0:2]
    a_wh = anchors[..., 2:4]

    # Reshape: (B, N, num_kpt, kpt_vals)
    B = tf.shape(pred_kpt_rel)[0]
    N = tf.shape(pred_kpt_rel)[1]

    pred_kpt = tf.reshape(pred_kpt_rel, (B, N, num_kpt, kpt_vals))

    # 取出 xy (前兩個數值)
    kp_xy = pred_kpt[..., 0:2]  # (B, N, K, 2)

    # 擴展 anchors 維度以匹配 K
    a_xy_exp = tf.expand_dims(a_xy, axis=2)  # (1, N, 1, 2)
    a_wh_exp = tf.expand_dims(a_wh, axis=2)  # (1, N, 1, 2)

    # Decode: kpt_x = anchor_cx + (pred_x - 0.5) * anchor_w * scale
    # 這裡假設 keypoints 範圍比較大，給予 4 倍 anchor 大小的搜尋範圍
    decoded_kp_xy = a_xy_exp + (kp_xy - 0.5) * a_wh_exp * 4.0

    # 組合回去 (如果有 visibility 屬性則保留原值)
    if kpt_vals > 2:
        kp_rest = pred_kpt[..., 2:]  # Visibility or others
        decoded_kpt = tf.concat([decoded_kp_xy, kp_rest], axis=-1)
    else:
        decoded_kpt = decoded_kp_xy

    # Flatten back
    return tf.reshape(decoded_kpt, (B, N, num_kpt * kpt_vals))


# ==============================================================================
# Main Loss Function (Route A: logits → sigmoid → anchor decode)
# ==============================================================================

def pose_loss_from_labels(
    y_true,
    y_pred,
    anchors=None,
    num_cls=1,
    num_kpt=17,
    kpt_vals=3,
    lambda_box=7.0,
    lambda_cls=1.0,
    lambda_kpt=14.0,
):
    """
    y_true: dict, 包含 'bboxes', 'cls', 'keypoints', 'num_objects'
        bboxes      shape: (B, M, 4),  [cx, cy, w, h]，0~1
        cls         shape: (B, M, 1) or (B, M)
        keypoints   shape: (B, M, K, V)   (x, y, v) normalized 0~1
        num_objects shape: (B, 1)

    y_pred: Tensor, shape (B, N, 4 + num_cls + num_kpt*kpt_vals)，
            模型輸出為 **logits**，需要先經 sigmoid 再 decode：
            [box_logit(4), cls_logit(num_cls), kpt_logit(num_kpt*kpt_vals)]

    anchors: Tensor, shape (N, 4)，anchor 的 [cx, cy, w, h] (0~1)

    回傳:
        total_loss, loss_box, loss_cls, loss_kpt
    """

    # -------------------------------------------------------------------------
    # 1. 解析模型輸出 (logits → sigmoid → anchor decode)
    # -------------------------------------------------------------------------
    C = 4 + num_cls + num_kpt * kpt_vals
    raw = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1], C))  # (B, N, C)

    # 分割 logits
    box_logit = raw[..., :4]  # (B, N, 4)

    if num_cls > 0:
        cls_logit = raw[..., 4:4 + num_cls]  # (B, N, num_cls)
        kpt_start = 4 + num_cls
    else:
        cls_logit = None
        kpt_start = 4

    if num_kpt > 0 and kpt_vals > 0:
        kpt_logit = raw[..., kpt_start : kpt_start + num_kpt * kpt_vals]
    else:
        kpt_logit = None

    # box / kpt 相對值 (0~1)
    box_rel = tf.sigmoid(box_logit)  # (B, N, 4)
    if kpt_logit is not None:
        kpt_rel = tf.sigmoid(kpt_logit)  # (B, N, K*V)
    else:
        kpt_rel = None

    # 解碼為絕對座標 (0~1)
    pred_box = decode_box(box_rel, anchors)  # (B, N, 4)
    if kpt_rel is not None and num_kpt > 0 and kpt_vals > 0:
        pred_kpt_flat = decode_kpt(kpt_rel, anchors, num_kpt, kpt_vals)  # (B, N, K*V)
    else:
        pred_kpt_flat = tf.zeros(
            (tf.shape(raw)[0], tf.shape(raw)[1], num_kpt * kpt_vals),
            dtype=raw.dtype,
        )

    # -------------------------------------------------------------------------
    # 2. 從 GT 取出 box / cls / keypoints，並做 IoU-based 匹配
    # -------------------------------------------------------------------------
    gt_boxes   = y_true["bboxes"]      # (B, M, 4)
    gt_classes = y_true["cls"]         # (B, M, 1) or (B, M)
    gt_kpts    = y_true["keypoints"]   # (B, M, K, V)
    num_objs   = y_true["num_objects"] # (B, 1)

    # IoU: (B, N, M)
    iou_map = iou_batch(pred_box, gt_boxes)

    # 每個 prediction 找到 IoU 最大的 GT
    best_gt_idx = tf.argmax(iou_map, axis=2, output_type=tf.int32)  # (B, N)
    best_iou    = tf.reduce_max(iou_map, axis=2)                    # (B, N)

    # 僅允許匹配到有效 GT (index < num_objects)
    num_objs_expand = tf.broadcast_to(num_objs, tf.shape(best_gt_idx))  # (B, N)
    valid_gt_mask   = best_gt_idx < num_objs_expand

    # IoU 閾值
    iou_thr = 0.2
    pos_mask   = (best_iou > iou_thr) & valid_gt_mask  # (B, N)
    pos_mask_f = tf.cast(pos_mask, tf.float32)         # (B, N)

    num_pos      = tf.reduce_sum(pos_mask_f)
    num_neg      = tf.reduce_sum(1.0 - pos_mask_f)
    num_pos_safe = tf.maximum(num_pos, 1.0)
    num_neg_safe = tf.maximum(num_neg, 1.0)

    # -------------------------------------------------------------------------
    # 3. 根據 best_gt_idx 把對應的 GT 拉出來
    # -------------------------------------------------------------------------
    batch_size = tf.shape(gt_boxes)[0]
    num_preds  = tf.shape(pred_box)[1]

    batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))  # (B, 1)
    batch_indices = tf.tile(batch_indices, [1, num_preds])             # (B, N)

    gather_idx = tf.stack([batch_indices, best_gt_idx], axis=-1)  # (B, N, 2)

    target_box = tf.gather_nd(gt_boxes,   gather_idx)  # (B, N, 4)
    target_cls = tf.gather_nd(gt_classes, gather_idx)  # (B, N, 1) or (B, N)
    target_kpt = tf.gather_nd(gt_kpts,    gather_idx)  # (B, N, K, V)

    target_kpt_flat = tf.reshape(
        target_kpt, (batch_size, num_preds, num_kpt * kpt_vals)
    )

    # -------------------------------------------------------------------------
    # 4. Box Loss：Huber on decoded boxes (only positives)
    # -------------------------------------------------------------------------
    box_diff = huber_no_reduce(target_box, pred_box)  # (B, N, 4)
    loss_box = tf.reduce_sum(tf.reduce_mean(box_diff, axis=-1) * pos_mask_f)
    loss_box = lambda_box * (loss_box / num_pos_safe)

    # -------------------------------------------------------------------------
    # 5. Class Loss：BCE with logits
    # -------------------------------------------------------------------------
    if num_cls > 0 and cls_logit is not None:
        # target_cls: (B,N,1) or (B,N)
        if len(target_cls.shape) == 3:
            target_cls = tf.squeeze(target_cls, axis=-1)  # (B, N)
        target_cls = tf.cast(target_cls, tf.int32)

        # one-hot: (B, N, num_cls)
        t_cls_onehot = tf.one_hot(target_cls, depth=num_cls, dtype=cls_logit.dtype)

        # 只有正樣本有 class label，負樣本全部視為 0
        pos_mask_exp = tf.expand_dims(pos_mask_f, axis=-1)  # (B, N, 1)
        t_cls_targets = t_cls_onehot * pos_mask_exp         # (B, N, num_cls)

        # BCE with logits
        cls_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=t_cls_targets,
            logits=cls_logit,
        )  # (B, N, num_cls)

        # 負樣本給較小權重，避免淹沒正樣本梯度
        neg_weight = 0.1
        weights = pos_mask_exp + (1.0 - pos_mask_exp) * neg_weight  # (B, N, 1)

        loss_cls = tf.reduce_sum(cls_loss_raw * weights) / num_pos_safe
        loss_cls = lambda_cls * loss_cls
    else:
        loss_cls = tf.constant(0.0, dtype=raw.dtype)

    # -------------------------------------------------------------------------
    # 6. Keypoint Loss：拆成 xy 與 v 兩部分
    # -------------------------------------------------------------------------
    if num_kpt > 0 and kpt_vals > 0:
        B = tf.shape(pred_kpt_flat)[0]
        N = tf.shape(pred_kpt_flat)[1]

        pred_kpt = tf.reshape(pred_kpt_flat, (B, N, num_kpt, kpt_vals))
        tgt_kpt  = tf.reshape(target_kpt_flat, (B, N, num_kpt, kpt_vals))

        pred_xy = pred_kpt[..., :2]   # (B, N, K, 2)
        tgt_xy  = tgt_kpt[..., :2]    # (B, N, K, 2)

        if kpt_vals > 2:
            pred_v = pred_kpt[..., 2]  # (B, N, K)
            tgt_v  = tgt_kpt[..., 2]   # (B, N, K)
        else:
            # 沒有 visibility 維度時，視為全部可見且不做 v loss
            pred_v = None
            tgt_v  = tf.ones((B, N, num_kpt), dtype=pred_xy.dtype)

        # ----- xy loss：只在「正樣本 + 可見 keypoints」上計算 -----
        vis_mask = tgt_v > 0.5                          # (B, N, K)
        vis_mask_f = tf.cast(vis_mask, pred_xy.dtype)   # (B, N, K)

        pos_mask_k = tf.expand_dims(pos_mask_f, axis=-1)    # (B, N, 1)
        pos_vis_mask = pos_mask_k * vis_mask_f              # (B, N, K)

        coord_diff = huber_no_reduce(tgt_xy, pred_xy)       # (B, N, K, 2)
        coord_diff = tf.reduce_mean(coord_diff, axis=-1)    # (B, N, K)

        denom_coord = tf.reduce_sum(pos_vis_mask)
        denom_coord = tf.maximum(denom_coord, 1.0)

        loss_kpt_xy = tf.reduce_sum(coord_diff * pos_vis_mask) / denom_coord

        # ----- v loss：只在正樣本 anchor 上計算 -----
        if pred_v is not None:
            v_diff = huber_no_reduce(tgt_v, pred_v)     # (B, N, K)
            v_diff = v_diff * pos_mask_k                # (B, N, K)

            denom_v = tf.reduce_sum(pos_mask_k) * tf.cast(num_kpt, pred_xy.dtype)
            denom_v = tf.maximum(denom_v, 1.0)

            loss_kpt_v = tf.reduce_sum(v_diff) / denom_v
        else:
            loss_kpt_v = tf.constant(0.0, dtype=pred_xy.dtype)

        # 總 keypoint loss：xy 為主，v 給較小權重
        loss_kpt = lambda_kpt * (loss_kpt_xy + 0.1 * loss_kpt_v)
    else:
        loss_kpt = tf.constant(0.0, dtype=raw.dtype)

    # -------------------------------------------------------------------------
    # 7. 總 loss & 回傳
    # -------------------------------------------------------------------------
    total_loss = loss_box + loss_cls + loss_kpt
    return total_loss, loss_box, loss_cls, loss_kpt
