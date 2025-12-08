import tensorflow as tf
import numpy as np
import config

# ==============================================================================
# Helper Functions
# ==============================================================================

def iou_batch(bboxes1, bboxes2):
    """
    計算 IoU (Intersection over Union)
    bboxes1: (B, N, 4) - Pred Boxes (cx,cy,w,h，0~1)
    bboxes2: (B, M, 4) - GT Boxes   (cx,cy,w,h，0~1)
    回傳: (B, N, M)
    """
    # 轉成 x1,y1,x2,y2
    b1_x1 = bboxes1[..., 0] - bboxes1[..., 2] / 2.0
    b1_y1 = bboxes1[..., 1] - bboxes1[..., 3] / 2.0
    b1_x2 = bboxes1[..., 0] + bboxes1[..., 2] / 2.0
    b1_y2 = bboxes1[..., 1] + bboxes1[..., 3] / 2.0

    b2_x1 = bboxes2[..., 0] - bboxes2[..., 2] / 2.0
    b2_y1 = bboxes2[..., 1] - bboxes2[..., 3] / 2.0
    b2_x2 = bboxes2[..., 0] + bboxes2[..., 2] / 2.0
    b2_y2 = bboxes2[..., 1] + bboxes2[..., 3] / 2.0

    # 展開維度做 broadcast
    b1_x1, b1_y1, b1_x2, b1_y2 = [tf.expand_dims(x, -1) for x in [b1_x1, b1_y1, b1_x2, b1_y2]]
    b2_x1, b2_y1, b2_x2, b2_y2 = [tf.expand_dims(x, 1)  for x in [b2_x1, b2_y1, b2_x2, b2_y2]]

    inter_x1 = tf.maximum(b1_x1, b2_x1)
    inter_y1 = tf.maximum(b1_y1, b2_y1)
    inter_x2 = tf.minimum(b1_x2, b2_x2)
    inter_y2 = tf.minimum(b1_y2, b2_y2)

    w = tf.maximum(0.0, inter_x2 - inter_x1)
    h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter = w * h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def huber_no_reduce(y_true, y_pred, delta=1.0):
    """Huber，不做 reduce_mean，保留 shape 方便加權"""
    error = y_true - y_pred
    is_small = tf.abs(error) <= delta
    sq = 0.5 * tf.square(error)
    lin = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small, sq, lin)


# ==============================================================================
# Main Loss Function (Route A)
# ==============================================================================

def pose_loss_from_labels(
    y_true,
    y_pred,
    anchors=None,          # 用來做 IoU matching 的 anchor 幾何資訊 (N,4)
    num_cls=1,
    num_kpt=17,
    kpt_vals=3,
    lambda_box=7.0,
    lambda_cls=1.0,
    lambda_kpt=1.0,
    class_weights=None,
):
    """
    Route A: y_pred 直接等於最終輸出（和 pred_save_model 一樣的語意）

    y_true: dict, 包含 'bboxes', 'cls', 'keypoints', 'num_objects'
        bboxes      (B, M, 4)  [cx, cy, w, h]，0~1
        cls         (B, M, 1) 或 (B, M)
        keypoints   (B, M, K, V)
        num_objects (B, 1)

    y_pred: Tensor, shape (B, N, 4 + num_cls + num_kpt*kpt_vals)，
        格式與 pred_save_model.py 相同：
        [cx, cy, w, h, cls_scores..., kpt0_x, kpt0_y, kpt0_v, ...] （全部 0~1）

    anchors: (N, 4) 的 anchor 幾何位置，只用來做 IoU matching，不參與 decode。
    """

    # ----------------------
    # 0) 解析 GT
    # ----------------------
    gt_boxes     = y_true["bboxes"]      # (B, M, 4)
    gt_classes   = y_true["cls"]         # (B, M, 1) or (B, M)
    gt_kpts      = y_true["keypoints"]   # (B, M, K, V)
    num_objects  = y_true["num_objects"] # (B, 1)

    batch_size = tf.shape(gt_boxes)[0]
    max_gt     = tf.shape(gt_boxes)[1]

    # 每個 batch 可能有 padding 的 GT，用 num_objects 過濾
    valid_gt_mask = tf.sequence_mask(
        tf.squeeze(num_objects, axis=-1),
        maxlen=max_gt,
        dtype=tf.float32
    )  # (B, M)

    # ----------------------
    # 1) 解析 Model Output（直接當最終輸出）
    # ----------------------
    C = 4 + num_cls + num_kpt * kpt_vals
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1], C))   # (B, N, C)

    idx_box_end = 4
    idx_cls_end = 4 + num_cls

    pred_box      = y_pred[..., :idx_box_end]          # (B, N, 4)，0~1 [cx,cy,w,h]
    pred_cls_prob = y_pred[..., idx_box_end:idx_cls_end]  # (B, N, num_cls)，視為機率 0~1
    pred_kpt_flat = y_pred[..., idx_cls_end:]          # (B, N, K*V)，0~1

    # 如果想強制 clamp 在 0~1，可以打開這三行：
    # pred_box      = tf.clip_by_value(pred_box, 0.0, 1.0)
    # pred_cls_prob = tf.clip_by_value(pred_cls_prob, 0.0, 1.0)
    # pred_kpt_flat = tf.clip_by_value(pred_kpt_flat, 0.0, 1.0)
    '''
    # ----------------------
    # 2) IoU-based 配對：用 anchors 做 assignment
    # ----------------------
    if anchors is not None:
        # anchors: (N, 4) -> [cx, cy, w, h]，0~1
        anchors_f = tf.cast(anchors, tf.float32)          # (N, 4)
        anchors_f = tf.expand_dims(anchors_f, axis=0)     # (1, N, 4)
        B_true    = tf.shape(gt_boxes)[0]
        anchors_f = tf.tile(anchors_f, [B_true, 1, 1])    # (B, N, 4)

        # ★ Assignment 用 anchors_f，回歸用 pred_box
        iou_source = anchors_f
    else:
        # 如果沒給 anchors，就退回用 pred_box 做 matching
        iou_source = pred_box

    # IoU: (B, N, M)
    ious = iou_batch(iou_source, gt_boxes)

    # 把 invalid GT 的 IoU 變成 -1e6，避免被選為 best GT
    valid_mask_exp = tf.expand_dims(valid_gt_mask, 1)  # (B, 1, M)
    ious = ious * valid_mask_exp + (1.0 - valid_mask_exp) * (-1e6)

    best_gt_idx  = tf.argmax(ious, axis=-1, output_type=tf.int32)  # (B, N)
    best_ious    = tf.reduce_max(ious, axis=-1)                    # (B, N)

    # IoU 門檻決定正負樣本
    iou_thr = 0.3
    pos_mask   = tf.cast(best_ious > iou_thr, tf.float32)  # (B, N)
    pos_mask_f = pos_mask

    num_pos = tf.reduce_sum(pos_mask_f)
    num_neg = tf.reduce_sum(1.0 - pos_mask_f)

    num_pos_safe = tf.maximum(num_pos, 1.0)
    num_neg_safe = tf.maximum(num_neg, 1.0)
    '''
    # ----------------------
    # 2) Anchor-free center-based assignment
    #     - 只用 anchors 的 (cx, cy) 當 grid point
    #     - 不用 IoU threshold 決定正樣本
    # ----------------------
    if anchors is None:
        raise ValueError("Anchor-free assignment 需要 anchors 提供 grid center (cx,cy).")

    # anchors: (N, 4) -> 取前兩維 [cx, cy]
    anchors_xy = tf.cast(anchors[:, :2], tf.float32)          # (N, 2)
    anchors_xy = tf.expand_dims(anchors_xy, axis=0)           # (1, N, 2)
    B_true     = tf.shape(gt_boxes)[0]
    grid_xy    = tf.tile(anchors_xy, [B_true, 1, 1])          # (B, N, 2)

    # GT box: (B, M, 4) [cx, cy, w, h] -> (x1,y1,x2,y2)
    gx, gy, gw, gh = tf.unstack(gt_boxes, axis=-1)            # (B, M) each
    x1 = gx - gw / 2.0
    y1 = gy - gh / 2.0
    x2 = gx + gw / 2.0
    y2 = gy + gh / 2.0

    # expand dims for broadcast: (B, 1, M)
    x1 = tf.expand_dims(x1, axis=1)
    y1 = tf.expand_dims(y1, axis=1)
    x2 = tf.expand_dims(x2, axis=1)
    y2 = tf.expand_dims(y2, axis=1)

    # grid_xy: (B, N, 2)
    cx = grid_xy[..., 0:1]   # (B, N, 1)
    cy = grid_xy[..., 1:2]   # (B, N, 1)

    # 每個 (B,N,1) 跟 (B,1,M) 比較 → (B,N,M)
    inside_x = (cx >= x1) & (cx <= x2)
    inside_y = (cy >= y1) & (cy <= y2)
    inside   = inside_x & inside_y            # (B, N, M)

    # 過濾 padding GT（num_objects 以外的 GT 不參與）
    valid_mask_exp = tf.expand_dims(valid_gt_mask, axis=1)   # (B, 1, M)
    inside = inside & tf.cast(valid_mask_exp, tf.bool)       # (B, N, M)

    # 若要只用「中心附近」的點，可加 center_radius：例如 2.5
    # center_radius = 2.5
    # cx_gt = tf.expand_dims(gx, axis=1)  # (B,1,M)
    # cy_gt = tf.expand_dims(gy, axis=1)
    # rx = (gw / 2.0) / center_radius     # (B,M)
    # ry = (gh / 2.0) / center_radius
    # rx = tf.expand_dims(rx, axis=1)     # (B,1,M)
    # ry = tf.expand_dims(ry, axis=1)
    # center_x1 = cx_gt - rx
    # center_x2 = cx_gt + rx
    # center_y1 = cy_gt - ry
    # center_y2 = cy_gt + ry
    # in_cx = (cx >= center_x1) & (cx <= center_x2)
    # in_cy = (cy >= center_y1) & (cy <= center_y2)
    # center_region = in_cx & in_cy
    # inside = inside & center_region   # 再進一步收窄候選區

    # 若某個 anchor 對多個 GT 都是 inside，選距離最近的 GT
    # 計算 grid point 到各 GT center 的 L1 距離
    gx_exp = tf.expand_dims(gx, axis=1)   # (B,1,M)
    gy_exp = tf.expand_dims(gy, axis=1)   # (B,1,M)

    dist_x = tf.abs(cx - gx_exp)         # (B,N,M)
    dist_y = tf.abs(cy - gy_exp)         # (B,N,M)
    dist   = dist_x + dist_y

    # 對於不在 inside 的位置給一個很大的距離，避免被選為 best GT
    big = tf.constant(1e6, dtype=dist.dtype)
    dist = tf.where(inside, dist, big)

    # 針對每個 anchor (B,N)，選擇距離最小的 GT
    best_gt_idx = tf.argmin(dist, axis=-1, output_type=tf.int32)   # (B,N)

    # pos_mask: 只要有任何 GT 把它當作 inside，就算正樣本
    pos_mask = tf.reduce_any(inside, axis=-1)    # (B,N) bool
    pos_mask_f = tf.cast(pos_mask, tf.float32)

    num_pos = tf.reduce_sum(pos_mask_f)
    num_neg = tf.reduce_sum(1.0 - pos_mask_f)
    num_pos_safe = tf.maximum(num_pos, 1.0)
    num_neg_safe = tf.maximum(num_neg, 1.0)

    # ----------------------
    # 3) 依照 best_gt_idx 取出對應 GT
    # ----------------------
    num_preds = tf.shape(pred_box)[1]

    batch_indices = tf.reshape(tf.range(batch_size, dtype=tf.int32), (batch_size, 1))
    batch_indices = tf.tile(batch_indices, [1, num_preds])  # (B, N)

    gather_idx = tf.stack([batch_indices, best_gt_idx], axis=-1)  # (B, N, 2)

    target_box = tf.gather_nd(gt_boxes,   gather_idx)  # (B, N, 4)
    target_cls = tf.gather_nd(gt_classes, gather_idx)  # (B, N, 1 或 (B,N))
    target_kpt = tf.gather_nd(gt_kpts,    gather_idx)  # (B, N, K, V)

    # flatten keypoints → (B, N, K*V)
    target_kpt_flat = tf.reshape(
        target_kpt, (batch_size, num_preds, num_kpt * kpt_vals)
    )

    # ----------------------
    # 4) Box Loss（Huber）
    # ----------------------
    box_diff = huber_no_reduce(target_box, pred_box)      # (B, N, 4)
    loss_box = tf.reduce_sum(tf.reduce_mean(box_diff, axis=-1) * pos_mask_f)
    loss_box = lambda_box * (loss_box / num_pos_safe)

    # ----------------------
    # 5) Class Loss（Huber + 正負樣本權重）
    # ----------------------
    if num_cls > 0:
        if len(target_cls.shape) == 3:
            target_cls = tf.squeeze(target_cls, -1)
        target_cls = tf.cast(target_cls, tf.int32)

        # one-hot → (B, N, num_cls)
        t_cls_onehot = tf.one_hot(target_cls, depth=num_cls, dtype=pred_cls_prob.dtype)

        # 只在正樣本上有 label（負樣本設成 0）
        pos_mask_exp = tf.expand_dims(pos_mask_f, -1)  # (B, N, 1)
        t_cls_onehot = t_cls_onehot * pos_mask_exp

        # 這裡把 pred_cls_prob 當成「已經是機率 0~1」
        cls_diff = huber_no_reduce(t_cls_onehot, pred_cls_prob)  # (B, N, num_cls)

        # 平衡Dataset cls 數量
        if class_weights is not None:
            cw = tf.convert_to_tensor(class_weights, dtype=cls_diff.dtype)  # (C,)
            # 保險起見再 normalize 一次平均值 ≈ 1
            cw = cw / tf.reduce_mean(cw)
            cw = tf.reshape(cw, (1, 1, -1))  # broadcast 到 (B, N, C)
            cls_diff = cls_diff * cw

        # 正樣本 1.0，負樣本 0.01
        neg_weight = 0.1
        weights = pos_mask_exp * 1.0 + (1.0 - pos_mask_exp) * neg_weight  # (B, N, 1)

        loss_cls = tf.reduce_sum(
            tf.reduce_mean(cls_diff, axis=-1) * tf.squeeze(weights, -1)
        )
        loss_cls = lambda_cls * (loss_cls / num_pos_safe)
    else:
        loss_cls = tf.constant(0.0, dtype=tf.float32)

    # ----------------------
    # 6) Keypoint Loss（直接對齊 pred_save_model 的 kpt domain）
    # ----------------------
    if num_kpt > 0 and kpt_vals > 0:
        kpt_diff = huber_no_reduce(target_kpt_flat, pred_kpt_flat)  # (B, N, K*V)
        loss_kpt = tf.reduce_sum(tf.reduce_mean(kpt_diff, axis=-1) * pos_mask_f)
        loss_kpt = lambda_kpt * (loss_kpt / num_pos_safe)
    else:
        loss_kpt = tf.constant(0.0, dtype=tf.float32)

    # ----------------------
    # 7) Total
    # ----------------------
    total_loss = loss_box + loss_cls + loss_kpt
    return total_loss, loss_box, loss_cls, loss_kpt
