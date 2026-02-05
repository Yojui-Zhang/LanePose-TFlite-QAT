import tensorflow as tf
import numpy as np
import config

# If True: treat keypoint v=0 as 'unlabeled' and ignore it in BOTH xy and visibility losses.
# If False: treat v=0 as 'not visible' (will supervise visibility to 0).
KPT_V0_IS_UNLABELED = getattr(config, 'KPT_V0_IS_UNLABELED', True)

# If True: still compute XY keypoint loss when GT v==0.
# Use this when your dataset stores valid (x,y) even when v=0 (e.g., v=0 means occluded / low-confidence),
# otherwise YOLO-style masking (v>0) will drop most keypoints from supervision.
KPT_SUPERVISE_XY_WHEN_V0 = getattr(config, 'KPT_SUPERVISE_XY_WHEN_V0', True)

# Optional: per-class keypoint usage mask to prevent different classes' keypoint topologies from interfering.
# Shape: (num_cls, num_kpt) with values in {0,1}. 1 = supervise this keypoint for that class; 0 = ignore.
# You can also provide a dict {class_id: [mask...]}.
KPT_CLASS_MASK = getattr(config, 'KPT_CLASS_MASK', None)


# ==============================================================================
# Helper Functions
# ==============================================================================


def bbox_ciou(pred_box, target_box, eps=1e-7):
    """
    pred_box, target_box: (B, N, 4)  [cx, cy, w, h]，都是 0~1
    回傳: (B, N) 的 CIoU loss = 1 - CIoU
    """
    px, py, pw, ph = tf.unstack(pred_box, axis=-1)
    gx, gy, gw, gh = tf.unstack(target_box, axis=-1)

    # 轉成 (x1, y1, x2, y2)
    px1 = px - pw / 2.0
    py1 = py - ph / 2.0
    px2 = px + pw / 2.0
    py2 = py + ph / 2.0

    gx1 = gx - gw / 2.0
    gy1 = gy - gh / 2.0
    gx2 = gx + gw / 2.0
    gy2 = gy + gh / 2.0

    # 交集
    inter_x1 = tf.maximum(px1, gx1)
    inter_y1 = tf.maximum(py1, gy1)
    inter_x2 = tf.minimum(px2, gx2)
    inter_y2 = tf.minimum(py2, gy2)

    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_p = tf.maximum(pw * ph, 0.0)
    area_g = tf.maximum(gw * gh, 0.0)
    union = area_p + area_g - inter_area
    iou = inter_area / (union + eps)

    # 最小包圍框對角線長度
    cw = tf.maximum(px2, gx2) - tf.minimum(px1, gx1)
    ch = tf.maximum(py2, gy2) - tf.minimum(py1, gy1)
    c2 = tf.square(cw) + tf.square(ch) + eps

    # 中心距離
    d2 = tf.square(px - gx) + tf.square(py - gy)

    # aspect ratio 項
    v = (4.0 / (np.pi ** 2)) * tf.square(
        tf.atan(gw / (gh + eps)) - tf.atan(pw / (ph + eps))
    )
    with tf.name_scope("ciou_alpha"):
            alpha = v / (1.0 - iou + v + eps)
            alpha = tf.stop_gradient(alpha)  # <--- 加上這行！

    ciou = iou - d2 / c2 - alpha * v
    ciou = tf.clip_by_value(ciou, -1.0, 1.0)

    return 1.0 - ciou   # 當成 loss


def bbox_iou_pair(pred_box, target_box, eps=1e-7):
    """
    pred_box, target_box: (B, N, 4)  [cx, cy, w, h]，皆為 0~1
    回傳: (B, N) 的 IoU (0~1)
    """
    px, py, pw, ph = tf.unstack(pred_box, axis=-1)
    gx, gy, gw, gh = tf.unstack(target_box, axis=-1)

    px1 = px - pw / 2.0
    py1 = py - ph / 2.0
    px2 = px + pw / 2.0
    py2 = py + ph / 2.0

    gx1 = gx - gw / 2.0
    gy1 = gy - gh / 2.0
    gx2 = gx + gw / 2.0
    gy2 = gy + gh / 2.0

    inter_x1 = tf.maximum(px1, gx1)
    inter_y1 = tf.maximum(py1, gy1)
    inter_x2 = tf.minimum(px2, gx2)
    inter_y2 = tf.minimum(py2, gy2)

    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_p = tf.maximum(pw * ph, 0.0)
    area_g = tf.maximum(gw * gh, 0.0)
    union = area_p + area_g - inter_area
    iou = inter_area / (union + eps)
    return tf.clip_by_value(iou, 0.0, 1.0)

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


def _get_kpt_class_mask_tensor(num_cls, num_kpt, dtype):
    """Build a (num_cls, num_kpt) float tensor from config.KPT_CLASS_MASK.
    - None: returns None (means all-ones).
    - 2D list/ndarray: interpreted as (num_cls, num_kpt) with auto pad/crop.
    - dict: {class_id: mask_list}, auto pad/crop; unspecified classes default to all-ones.
    """
    m = KPT_CLASS_MASK
    if m is None:
        return None

    # dict form
    if isinstance(m, dict):
        out = np.ones((int(num_cls), int(num_kpt)), dtype=np.float32)
        for k, v in m.items():
            ci = int(k)
            if ci < 0 or ci >= int(num_cls):
                continue
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            out[ci, :min(int(num_kpt), arr.size)] = arr[:min(int(num_kpt), arr.size)]
        return tf.constant(out, dtype=dtype)

    # 2D table form
    arr = np.asarray(m, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("config.KPT_CLASS_MASK must be 2D (num_cls, num_kpt) or dict{cls_id:mask_list}.")
    out = np.ones((int(num_cls), int(num_kpt)), dtype=np.float32)
    r = min(int(num_cls), arr.shape[0])
    c = min(int(num_kpt), arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return tf.constant(out, dtype=dtype)

def decode_yolo_pose_outputs(
    y_pred_raw,
    num_cls,
    num_kpt,
    kpt_vals,
    box_act="sigmoid",
    cls_act="sigmoid",
    kpt_xy_act=None,      # None: identity + clip, "sigmoid": sigmoid
    kpt_v_act="sigmoid",  # visibility 建議 sigmoid
):
    """
    輸出:
      pred_box       : (B,N,4)    [cx,cy,w,h] in [0,1]
      pred_cls_logit : (B,N,C)    logits
      pred_cls_prob  : (B,N,C)    prob (sigmoid/softmax)
      pred_kpt_flat  : (B,N,K*V)  kpt in [0,1]（xy保證在0~1，v為sigmoid）
    """

    C = 4 + num_cls + num_kpt * kpt_vals

    y = y_pred_raw
    # (B,C,N) -> (B,N,C)
    if tf.shape(y)[1] < tf.shape(y)[2]:
        y = tf.transpose(y, perm=[0, 2, 1])
    y = tf.reshape(y, (-1, tf.shape(y)[1], C))

    box_logits = y[..., :4]
    cls_logits = y[..., 4:4 + num_cls]
    kpt_logits = y[..., 4 + num_cls:]  # (B,N,K*V)

    # ---- box ----
    if box_act == "sigmoid":
        pred_box = tf.sigmoid(box_logits)
    elif box_act == "tanh":
        pred_box = (tf.tanh(box_logits) + 1.0) / 2.0
    elif box_act is None:
        pred_box = tf.clip_by_value(box_logits, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown box_act: {box_act}")

    # ---- cls ----
    pred_cls_logit = cls_logits
    if cls_act == "sigmoid":
        pred_cls_prob = tf.sigmoid(cls_logits)
    elif cls_act == "softmax":
        pred_cls_prob = tf.nn.softmax(cls_logits, axis=-1)
    elif cls_act is None:
        pred_cls_prob = cls_logits
    else:
        raise ValueError(f"Unknown cls_act: {cls_act}")

    # ---- kpt ----
    if num_kpt <= 0:
        pred_kpt_flat = tf.zeros_like(kpt_logits)
        return pred_box, pred_cls_logit, pred_cls_prob, pred_kpt_flat

    kpt = tf.reshape(kpt_logits, (-1, tf.shape(y)[1], num_kpt, kpt_vals))

    if kpt_vals == 2:
        kxy = kpt[..., :2]
        if kpt_xy_act == "sigmoid":
            kxy = tf.sigmoid(kxy)
        else:
            kxy = tf.clip_by_value(kxy, 0.0, 1.0)
        pred_kpt_flat = tf.reshape(kxy, (-1, tf.shape(y)[1], num_kpt * 2))

    elif kpt_vals == 3:
        kxy = kpt[..., :2]
        kv  = kpt[..., 2:3]

        if kpt_xy_act == "sigmoid":
            kxy = tf.sigmoid(kxy)
        else:
            kxy = tf.clip_by_value(kxy, 0.0, 1.0)

        if kpt_v_act == "sigmoid":
            kv = tf.sigmoid(kv)
        else:
            kv = tf.clip_by_value(kv, 0.0, 1.0)

        k_all = tf.concat([kxy, kv], axis=-1)
        pred_kpt_flat = tf.reshape(k_all, (-1, tf.shape(y)[1], num_kpt * 3))
    else:
        raise ValueError(f"Unsupported kpt_vals={kpt_vals}")

    return pred_box, pred_cls_logit, pred_cls_prob, pred_kpt_flat


def _cxcywh_to_xyxy(box):
    cx, cy, w, h = tf.unstack(box, axis=-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return tf.stack([x1, y1, x2, y2], axis=-1)

def task_aligned_assigner_single(
    pred_box,          # (N,4) in [0,1]
    pred_cls_prob,     # (N,C) in [0,1]
    anchors_all,       # (N,4) [ax,ay,cw,ch] in [0,1]
    gt_boxes,          # (M,4) cxcywh in [0,1]
    gt_cls,            # (M,) int
    topk=10,
    alpha=0.5,
    beta=6.0,
    center_radius=2.5,
    eps=1e-9
):
    """
    回傳:
      assigned_gt_idx : (N,) int32, -1 means negative
      assigned_cls    : (N,) int32, -1 means negative
      quality         : (N,) float32 in [0,1] (use IoU)
      pos_mask        : (N,) bool
    """
    N = tf.shape(pred_box)[0]
    M = tf.shape(gt_boxes)[0]

    # no gt
    def _no_gt():
        neg1 = tf.fill([N], tf.constant(-1, tf.int32))
        q0 = tf.zeros([N], tf.float32)
        pm = tf.zeros([N], tf.bool)
        return neg1, neg1, q0, pm

    def _has_gt():
        anchors_xy = anchors_all[:, :2]   # (N,2)
        cell_wh    = anchors_all[:, 2:4]  # (N,2)

        gt_xyxy = _cxcywh_to_xyxy(gt_boxes)  # (M,4)
        x1, y1, x2, y2 = tf.unstack(gt_xyxy, axis=-1)  # each (M,)

        ax = anchors_xy[:, 0][None, :]  # (1,N)
        ay = anchors_xy[:, 1][None, :]  # (1,N)

        # in-gt
        in_gt = (ax >= x1[:, None]) & (ax <= x2[:, None]) & (ay >= y1[:, None]) & (ay <= y2[:, None])

        # center sampling (radius proportional to cell size)
        gcx = gt_boxes[:, 0][:, None]  # (M,1)
        gcy = gt_boxes[:, 1][:, None]  # (M,1)
        rx  = center_radius * cell_wh[:, 0][None, :]  # (1,N)
        ry  = center_radius * cell_wh[:, 1][None, :]
        in_center = (tf.abs(ax - gcx) <= rx) & (tf.abs(ay - gcy) <= ry)

        candidates = in_gt & in_center
        # fallback: 若某 gt 沒候選，就放寬成 in_gt
        cand_any = tf.reduce_any(candidates, axis=1, keepdims=True)
        candidates = tf.where(cand_any, candidates, in_gt)

        # IoU matrix: (M,N)
        ious_NM = iou_batch(pred_box[None, ...], gt_boxes[None, ...])[0]  # (N,M)
        ious = tf.transpose(ious_NM, [1, 0])                              # (M,N)
        ious = ious * tf.cast(candidates, ious.dtype)

        # cls score of gt class: (M,N)
        cls_scores_NM = tf.gather(pred_cls_prob, gt_cls, axis=1)  # (N,M)
        cls_scores = tf.transpose(cls_scores_NM, [1, 0])          # (M,N)
        cls_scores = cls_scores * tf.cast(candidates, cls_scores.dtype)

        align = tf.pow(cls_scores + eps, alpha) * tf.pow(ious + eps, beta)  # (M,N)

        k = tf.minimum(topk, tf.shape(align)[1])
        topk_val, topk_idx = tf.math.top_k(align, k=k)  # (M,k)
        valid = topk_val > 0

        gt_ids = tf.repeat(tf.range(tf.shape(align)[0]), k)         # (M*k,)
        anc_ids = tf.reshape(topk_idx, [-1])                        # (M*k,)
        values = tf.reshape(tf.cast(valid, tf.float32), [-1])       # (M*k,)
        match = tf.scatter_nd(tf.stack([gt_ids, anc_ids], axis=1), values,
                              [tf.shape(align)[0], tf.shape(align)[1]])  # (M,N)
        match = match > 0

        match_float = tf.cast(match, tf.float32) * align
        best_gt = tf.argmax(match_float, axis=0, output_type=tf.int32)  # (N,)
        best_val = tf.reduce_max(match_float, axis=0)                   # (N,)
        pos = best_val > 0

        # quality = IoU(best_gt, anchor)
        iou_best = tf.gather_nd(ious, tf.stack([best_gt, tf.range(N, dtype=tf.int32)], axis=1))
        quality = tf.where(pos, tf.clip_by_value(iou_best, 0.0, 1.0), tf.zeros_like(iou_best))

        assigned_gt = tf.where(pos, best_gt, tf.fill([N], -1))
        assigned_cls = tf.where(pos, tf.gather(gt_cls, best_gt), tf.fill([N], -1))
        return assigned_gt, assigned_cls, tf.cast(quality, tf.float32), pos

    return tf.cond(M > 0, _has_gt, _no_gt)


# ==============================================================================
# Main Loss Function (Route A)
# ==============================================================================
def pose_loss_from_labels(
    y_pred,
    batch_dict,
    anchors,
    lambda_box=7.5,
    lambda_cls=1.0,
    lambda_kpt=1.0,
    num_cls=config.NUM_CLS,
    num_kpt=config.NUM_KPT,
    kpt_vals=config.KPT_VALS,
    eps=1e-7,
    class_weights=None
):
    """
    y_pred: (B,N,C) or (B,C,N)
    batch_dict:
      - "boxes": (B,M,4) cxcywh in [0,1]
      - "classes": (B,M,1) or (B,M)
      - "kpts": (B,M,K,V) in [0,1]
      - "valid_mask": (B,M) 0/1
    anchors: (N,4) [ax,ay,cw,ch] in [0,1]
    """



    gt_boxes   = batch_dict["bboxes"]
    gt_classes = batch_dict["cls"]
    gt_kpts    = batch_dict["keypoints"]
    valid_mask = batch_dict["valid_mask"]

    batch_size = tf.shape(gt_boxes)[0]
    max_gt     = tf.shape(gt_boxes)[1]

    if len(gt_classes.shape) == 3:
        gt_classes = tf.squeeze(gt_classes, axis=-1)
    gt_classes = tf.cast(gt_classes, tf.int32)  # (B,M)

    # ---- decode preds (IMPORTANT) ----
    pred_box, pred_cls_logit, pred_cls_prob, pred_kpt_flat = decode_yolo_pose_outputs(
        y_pred,
        num_cls=num_cls,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        box_act="sigmoid",      # 強制 box in [0,1]
        cls_act="sigmoid",      # cls prob for alignment
        kpt_xy_act=None,        # xy identity+clip
        kpt_v_act="sigmoid"     # v sigmoid
    )

    num_preds = tf.shape(pred_box)[1]
    anchors = tf.cast(anchors, pred_box.dtype)  # (N,4)
    anchors = tf.reshape(anchors, (1, num_preds, 4))
    anchors = tf.tile(anchors, [batch_size, 1, 1])  # (B,N,4)

    # ----------------------
    # 1) TAL assignment (per-image)
    # ----------------------
    TAL_TOPK = getattr(config, "TAL_TOPK", 10)
    TAL_ALPHA = getattr(config, "TAL_ALPHA", 0.5)
    TAL_BETA  = getattr(config, "TAL_BETA", 6.0)
    TAL_CENTER_RADIUS = getattr(config, "TAL_CENTER_RADIUS", 2.5)

    def _assign_one(inputs):
        pbox_i, pcl_i, anch_i, gtb_i, gtc_i, vm_i = inputs

        # valid gt slice
        m = tf.cast(tf.reduce_sum(vm_i), tf.int32)
        gtb = gtb_i[:m]
        gtc = gtc_i[:m]

        assigned_gt, assigned_cls, quality, pos = task_aligned_assigner_single(
            pred_box=pbox_i,
            pred_cls_prob=pcl_i,
            anchors_all=anch_i,
            gt_boxes=gtb,
            gt_cls=gtc,
            topk=TAL_TOPK,
            alpha=TAL_ALPHA,
            beta=TAL_BETA,
            center_radius=TAL_CENTER_RADIUS,
            eps=eps
        )
        return assigned_gt, assigned_cls, quality, pos

    out_sig = (
        tf.TensorSpec(shape=(None,), dtype=tf.int32),   # assigned_gt
        tf.TensorSpec(shape=(None,), dtype=tf.int32),   # assigned_cls
        tf.TensorSpec(shape=(None,), dtype=tf.float32), # quality
        tf.TensorSpec(shape=(None,), dtype=tf.bool),    # pos
    )

    assigned_gt_idx, assigned_cls, quality, pos_mask = tf.map_fn(
        _assign_one,
        (pred_box, pred_cls_prob, anchors, gt_boxes, gt_classes, tf.cast(valid_mask, tf.float32)),
        fn_output_signature=out_sig
    )

    pos_mask_f = tf.cast(pos_mask, tf.float32)               # (B,N)
    q = tf.stop_gradient(tf.clip_by_value(quality, 0.0, 1.0))# (B,N)
    pos_q = pos_mask_f * q                                   # quality-weighted positives
    den_pos_q = tf.reduce_sum(pos_q) + eps
    num_pos = tf.reduce_sum(pos_mask_f)
    num_pos_safe = tf.maximum(num_pos, 1.0)

    # ----------------------
    # 2) gather matched GT targets
    # ----------------------
    # safe index for negatives
    assigned_gt_safe = tf.where(pos_mask, assigned_gt_idx, tf.zeros_like(assigned_gt_idx))

    batch_indices = tf.reshape(tf.range(batch_size, dtype=tf.int32), (batch_size, 1))
    batch_indices = tf.tile(batch_indices, [1, num_preds])  # (B,N)
    gather_idx = tf.stack([batch_indices, assigned_gt_safe], axis=-1)  # (B,N,2)

    target_box = tf.gather_nd(gt_boxes, gather_idx)  # (B,N,4)
    target_kpt = tf.gather_nd(gt_kpts, gather_idx)   # (B,N,K,V)
    target_kpt_flat = tf.reshape(target_kpt, (batch_size, num_preds, num_kpt * kpt_vals))

    # ----------------------
    # 3) Box loss (CIoU) with quality weighting
    # ----------------------
    ciou_loss = bbox_ciou(pred_box, target_box)  # (B,N) = 1-CIoU

    # small object reweight (keep your original idea)
    box_area = target_box[..., 2] * target_box[..., 3]
    box_scale = 3.0 - box_area
    box_scale = tf.clip_by_value(box_scale, 1.0, 2.0)

    per_anchor_box = ciou_loss * box_scale

    # quality-weighted positives
    loss_box = tf.reduce_sum(per_anchor_box * pos_q) / den_pos_q
    loss_box = lambda_box * loss_box

    # ----------------------
    # 4) Class loss: Varifocal-style quality-aware BCE (NO obj head needed)
    # ----------------------
    # target: onehot * quality; negatives are all-zero
    cls_safe = tf.where(pos_mask, assigned_cls, tf.zeros_like(assigned_cls))
    t_onehot = tf.one_hot(cls_safe, depth=num_cls, dtype=pred_cls_logit.dtype)  # (B,N,C)
    t = t_onehot * tf.expand_dims(q, axis=-1) * tf.expand_dims(pos_mask_f, axis=-1)

    # Varifocal weighting
    VFL_ALPHA = getattr(config, "VFL_ALPHA", 0.75)
    VFL_GAMMA = getattr(config, "VFL_GAMMA", 2.0)

    p = tf.sigmoid(pred_cls_logit)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=pred_cls_logit)  # (B,N,C)

    # weight = t + alpha * p^gamma * (1 - t)
    weight = t + VFL_ALPHA * tf.pow(p, VFL_GAMMA) * (1.0 - t)
    loss_cls = tf.reduce_sum(bce * weight) / (den_pos_q + eps)
    loss_cls = lambda_cls * loss_cls

    # ----------------------
    # 5) Keypoint loss: keep your OKS + vis BCE, but use quality weighting
    # ----------------------
    if num_kpt > 0:
        pred_kpt = tf.reshape(pred_kpt_flat, (batch_size, num_preds, num_kpt, kpt_vals))
        targ_kpt = tf.reshape(target_kpt_flat, (batch_size, num_preds, num_kpt, kpt_vals))

        p_xy = pred_kpt[..., :2]
        t_xy = targ_kpt[..., :2]

        if kpt_vals == 3:
            p_v = pred_kpt[..., 2]
            t_v_raw = targ_kpt[..., 2]
            # label: >0 視為有標註/可見（依你資料定義可再微調）
            t_v = tf.cast(t_v_raw > 0.0, p_xy.dtype)
        else:
            p_v = None
            t_v = None

        # image size for pixel OKS
        imgsz = config.IMGSZ
        if isinstance(imgsz, (tuple, list)):
            H, W = imgsz
        else:
            H = W = imgsz
        H = tf.cast(H, p_xy.dtype)
        W = tf.cast(W, p_xy.dtype)

        p_xy_px = tf.stack([p_xy[..., 0] * W, p_xy[..., 1] * H], axis=-1)
        t_xy_px = tf.stack([t_xy[..., 0] * W, t_xy[..., 1] * H], axis=-1)
        d2 = tf.reduce_sum(tf.square(t_xy_px - p_xy_px), axis=-1)  # (B,N,K)

        area_px = (target_box[..., 2] * W) * (target_box[..., 3] * H)
        area_px = tf.maximum(area_px, 1.0)
        area_px = tf.expand_dims(area_px, axis=-1)  # (B,N,1)

        if num_kpt == 17:
            sigmas = tf.constant(
                [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
                 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
                dtype=p_xy.dtype
            ) / 10.0
        else:
            sigmas = tf.fill([num_kpt], tf.cast(0.05, p_xy.dtype))
        k = 2.0 * sigmas
        k2 = tf.reshape(tf.square(k), (1, 1, num_kpt))

        # visibility mask
        if t_v is not None:
            vis_mask = tf.cast(t_v > 0.0, p_xy.dtype)
        else:
            vis_mask = tf.ones_like(d2, dtype=p_xy.dtype)

        denom = 2.0 * area_px * k2 + eps
        oks_k = tf.exp(-d2 / denom) * vis_mask
        den_xy = tf.reduce_sum(vis_mask, axis=-1)  # (B,N)
        oks = tf.reduce_sum(oks_k, axis=-1) / (den_xy + eps)  # (B,N)

        oks_loss = 1.0 - oks
        oks_loss = tf.where(den_xy > 0.0, oks_loss, tf.zeros_like(oks_loss))

        # quality-weighted
        loss_xy = tf.reduce_sum(oks_loss * pos_q) / den_pos_q

        # visibility BCE (optional, small weight)
        if t_v is not None:
            p_v = tf.clip_by_value(p_v, eps, 1.0 - eps)
            v_bce = -(t_v * tf.math.log(p_v) + (1.0 - t_v) * tf.math.log(1.0 - p_v))  # (B,N,K)

            # weight by pos_q (B,N,1)
            v_mask = tf.expand_dims(pos_q, -1)
            v_bce = v_bce * v_mask
            loss_v = tf.reduce_sum(v_bce) / (tf.reduce_sum(v_mask) + eps)
        else:
            loss_v = tf.constant(0.0, dtype=tf.float32)

        kpt_vis_w = getattr(config, "KPT_VIS_W", 0.5)
        loss_kpt = lambda_kpt * (loss_xy + kpt_vis_w * loss_v)
    else:
        loss_kpt = tf.constant(0.0, dtype=tf.float32)

    total_loss = loss_box + loss_cls + loss_kpt
    return total_loss, loss_box, loss_cls, loss_kpt
