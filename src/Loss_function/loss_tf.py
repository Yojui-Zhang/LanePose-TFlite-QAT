import tensorflow as tf
import numpy as np
import config

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

def decode_yolo_pose_outputs(
    y_pred_raw,
    num_cls,
    num_kpt,
    kpt_vals,
    box_act="sigmoid",
    cls_act=None,
    kpt_act="sigmoid",
):
    """
    將模型輸出的 logits 解析成:
    - pred_box       : (B, N, 4)      -> [cx, cy, w, h] in [0,1]
    - pred_cls_logit : (B, N, num_cls)-> 分類 logits
    - pred_cls_prob  : (B, N, num_cls)-> 分類機率 (如需要)
    - pred_kpt_flat  : (B, N, K*V)    -> 關鍵點 (0~1)

    y_pred_raw: (B, N, 4 + num_cls + num_kpt*kpt_vals) 或 (B, C, N)
                （你在 Train_Model 裡已經轉成 BNC 了）
    """

    C = 4 + num_cls + num_kpt * kpt_vals

    # 保險：如果是 (B, C, N) 就轉成 (B, N, C)
    y = y_pred_raw
    if tf.shape(y)[1] < tf.shape(y)[2]:  # 例如 (B, 56, 8400)
        y = tf.transpose(y, perm=[0, 2, 1])  # -> (B, 8400, 56)

    y = tf.reshape(y, (-1, tf.shape(y)[1], C))  # (B, N, C)

    box_logits = y[..., :4]                       # (B, N, 4)
    cls_logits = y[..., 4:4+num_cls]             # (B, N, num_cls)
    kpt_logits = y[..., 4+num_cls:]              # (B, N, K*V)

    # --- boxes ---
    if box_act == "sigmoid":
        pred_box = tf.sigmoid(box_logits)        # clamp 到 0~1
    elif box_act == "tanh":
        pred_box = (tf.tanh(box_logits) + 1.0) / 2.0
    elif box_act is None:
        pred_box = box_logits                    # 不做任何 decode
    else:
        raise ValueError(f"Unknown box_act: {box_act}")

    # --- class ---
    pred_cls_logit = cls_logits
    if cls_act == "sigmoid":
        pred_cls_prob = tf.sigmoid(cls_logits)
    elif cls_act == "softmax":
        pred_cls_prob = tf.nn.softmax(cls_logits, axis=-1)
    elif cls_act is None:
        pred_cls_prob = cls_logits               # 只當 logits，用在 KD
    else:
        raise ValueError(f"Unknown cls_act: {cls_act}")

    # --- keypoints ---
    if kpt_act == "sigmoid":
        pred_kpt_flat = tf.sigmoid(kpt_logits)   # (B, N, K*V) in [0,1]
    elif kpt_act is None:
        pred_kpt_flat = kpt_logits
    else:
        raise ValueError(f"Unknown kpt_act: {kpt_act}")

    return pred_box, pred_cls_logit, pred_cls_prob, pred_kpt_flat

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
    lambda_cls=7.0,
    lambda_kpt=14.0,
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
    # C = 4 + num_cls + num_kpt * kpt_vals
    # y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[1], C))   # (B, N, C)

    # idx_box_end = 4
    # idx_cls_end = 4 + num_cls

    # pred_box      = y_pred[..., :idx_box_end]          # (B, N, 4)，0~1 [cx,cy,w,h]
    # pred_cls_prob = y_pred[..., idx_box_end:idx_cls_end]  # (B, N, num_cls)，視為機率 0~1
    # pred_kpt_flat = y_pred[..., idx_cls_end:]          # (B, N, K*V)，0~1

    # ----------------------
    pred_box, pred_cls_logit, pred_cls_prob, pred_kpt_flat = decode_yolo_pose_outputs(
        y_pred,
        num_cls=num_cls,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        box_act=None,    # xywh -> 0~1
        cls_act="sigmoid",         # 保留 logits 給 loss，用不到機率就不 decode
        kpt_act=None,    # kpt -> 0~1
    )
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

    # box_diff = huber_no_reduce(target_box, pred_box)      # (B, N, 4)
    # loss_box = tf.reduce_sum(tf.reduce_mean(box_diff, axis=-1) * pos_mask_f)
    # loss_box = lambda_box * (loss_box / num_pos_safe)
    # -------------------
    # ciou_loss = bbox_ciou(pred_box, target_box)    # (B, N)
    # loss_box = tf.reduce_sum(ciou_loss * pos_mask_f)
    # loss_box = lambda_box * (loss_box / num_pos_safe)
    # -------------------
    # 4.1 CIoU 主要項
    ciou_loss = bbox_ciou(pred_box, target_box)    # (B, N), = 1 - CIoU
    # ciou_loss = tf.square(ciou_loss)             # 小誤差權重放大

    # 4.2 把 cx,cy,w,h 轉成像素做 Huber（讓微小偏移也有明顯罰則）
    # -----------
    imgsz = config.IMGSZ
    if isinstance(imgsz, (tuple, list)):
        H, W = imgsz
    else:
        H = W = imgsz

    H = tf.cast(H, pred_box.dtype)
    W = tf.cast(W, pred_box.dtype)

    # 用 tf.stack 堆疊 scalar tensor，而不是 tf.constant 包 list
    scale_px = tf.stack([W, H, W, H])          # 形狀 (4,)
    scale_px = tf.reshape(scale_px, (1, 1, 4)) # → (1,1,4) 方便 broadcast

    # -----------

    target_box_px = target_box * scale_px                      # (B, N, 4)
    pred_box_px   = pred_box   * scale_px                      # (B, N, 4)

    box_diff  = huber_no_reduce(target_box_px, pred_box_px)    # (B, N, 4)
    box_huber = tf.reduce_mean(box_diff, axis=-1)              # (B, N) 對 4 個參數平均

    # 4.3 對小物件加權（類 YOLOv5：小 box loss 放大）
    box_area = target_box[..., 2] * target_box[..., 3]         # (B, N), normalized area
    box_scale = 2.0 - box_area                                 # 小 box → 係數接近 2
    box_scale = tf.clip_by_value(box_scale, 1.0, 2.0)          # 避免太極端

    # 4.4 組合成單一 box loss
    alpha = 0.25  # Huber 權重（超參數，可調）
    per_anchor_box = (ciou_loss + alpha * box_huber) * box_scale   # (B, N)

    # 只對正樣本 anchor 做 box loss
    loss_box = tf.reduce_sum(per_anchor_box * pos_mask_f)      # scalar
    loss_box = lambda_box * (loss_box / num_pos_safe)

    # ----------------------
    # 5) Class Loss（使用 logits 的 BCE + 正負樣本權重）
    # ----------------------
    # if num_cls > 0:
    #     if len(target_cls.shape) == 3:
    #         target_cls = tf.squeeze(target_cls, -1)
    #     target_cls = tf.cast(target_cls, tf.int32)      # (B, N)

    #     # one-hot → (B, N, num_cls)
    #     t_cls_onehot = tf.one_hot(target_cls, depth=num_cls, dtype=pred_cls_logit.dtype)

    #     # 只在正樣本上有 label（負樣本當作全 0）
    #     pos_mask_exp = tf.expand_dims(pos_mask_f, -1)   # (B, N, 1)
    #     t_cls_onehot = t_cls_onehot * pos_mask_exp      # 背景 anchor label = 全 0

    #     # logits 版 Binary Cross Entropy:
    #     # loss = max(x,0) - x*z + log(1 + exp(-|x|))
    #     x = pred_cls_logit
    #     z = t_cls_onehot

    #     cls_loss_per_class = (
    #         tf.nn.relu(x) - x * z + tf.math.log1p(tf.exp(-tf.abs(x)))
    #     )  # (B, N, num_cls)

    #     # 類別權重（可選）
    #     if class_weights is not None:
    #         cw = tf.convert_to_tensor(class_weights, dtype=cls_loss_per_class.dtype)
    #         cw = tf.reshape(cw, [1, 1, num_cls])   # (1, 1, C)
    #         cls_loss_per_class = cls_loss_per_class * cw

    #     # 對類別做 sum → (B, N)
    #     cls_loss_per_anchor = tf.reduce_sum(cls_loss_per_class, axis=-1)

    #     # 正樣本 1.0，負樣本 0.1 （你可以依需要調）
    #     neg_weight = 0.5
    #     weights = pos_mask_f * 1.0 + (1.0 - pos_mask_f) * neg_weight  # (B, N)

    #     loss_cls = tf.reduce_sum(cls_loss_per_anchor * weights)
    #     loss_cls = lambda_cls * (loss_cls / num_pos_safe)
    # else:
    #     loss_cls = tf.constant(0.0, dtype=tf.float32)
    
    # ------------------
    
    if num_cls > 0:
        # 先把 target_cls 攤平成 (B, N)
        if len(target_cls.shape) == 3:
            target_cls = tf.squeeze(target_cls, -1)
        target_cls = tf.cast(target_cls, tf.int32)      # (B, N)

        # one-hot → (B, N, num_cls)，只對正樣本 anchor 有有效 label
        t_cls_onehot = tf.one_hot(target_cls, depth=num_cls,
                                  dtype=pred_cls_logit.dtype)  # (B, N, C)
        pos_mask_exp = tf.expand_dims(pos_mask_f, -1)          # (B, N, 1)
        t_cls_onehot = t_cls_onehot * pos_mask_exp             # 負樣本的 one-hot 全 0

        # 預測機率 (B, N, C)
        p_cls = pred_cls_prob
        eps = 1e-6

        # --------------------------------------------------
        # 5.1 隱式 Objectness Loss：p_obj = Σ_c p_cls,c
        # --------------------------------------------------
        # p_obj 代表「這個 anchor 有物件」的機率：所有類別機率加總，clamp 到 [0,1]
        p_obj = tf.reduce_sum(p_cls, axis=-1)           # (B, N)
        p_obj = tf.clip_by_value(p_obj, 0.0, 1.0)

        t_obj = pos_mask_f                              # (B, N)，有物件就 1，背景 0

        # BCE on objectness
        obj_ce = - (t_obj * tf.math.log(p_obj + eps) +
                    (1.0 - t_obj) * tf.math.log(1.0 - p_obj + eps))   # (B, N)

        # Focal-like weight for objectness
        gamma_obj = 2.0
        pt_obj = t_obj * p_obj + (1.0 - t_obj) * (1.0 - p_obj)        # (B, N)
        focal_w_obj = tf.pow(1.0 - pt_obj, gamma_obj)

        loss_obj = tf.reduce_sum(obj_ce * focal_w_obj) / num_pos_safe

        # --------------------------------------------------
        # 5.2 Conditional Class Loss：只在正樣本上做 Focal BCE
        # --------------------------------------------------
        z = t_cls_onehot                                     # (B, N, C)
        x = pred_cls_logit                                   # (B, N, C)
        p = p_cls                                            # sigmoid(x)

        # standard BCE
        ce_cls = - (z * tf.math.log(p + eps) +
                    (1.0 - z) * tf.math.log(1.0 - p + eps))  # (B, N, C)

        # focal weight：對正樣本較高，避免少樣本被 easy negative 淹沒
        gamma_cls = 2.0
        # pt = p if z=1 else 1-p
        pt = z * p + (1.0 - z) * (1.0 - p)
        alpha_pos = 0.75
        alpha_neg = 0.25
        alpha_factor = alpha_pos * z + alpha_neg * (1.0 - z)

        focal_w_cls = alpha_factor * tf.pow(1.0 - pt, gamma_cls)

        cls_loss_per_class = ce_cls * focal_w_cls          # (B, N, C)

        # 類別權重（針對少樣本類別可加大權重）
        if class_weights is not None:
            cw = tf.convert_to_tensor(class_weights,
                                      dtype=cls_loss_per_class.dtype)  # (C,)
            cw = tf.reshape(cw, [1, 1, num_cls])                        # (1,1,C)
            cls_loss_per_class = cls_loss_per_class * cw

        # 把負樣本的 class loss 壓得更低（主要靠 obj_loss 處理背景）
        # pos: 1.0, neg: small
        neg_cls_weight = 0.5
        anchor_weight = pos_mask_exp + (1.0 - pos_mask_exp) * neg_cls_weight  # (B,N,1)
        cls_loss_per_class = cls_loss_per_class * anchor_weight

        # 對類別 sum → (B, N)，再 sum → scalar
        cls_loss_per_anchor = tf.reduce_sum(cls_loss_per_class, axis=-1)      # (B,N)
        loss_cls_cond = tf.reduce_sum(cls_loss_per_anchor) / num_pos_safe

        # --------------------------------------------------
        # 5.3 合併 obj + cls
        # --------------------------------------------------
        lambda_obj = 1.0   # 你可以視情況調整這個係數
        loss_cls = lambda_cls * (loss_cls_cond + lambda_obj * loss_obj)

    else:
        loss_cls = tf.constant(0.0, dtype=tf.float32)



    # ----------------------
    # 6) Keypoint Loss（直接對齊 pred_save_model 的 kpt domain）
    # ----------------------
    # if num_kpt > 0 and kpt_vals > 0:
    #     kpt_diff = huber_no_reduce(target_kpt_flat, pred_kpt_flat)  # (B, N, K*V)
    #     loss_kpt = tf.reduce_sum(tf.reduce_mean(kpt_diff, axis=-1) * pos_mask_f)
    #     loss_kpt = lambda_kpt * (loss_kpt / num_pos_safe)
    # else:
    #     loss_kpt = tf.constant(0.0, dtype=tf.float32)

    # ---------- Re Image ------------
    if num_kpt > 0 and kpt_vals > 0:
        # target_kpt_flat, pred_kpt_flat: (B, N, K*V)
        B = tf.shape(pred_kpt_flat)[0]
        N = tf.shape(pred_kpt_flat)[1]

        # 回復成 (B, N, K, V)
        t_kpt = tf.reshape(target_kpt_flat, (B, N, num_kpt, kpt_vals))
        p_kpt = tf.reshape(pred_kpt_flat, (B, N, num_kpt, kpt_vals))

        # 取得影像尺寸 (假設 config.IMGSZ 是單一邊長，例如 640)
        img_size = tf.cast(config.IMGSZ, t_kpt.dtype)

        # 只把 x, y 放大到像素座標，v 繼續保留 0~1
        # t_xy, p_xy: (B, N, K, 2) 單位 = pixel
        t_xy = t_kpt[..., :2] * img_size
        p_xy = p_kpt[..., :2] * img_size

        if kpt_vals > 2:
            # 其餘維度（通常是 visibility）可以維持原本 0~1 domain
            t_rest = t_kpt[..., 2:]  # (B, N, K, V-2)
            p_rest = p_kpt[..., 2:]

            # 決定要不要對 v 也做 Huber：
            # 方案1：也對 v 做 Huber（但不縮放）
            t_kpt_scaled = tf.concat([t_xy, t_rest], axis=-1)
            p_kpt_scaled = tf.concat([p_xy, p_rest], axis=-1)
        else:
            # 只有 x,y
            t_kpt_scaled = t_xy      # (B, N, K, 2)
            p_kpt_scaled = p_xy

        # 在像素座標 (以及 v) 上算 Huber，不做 reduce
        kpt_diff = huber_no_reduce(t_kpt_scaled, p_kpt_scaled)  # (B, N, K, ?)

        # 攤回 (B, N, K*?)
        kpt_diff = tf.reshape(kpt_diff, (B, N, -1))             # (B, N, K*V')

        # 先在最後一維平均 → 每個 anchor 一個 scalar loss
        per_anchor_kpt = tf.reduce_mean(kpt_diff, axis=-1)      # (B, N)

        # 只對正樣本 anchor 做 keypoint loss
        loss_kpt = tf.reduce_sum(per_anchor_kpt * pos_mask_f)   # scalar

        # 做正樣本數量的平均、加上權重
        loss_kpt = lambda_kpt * (loss_kpt / num_pos_safe)
    else:
        loss_kpt = tf.constant(0.0, dtype=tf.float32)



    # ----------------------
    # 7) Total
    # ----------------------
    total_loss = loss_box + loss_cls + loss_kpt
    return total_loss, loss_box, loss_cls, loss_kpt
