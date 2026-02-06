"""
PoseLabelLoss (TensorFlow): Train directly from YOLOv8-style pose labels.
Targets:
- boxes: [cx, cy, w, h] normalized to 0..1
- cls: one-hot (num_classes)
- kpts: [x, y, v] * K, where x,y are normalized 0..1 and v in {0,1,2} (2 will be mapped to 1)

This implements a light-weight approximation of Ultralytics v8 Pose loss:
- BCEWithLogits for classification
- CIOU for boxes (optionally SmoothL1 on xywh)
- Keypoint OKS-like loss + keypoint-objectness (kobj) BCE

References:
- Ultralytics loss reference (KeypointLoss, DFL, Varifocal etc.)
- v8PoseLoss uses KeypointLoss with OKS sigmas and a BCE head for pose visibility (kobj).
"""

import tensorflow as tf
from typing import Dict, Tuple
import numpy as np

# --- CIOU helpers (TensorFlow) ---
def bbox_xywh_to_xyxy(b):  # (...,4)
    cx, cy, w, h = tf.unstack(b, axis=-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return tf.stack([x1, y1, x2, y2], axis=-1)

def bbox_iou_xyxy(a, b, eps=1e-9):  # pairwise IoU for same-shape (...,4)
    ax1, ay1, ax2, ay2 = tf.unstack(a, axis=-1)
    bx1, by1, bx2, by2 = tf.unstack(b, axis=-1)
    inter_x1 = tf.maximum(ax1, bx1); inter_y1 = tf.maximum(ay1, by1)
    inter_x2 = tf.minimum(ax2, bx2); inter_y2 = tf.minimum(ay2, by2)
    iw = tf.maximum(0.0, inter_x2 - inter_x1)
    ih = tf.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    aa = tf.maximum(0.0, (ax2-ax1)) * tf.maximum(0.0, (ay2-ay1))
    bb = tf.maximum(0.0, (bx2-bx1)) * tf.maximum(0.0, (by2-by1))
    union = aa + bb - inter + eps
    return inter / union

def bbox_ciou_xywh(p, t, eps=1e-9):
    # Complete IoU (https://arxiv.org/abs/1911.08287)
    pxyxy = bbox_xywh_to_xyxy(p)
    txyxy = bbox_xywh_to_xyxy(t)
    iou = bbox_iou_xyxy(pxyxy, txyxy, eps)

    # center distance
    pcx, pcy, pw, ph = tf.unstack(p, axis=-1)
    tcx, tcy, tw, th = tf.unstack(t, axis=-1)
    c_dist = (pcx - tcx)**2 + (pcy - tcy)**2

    # enclosing box diagonal
    x1 = tf.minimum(pxyxy[...,0], txyxy[...,0]); y1 = tf.minimum(pxyxy[...,1], txyxy[...,1])
    x2 = tf.maximum(pxyxy[...,2], txyxy[...,2]); y2 = tf.maximum(pxyxy[...,3], txyxy[...,3])
    cw = tf.maximum(0.0, x2 - x1); ch = tf.maximum(0.0, y2 - y1)
    c_diag = cw**2 + ch**2 + eps

    # aspect ratio term
    v = (4 / (3.14159265**2)) * tf.square(tf.atan(tw/(th+eps)) - tf.atan(pw/(ph+eps)))
    with tf.device("/CPU:0"):
        alpha = tf.stop_gradient( v / (1 - iou + v + eps) )
    ciou = iou - c_dist / c_diag - alpha * v
    return ciou

def oks_loss(pred_kxy, gt_kxy, kpt_v, bbox_area, sigmas):
    """
    pred_kxy: (B,N,K,2) predicted keypoint xy in 0..1 domain
    gt_kxy:   (B,N,K,2) target keypoint xy in 0..1
    kpt_v:    (B,N,K,1) visibility flag in {0,1} (2 is mapped to 1 upstream)
    bbox_area:(B,N,1)   target box area in 0..1^2
    sigmas:   (K,)      per-kpt sigma like COCO OKS
    returns:  scalar loss
    """
    # squared distance in absolute pixel fraction
    d2 = tf.reduce_sum(tf.square(pred_kxy - gt_kxy), axis=-1, keepdims=True)  # (B,N,K,1)
    # scale by area and sigma (broadcast)
    sig = tf.reshape(tf.convert_to_tensor(sigmas, dtype=tf.float32), [1,1,-1,1])
    denom = 2.0 * tf.square(sig) * (bbox_area[...,None] + 1e-9)  # (B,N,1,1)->(B,N,1,1)
    e = tf.exp(-d2 / tf.maximum(denom, 1e-9))
    oks = (1.0 - e) * kpt_v  # only count visible
    # mean over visible points
    num_vis = tf.reduce_sum(kpt_v, axis=[2,3]) + 1e-9  # (B,N)
    per_inst = tf.reduce_sum(oks, axis=[2,3]) / num_vis
    return tf.reduce_mean(per_inst)

class PoseLabelLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes:int, num_kpt:int, kpt_vals:int=3,
                 lambda_box=7.5, lambda_cls=1.0, lambda_kpt=10.0, lambda_kobj=2.0,
                 use_ciou=True, sigmas=None, name="PoseLabelLoss"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.num_kpt = num_kpt
        self.kpt_vals = kpt_vals
        self.lambda_box = float(lambda_box)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)
        self.lambda_kobj = float(lambda_kobj)
        self.use_ciou = bool(use_ciou)
        if sigmas is None:
            # COCO 17 default; if num_kpt!=17 we fallback to uniform
            if num_kpt == 17:
                self.sigmas = [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79,
                               0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
            else:
                self.sigmas = [1.0/num_kpt]*num_kpt
        else:
            self.sigmas = list(sigmas)

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.smoothl1 = tf.keras.losses.Huber(delta=1.0, reduction="none")

    def call(self, y_true, y_pred):
        """
        y_true: target dict (含 t_box/t_cls/t_kxy/t_kv/t_area/pos_mask)
        y_pred: pred_BNC, shape (B,N,C)
        """
        target   = y_true
        pred_BNC = y_pred
        B  = tf.shape(pred_BNC)[0]
        Np = tf.shape(pred_BNC)[1]
        # 取 pm 並確保 rank=3
        pm = tf.cast(target.get('pos_mask'), tf.float32)
        pm = tf.reshape(pm, [tf.shape(pm)[0], tf.shape(pm)[1], 1])
        Nt = tf.shape(pm)[1]
        # 對齊 N（避免 broadcast error）
        minN = tf.minimum(Np, Nt)
        def _slice_to_minN(x): return (x[:, :minN, ...] if x is not None else None)
        pred_BNC = tf.cond(tf.not_equal(Np, Nt),
                           lambda: pred_BNC[:, :minN, :],
                           lambda: pred_BNC)
        pm             = tf.cond(tf.not_equal(Np, Nt), lambda: pm[:, :minN, :], lambda: pm)
        target         = dict(target)
        target['t_box']  = _slice_to_minN(target['t_box'])
        target['t_kxy']  = _slice_to_minN(target['t_kxy'])
        target['t_kv']   = _slice_to_minN(target['t_kv'])
        target['t_area'] = _slice_to_minN(target['t_area'])
        if 't_cls' in target and target['t_cls'] is not None:
            target['t_cls'] = _slice_to_minN(target['t_cls'])
        N = tf.shape(pred_BNC)[1]

        # slice channels
        c_box = 4
        c_cls = self.num_classes
        c_kpt = self.num_kpt * self.kpt_vals
        box = pred_BNC[...,:4]
        cls_logits = pred_BNC[...,4:4+c_cls] if c_cls>0 else None
        kpt = pred_BNC[...,4+c_cls:4+c_cls+c_kpt]

        # reshape kpts
        kpt = tf.reshape(kpt, [B, N, self.num_kpt, self.kpt_vals])
        kxy = kpt[...,:2]
        kv_logit = kpt[...,2:3] if self.kpt_vals>=3 else None

        # masks
        pm = tf.cast(pm, tf.float32)        # (B,N,1)

        w   = tf.cast(tf.squeeze(pm, -1), tf.float32)          # (B, N)
        den = tf.reduce_sum(w) + 1e-9
        # --- Box loss (只算正樣本) ---
        t_box = target['t_box']
        if self.use_ciou:
            ciou = bbox_ciou_xywh(box, t_box)                  # (B, N)
            l_box = tf.reduce_sum((1.0 - ciou) * w) / den
        else:
            box_l1 = tf.reduce_sum(self.smoothl1(box, t_box), axis=-1)  # (B, N)
            l_box  = tf.reduce_sum(box_l1 * w) / den



        
        # --- Cls loss (BCE on positives only) ---
        l_cls = tf.constant(0.0, tf.float32)
        if c_cls > 0:
            # 取得 per-class BCE，不做任何隱性 reduce：形狀 (B, N, C)
            bce_per_class = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(target['t_cls'], tf.float32),
                logits=tf.cast(cls_logits, tf.float32)
            )
            # 先把 per-class 平均成 per-anchor：形狀 (B, N)
            cls_per_anchor = tf.reduce_mean(bce_per_class, axis=-1)
            # 防呆（可留著）
            tf.debugging.assert_equal(tf.shape(cls_per_anchor), tf.shape(w),
                                    message="cls_per_anchor vs pos_mask shape mismatch")
            # 只用正樣本做加權平均
            l_cls = tf.cond(
                tf.greater(den, 0.0),
                lambda: tf.reduce_sum(cls_per_anchor * w) / den,
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        # --- Keypoint losses ---
        l_kpt  = tf.constant(0.0, tf.float32)
        l_kobj = tf.constant(0.0, tf.float32)
        if self.num_kpt > 0:
            # 你原本的 OKS（標量）不動
            l_kpt = oks_loss(kxy, target['t_kxy'], target['t_kv'], target['t_area'], self.sigmas)

            if kv_logit is not None:
                # 可見度同樣用低階 BCE，保留完整形狀 (B, N, K, 1)
                kobj_per_el = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(target['t_kv'], tf.float32),
                    logits=tf.cast(kv_logit, tf.float32)
                )  # (B, N, K, 1)
                # 先在 (K,1) 平均到 per-anchor：得到 (B, N)
                kobj_per_inst = tf.reduce_mean(kobj_per_el, axis=[2, 3])  # (B, N)
                tf.debugging.assert_equal(tf.shape(kobj_per_inst), tf.shape(w),
                                        message="kobj_per_inst vs pos_mask shape mismatch")
                l_kobj = tf.cond(
                    tf.greater(den, 0.0),
                    lambda: tf.reduce_sum(kobj_per_inst * w) / den,
                    lambda: tf.constant(0.0, dtype=tf.float32)
                )

        # # --- Box loss (only positives) ---
        # t_box = target['t_box']
        # if self.use_ciou:
        #     ciou = bbox_ciou_xywh(box, t_box)  # (B,N)
        #     # ciou loss = 1 - ciou
        #     l_box = tf.reduce_sum((1.0 - ciou) * tf.squeeze(pm, -1)) / (tf.reduce_sum(pm) + 1e-9)
        # else:
        #     l_box = tf.reduce_sum(tf.reduce_sum(self.smoothl1(box, t_box), axis=-1) * tf.squeeze(pm,-1)) / (tf.reduce_sum(pm) + 1e-9)

        # # --- Cls loss (BCE on positives only; background ignored here) ---
        # l_cls = 0.0
        # if c_cls>0:
        #     l_cls = self.bce(target['t_cls'], cls_logits)  # (B,N,C)
        #     l_cls = tf.reduce_sum(tf.reduce_mean(l_cls, axis=-1) * tf.squeeze(pm,-1)) / (tf.reduce_sum(pm) + 1e-9)

        # # --- Keypoint losses ---
        # l_kpt = 0.0
        # l_kobj = 0.0
        # if self.num_kpt>0:
        #     l_kpt = oks_loss(kxy, target['t_kxy'], target['t_kv'], target['t_area'], self.sigmas)  # scalar
        #     if kv_logit is not None:
        #         l_kobj_raw = self.bce(target['t_kv'], kv_logit)  # (B,N,K,1)
        #         # only positives (instances) contribute
        #         l_kobj = tf.reduce_sum(tf.reduce_mean(l_kobj_raw, axis=[2,3]) * tf.squeeze(pm,-1)) / (tf.reduce_sum(pm) + 1e-9)

        total = self.lambda_box*l_box + self.lambda_cls*l_cls + self.lambda_kpt*l_kpt + self.lambda_kobj*l_kobj
        return total, {'box': l_box, 'cls': l_cls, 'kpt': l_kpt, 'kobj': l_kobj}

# --- Target builder ---
def build_grid_shapes(imgsz:int=640, strides=(8,16,32)):
    shapes = [(imgsz//s, imgsz//s) for s in strides]
    N = sum(h*w for (h,w) in shapes)
    return shapes, N

def build_targets_from_labels(batch_labels, num_classes:int, num_kpt:int, kpt_vals:int, imgsz:int=640, strides=(8,16,32)):
    """
    batch_labels: list of Ragged/np arrays per image, each is [M_i, 5 + num_kpt*kpt_vals] in YOLO format (cx,cy,w,h normalized)
                  the first column is integer class id in [0,num_classes-1]
    Return:
      target dict tensors of shapes:
        t_box:  (B,N,4), t_cls:(B,N,C), t_kxy:(B,N,K,2), t_kv:(B,N,K,1), t_area:(B,N,1)
        pos_mask: (B,N,1)
    Assignment strategy: one positive per GT per pyramid level (nearest grid cell to GT center).
    """
    B = len(batch_labels)
    shapes, N = build_grid_shapes(imgsz, strides)
    Hs = [h for (h,w) in shapes]; Ws = [w for (h,w) in shapes]

    # allocate
    t_box  = np.zeros((B, N, 4), dtype=np.float32)
    t_cls  = np.zeros((B, N, num_classes), dtype=np.float32) if num_classes>0 else None
    t_kxy  = np.zeros((B, N, num_kpt, 2), dtype=np.float32)
    t_kv   = np.zeros((B, N, num_kpt, 1), dtype=np.float32)
    t_area = np.zeros((B, N, 1), dtype=np.float32)
    pos    = np.zeros((B, N, 1), dtype=np.float32)

    # offsets per level within N
    level_offsets = [0]
    for idx in range(len(shapes)-1):
        level_offsets.append(level_offsets[-1] + Hs[idx]*Ws[idx])

    for b in range(B):
        gts = np.asarray(batch_labels[b], dtype=np.float32)
        if gts.size == 0:
            continue
        # columns: cls, cx, cy, w, h, (kpts...)
        cls_ids = gts[:,0].astype(np.int32)
        cx, cy, w, h = gts[:,1], gts[:,2], gts[:,3], gts[:,4]
        area = w*h
        kpts = gts[:,5:5+num_kpt*kpt_vals].reshape(-1, num_kpt, kpt_vals) if num_kpt>0 else None

        for gi in range(gts.shape[0]):
            for lvl,(H,W) in enumerate(shapes):
                # choose nearest grid cell to center
                ix = int(np.clip(np.floor(cx[gi] * W), 0, W-1))
                iy = int(np.clip(np.floor(cy[gi] * H), 0, H-1))
                n  = level_offsets[lvl] + iy*W + ix

                pos[b, n, 0] = 1.0
                t_box[b, n, :] = [cx[gi], cy[gi], w[gi], h[gi]]
                if num_classes>0:
                    t_cls[b, n, cls_ids[gi]] = 1.0
                t_area[b, n, 0] = area[gi]
                if num_kpt>0:
                    xy = kpts[gi,:,:2]  # (K,2)
                    v  = kpts[gi,:,2:3] if kpts.shape[-1]>=3 else np.ones((num_kpt,1), np.float32)
                    v = (v>0).astype(np.float32)  # map {0,1,2}->{0,1}
                    t_kxy[b, n, :, :] = xy
                    t_kv[b, n, :, :]  = v

    # convert to tensors
    out = {
        't_box' : tf.convert_to_tensor(t_box),
        't_kxy' : tf.convert_to_tensor(t_kxy),
        't_kv'  : tf.convert_to_tensor(t_kv),
        't_area': tf.convert_to_tensor(t_area),
        'pos_mask': tf.convert_to_tensor(pos),
    }
    if num_classes>0:
        out['t_cls'] = tf.convert_to_tensor(t_cls)
    pos_mask = tf.convert_to_tensor(pos)
    out['pos_mask'] = pos_mask
    return out, pos_mask, shapes


