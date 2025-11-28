# src/Loss_function/loss_tf.py
import tensorflow as tf
import math

import config
from src.process.pred_model import (split_BNC, ensure_BNC_static)


# ---------- 主 Loss 類別 ----------

# ==========================
# box IoU helper
# ==========================

def huber_no_reduce(y_true, y_pred, delta=1.0):
    """
    Huber loss，不做任何 reduction，回傳 shape = y_true / y_pred 的 shape。
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    diff = y_pred - y_true
    abs_diff = tf.abs(diff)
    small = 0.5 * tf.square(diff)
    big   = delta * (abs_diff - 0.5 * delta)
    return tf.where(abs_diff <= delta, small, big)  # same shape

def xywh_to_xyxy(box_xywh):
    """
    box_xywh: (...,4) with (x,y,w,h), x,y 為中心座標 (0~1)，w,h 為寬高 (0~1)
    回傳 xyxy: (...,4) => x1,y1,x2,y2 (0~1)
    """
    x, y, w, h = tf.unstack(box_xywh, axis=-1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return tf.stack([x1, y1, x2, y2], axis=-1)


def bbox_iou_matrix(boxes1, boxes2, eps=1e-9):
    """
    計算 IoU 矩陣:
    boxes1: (M,4) xyxy
    boxes2: (N,4) xyxy
    回傳: (M,N)
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)

    b1 = tf.expand_dims(boxes1, 1)  # (M,1,4)
    b2 = tf.expand_dims(boxes2, 0)  # (1,N,4)

    inter_x1 = tf.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = tf.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = tf.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = tf.minimum(b1[..., 3], b2[..., 3])

    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (M,)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (N,)

    area1 = tf.expand_dims(area1, 1)
    area2 = tf.expand_dims(area2, 0)

    union = area1 + area2 - inter_area + eps
    iou = inter_area / union
    return iou
def pose_loss_from_labels(
    pred_raw,
    batch_dict,
    num_cls,
    num_kpt,
    kpt_vals,
    img_size,
    lambda_box=5.0,
    lambda_kxy=9.0,
    lambda_v=1.0,
    lambda_cls=1.0,
):
    """
    pred_raw: student label 分支原始輸出（跟 distill 的 kd_raw / deploy_raw 同格式）
    batch_dict: build_batch_dict_from_targets() 的結果，需包含：
        - 'batch_idx': (G,) 或 (G,1)
        - 'cls':       (G,) 或 (G,1)
        - 'bboxes':    (G,4)  (xywh, 0~1 normalized)
        - 'keypoints': (G,num_kpt,kpt_vals)  常見 [x,y,v]
    img_size: 若 keypoints 是像素，這裡給 (W,H)；若已是 0~1，可以給 (1.0,1.0)
    """

    # ------------------------------------------------------
    # 0. Pred 展平 + 拆解（跟 distill 一模一樣）
    # ------------------------------------------------------
    C = 4 + num_cls + num_kpt * kpt_vals

    BNC = ensure_BNC_static(pred_raw, C)   # (B, N, C)
    box_pred, cls_pred, kxy_pred, v_pred = split_BNC(BNC, num_cls, num_kpt, kpt_vals)

    box_pred = tf.cast(box_pred, tf.float32)  # (B,N,4)
    if cls_pred is not None:
        cls_pred = tf.cast(cls_pred, tf.float32)  # (B,N,num_cls)
    if kxy_pred is not None:
        kxy_pred = tf.cast(kxy_pred, tf.float32)  # (B,N,num_kpt,2)  <-- 重要：沿用你 split_BNC 的格式
    if v_pred is not None:
        v_pred = tf.cast(v_pred, tf.float32)      # (B,N,num_kpt)


    # 先取得 B, N（從 box_pred 就可以）
    B_dyn = tf.shape(box_pred)[0]
    N_dyn = tf.shape(box_pred)[1]

    # 🔧 關鍵：把 v_pred 壓成 (B,N,num_kpt)
    if v_pred is not None:
        v_pred = tf.reshape(v_pred, [B_dyn, N_dyn, num_kpt])   # 不管原本是 (B,N,K) 還是 (B,N,K,1)，都會變 (B,N,K)

    # ------------------------------------------------------
    # 1. GT 展開（壓成一維）
    # ------------------------------------------------------
    gt_batch = tf.cast(tf.reshape(batch_dict["batch_idx"], [-1]), tf.int32)   # (G,)
    gt_cls   = tf.cast(tf.reshape(batch_dict["cls"],       [-1]), tf.int32)   # (G,)
    gt_box   = tf.cast(batch_dict["bboxes"],    tf.float32)                   # (G,4)
    gt_kpts  = tf.cast(batch_dict["keypoints"], tf.float32)                   # (G,num_kpt,kpt_vals)

    # img_size → tensor
    if isinstance(img_size, (tuple, list)):
        img_w, img_h = float(img_size[0]), float(img_size[1])
    else:
        img_w = img_h = float(img_size)
    img_size_xy = tf.constant([img_w, img_h], dtype=tf.float32)  # (2,)

    # ------------------------------------------------------
    # 2. 逐 batch sample 建 target：shape 全部用 zeros_like 對齊 pred
    # ------------------------------------------------------
    box_t_list = []
    cls_t_list = [] if (cls_pred is not None) else None
    kxy_t_list = [] if (kxy_pred is not None) else None
    v_t_list   = [] if (v_pred   is not None) else None
    pos_mask_list = []  # (B,N)

    B_int = int(B_dyn.numpy())

    for b in range(B_int):
        # 2.1 初始化本 batch target，全 0
        box_t_b = tf.zeros_like(box_pred[b])            # (N,4)
        pos_mask_b = tf.zeros([N_dyn], tf.float32)      # (N,)

        if cls_pred is not None:
            cls_t_b = tf.zeros_like(cls_pred[b])        # (N,num_cls)
        else:
            cls_t_b = None

        if kxy_pred is not None:
            kxy_t_b = tf.zeros_like(kxy_pred[b])        # (N,num_kpt,2)
        else:
            kxy_t_b = None

        if v_pred is not None:
            v_t_b = tf.zeros_like(v_pred[b])            # (N,num_kpt)
        else:
            v_t_b = None

        # 2.2 找出屬於 batch b 的所有 GT index
        b_tensor = tf.cast(b, gt_batch.dtype)
        gt_mask_b = tf.where(tf.equal(gt_batch, b_tensor))  # (Mb,1)
        gt_mask_b = tf.reshape(gt_mask_b, [-1])             # (Mb,)
        Mb = int(tf.shape(gt_mask_b)[0].numpy())

        if Mb > 0:
            # 2.3 取出本 batch 的 GT
            boxes_b = tf.gather(gt_box,  gt_mask_b)         # (Mb,4)
            cls_b   = tf.gather(gt_cls,  gt_mask_b)         # (Mb,)
            cls_b   = tf.reshape(cls_b, [-1])               # (Mb,)
            kpts_b  = tf.gather(gt_kpts, gt_mask_b)         # (Mb,num_kpt,kpt_vals)

            # 2.4 用 IoU 選擇每個 GT 對應的 anchor index
            pred_xyxy = xywh_to_xyxy(box_pred[b])           # (N,4)
            gt_xyxy   = xywh_to_xyxy(boxes_b)               # (Mb,4)
            ious      = bbox_iou_matrix(gt_xyxy, pred_xyxy) # (Mb,N)
            best_idx  = tf.argmax(ious, axis=1, output_type=tf.int32)  # (Mb,)

            # 2.5 box target（直接填 xywh）
            box_t_b = tf.tensor_scatter_nd_update(
                box_t_b,                                    # (N,4)
                indices=tf.expand_dims(best_idx, 1),        # (Mb,1)
                updates=boxes_b                             # (Mb,4)
            )

            # 正樣本 mask
            pos_updates = tf.ones((Mb,), dtype=tf.float32)
            pos_mask_b = tf.tensor_scatter_nd_update(
                pos_mask_b,                                 # (N,)
                indices=tf.expand_dims(best_idx, 1),        # (Mb,1)
                updates=pos_updates                         # (Mb,)
            )

            # 2.6 cls one-hot target
            if cls_pred is not None:
                one_hot = tf.one_hot(cls_b, depth=num_cls, dtype=tf.float32)  # (Mb,num_cls)
                cls_t_b = tf.tensor_scatter_nd_update(
                    cls_t_b,                                # (N,num_cls)
                    indices=tf.expand_dims(best_idx, 1),    # (Mb,1)
                    updates=one_hot                         # (Mb,num_cls)
                )

            # 2.7 kpt xy target （⚠️ 重點：直接用 (Mb,num_kpt,2)，不要 flatten）
            if kxy_pred is not None:
                # 如果你的 kpts 是像素座標，這裡除以 img_size_xy；若已是 0~1，就把這行拿掉
                kpts_xy = kpts_b[..., 0:2]                  # (Mb,num_kpt,2)
                kpts_xy = kpts_xy / img_size_xy            # normalize to 0~1

                # updates: (Mb,num_kpt,2)，對應 output: (N,num_kpt,2)
                kxy_t_b = tf.tensor_scatter_nd_update(
                    kxy_t_b,                                # (N,num_kpt,2)
                    indices=tf.expand_dims(best_idx, 1),    # (Mb,1)
                    updates=kpts_xy                         # (Mb,num_kpt,2)
                )

            # 2.8 kpt visibility target
            if v_pred is not None:
                if kpt_vals > 2:
                    vis = kpts_b[..., 2]                    # (Mb,num_kpt)
                else:
                    vis = tf.ones((Mb, num_kpt), tf.float32)
                v_t_b = tf.tensor_scatter_nd_update(
                    v_t_b,                                  # (N,num_kpt)
                    indices=tf.expand_dims(best_idx, 1),    # (Mb,1)
                    updates=vis                             # (Mb,num_kpt)
                )

        # 收集本 batch 的 target
        box_t_list.append(box_t_b)
        pos_mask_list.append(pos_mask_b)
        if cls_pred is not None:
            cls_t_list.append(cls_t_b)
        if kxy_pred is not None:
            kxy_t_list.append(kxy_t_b)
        if v_pred is not None:
            v_t_list.append(v_t_b)

    # ------------------------------------------------------
    # 3. stack 回 (B,N,...) 形式
    # ------------------------------------------------------
    box_t = tf.stack(box_t_list, axis=0)         # (B,N,4)
    pos_mask = tf.stack(pos_mask_list, axis=0)   # (B,N)
    pos_mask_f = tf.cast(pos_mask, tf.float32)   # (B,N)

    if cls_pred is not None:
        cls_t = tf.stack(cls_t_list, axis=0)     # (B,N,num_cls)
    else:
        cls_t = None

    if kxy_pred is not None:
        kxy_t = tf.stack(kxy_t_list, axis=0)     # (B,N,num_kpt,2)
    else:
        kxy_t = None

    if v_pred is not None:
        v_t = tf.stack(v_t_list, axis=0)         # (B,N,num_kpt)
    else:
        v_t = None

    # ------------------------------------------------------
    # 4. 計算各項 loss
    # ------------------------------------------------------
    # 4.1 box Huber（只算正樣本）
    box_diff_elem = huber_no_reduce(box_t, box_pred)  # (B,N,4)
    box_diff = tf.reduce_sum(box_diff_elem, axis=-1)  # (B,N)
    box_loss_num = tf.reduce_sum(box_diff * pos_mask_f)
    num_pos = tf.maximum(tf.reduce_sum(pos_mask_f), 1.0)
    box_loss = lambda_box * (box_loss_num / num_pos)

    # 4.2 cls BCE（全 anchor；負樣本 target=0）
    if cls_pred is not None:
        cls_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=cls_t, logits=cls_pred
        )                             # (B,N,num_cls)
        cls_loss = tf.reduce_mean(cls_loss_raw)
        cls_loss = lambda_cls * cls_loss
    else:
        cls_loss = tf.constant(0.0, tf.float32)

    # 4.3 kpt xy Huber（正樣本 + v=1 的點）
    if (kxy_pred is not None) and (v_pred is not None):
        # per coordinate diff: (B,N,num_kpt,2)
        kxy_diff_elem = huber_no_reduce(kxy_t, kxy_pred)

        # visibility mask: (B,N,num_kpt) -> (B,N,num_kpt,1)
        vis_mask = v_t[..., None]                       # (B,N,num_kpt,1)

        # anchor 正樣本 mask: (B,N) -> (B,N,1,1)
        anchor_mask = pos_mask_f[..., None, None]      # (B,N,1,1)

        total_mask = vis_mask * anchor_mask            # (B,N,num_kpt,1)
        # broadcast 到最後一個座標維度 2
        total_mask = tf.broadcast_to(total_mask, tf.shape(kxy_diff_elem))  # (B,N,num_kpt,2)

        kxy_loss_num = tf.reduce_sum(kxy_diff_elem * total_mask)
        num_vis_coords = tf.maximum(tf.reduce_sum(total_mask), 1.0)
        kxy_loss = lambda_kxy * (kxy_loss_num / num_vis_coords)
    else:
        kxy_loss = tf.constant(0.0, tf.float32)

    # 4.4 kpt visibility Huber（只算正樣本 anchor）
    if (v_pred is not None) and (v_t is not None):
        v_diff_elem = huber_no_reduce(v_t, v_pred)      # (B,N,num_kpt)
        anchor_mask_v = pos_mask_f[..., None]          # (B,N,1)
        anchor_mask_v = tf.broadcast_to(anchor_mask_v, tf.shape(v_diff_elem))  # (B,N,num_kpt)
        v_loss_num = tf.reduce_sum(v_diff_elem * anchor_mask_v)
        num_pos_kpt = tf.maximum(tf.reduce_sum(anchor_mask_v), 1.0)
        v_loss = lambda_v * (v_loss_num / num_pos_kpt)
    else:
        v_loss = tf.constant(0.0, tf.float32)

    total_loss = box_loss + cls_loss + kxy_loss + v_loss

    logs = {
        "loss_box": box_loss,
        "loss_cls": cls_loss,
        "loss_kxy": kxy_loss,
        "loss_v":   v_loss,
        "num_pos":  num_pos,
    }
    return total_loss, logs
