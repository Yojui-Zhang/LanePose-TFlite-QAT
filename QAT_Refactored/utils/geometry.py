import tensorflow as tf
import numpy as np

def bbox_xywh_to_xyxy(b):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""

    b = tf.convert_to_tensor(b)
    # Why: tf.function/map_fn 會把 shape trace 成 (None, None)，unstack 需明確 num 才能建圖
    tf.debugging.assert_rank_at_least(b, 1, message="bbox_xywh_to_xyxy: expected rank>=1 (...,4)")
    tf.debugging.assert_equal(tf.shape(b)[-1], 4, message="bbox_xywh_to_xyxy: last dim must be 4 (cx,cy,w,h)")

    cx, cy, w, h = tf.unstack(b, num=4, axis=-1)

    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return tf.stack([x1, y1, x2, y2], axis=-1)

def bbox_iou_pair(pred_box, target_box, eps=1e-7):
    """
    計算成對的 IoU (Element-wise).
    pred_box, target_box: (..., 4) [cx, cy, w, h]
    """
    p_xyxy = bbox_xywh_to_xyxy(pred_box)
    t_xyxy = bbox_xywh_to_xyxy(target_box)
    
    px1, py1, px2, py2 = tf.unstack(p_xyxy, num=4, axis=-1)
    gx1, gy1, gx2, gy2 = tf.unstack(t_xyxy, num=4, axis=-1)

    inter_x1 = tf.maximum(px1, gx1)
    inter_y1 = tf.maximum(py1, gy1)
    inter_x2 = tf.minimum(px2, gx2)
    inter_y2 = tf.minimum(py2, gy2)

    inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
    
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union = area_p + area_g - inter_area
    
    return tf.clip_by_value(inter_area / (union + eps), 0.0, 1.0)

def bbox_ciou(pred_box, target_box, eps=1e-7):
    """
    計算 CIoU Loss = 1 - CIoU.
    pred_box, target_box: (B, N, 4) [cx, cy, w, h]
    """
    iou = bbox_iou_pair(pred_box, target_box, eps)
    
    px, py, pw, ph = tf.unstack(pred_box, num=4, axis=-1)
    gx, gy, gw, gh = tf.unstack(target_box, num=4, axis=-1)

    # 中心點距離平方
    d2 = tf.square(px - gx) + tf.square(py - gy)

    # 最小包圍框對角線平方
    p_xyxy = bbox_xywh_to_xyxy(pred_box)
    t_xyxy = bbox_xywh_to_xyxy(target_box)
    
    # Enclosing box
    ex1 = tf.minimum(p_xyxy[..., 0], t_xyxy[..., 0])
    ey1 = tf.minimum(p_xyxy[..., 1], t_xyxy[..., 1])
    ex2 = tf.maximum(p_xyxy[..., 2], t_xyxy[..., 2])
    ey2 = tf.maximum(p_xyxy[..., 3], t_xyxy[..., 3])
    
    c2 = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps

    # Aspect Ratio penalty
    v = (4.0 / (np.pi ** 2)) * tf.square(
        tf.atan(gw / (gh + eps)) - tf.atan(pw / (ph + eps))
    )
    
    # Stop gradient for alpha
    with tf.name_scope("ciou_alpha"):
        alpha = tf.stop_gradient(v / (1.0 - iou + v + eps))

    ciou = iou - (d2 / c2) - (alpha * v)
    ciou = tf.clip_by_value(ciou, -1.0, 1.0)
    
    return 1.0 - ciou

def iou_batch(bboxes1, bboxes2):
    """
    計算 Batch IoU (M vs N).
    bboxes1: (B, N, 4)
    bboxes2: (B, M, 4)
    Returns: (B, N, M)
    """
    # 這裡為了 Assign logic 簡化，假設輸入已調整為 (N, 4) 與 (M, 4) 或利用 broadcasting
    # 為配合 TaskAlignedAssigner，我們實作單張圖片的 IoU (N, M)
    
    b1_xyxy = bbox_xywh_to_xyxy(bboxes1) # (N, 4)
    b2_xyxy = bbox_xywh_to_xyxy(bboxes2) # (M, 4)
    
    # Expand for broadcasting: (N, 1, 4) vs (1, M, 4)
    b1 = tf.expand_dims(b1_xyxy, 1)
    b2 = tf.expand_dims(b2_xyxy, 0)
    
    inter_x1 = tf.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = tf.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = tf.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = tf.minimum(b1[..., 3], b2[..., 3])
    
    inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
    
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-9)