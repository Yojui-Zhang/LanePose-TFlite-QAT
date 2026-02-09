import tensorflow as tf
from typing import Tuple, Optional
from QAT_Refactored.utils.geometry import iou_batch, bbox_xywh_to_xyxy

class TaskAlignedAssigner:
    """YOLOv8-style Positive/Negative Sample Assigner - Ultralytics compatible."""
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, 
                 stride: list = None, eps: float = 1e-9):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = float(self.stride[1]) if len(self.stride) > 1 else float(self.stride[0])
        self.eps = eps

    def assign(self, pred_box, pred_cls_prob, anchors, gt_boxes, gt_cls, valid_mask):
        valid_mask_int = tf.cast(valid_mask, tf.int32)
        num_valid = tf.reduce_sum(valid_mask_int)
        
        def _no_gt():
            N = tf.shape(pred_box)[0]
            neg = tf.fill([N], -1)
            qual = tf.zeros([N], tf.float32)
            pm = tf.zeros([N], tf.bool)
            return neg, neg, qual, pm

        def _has_gt():
            gtb = gt_boxes[:num_valid]
            gtc = tf.cast(gt_cls[:num_valid], tf.int32)

            anch_xy = anchors[:, :2]
            gt_xyxy = bbox_xywh_to_xyxy(gtb)

            ax = anch_xy[:, 0][:, None]
            ay = anch_xy[:, 1][:, None]

            x1, y1, x2, y2 = tf.unstack(gt_xyxy, num=4, axis=-1)
            is_in_bbox = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2)

            # GT boxes are normalized to [0,1], so min-size protection must also be normalized.
            # Use the smallest anchor cell size as lower bound (e.g. 8/imgsz), avoiding pixel-vs-normalized mismatch.
            gtb_wh = gtb[..., 2:4]
            min_cell_norm = tf.reduce_min(anchors[:, 2:4])  # scalar in normalized units
            gtb_wh_fixed = tf.maximum(gtb_wh, min_cell_norm)
            gtb_fixed = tf.concat([gtb[..., :2], gtb_wh_fixed], axis=-1)
            gt_xyxy_fixed = bbox_xywh_to_xyxy(gtb_fixed)
            
            x1_f, y1_f, x2_f, y2_f = tf.unstack(gt_xyxy_fixed, num=4, axis=-1)
            is_in_bbox_fixed = (ax >= x1_f) & (ax <= x2_f) & (ay >= y1_f) & (ay <= y2_f)
            
            candidates = is_in_bbox_fixed

            ious = iou_batch(pred_box, gtb)
            cls_scores = tf.gather(pred_cls_prob, gtc, axis=1)

            mask_float = tf.cast(candidates, tf.float32)
            ious *= mask_float
            cls_scores *= mask_float

            align_metric = (cls_scores ** self.alpha) * (ious ** self.beta)

            k = tf.minimum(self.topk, tf.shape(align_metric)[0])
            topk_metrics, topk_idxs = tf.nn.top_k(tf.transpose(align_metric), k=k)
            
            m_idx = tf.range(num_valid, dtype=tf.int32)[:, None]
            m_idx_rep = tf.repeat(m_idx, k, axis=1)
            
            indices = tf.stack([topk_idxs, m_idx_rep], axis=-1)
            indices = tf.reshape(indices, [-1, 2])
            updates = tf.ones([tf.shape(indices)[0]], dtype=tf.float32)
            
            is_topk = tf.scatter_nd(indices, updates, [tf.shape(pred_box)[0], num_valid]) > 0
            is_topk = tf.cast(is_topk, tf.float32)
            
            valid_metric = align_metric * is_topk

            best_metric_per_anchor = tf.reduce_max(valid_metric, axis=1)
            best_gt_idx = tf.argmax(valid_metric, axis=1, output_type=tf.int32)

            pos_mask = best_metric_per_anchor > 0

            row_idx = tf.range(tf.shape(pred_box)[0], dtype=tf.int32)
            gather_idx = tf.stack([row_idx, best_gt_idx], axis=1)
            iou_best = tf.gather_nd(ious, gather_idx)

            quality = tf.where(pos_mask, iou_best, tf.zeros_like(iou_best))
            quality = tf.clip_by_value(quality, 0.0, 1.0)

            assigned_gt = tf.where(pos_mask, best_gt_idx, tf.fill(tf.shape(best_gt_idx), -1))
            assigned_cls = tf.where(pos_mask, tf.gather(gtc, best_gt_idx), tf.fill(tf.shape(best_gt_idx), -1))

            return assigned_gt, assigned_cls, quality, pos_mask

        return tf.cond(num_valid > 0, _has_gt, _no_gt)
