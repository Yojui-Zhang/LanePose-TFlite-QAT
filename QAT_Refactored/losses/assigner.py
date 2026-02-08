# File: QAT_Refactored/losses/assigner.py

import tensorflow as tf
from typing import Tuple, Optional
from QAT_Refactored.utils.geometry import iou_batch, bbox_xywh_to_xyxy

class TaskAlignedAssigner:
    """
    YOLOv8-style Positive/Negative Sample Assigner.
    Assigns Ground Truths (GT) based on predicted Class Score and IoU (Alignment Metric).
    """
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0, center_radius: float = 2.5):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius
        self.eps = 1e-9

    def assign(
        self, 
        pred_box: tf.Tensor, 
        pred_cls_prob: tf.Tensor, 
        anchors: tf.Tensor, 
        gt_boxes: tf.Tensor, 
        gt_cls: tf.Tensor, 
        valid_mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Assigns targets for a single image (Unbatched).
        
        Args:
            pred_box: (N, 4) [cx, cy, w, h]
            pred_cls_prob: (N, C)
            anchors: (N, 4) [cx, cy, w, h]
            gt_boxes: (M, 4) [cx, cy, w, h]
            gt_cls: (M, )
            valid_mask: (M, ) 0/1 mask
        """
        # [CRITICAL FIX]: tf.reduce_sum does not support bool input in strict graph mode.
        # Cast to int32 BEFORE reduction.
        valid_mask_int = tf.cast(valid_mask, tf.int32)
        num_valid = tf.reduce_sum(valid_mask_int)
        
        def _no_gt():
            N = tf.shape(pred_box)[0]
            neg = tf.fill([N], -1)
            qual = tf.zeros([N], tf.float32)
            pm = tf.zeros([N], tf.bool)
            return neg, neg, qual, pm

        def _has_gt():
            # Slice valid GTs to ignore padding
            gtb = gt_boxes[:num_valid]       # (m, 4)
            gtc = tf.cast(gt_cls[:num_valid], tf.int32) # (m, )

            # --- 1. Coarse Filtering (In-GT & Center Sampling) ---
            # Anchors: (N, 4) -> xy (N, 2)
            anch_xy = anchors[:, :2]
            gt_xyxy = bbox_xywh_to_xyxy(gtb) # (m, 4)
            
            # Broadcast comparison: (N, 1, 2) vs (1, m, 4)
            ax = anch_xy[:, 0][:, None]
            ay = anch_xy[:, 1][:, None]
            
            # Why: map_fn 下靜態 shape 可能丟失，unstack 需固定 num
            x1, y1, x2, y2 = tf.unstack(gt_xyxy, num=4, axis=-1) # (1, m) implicitly
            
            # Is anchor inside GT box?
            is_in_bbox = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2)
            
            # Center Sampling: Check if anchor is within radius of GT center
            gt_cx, gt_cy, _, _ = tf.unstack(gtb, num=4, axis=-1)
            stride_w = anchors[:, 2] # Use anchor width as stride proxy
            stride_h = anchors[:, 3]
            
            rx = self.center_radius * stride_w[:, None]
            ry = self.center_radius * stride_h[:, None]
            
            is_in_center = (tf.abs(ax - gt_cx) <= rx) & (tf.abs(ay - gt_cy) <= ry)
            
            candidates = is_in_bbox & is_in_center
            
            # Fallback: if a GT has no candidates via center sampling, accept any inside bbox
            cand_any = tf.reduce_any(candidates, axis=0) # (m, )
            candidates = tf.where(cand_any[None, :], candidates, is_in_bbox)
            
            # --- 2. Calculate Alignment Metric ---
            # IoU (N, m)
            ious = iou_batch(pred_box, gtb)
            
            # Cls Scores for the specific GT class (N, m)
            # Gather relevant class scores: pred_cls_prob (N, C) -> (N, m) using gtc
            cls_scores = tf.gather(pred_cls_prob, gtc, axis=1)
            
            # Mask out non-candidates
            mask_float = tf.cast(candidates, tf.float32)
            ious *= mask_float
            cls_scores *= mask_float
            
            # Alignment Metric: s^alpha * u^beta
            align_metric = (cls_scores ** self.alpha) * (ious ** self.beta)
            
            # --- 3. Top-K Selection ---
            # For each GT, select top-k anchors with highest metric
            k = tf.minimum(self.topk, tf.shape(align_metric)[0])
            _, topk_ind = tf.math.top_k(tf.transpose(align_metric), k=k) # (m, k)
            
            # Create a mask for Top-K
            m_idx = tf.range(num_valid, dtype=tf.int32)[:, None] # (m, 1)
            m_idx_rep = tf.repeat(m_idx, k, axis=1) # (m, k)
            
            indices = tf.stack([topk_ind, m_idx_rep], axis=-1) # (m, k, 2)
            indices = tf.reshape(indices, [-1, 2])
            updates = tf.ones([tf.shape(indices)[0]], dtype=tf.float32)
            
            # Scatter to (N, m)
            is_topk = tf.scatter_nd(indices, updates, [tf.shape(pred_box)[0], num_valid]) > 0
            
            # Filter low quality or non-topk
            valid_metric = align_metric * tf.cast(is_topk, tf.float32)
            
            # --- 4. Resolve Ambiguity ---
            # If an anchor is assigned to multiple GTs, pick the one with max metric
            best_metric_per_anchor = tf.reduce_max(valid_metric, axis=1) # (N, )
            best_gt_idx = tf.argmax(valid_metric, axis=1, output_type=tf.int32) # (N, )
            
            pos_mask = best_metric_per_anchor > 0
            
            # Quality is the IoU of the matched GT
            row_idx = tf.range(tf.shape(pred_box)[0], dtype=tf.int32)
            gather_idx = tf.stack([row_idx, best_gt_idx], axis=1)
            iou_best = tf.gather_nd(ious, gather_idx)
            
            quality = tf.where(pos_mask, iou_best, tf.zeros_like(iou_best))
            quality = tf.clip_by_value(quality, 0.0, 1.0)
            
            assigned_gt = tf.where(pos_mask, best_gt_idx, tf.fill(tf.shape(best_gt_idx), -1))
            assigned_cls = tf.where(pos_mask, tf.gather(gtc, best_gt_idx), tf.fill(tf.shape(best_gt_idx), -1))
            
            return assigned_gt, assigned_cls, quality, pos_mask

        # If no valid GTs in this image, return empty assignment
        return tf.cond(num_valid > 0, _has_gt, _no_gt)