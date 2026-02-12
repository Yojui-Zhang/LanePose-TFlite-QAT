# Source File: QAT_Refactored/losses/pose_loss.py

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional, Any

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.utils.geometry import bbox_ciou
from QAT_Refactored.losses.assigner import TaskAlignedAssigner
from QAT_Refactored.utils.tensor_layout import ensure_bnc_tf

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_anchors(imgsz: int = 640, strides: Optional[list[int]] = None, grid_cell_offset: float = 0.5) -> tf.Tensor:
    """
    Generates anchor points (priors) for the model based on strides.
    Output anchors are normalized to [0,1] in (cx, cy, w, h).
    Why: assigner 的 candidates/center sampling 與 GT 都是 normalized；anchors 若是 pixel 會導致永遠指派不到正樣本。
    """
    if imgsz <= 0:
        raise ValueError("get_anchors: imgsz must be > 0")
    if strides is None:
        strides = [8, 16, 32]
    if not strides:
        raise ValueError("get_anchors: strides must be non-empty")

    anchors_list = []
    imgsz_f = tf.cast(imgsz, tf.float32)
    for stride in strides:
        if stride <= 0:
            raise ValueError("get_anchors: stride must be > 0")
        h = imgsz // stride
        w = imgsz // stride
        if h <= 0 or w <= 0:
            raise ValueError("get_anchors: invalid grid size (check imgsz/stride)")        

        # Create Meshgrid
        xv, yv = tf.meshgrid(tf.range(w), tf.range(h))
        xv = tf.cast(xv, tf.float32)
        yv = tf.cast(yv, tf.float32)

        # Calculate Center Points
        stride_f = tf.cast(stride, tf.float32)
        cx = ((xv + grid_cell_offset) * stride_f) / imgsz_f
        cy = ((yv + grid_cell_offset) * stride_f) / imgsz_f
        
        # Use stride as width/height placeholder for decoding
        sw = tf.ones_like(cx) * (stride_f / imgsz_f)
        sh = tf.ones_like(cy) * (stride_f / imgsz_f)
        
        # Stack: (H, W, 4)
        anchors = tf.stack([cx, cy, sw, sh], axis=-1)
        anchors_list.append(tf.reshape(anchors, [-1, 4]))
    
    return tf.concat(anchors_list, axis=0)

def build_batch_dict_from_padded_labels(
    batch_labels: tf.Tensor,
    num_cls: int,
    num_kpt: int,
    kpt_vals: int = 3,
) -> Dict[str, tf.Tensor]:
    """
    Parses the raw padded batch tensor into a structured dictionary for loss calculation.
    """
    tf.debugging.assert_all_finite(batch_labels, "labels contain NaN/Inf")
    tf.debugging.assert_greater(num_cls, 0, "num_cls must be > 0")
    tf.debugging.assert_greater_equal(num_kpt, 0, "num_kpt must be >= 0")
    tf.debugging.assert_greater_equal(kpt_vals, 0, "kpt_vals must be >= 0")

    cls_ids = batch_labels[..., 0:1] # (B, M, 1)
    bboxes = batch_labels[..., 1:5]  # (B, M, 4) [cx, cy, w, h]
    kpts = batch_labels[..., 5:]     # (B, M, K*V)
    
    # Identify padding rows (real boxes must have width > 0)
    valid_mask = bboxes[..., 2] > 0.0 # (B, M)
    
    # --- Fail-fast: class id range for valid rows ---
    cls_i = tf.cast(tf.squeeze(cls_ids, axis=-1), tf.int32)  # (B,M)
    valid_cls = tf.boolean_mask(cls_i, valid_mask)
    tf.debugging.assert_greater_equal(valid_cls, 0, "label class_id < 0 found")
    tf.debugging.assert_less(valid_cls, tf.cast(num_cls, tf.int32), "label class_id >= num_cls found")

    # --- Fail-fast: bbox sanity for valid rows (normalized) ---
    vb = tf.boolean_mask(bboxes, valid_mask)  # (V,4)
    # Why: 允許極小浮動誤差，但不允許明顯越界
    tol = tf.constant(1e-3, tf.float32)
    tf.debugging.assert_greater_equal(vb, -tol, "bbox < 0 found (beyond tolerance)")
    tf.debugging.assert_less_equal(vb, 1.0 + tol, "bbox > 1 found (beyond tolerance)")
    tf.debugging.assert_greater(vb[:, 2], 0.0, "bbox w<=0 found")
    tf.debugging.assert_greater(vb[:, 3], 0.0, "bbox h<=0 found")

    # --- Keypoints: clip visible coords to [0,1] ---
    if num_kpt > 0 and kpt_vals >= 2:
        B = tf.shape(batch_labels)[0]
        M = tf.shape(batch_labels)[1]
        need_k = tf.cast(num_kpt * kpt_vals, tf.int32)
        tf.debugging.assert_equal(tf.shape(kpts)[-1], need_k, "kpt dims mismatch")

        k = tf.reshape(kpts, [B, M, num_kpt, kpt_vals])
        kx = k[..., 0]
        ky = k[..., 1]
        if kpt_vals >= 3:
            v = k[..., 2]
            vis = v > 0.0
        else:
            vis = tf.ones_like(kx, dtype=tf.bool)

        kx = tf.where(vis, tf.clip_by_value(kx, 0.0, 1.0), tf.zeros_like(kx))
        ky = tf.where(vis, tf.clip_by_value(ky, 0.0, 1.0), tf.zeros_like(ky))

        parts = [kx[..., None], ky[..., None]]
        if kpt_vals >= 3:
            parts.append(k[..., 2:])
        k2 = tf.concat(parts, axis=-1)
        kpts = tf.reshape(k2, [B, M, num_kpt * kpt_vals])

    return {
        "cls": cls_ids,
        "bboxes": bboxes,
        "keypoints": kpts,
        "valid_mask": valid_mask
    }

# ==============================================================================
# Loss Functions
# ==============================================================================

def oks_loss(pred_kxy: tf.Tensor, gt_kxy: tf.Tensor, kpt_v: tf.Tensor, 
             bbox_area: tf.Tensor, sigmas: np.ndarray, use_ultralytics: bool = False) -> tf.Tensor:
    """Computes Object Keypoint Similarity (OKS) Loss.
    
    Args:
        pred_kxy, gt_kxy: (B, N, K, 2) predicted and ground truth keypoints
        kpt_v: (B, N, K, 1) visibility mask
        bbox_area: (B, N, 1) bounding box area in pixels
        sigmas: np.ndarray of keypoint sigmas
        use_ultralytics: if True, use Ultralytics OKS formula
    """
    # (B, N, K, 2) -> (B, N, K, 1) distance squared
    d2 = tf.reduce_sum(tf.square(pred_kxy - gt_kxy), axis=-1, keepdims=True) 
    
    # Broadcast sigmas: (1, 1, K, 1)
    sig = tf.cast(tf.reshape(sigmas, [1, 1, -1, 1]), dtype=tf.float32)
    
    if use_ultralytics:
        # Ultralytics formula: e = d² / ((2σ)² * area * 2)
        # = d² / (8 * σ² * area)
        denom = 8.0 * tf.square(sig) * (bbox_area[..., None] + 1e-9)
    else:
        # Original QAT formula: e = d² / (2 * σ² * area)
        denom = 2.0 * tf.square(sig) * (bbox_area[..., None] + 1e-9)
    
    e = tf.exp(-d2 / tf.maximum(denom, 1e-9))
    
    # Only calculate loss for visible keypoints
    oks = (1.0 - e) * kpt_v 
    
    # Sum over keypoints, normalize by number of visible keypoints
    visible_count = tf.reduce_sum(kpt_v, axis=[2, 3]) + 1e-9
    return tf.reduce_mean(tf.reduce_sum(oks, axis=[2, 3]) / visible_count)

class PoseLabelLoss(tf.keras.layers.Layer):
    """
    Main Training Loss Logic.
    Decoupled from Trainer to ensure SRP.
    """
    def __init__(self, cfg: AppConfig, use_ultralytics: bool = False, **kwargs):
        super().__init__(name="PoseLabelLoss", **kwargs)
        self.cfg = cfg
        self.use_ultralytics = use_ultralytics
        
        if use_ultralytics:
            # Ultralytics default TAL settings.
            self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0, stride=[8, 16, 32])
        else:
            # Legacy QAT assigner behavior.
            self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
        
        # Initialize OKS Sigmas (COCO or Uniform)
        if self.cfg.NUM_KPT == 17:
             raw_sigmas = [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 
                           0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
             self.sigmas = np.array(raw_sigmas) / 10.0
        else:
             self.sigmas = np.array([0.05] * self.cfg.NUM_KPT)


    def decode_preds(self, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Decodes raw model output into Box, Cls, Keypoints.
        Why: 訓練 head 輸出為 (B, C, N)，loss/assigner 統一以 (B, N, C) 處理以避免切片維度錯誤。
        """
        y_pred = ensure_bnc_tf(y_pred, self.cfg.total_output_channels)
        # Why: 轉成 (B,N,C) 後，最後一維必須等於 total_output_channels，否則後續切片會 silent wrong
        tf.debugging.assert_equal(
            tf.shape(y_pred)[-1],
            tf.cast(self.cfg.total_output_channels, tf.int32),
            message="PoseLabelLoss.decode_preds: channel dim mismatch after ensure_bnc_tf"
        )

        box = tf.sigmoid(y_pred[..., :4])
        cls_logits = y_pred[..., 4: 4 + self.cfg.NUM_CLS]
        # Why: map_fn tracing 時常丟失靜態 shape；固定最後一維能避免後續解包/reshape 再爆
        box = tf.ensure_shape(box, [None, None, 4])
        cls_logits = tf.ensure_shape(cls_logits, [None, None, None])  # 先保守，下面再用 assert 鎖 NUM_CLS
 

        kpt_raw = y_pred[..., 4 + self.cfg.NUM_CLS:]
        B = tf.shape(y_pred)[0]
        N = tf.shape(y_pred)[1]

        tf.debugging.assert_equal(
            tf.shape(box)[-1], 4,
            message="PoseLabelLoss.decode_preds: pred_box last dim must be 4"
        )
        tf.debugging.assert_equal(
            tf.shape(cls_logits)[-1], tf.cast(self.cfg.NUM_CLS, tf.int32),
            message="PoseLabelLoss.decode_preds: pred_cls_logits last dim must be NUM_CLS"
        )


        kpt_raw = tf.reshape(kpt_raw, [B, N, self.cfg.NUM_KPT, self.cfg.KPT_VALS])
        return box, cls_logits, kpt_raw


    def call(
        self,
        y_pred: tf.Tensor,
        batch_dict: Dict[str, tf.Tensor],
        anchors: tf.Tensor,
        class_weights: Optional[tf.Tensor] = None,
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        """
        Calculates total loss.
        """
        # 1. Decode Predictions
        pred_box, pred_cls_logits, pred_kpt_raw = self.decode_preds(y_pred)
        pred_cls_prob = tf.sigmoid(pred_cls_logits)
        
        B = tf.shape(pred_box)[0]
        N = tf.shape(pred_box)[1] 


        # Why: assigner 以 anchors 對應每個 pred；N 必須等於 anchors 數量，否則會產生錯配但不一定立刻報錯
        tf.debugging.assert_equal(
            N, tf.shape(anchors)[0],
            message="PoseLabelLoss.call: N must match anchors.shape[0]"
        )

        # 2. Assign Targets (Map over batch)
        # Using tf.map_fn to handle variable assignments per image
        def assign_fn(args):
            pbox, pcls, gtb, gtc, vm = args

            tf.debugging.assert_equal(tf.shape(pbox)[-1], 4, message="assign_fn: pbox last dim must be 4")
            tf.debugging.assert_equal(
                tf.shape(pcls)[-1], tf.cast(self.cfg.NUM_CLS, tf.int32),
                message="assign_fn: pcls last dim must be NUM_CLS"
            )

            gtc = tf.cast(tf.squeeze(gtc, -1), tf.int32)
            return self.assigner.assign(pbox, pcls, anchors, gtb, gtc, vm)

        # Why: 若 anchors.shape[0] 可靜態取得（常見 8400），把它灌進 TensorSpec 可顯著提升 map_fn tracing 穩定性
        n_spec = anchors.shape[0]  # int or None
        out_spec = (
            tf.TensorSpec([n_spec], tf.int32),   # assigned_gt : (N,)
            tf.TensorSpec([n_spec], tf.int32),   # assigned_cls: (N,)
            tf.TensorSpec([n_spec], tf.float32), # quality     : (N,)
            tf.TensorSpec([n_spec], tf.bool)     # pos_mask    : (N,)
        )
        
        assigned_gt, assigned_cls, quality, pos_mask = tf.map_fn(
            assign_fn,
            (pred_box, pred_cls_prob, batch_dict['bboxes'], batch_dict['cls'], batch_dict['valid_mask']),
            fn_output_signature=out_spec
        )

        # --- Metrics (debug/monitor) ---
        # Why: loss 下降不代表有學到偵測；pos_count=0 或 max_score→0 會導致推論無框但 loss 仍可下降。
        gt_cnt = tf.reduce_sum(tf.cast(batch_dict["valid_mask"], tf.float32), axis=1)     # (B,)
        pos_cnt = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1)                    # (B,)
        pos_mean = tf.reduce_mean(pos_cnt)
        gt_mean = tf.reduce_mean(gt_cnt)

        # per-image max classification score across all anchors/classes
        max_score_img = tf.reduce_max(pred_cls_prob, axis=[1, 2])                         # (B,)
        max_score_mean = tf.reduce_mean(max_score_img)


        # Why: 若圖片內有 GT 但 pos_count=0，訓練會退化成「全背景/全低置信度」，loss 仍會下降但推論無框。
        # gt_cnt = tf.reduce_sum(tf.cast(batch_dict["valid_mask"], tf.int32), axis=1)          # (B,)
        # pos_cnt = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=1)                          # (B,)
        # bad = tf.logical_and(gt_cnt > 0, pos_cnt == 0)                                        # (B,)
        # tf.debugging.Assert(
        #     tf.logical_not(tf.reduce_any(bad)),
        #     data=[
        #         "PoseLabelLoss: images have GT but got zero positive assignments. "
        #         "Common cause: anchors/GT coordinate units mismatch (pixel vs normalized).",
        #         "gt_cnt:", gt_cnt,
        #         "pos_cnt:", pos_cnt
        #     ],
        #     summarize=16
        # )


        # 3. Compute Losses
        pos_mask_f = tf.cast(pos_mask, tf.float32)
        safe_assigned_cls = tf.maximum(assigned_cls, 0)
        target_scores = tf.one_hot(safe_assigned_cls, self.cfg.NUM_CLS, dtype=tf.float32)
        target_scores *= tf.where(pos_mask[..., None], quality[..., None], tf.zeros_like(target_scores))
        target_scores_sum = tf.maximum(tf.reduce_sum(target_scores), 1.0)
        
        # Gather matched GTs
        batch_idx = tf.repeat(tf.range(B, dtype=tf.int32)[:, None], N, axis=1)
        safe_gt_idx = tf.maximum(assigned_gt, 0)
        gather_idx = tf.stack([batch_idx, safe_gt_idx], axis=-1)
        
        target_box = tf.gather_nd(batch_dict['bboxes'], gather_idx)
        target_kpt_flat = tf.gather_nd(batch_dict['keypoints'], gather_idx) # (B, N, K*V)
        
        # --- A. Box Loss (CIoU, Ultralytics-style weighting) ---
        ciou = bbox_ciou(pred_box, target_box) 
        box_weight = tf.reduce_sum(target_scores, axis=-1)
        l_box = tf.reduce_sum(ciou * box_weight) / target_scores_sum
        l_box *= self.cfg.W_BOX
        
        # --- B. Class Loss (Ultralytics BCE target scores) ---
        l_cls_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_scores, logits=pred_cls_logits)
        
        if class_weights is not None:
             cw = tf.reshape(class_weights, [1, 1, -1])
             l_cls_raw *= cw

        l_cls = self.cfg.W_CLS * (tf.reduce_sum(l_cls_raw) / target_scores_sum)
        
        # --- C. Keypoint Loss (OKS) ---
        l_kpt = tf.constant(0.0, dtype=tf.float32)
        
        if self.cfg.NUM_KPT > 0:
            # Pred: (B, N, K, V)
            pkpt = pred_kpt_raw
            pxy = tf.sigmoid(pkpt[..., :2])
            
            # [CRITICAL FIX]: Reshape target to (B, N, K, V) before slicing
            # Previous code failed here because target_kpt_flat was (B, N, K*V)
            tkpt = tf.reshape(target_kpt_flat, [B, N, self.cfg.NUM_KPT, self.cfg.KPT_VALS])
            txy = tkpt[..., :2]
            
            if self.cfg.KPT_VALS >= 3:
                tv = tkpt[..., 2:3]
            else:
                tv = tf.ones_like(txy[..., :1])

            tv_mask = tf.cast(tv > 0, tf.float32)

            # Scale area to pixels for OKS
            area_px = (target_box[..., 2] * self.cfg.IMGSZ) * (target_box[..., 3] * self.cfg.IMGSZ)

            # oks_loss() already returns a loss value; do not invert it.
            l_kpt_xy = oks_loss(
                pxy, txy, tv_mask, area_px[..., None], self.sigmas, use_ultralytics=self.use_ultralytics
            )
            l_kpt = self.cfg.W_KPT_XY * l_kpt_xy

            if self.cfg.KPT_VALS >= 3:
                pv_logits = pkpt[..., 2:3]
                kobj_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tv_mask, logits=pv_logits)
                kobj_weight = pos_mask_f[..., None, None]
                den_kobj = tf.maximum(tf.reduce_sum(kobj_weight), 1.0)
                l_kobj = self.cfg.W_KPT_V * (tf.reduce_sum(kobj_bce * kobj_weight) / den_kobj)
                l_kpt += l_kobj
            
        total_loss = l_box + l_cls + l_kpt
        return total_loss, l_box, l_cls, l_kpt, pos_mean, gt_mean, max_score_mean
