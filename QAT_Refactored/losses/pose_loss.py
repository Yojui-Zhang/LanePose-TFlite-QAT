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

def get_anchors(imgsz: int = 640, strides: list = None, grid_cell_offset: float = 0.5) -> tf.Tensor:
    """
    Generates anchor points (priors) for the model based on strides.
    """
    if strides is None:
        strides = [8, 16, 32]

    anchors_list = []
    for stride in strides:
        h = imgsz // stride
        w = imgsz // stride
        
        # Create Meshgrid
        xv, yv = tf.meshgrid(tf.range(w), tf.range(h))
        xv = tf.cast(xv, tf.float32)
        yv = tf.cast(yv, tf.float32)

        # Calculate Center Points
        cx = (xv + grid_cell_offset) * stride
        cy = (yv + grid_cell_offset) * stride
        
        # Use stride as width/height placeholder for decoding
        sw = tf.ones_like(cx) * stride
        sh = tf.ones_like(cy) * stride
        
        # Stack: (H, W, 4)
        anchors = tf.stack([cx, cy, sw, sh], axis=-1)
        anchors_list.append(tf.reshape(anchors, [-1, 4]))
    
    return tf.concat(anchors_list, axis=0)

def build_batch_dict_from_padded_labels(batch_labels: tf.Tensor, num_kpt: int, kpt_vals: int = 3) -> Dict[str, tf.Tensor]:
    """
    Parses the raw padded batch tensor into a structured dictionary for loss calculation.
    """
    cls_ids = batch_labels[..., 0:1] # (B, M, 1)
    bboxes = batch_labels[..., 1:5]  # (B, M, 4) [cx, cy, w, h]
    kpts = batch_labels[..., 5:]     # (B, M, K*V)
    
    # Identify padding rows (real boxes must have width > 0)
    valid_mask = bboxes[..., 2] > 0.0 # (B, M)
    
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
              bbox_area: tf.Tensor, sigmas: np.ndarray) -> tf.Tensor:
    """
    Computes Object Keypoint Similarity (OKS) Loss.
    """
    # (B, N, K, 1)
    d2 = tf.reduce_sum(tf.square(pred_kxy - gt_kxy), axis=-1, keepdims=True) 
    
    # Broadcast sigmas: (1, 1, K, 1)
    sig = tf.cast(tf.reshape(sigmas, [1, 1, -1, 1]), dtype=tf.float32)
    
    # Denominator
    denom = 2.0 * tf.square(sig) * (bbox_area[..., None] + 1e-12)
    
    e = tf.exp(-d2 / tf.maximum(denom, 1e-12))
    
    # Only calculate loss for visible keypoints
    oks = (1.0 - e) * kpt_v 
    
    # Sum over keypoints, normalize by number of visible keypoints
    visible_count = tf.reduce_sum(kpt_v, axis=[2, 3]) + 1e-9
    return tf.reduce_mean(tf.reduce_sum(oks, axis=[2, 3]) / visible_count)

def oks_loss(pred_kxy: tf.Tensor, gt_kxy: tf.Tensor, kpt_v: tf.Tensor, 
             bbox_area: tf.Tensor, sigmas: np.ndarray) -> tf.Tensor:
    """Computes Object Keypoint Similarity (OKS) Loss."""
    # (B, N, K, 2) -> (B, N, K, 1) distance squared
    d2 = tf.reduce_sum(tf.square(pred_kxy - gt_kxy), axis=-1, keepdims=True) 
    
    # Broadcast sigmas: (1, 1, K, 1)
    sig = tf.cast(tf.reshape(sigmas, [1, 1, -1, 1]), dtype=tf.float32)
    
    # Denominator
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
    def __init__(self, cfg: AppConfig, **kwargs):
        super().__init__(name="PoseLabelLoss", **kwargs)
        self.cfg = cfg
        self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        
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
        cls = tf.sigmoid(y_pred[..., 4: 4 + self.cfg.NUM_CLS])
        # Why: map_fn tracing 時常丟失靜態 shape；固定最後一維能避免後續解包/reshape 再爆
        box = tf.ensure_shape(box, [None, None, 4])
        cls = tf.ensure_shape(cls, [None, None, None])  # 先保守，下面再用 assert 鎖 NUM_CLS
 

        kpt_raw = y_pred[..., 4 + self.cfg.NUM_CLS:]
        B = tf.shape(y_pred)[0]
        N = tf.shape(y_pred)[1]

        tf.debugging.assert_equal(
            tf.shape(box)[-1], 4,
            message="PoseLabelLoss.decode_preds: pred_box last dim must be 4"
        )
        tf.debugging.assert_equal(
            tf.shape(cls)[-1], tf.cast(self.cfg.NUM_CLS, tf.int32),
            message="PoseLabelLoss.decode_preds: pred_cls last dim must be NUM_CLS"
        )


        kpt = tf.reshape(kpt_raw, [B, N, self.cfg.NUM_KPT, self.cfg.KPT_VALS])
        kxy = tf.sigmoid(kpt[..., :2])

        if self.cfg.KPT_VALS >= 3:
            kv = tf.sigmoid(kpt[..., 2:3])
            kpt_out = tf.concat([kxy, kv], axis=-1)
        else:
            kpt_out = kxy

        kpt_flat = tf.reshape(kpt_out, [B, N, -1])
        return box, cls, kpt_flat


    def call(self, y_pred: tf.Tensor, batch_dict: Dict[str, tf.Tensor], 
             anchors: tf.Tensor, class_weights: Optional[tf.Tensor] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Calculates total loss.
        """
        # 1. Decode Predictions
        pred_box, pred_cls, pred_kpt = self.decode_preds(y_pred)
        
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
            (pred_box, pred_cls, batch_dict['bboxes'], batch_dict['cls'], batch_dict['valid_mask']),
            fn_output_signature=out_spec
        )
        
        # 3. Compute Losses
        pos_mask_f = tf.cast(pos_mask, tf.float32)
        weight = pos_mask_f * quality
        den = tf.maximum(tf.reduce_sum(weight), 1.0) 
        
        # Gather matched GTs
        batch_idx = tf.repeat(tf.range(B, dtype=tf.int32)[:, None], N, axis=1)
        safe_gt_idx = tf.maximum(assigned_gt, 0)
        gather_idx = tf.stack([batch_idx, safe_gt_idx], axis=-1)
        
        target_box = tf.gather_nd(batch_dict['bboxes'], gather_idx)
        target_kpt_flat = tf.gather_nd(batch_dict['keypoints'], gather_idx) # (B, N, K*V)
        
        # --- A. Box Loss (CIoU) ---
        ciou = bbox_ciou(pred_box, target_box) 
        box_area = target_box[..., 2] * target_box[..., 3]
        box_scale = 3.0 - box_area 
        
        l_box = tf.reduce_sum(ciou * weight * box_scale) / den
        l_box *= self.cfg.W_BOX
        
        # --- B. Class Loss (VFL-like) ---
        t_cls = tf.one_hot(assigned_cls, self.cfg.NUM_CLS) * quality[..., None]
        t_cls = tf.where(pos_mask[..., None], t_cls, tf.zeros_like(t_cls))
        
        eps = 1e-9
        p_cls = tf.clip_by_value(pred_cls, eps, 1.0 - eps)
        l_cls_raw = -(t_cls * tf.math.log(p_cls) + (1.0 - t_cls) * tf.math.log(1.0 - p_cls))
        
        if class_weights is not None:
             cw = tf.reshape(class_weights, [1, 1, -1])
             l_cls_raw *= cw

        l_cls = self.cfg.W_CLS * (tf.reduce_sum(l_cls_raw) / den)
        
        # --- C. Keypoint Loss (OKS) ---
        l_kpt = tf.constant(0.0, dtype=tf.float32)
        
        if self.cfg.NUM_KPT > 0:
            # Pred: (B, N, K, V)
            pkpt = tf.reshape(pred_kpt, [B, N, self.cfg.NUM_KPT, self.cfg.KPT_VALS])
            pxy = pkpt[..., :2]
            
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

            oks = oks_loss(pxy, txy, tv_mask, area_px[..., None], self.sigmas)
            l_kpt_xy = (1.0 - oks)
            
            l_kpt = self.cfg.W_KPT_XY * (tf.reduce_sum(l_kpt_xy * weight) / den)
            
        total_loss = l_box + l_cls + l_kpt
        return total_loss, l_box, l_cls, l_kpt