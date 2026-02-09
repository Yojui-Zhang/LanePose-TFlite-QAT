import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from QAT_Refactored.utils.tensor_layout import ensure_bnc_np


# 定義一組顯眼的顏色 (RGB format for Matplotlib, BGR for OpenCV)
class Colors:
    def __init__(self):
        self.palette = [
            (255, 56, 56),   # Red
            (255, 157, 151), # Light Red
            (255, 112, 31),  # Orange
            (255, 178, 29),  # Yellow
            (207, 210, 49),  # Lime
            (72, 249, 10),   # Green
            (146, 204, 23),  # Dark Green
            (61, 219, 134),  # Cyan-Green
            (26, 147, 52),   # Forest
            (0, 212, 187),   # Cyan
            (44, 153, 168),  # Teal
            (0, 194, 255),   # Sky Blue
            (52, 69, 147),   # Blue
            (100, 115, 255), # Light Blue
            (0, 24, 236),    # Dark Blue
            (132, 56, 255),  # Purple
            (82, 0, 133),    # Dark Purple
            (203, 56, 255),  # Magenta
            (255, 149, 200), # Pink
            (255, 55, 199)   # Hot Pink
        ]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

colors = Colors()

def _cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] boxes to [x1, y1, x2, y2]."""
    b = np.asarray(boxes, dtype=np.float32)
    if b.size == 0:
        return b.reshape(0, 4)
    x1 = b[:, 0] - 0.5 * b[:, 2]
    y1 = b[:, 1] - 0.5 * b[:, 3]
    x2 = b[:, 0] + 0.5 * b[:, 2]
    y2 = b[:, 1] + 0.5 * b[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)

def _iou_with_one_np(one_box_xyxy: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and many boxes (xyxy format)."""
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(one_box_xyxy[0], boxes_xyxy[:, 0])
    y1 = np.maximum(one_box_xyxy[1], boxes_xyxy[:, 1])
    x2 = np.minimum(one_box_xyxy[2], boxes_xyxy[:, 2])
    y2 = np.minimum(one_box_xyxy[3], boxes_xyxy[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, one_box_xyxy[2] - one_box_xyxy[0]) * np.maximum(0.0, one_box_xyxy[3] - one_box_xyxy[1])
    area_b = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)

def tflite_style_nms_indices(
    boxes_cxcywh: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh_bbox: float = 0.45,
    iou_thresh_lane: float = 0.45,
    lane_class_id: int = 0,
    max_det: int = 300,
) -> np.ndarray:
    """
    NMS close to TFlite.h logic:
    - if both classes are non-lane -> use bbox threshold
    - otherwise -> use lane threshold
    """
    boxes = np.asarray(boxes_cxcywh, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)

    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    order = np.argsort(scores)[::-1]
    boxes_xyxy = _cxcywh_to_xyxy_np(np.clip(boxes, 0.0, 1.0))
    keep = []

    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = _iou_with_one_np(boxes_xyxy[i], boxes_xyxy[rest])

        cls_i = class_ids[i]
        cls_rest = class_ids[rest]
        both_non_lane = np.logical_and(cls_i != lane_class_id, cls_rest != lane_class_id)
        suppress_bbox = ious > float(iou_thresh_bbox)
        suppress_lane = ious > float(iou_thresh_lane)
        suppress = np.where(both_non_lane, suppress_bbox, suppress_lane)
        order = rest[~suppress]

    return np.asarray(keep, dtype=np.int32)

def draw_on_image(
    img: np.ndarray,
    boxes: np.ndarray,
    kpts: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    conf_thres: float = 0.25,
) -> np.ndarray:
    """
    在單張圖片上繪製 Box 與 Keypoints。
    img: numpy array (H, W, 3) range [0, 255], uint8, RGB
    boxes: (N, 4) [cx, cy, w, h] normalized [0, 1]
    kpts: (N, K*V) 或 (N, K, V) normalized [0, 1]
    return: BGR image (for OpenCV drawing and later cvtColor(BGR2RGB) in matplotlib)
    """
    h, w = img.shape[:2]
    # Batch pipeline yields RGB tensors; convert once so OpenCV draws on BGR consistently.
    img_draw = cv2.cvtColor(np.ascontiguousarray(img.copy()), cv2.COLOR_RGB2BGR)

    if boxes is None or len(boxes) == 0:
        return img_draw

    for i, box in enumerate(boxes):
        # Filter by score if provided
        if scores is not None and scores[i] < conf_thres:
            continue

        # Color
        cls_id = int(class_ids[i]) if class_ids is not None else 0
        color = colors(cls_id, bgr=True)
        
        # Decode Box (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw Box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        label = f"id:{cls_id}"
        if scores is not None:
            label += f" {scores[i]:.2f}"
            
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        cv2.rectangle(img_draw, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Draw Keypoints
        if kpts is not None and len(kpts) > i:
            # Why: 同時支援 flat 與 (K,V)；避免用 len(k)//3 猜測導致 reshape/繪製錯誤。
            k = np.asarray(kpts[i])
            if k.ndim == 2:
                k2 = k
            elif k.ndim == 1:
                if (k.size % 3) == 0:
                    k2 = k.reshape((-1, 3))
                elif (k.size % 2) == 0:
                    k2 = k.reshape((-1, 2))
                else:
                    continue
            else:
                continue

            for j in range(k2.shape[0]):
                kx, ky = float(k2[j, 0]), float(k2[j, 1])
                vis = float(k2[j, 2]) if k2.shape[1] > 2 else 1.0
                
                if vis > 0.5 and kx != 0 and ky != 0:
                    px, py = int(kx * w), int(ky * h)
                    # 依據關鍵點索引變色
                    k_color = colors(j, bgr=True)
                    cv2.circle(img_draw, (px, py), 3, k_color, -1)

    return img_draw

def save_gt_and_plot(batch_imgs, batch_labels, output_dir, step, prefix="val_gt", max_imgs=4):
    """
    儲存 Batch 中的 GT 圖像。
    Args:
        batch_imgs: Tensor (B, H, W, 3) float [0,1]
        batch_labels: Tensor (B, M, D)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    B = batch_imgs.shape[0]
    num_plot = min(B, max_imgs)
    
    # Convert to numpy
    imgs_np = (batch_imgs.numpy() * 255).astype(np.uint8)
    labels_np = batch_labels.numpy()
    
    fig, axes = plt.subplots(1, num_plot, figsize=(15, 5))
    if num_plot == 1: axes = [axes]
    
    for i in range(num_plot):
        img = imgs_np[i]
        lbl = labels_np[i] # (M, D)
        
        # Filter padding (0,0,0,0)
        mask = np.any(lbl[:, 1:5] > 0, axis=1)
        valid_lbl = lbl[mask]
        
        # Parse Format: cls, cx, cy, w, h, kpts...
        if len(valid_lbl) > 0:
            cls_ids = valid_lbl[:, 0]
            boxes = valid_lbl[:, 1:5]
            kpts = valid_lbl[:, 5:]
        else:
            cls_ids, boxes, kpts = [], [], []

        img_res = draw_on_image(img, boxes, kpts, class_ids=cls_ids)
        
        axes[i].imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"GT: {i}")

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_step{step}.jpg"
    plt.savefig(save_path)
    plt.close()
    print(f"[Vis] Saved GT plot to {save_path}")

def save_pred_and_plot(
    batch_imgs: tf.Tensor,
    preds: tf.Tensor,
    output_dir,
    step: int,
    prefix: str = "val_pred",
    max_imgs: int = 4,
    conf_thres: float = 0.25,
    num_cls: int = 7,
    num_kpt: int = 17,
    kpt_vals: int = 3,
    total_C: Optional[int] = None,
    force_draw_topk_if_empty: int = 1,
    nms_iou_thres_bbox: float = 0.45,
    nms_iou_thres_lane: float = 0.45,
    lane_class_id: int = 0,
    max_det: int = 300,
) -> None:
    """
    preds: Tensor (B, C, N) 或 (B, N, C)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B = batch_imgs.shape[0]
    num_plot = min(B, max_imgs)

    imgs_np = (batch_imgs.numpy() * 255).astype(np.uint8)
    preds_np = preds.numpy()

    # --- 1) 統一成 (B, N, C)（total_C 強約束，失配直接報錯）---
    preds_np = ensure_bnc_np(preds_np, total_c=total_C, num_cls=num_cls)

    fig, axes = plt.subplots(1, num_plot, figsize=(15, 5))
    if num_plot == 1:
        axes = [axes]

    for i in range(num_plot):
        img = imgs_np[i]
        p = preds_np[i]  # (N, C)

        needed_c = 4 + num_cls + (num_kpt * kpt_vals)
        if p.shape[1] < needed_c:
            raise ValueError(f"[Vis] invalid channel count: C={p.shape[1]} < needed={needed_c}")
 

        # --- 2) decode ---
        box = p[:, :4]
        if np.min(box) < 0 or np.max(box) > 1.0:
            box = 1 / (1 + np.exp(-box))  # sigmoid

        cls_raw = p[:, 4:4+num_cls]
        # Why: avoid double-sigmoid when upstream already activated.
        if np.min(cls_raw) < 0 or np.max(cls_raw) > 1.0:
            cls_prob = 1 / (1 + np.exp(-cls_raw))  # sigmoid
        else:
            cls_prob = cls_raw

        kpt_raw = p[:, 4+num_cls:4+num_cls+(num_kpt*kpt_vals)]
        if kpt_raw.size == 0:
            kpt = None
        else:
            if np.min(kpt_raw) < 0 or np.max(kpt_raw) > 1.0:
                kpt_raw = 1 / (1 + np.exp(-kpt_raw))  # sigmoid
            kpt = kpt_raw.reshape((-1, num_kpt, kpt_vals))  # (N,K,V)


        scores = np.max(cls_prob, axis=1).astype(np.float32)
        class_ids = np.argmax(cls_prob, axis=1).astype(np.int32)

        candidate_idx = np.where(scores > conf_thres)[0]
        used_topk_fallback = False

        if candidate_idx.size > 0:
            keep_local = tflite_style_nms_indices(
                boxes_cxcywh=box[candidate_idx],
                scores=scores[candidate_idx],
                class_ids=class_ids[candidate_idx],
                iou_thresh_bbox=nms_iou_thres_bbox,
                iou_thresh_lane=nms_iou_thres_lane,
                lane_class_id=lane_class_id,
                max_det=max_det,
            )
            keep_idx = candidate_idx[keep_local]
            valid_box = box[keep_idx]
            valid_scores = scores[keep_idx]
            valid_cls = class_ids[keep_idx]
            valid_kpt = (kpt[keep_idx] if (kpt is not None) else None)
        else:
            valid_box = np.zeros((0, 4), dtype=np.float32)
            valid_scores = np.zeros((0,), dtype=np.float32)
            valid_cls = np.zeros((0,), dtype=np.int32)
            valid_kpt = (np.zeros((0, num_kpt, kpt_vals), dtype=np.float32) if (kpt is not None) else None)

        # Why: 若全數 < conf_thres，圖會「完全空白」造成誤判；至少畫出 top-k 便於判讀分數是否被壓低。
        if valid_box.shape[0] == 0 and force_draw_topk_if_empty > 0 and box.shape[0] > 0:
            topk = int(min(force_draw_topk_if_empty, scores.shape[0]))
            top_idx = np.argsort(scores)[-topk:][::-1]
            valid_box = box[top_idx]
            valid_scores = scores[top_idx]
            valid_cls = class_ids[top_idx]
            valid_kpt = (kpt[top_idx] if (kpt is not None) else None)
            used_topk_fallback = True

        # console summary (debug)
        try:
            mx = float(np.max(scores)) if scores.size else 0.0
            cnt = int(candidate_idx.size) if scores.size else 0
            cnt_nms = int(valid_box.shape[0])
            print(f"[Vis] step={step} img={i} max_score={mx:.4f} cnt>{conf_thres}={cnt} nms={cnt_nms}")
        except Exception:
            pass

        draw_conf_thres = 0.0 if used_topk_fallback else conf_thres
        img_res = draw_on_image(
            img,
            valid_box,
            kpts=valid_kpt,
            class_ids=valid_cls,
            scores=valid_scores,
            conf_thres=draw_conf_thres,
        )

        axes[i].imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {i}")

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_step{step}.jpg"
    plt.savefig(save_path)
    plt.close()
