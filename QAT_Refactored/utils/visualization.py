import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
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

def draw_on_image(img, boxes, kpts=None, class_ids=None, scores=None, conf_thres=0.25):
    """
    在單張圖片上繪製 Box 與 Keypoints。
    img: numpy array (H, W, 3) range [0, 255], uint8
    boxes: (N, 4) [cx, cy, w, h] normalized [0, 1]
    kpts: (N, K*V) normalized [0, 1]
    """
    h, w = img.shape[:2]
    img_draw = img.copy() # Avoid modifying original

    # 確保是 contiguous array 以便 OpenCV 繪圖
    img_draw = np.ascontiguousarray(img_draw)

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
            # Reshape flat kpts to (K, V)
            # 假設 V=3 (x, y, v) 或 V=2 (x, y)
            k = kpts[i]
            num_points = len(k) // 3 # Guessing V=3
            if len(k) % 3 != 0: 
                num_points = len(k) // 2 # Fallback to V=2

            k = k.reshape((num_points, -1))
            
            for j in range(num_points):
                kx, ky = k[j, 0], k[j, 1]
                vis = k[j, 2] if k.shape[1] > 2 else 1.0
                
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

def save_pred_and_plot(batch_imgs, preds, output_dir, step, prefix="val_pred",
                       max_imgs=4, conf_thres=0.25, num_cls=7, total_C=None):
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

        if p.shape[1] < (4 + num_cls):
            raise ValueError(f"[Vis] invalid channel count: C={p.shape[1]} < 4+num_cls={4+num_cls}")

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

        scores = np.max(cls_prob, axis=1)
        class_ids = np.argmax(cls_prob, axis=1)

        mask = scores > conf_thres
        valid_box = box[mask]
        valid_scores = scores[mask]
        valid_cls = class_ids[mask]

        img_res = draw_on_image(img, valid_box, class_ids=valid_cls, scores=valid_scores, conf_thres=conf_thres)

        axes[i].imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {i}")

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_step{step}.jpg"
    plt.savefig(save_path)
    plt.close()
