import tensorflow as tf
import config

import os
import cv2
import pandas as pd
import numpy as np

from src.process.pred_model import ensure_BNC_static

if config.PLOT_Switch == True:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

def plot_and_save_lr_schedule(schedule, total_steps, save_path):
    """繪製學習率變化曲線並儲存。"""
    steps = np.arange(total_steps)
    lrs = [schedule(step) for step in steps]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Learning rate schedule plot saved to {save_path}")

def plot_and_save_loss_curve(history, save_path, y_key="train_total"):
    """支援:
      1) history = [float, float, ...]
      2) history = [{"epoch":1, "train_total":..., "val_total":...}, ...]
    """

    if history is None or len(history) == 0:
        print("[plot] history is empty, skip.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # case: list of dict
    if isinstance(history, (list, tuple)) and isinstance(history[0], dict):
        xs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
        ys = [h.get(y_key, np.nan) for h in history]
    else:
        xs = list(range(1, len(history) + 1))
        ys = history

    ys = np.asarray(ys, dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, label=y_key)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve plot saved to {save_path}")


    
def save_gt_and_plot(step, batch_imgs, batch_dict,
                     num_kpt, kpt_vals,
                     out_dir="debug_gt", max_images=1):
    """
    修正版：
    1. 確保所有陣列在 concatenate 前都 reshape 成 2D。
    2. 使用 num_objects 過濾掉 Padding 的 0 值數據，讓 CSV 和圖片更乾淨。
    """
    os.makedirs(out_dir, exist_ok=True)

    bd = batch_dict

    # 1. 取得原始數據 (B, M, ...)
    # -------------------------------------------------
    # 假設 shape:
    # batch_idx: (B, M, 1)
    # cls:       (B, M, 1)
    # bboxes:    (B, M, 4)
    # kpts:      (B, M, K, V)
    
    # 轉 numpy 並展平主要維度 (B*M)
    raw_batch_idx = bd['batch_idx'].numpy().reshape(-1)      # (G,)
    raw_cls       = bd['cls'].numpy().reshape(-1)            # (G,)
    raw_bboxes    = bd['bboxes'].numpy().reshape(-1, 4)      # (G, 4) <--- 關鍵修正：強制轉為 2D
    
    # Keypoints 處理
    raw_kpts      = bd['keypoints'].numpy()
    G = raw_batch_idx.shape[0]
    raw_kpts_flat = raw_kpts.reshape(G, -1)                  # (G, K*V)

    # 2. 過濾 Padding (Optional but Recommended)
    # -------------------------------------------------
    # 如果 batch_dict 裡有 num_objects，我們可以用它來過濾掉補 0 的部分
    if 'num_objects' in bd:
        num_objs = bd['num_objects'].numpy().reshape(-1) # (B,)
        B_dim = num_objs.shape[0]
        # 計算 M (Max objects per image)
        M_dim = G // B_dim 
        
        # 建立一個 mask 來標示哪些是真實物件
        # 每個 batch 內部的 index: [0, 1, 2, ... M-1, 0, 1, ...]
        idx_in_batch = np.tile(np.arange(M_dim), B_dim) 
        
        # 對應的物件數量限制: [N0, N0, ..., N1, N1, ...]
        objs_limit = np.repeat(num_objs, M_dim)
        
        # 若 index < limit 則為真
        valid_mask = idx_in_batch < objs_limit
        
        # 應用 Mask
        batch_idx_fin = raw_batch_idx[valid_mask]
        cls_fin       = raw_cls[valid_mask]
        bboxes_fin    = raw_bboxes[valid_mask]
        kpts_fin      = raw_kpts_flat[valid_mask]
    else:
        # 如果沒有 num_objects，就全部保留 (或是過濾掉 w=0 的)
        batch_idx_fin = raw_batch_idx
        cls_fin       = raw_cls
        bboxes_fin    = raw_bboxes
        kpts_fin      = raw_kpts_flat

    # 3. 製作 CSV 數據 (Concatenate)
    # -------------------------------------------------
    if len(batch_idx_fin) > 0:
        data = np.concatenate([
            batch_idx_fin[:, None],    # (N_valid, 1)
            cls_fin[:, None],          # (N_valid, 1)
            bboxes_fin,                # (N_valid, 4)
            kpts_fin                   # (N_valid, K*V)
        ], axis=1)

        columns = ['batch_idx', 'cls', 'x', 'y', 'w', 'h']
        name_per_val = ['x', 'y', 'v'][:kpt_vals]
        for i in range(num_kpt):
            for j in range(kpt_vals):
                suffix = name_per_val[j] if j < len(name_per_val) else str(j)
                columns.append(f'kpt{i}_{suffix}')

        df = pd.DataFrame(data, columns=columns)
        csv_path = os.path.join(out_dir, f"gt_step{step:06d}.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
    
    # 4. 繪圖 (Matplotlib)
    # -------------------------------------------------
    imgs_np = batch_imgs.numpy()
    B, H, W, C = imgs_np.shape
    imgs_np = np.clip(imgs_np, 0.0, 1.0)

    for b in range(min(B, max_images)):
        img = imgs_np[b]

        fig, ax = plt.subplots()
        ax.imshow(img)

        # 從「過濾後」的數據中找出屬於這張圖 (batch_idx == b) 的資料
        mask_b = (batch_idx_fin == b)
        
        if np.any(mask_b):
            boxes_b = bboxes_fin[mask_b]           
            kpts_b  = kpts_fin[mask_b].reshape(-1, num_kpt, kpt_vals)

            for box, kpt_obj in zip(boxes_b, kpts_b):
                cx, cy, bw, bh = box
                # 轉回絕對座標
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                w_px = bw * W
                h_px = bh * H

                # 畫 bbox
                rect = Rectangle((x1, y1), w_px, h_px,
                                 fill=False, linewidth=1.5, edgecolor='green')
                ax.add_patch(rect)

                # 畫 keypoints
                for k in range(num_kpt):
                    kx = kpt_obj[k, 0] * W
                    ky = kpt_obj[k, 1] * H
                    
                    if kpt_vals > 2:
                        vis = kpt_obj[k, 2]
                        if vis < 0.5: continue

                    ax.scatter(kx, ky, s=10, c='magenta')

        ax.set_title(f"GT step {step}, img {b}")
        ax.axis("off")

        img_out_path = os.path.join(out_dir, f"gt_step{step:06d}_img{b}.png")
        plt.savefig(img_out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        

def save_pred_and_plot(
    step,
    batch_imgs,
    pred_raw,
    num_cls,
    num_kpt,
    kpt_vals,
    anchors,
    out_dir,
    max_images=1,
    score_thr=0.3,
):
    """
    將模型預測結果 decode 成 bbox + kpt 並畫在圖片上，同時輸出 CSV。

    ★ 重點：這裡的 decode 完全模仿 pred_save_model.py：
        pred[i] = [cx, cy, w, h, cls0, cls1, ..., kpt0_x, kpt0_y, kpt0_v, ...]
        全部都是 0~1 的 normalized 值
    """
    os.makedirs(out_dir, exist_ok=True)

    expected_C = 4 + num_cls + num_kpt * kpt_vals

    # ---- 1) 轉成 (B, N, C) ----
    # pred_raw 可能是 list / dict / tensor，統一轉成 BNC
    preds_BNC = ensure_BNC_static(
        pred_raw,
        expected_C=expected_C,   # 只傳 expected_C 這一個參數
    )  # (B, N, C)

    preds_BNC = preds_BNC.numpy() if hasattr(preds_BNC, "numpy") else np.array(preds_BNC)

    B, N, C = preds_BNC.shape
    
    if C != expected_C:
        # 保險：只取前 expected_C 維度
        preds_BNC = preds_BNC[:, :, :expected_C]
        C = expected_C

    # ---- 2) 每張圖處理 ----
    num_imgs = min(B, max_images)

    for b in range(num_imgs):
        img = batch_imgs[b]  # tensor 或 numpy
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img)

        # 假設 img 為 (H, W, 3)，0~1 → 0~255
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]

        pred = preds_BNC[b]  # (N, C)

        # ---- split columns ----
        boxes   = pred[:, 0:4]                      # (N, 4) cx,cy,w,h (0~1)
        cls_all = pred[:, 4:4 + num_cls]            # (N, num_cls)
        kpt_all = pred[:, 4 + num_cls:]             # (N, num_kpt * kpt_vals)


        # Sigmoid ***********************
        # cls_all = tf.sigmoid(cls_all)
        # kpt_all[...,2] = tf.sigmoid(kpt_all[...,2])


        # 選出 conf & cls_id
        cls_ids  = np.argmax(cls_all, axis=-1)      # (N,)
        cls_conf = np.max(cls_all, axis=-1)         # (N,)

        keep = cls_conf > score_thr
        boxes   = boxes[keep]
        cls_ids = cls_ids[keep]
        cls_conf_keep = cls_conf[keep]
        kpt_all = kpt_all[keep]

        # 用來存 CSV
        csv_rows = []

        for i in range(boxes.shape[0]):
            cx, cy, bw, bh = boxes[i]

            # 轉成左上右下像素座標（跟 pred_save_model 邏輯一致）
            x1 = int((cx - bw / 2.0) * W)
            y1 = int((cy - bh / 2.0) * H)
            x2 = int((cx + bw / 2.0) * W)
            y2 = int((cy + bh / 2.0) * H)

            # clamp
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            # 畫 bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"id:{int(cls_ids[i])} {cls_conf_keep[i]:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ---- keypoints ----
            if num_kpt > 0 and kpt_vals > 0 and kpt_all.size > 0:
                kpt_flat = kpt_all[i]  # (num_kpt * kpt_vals,)
                kpt = kpt_flat.reshape(num_kpt, kpt_vals)

                for ki in range(num_kpt):
                    kx = kpt[ki, 0]
                    ky = kpt[ki, 1]
                    kv = kpt[ki, 2] if kpt_vals > 2 else 1.0

                    # 跟 pred_save_model 一樣，用 kv > 0.5 當作可見
                    if kv > 0.5:
                        px = int(kx * W)
                        py = int(ky * H)
                        cv2.circle(img, (px, py), 3, (0, 0, 255), -1)

            # 組 CSV 一列：
            # [cx, cy, w, h, cls_id, conf, kpt0_x, kpt0_y, kpt0_v, ...]
            row = [cx, cy, bw, bh, int(cls_ids[i]), float(cls_conf_keep[i])]
            if num_kpt > 0 and kpt_vals > 0 and kpt_all.size > 0:
                row.extend(kpt_flat.tolist())
            csv_rows.append(row)

        # ---- 存圖 ----
        img_name = os.path.join(out_dir, f"pred_step{step:06d}_img{b}.png")
        cv2.imwrite(img_name, img)

        # ---- 存 CSV ----
        import pandas as pd
        columns = ["cx", "cy", "w", "h", "cls_id", "conf"]

        if num_kpt > 0 and kpt_vals > 0:
            # kpt0_x, kpt0_y, kpt0_v, kpt1_x, ...
            name_per_val = ['x', 'y', 'v'][:kpt_vals]
            for ki in range(num_kpt):
                for vj in range(kpt_vals):
                    columns.append(f"kpt{ki}_{name_per_val[vj] if vj < len(name_per_val) else vj}")

        csv_path = os.path.join(out_dir, f"pred_step{step:06d}.csv")
        if len(csv_rows) == 0:
            # 沒有任何 detection，仍然建一個空檔方便 debug
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(csv_rows, columns=columns)

        df.to_csv(csv_path, index=False, float_format="%.6f")




