
import config

import os
import pandas as pd
import numpy as np

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

def plot_and_save_loss_curve(history, save_path):
    """繪製損失曲線並儲存。"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.title('Distillation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
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

def save_pred_and_plot(step, batch_imgs, pred_raw, 
                       num_cls, num_kpt, kpt_vals,
                       out_dir="debug_pred", max_images=1, score_thr=0.05):
    """
    繪製模型的預測結果 (Prediction) 並儲存 CSV。
    
    pred_raw: 模型輸出的 Tensor (B, N, C) (尚未經過 Sigmoid)
    score_thr: 分數門檻值，只儲存/繪製置信度大於此值的預測
    """
    os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------
    preds_prob = pred_raw.numpy().astype(np.float32)
    '''
    # 1) 轉成 Numpy 並做 Sigmoid (因為 loss_tf 顯示模型輸出是 linear)
    # pred_raw shape: (B, N, C)
    preds_np = pred_raw.numpy()
    
    # 簡單的 Sigmoid 函數
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    preds_prob = sigmoid(preds_np)  # 全部轉成 0~1 機率/座標
    '''
# ---------------------------------------
    B, N, C = preds_prob.shape
    imgs_np = batch_imgs.numpy()
    imgs_np = np.clip(imgs_np, 0.0, 1.0)
    _, H, W, _ = imgs_np.shape
    
    # 用來收集所有 batch 的數據寫入 CSV
    csv_data_list = []
    
    # 定義 CSV 欄位
    columns = ['batch_idx', 'anchor_idx', 'score', 'cls_id', 'x', 'y', 'w', 'h']
    name_per_val = ['x', 'y', 'v'][:kpt_vals]
    for i in range(num_kpt):
        for j in range(kpt_vals):
            suffix = name_per_val[j] if j < len(name_per_val) else str(j)
            columns.append(f'kpt{i}_{suffix}')

    # 2) 逐張圖片處理
    for b in range(min(B, max_images)):
        img = imgs_np[b]
        pred_b = preds_prob[b] # (N, C)
        
        # --- 解析 pred_b ---
        # 結構通常是: [box(4), cls(num_cls), kpts(num_kpt*kpt_vals)]
        
        # 2.1 Box
        boxes = pred_b[:, 0:4] # xywh (0~1)
        
        # 2.2 Class & Score
        if num_cls > 0:
            cls_probs = pred_b[:, 4 : 4 + num_cls]
            # 找出每個 anchor 最大機率的類別
            cls_ids = np.argmax(cls_probs, axis=1)
            scores  = np.max(cls_probs, axis=1)
            kpt_start_idx = 4 + num_cls
        else:
            # 若無分類頭，假設 score 為 1 或使用 objectness (視模型而定，這邊先簡單設為 1)
            cls_ids = np.zeros(N, dtype=int)
            scores  = np.ones(N, dtype=float) 
            kpt_start_idx = 4
            
        # 2.3 Keypoints
        kpts_flat = pred_b[:, kpt_start_idx : kpt_start_idx + num_kpt * kpt_vals]
        kpts = kpts_flat.reshape(N, num_kpt, kpt_vals)
        
        # --- 過濾: 只保留 score > thr 的 anchors ---
        mask = scores > score_thr
        
        valid_boxes = boxes[mask]
        valid_scores = scores[mask]
        valid_cls = cls_ids[mask]
        valid_kpts = kpts[mask]
        valid_indices = np.where(mask)[0] # 原始 anchor index
        

        MAX_BOXES_DRAW = 50   # 你可以依需求調整

        num_valid = len(valid_scores)
        if num_valid > 0:
            # 先取 top-k box，避免畫太多東西
            if num_valid > MAX_BOXES_DRAW:
                top_idx = np.argsort(-valid_scores)[:MAX_BOXES_DRAW]  # 分數由大到小
                valid_boxes  = valid_boxes[top_idx]
                valid_scores = valid_scores[top_idx]
                valid_cls    = valid_cls[top_idx]
                valid_kpts   = valid_kpts[top_idx]
                valid_indices = valid_indices[top_idx]
                num_valid = MAX_BOXES_DRAW


        # 3) 收集 CSV 資料
        num_valid = len(valid_scores)
        if num_valid > 0:
            # 建構這張圖的數據矩陣
            # 格式: [b, anchor_idx, score, cls, x, y, w, h, kpt...]
            batch_col = np.full((num_valid, 1), b)
            anch_col  = valid_indices[:, None]
            score_col = valid_scores[:, None]
            cls_col   = valid_cls[:, None]
            box_col   = valid_boxes
            kpt_col   = valid_kpts.reshape(num_valid, -1)
            
            row_data = np.concatenate([
                batch_col, anch_col, score_col, cls_col, box_col, kpt_col
            ], axis=1)
            csv_data_list.append(row_data)

        # 4) 畫圖 (matplotlib)
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        for i in range(num_valid):
            cx, cy, bw, bh = valid_boxes[i]
            # 轉回絕對座標
            cx *= W; cy *= H; bw *= W; bh *= H
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            
            # 畫 Box
            rect = Rectangle((x1, y1), bw, bh, fill=False, color='red', linewidth=1.5)
            ax.add_patch(rect)
            
            # 標示分數
            ax.text(x1, y1, f"{valid_scores[i]:.2f}", color='yellow', fontsize=8, backgroundcolor='black')

            # 畫 Keypoints
            kp = valid_kpts[i] # (K, V)
            for k in range(num_kpt):
                kx = kp[k, 0] * W
                ky = kp[k, 1] * H
                
                # 若有 visibility
                if kpt_vals > 2:
                    vis = kp[k, 2]
                    if vis < 0.5: continue # 預測不可見就不畫
                
                ax.scatter(kx, ky, s=10, c='cyan')

        ax.set_title(f"Pred step {step}, img {b} (thr={score_thr})")
        ax.axis("off")
        
        img_out_path = os.path.join(out_dir, f"pred_step{step:06d}_img{b}.png")
        plt.savefig(img_out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    # 5) 儲存 CSV
    if csv_data_list:
        all_data = np.concatenate(csv_data_list, axis=0)
        df = pd.DataFrame(all_data, columns=columns)
        csv_path = os.path.join(out_dir, f"pred_step{step:06d}.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"📈 Pred plots & CSV saved to {out_dir}")
    else:
        # print(f"⚠️ Step {step}: No predictions > {score_thr}")
        pass