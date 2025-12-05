
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
                       anchors,
                       out_dir="debug_pred", max_images=1, score_thr=0.4):
    """
    Route A 版本：
    - pred_raw: 模型輸出的 Tensor (B, N, C)，仍是 logits
    - anchors:  anchor tensor/ndarray, 形狀 (N, 4)，為 0~1 的 [cx, cy, w, h]

    流程：
    1) 將 box / kpt / cls logits 做 sigmoid → 0~1 的相對數值
    2) 依照 anchors (與 loss_tf.decode_box / decode_kpt 相同的公式) 解碼為
       0~1 的絕對座標
    3) 依 score_thr 篩選 → 畫在影像上並輸出 CSV
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 轉成 numpy ----------
    if hasattr(batch_imgs, "numpy"):
        imgs_np = batch_imgs.numpy()
    else:
        imgs_np = np.asarray(batch_imgs, dtype=np.float32)
    imgs_np = np.clip(imgs_np, 0.0, 1.0)     # 確保在 [0,1]

    if hasattr(pred_raw, "numpy"):
        preds_np = pred_raw.numpy()
    else:
        preds_np = np.asarray(pred_raw, dtype=np.float32)

    B, H, W, _ = imgs_np.shape
    B2, N, C = preds_np.shape
    assert B2 == B, f"B dimension mismatch: imgs {B}, preds {B2}"

    # ---------- anchors 處理 ----------
    if anchors is None:
        raise ValueError("Route A 版本的 save_pred_and_plot 需要 anchors 參數")

    if hasattr(anchors, "numpy"):
        anchors_np = anchors.numpy()
    else:
        anchors_np = np.asarray(anchors, dtype=np.float32)

    # 確認 anchor 個數與 N 一致，若不一致則取 min(N) 以避免 crash
    if anchors_np.shape[0] != N:
        minN = min(N, anchors_np.shape[0])
        preds_np   = preds_np[:, :minN, :]
        anchors_np = anchors_np[:minN, :]
        N = minN

    anchors_np = anchors_np.astype(np.float32)          # (N, 4)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- 先把所有 batch 的 box / cls / kpt decode 完 ----------
    # 1) box logits → sigmoid → 0~1 相對值
    box_logits = preds_np[..., :4]                      # (B, N, 4)
    box_rel    = sigmoid(box_logits)                    # 0~1，相對 anchor

    # 2) class logits
    if num_cls > 0:
        cls_logits    = preds_np[..., 4 : 4 + num_cls]  # (B, N, num_cls)
        cls_probs_all = sigmoid(cls_logits)             # 0~1
        kpt_start_idx = 4 + num_cls
    else:
        cls_probs_all = None
        kpt_start_idx = 4

    # 3) keypoint logits
    if num_kpt > 0:
        kpt_logits = preds_np[..., kpt_start_idx : kpt_start_idx + num_kpt * kpt_vals]
        kpt_rel    = sigmoid(kpt_logits)                # (B, N, K*V)
    else:
        kpt_logits = None
        kpt_rel    = None

    # ---- anchors: (N,4) → (1,N,4) ----
    anchors_b1 = anchors_np[None, ...]                  # (1, N, 4)
    a_xy = anchors_b1[..., 0:2]                         # (1, N, 2)
    a_wh = anchors_b1[..., 2:4]                         # (1, N, 2)

    # ---------- box decode：對齊 loss_tf.decode_box ----------
    # decoded_xy = a_xy + (p_xy - 0.5) * a_wh
    # decoded_wh = a_wh * (2 * p_wh)^2
    p_xy = box_rel[..., 0:2]                            # (B, N, 2)
    p_wh = box_rel[..., 2:4]                            # (B, N, 2)
    decoded_xy = a_xy + (p_xy - 0.5) * a_wh             # (B, N, 2)
    decoded_wh = a_wh * np.square(p_wh * 2.0)           # (B, N, 2)
    boxes_decoded = np.concatenate([decoded_xy, decoded_wh], axis=-1)  # (B, N, 4)

    # ---------- kpt decode：對齊 loss_tf.decode_kpt ----------
    if num_kpt > 0 and kpt_rel is not None:
        Bn, Nn, _ = kpt_rel.shape
        assert Bn == B and Nn == N
        pred_kpt = kpt_rel.reshape(B, N, num_kpt, kpt_vals)  # (B, N, K, V)
        kp_xy    = pred_kpt[..., 0:2]                        # (B, N, K, 2)

        # a_xy_exp / a_wh_exp: (1, N, 1, 2)，透過 broadcast 擴到 (B, N, K, 2)
        a_xy_exp = a_xy[..., None, :]                        # (1, N, 1, 2)
        a_wh_exp = a_wh[..., None, :]                        # (1, N, 1, 2)

        # decoded_kp_xy = a_xy + (kp_xy - 0.5) * a_wh * 4.0
        decoded_kp_xy = a_xy_exp + (kp_xy - 0.5) * a_wh_exp * 4.0

        if kpt_vals > 2:
            kp_rest = pred_kpt[..., 2:]                      # (B, N, K, V-2)
            decoded_kpt = np.concatenate([decoded_kp_xy, kp_rest], axis=-1)
        else:
            decoded_kpt = decoded_kp_xy

        kpts_decoded_flat = decoded_kpt.reshape(B, N, num_kpt * kpt_vals)
    else:
        kpts_decoded_flat = np.zeros((B, N, num_kpt * kpt_vals), dtype=np.float32)

    # ---------- 準備 CSV 欄位 ----------
    csv_data_list = []
    columns = ['batch_idx', 'anchor_idx', 'score', 'cls_id', 'x', 'y', 'w', 'h']
    name_per_val = ['x', 'y', 'v'][:kpt_vals]
    for i in range(num_kpt):
        for j in range(kpt_vals):
            suffix = name_per_val[j] if j < len(name_per_val) else str(j)
            columns.append(f'kpt{i}_{suffix}')

    # ---------- 逐張圖片處理 & 畫圖 ----------
    for b in range(min(B, max_images)):
        img = imgs_np[b]
        boxes_b      = boxes_decoded[b]                  # (N, 4)
        kpts_flat_b  = kpts_decoded_flat[b]              # (N, K*V)
        kpts_b       = kpts_flat_b.reshape(N, num_kpt, kpt_vals) if num_kpt > 0 else None

        # class score / id
        if num_cls > 0:
            cls_probs_b = cls_probs_all[b]               # (N, num_cls)
            cls_ids     = np.argmax(cls_probs_b, axis=1)
            scores      = np.max(cls_probs_b, axis=1)
        else:
            cls_ids = np.zeros(N, dtype=int)
            scores  = np.ones(N, dtype=float)

        # 依 score_thr 篩選
        mask = scores > score_thr
        if not np.any(mask):
            continue

        valid_boxes   = boxes_b[mask]
        valid_scores  = scores[mask]
        valid_cls     = cls_ids[mask]
        valid_kpts    = kpts_b[mask] if kpts_b is not None else None
        valid_indices = np.where(mask)[0]

        MAX_BOXES_DRAW = 50
        num_valid = len(valid_scores)
        if num_valid > MAX_BOXES_DRAW:
            top_idx      = np.argsort(-valid_scores)[:MAX_BOXES_DRAW]
            valid_boxes  = valid_boxes[top_idx]
            valid_scores = valid_scores[top_idx]
            valid_cls    = valid_cls[top_idx]
            if valid_kpts is not None:
                valid_kpts = valid_kpts[top_idx]
            valid_indices = valid_indices[top_idx]
            num_valid     = MAX_BOXES_DRAW

        # 3) 收集 CSV
        if num_valid > 0:
            batch_col = np.full((num_valid, 1), b)
            anch_col  = valid_indices[:, None]
            score_col = valid_scores[:, None]
            cls_col   = valid_cls[:, None]
            box_col   = valid_boxes
            if valid_kpts is not None:
                kpt_col = valid_kpts.reshape(num_valid, -1)
            else:
                kpt_col = np.zeros((num_valid, num_kpt * kpt_vals), dtype=np.float32)

            row_data = np.concatenate(
                [batch_col, anch_col, score_col, cls_col, box_col, kpt_col],
                axis=1
            )
            csv_data_list.append(row_data)

        # 4) 繪圖
        if config.PLOT_Switch:
            fig, ax = plt.subplots()
            ax.imshow(img)

            for i in range(num_valid):
                cx, cy, bw, bh = valid_boxes[i]
                # 轉回 pixel
                cx *= W; cy *= H; bw *= W; bh *= H
                x1 = cx - bw / 2
                y1 = cy - bh / 2

                rect = Rectangle(
                    (x1, y1), bw, bh,
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                lbl = f"{valid_cls[i]}:{valid_scores[i]:.2f}"
                ax.text(
                    x1, y1 - 5, lbl, fontsize=8, color='yellow',
                    bbox=dict(facecolor='black', alpha=0.7, pad=1)
                )

                if valid_kpts is not None:
                    kp = valid_kpts[i]
                    for k in range(num_kpt):
                        kx = kp[k, 0] * W
                        ky = kp[k, 1] * H
                        if kpt_vals > 2:
                            vis = kp[k, 2]
                            if vis < 0.5:
                                continue
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
