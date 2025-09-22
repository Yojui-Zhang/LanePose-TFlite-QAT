# src/Loss_function/loss_tf.py
import tensorflow as tf
import math

import config

# ---------- 小工具 ----------

def sigmoid_focal_bce(logits, targets):
    # 和普通 BCE 一樣用 sigmoid，再加一點穩定
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    return tf.reduce_mean(loss)

def bbox_xywh_to_xyxy(xywh):
    x, y, w, h = tf.split(xywh, 4, axis=-1)
    half_w, half_h = w / 2.0, h / 2.0
    return tf.concat([x - half_w, y - half_h, x + half_w, y + half_h], axis=-1)

def bbox_iou_xyxy(a, b, eps=1e-7):
    # a: (N,4), b: (M,4) → IoU: (N,M)
    a = tf.expand_dims(a, 1)  # (N,1,4)
    b = tf.expand_dims(b, 0)  # (1,M,4)
    inter_x1 = tf.maximum(a[...,0], b[...,0])
    inter_y1 = tf.maximum(a[...,1], b[...,1])
    inter_x2 = tf.minimum(a[...,2], b[...,2])
    inter_y2 = tf.minimum(a[...,3], b[...,3])
    inter_w  = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h  = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter    = inter_w * inter_h

    area_a = (a[...,2]-a[...,0]) * (a[...,3]-a[...,1])
    area_b = (b[...,2]-b[...,0]) * (b[...,3]-b[...,1])
    union  = area_a + area_b - inter + eps
    return inter / union

def bbox_ciou_xyxy(pred, target, eps=1e-7):
    # pred/target: (P,4)
    px1, py1, px2, py2 = tf.split(pred, 4, axis=-1)
    tx1, ty1, tx2, ty2 = tf.split(target, 4, axis=-1)
    pw = tf.maximum(px2 - px1, eps)
    ph = tf.maximum(py2 - py1, eps)
    tw = tf.maximum(tx2 - tx1, eps)
    th = tf.maximum(ty2 - ty1, eps)

    # IoU
    inter_x1 = tf.maximum(px1, tx1)
    inter_y1 = tf.maximum(py1, ty1)
    inter_x2 = tf.minimum(px2, tx2)
    inter_y2 = tf.minimum(py2, ty2)
    inter_w  = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h  = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter    = inter_w * inter_h
    area_p   = pw * ph
    area_t   = tw * th
    union    = area_p + area_t - inter + eps
    iou      = inter / union

    # DIoU/CIoU 補償
    pcx = (px1 + px2) / 2.0; pcy = (py1 + py2) / 2.0
    tcx = (tx1 + tx2) / 2.0; tcy = (ty1 + ty2) / 2.0
    cw  = tf.maximum(px2, tx2) - tf.minimum(px1, tx1)
    ch  = tf.maximum(py2, ty2) - tf.minimum(py1, ty1)
    c2  = tf.maximum(cw, 0.0) ** 2 + tf.maximum(ch, 0.0) ** 2 + eps
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    v = (4 / (math.pi ** 2)) * tf.square(tf.atan(tw/th) - tf.atan(pw/ph))
    with tf.device('/CPU:0'):
        # 避免 fp16 underflow
        S = 1 - iou
        alpha = v / (S + v + eps)
    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou

def make_anchors(h, w, stride, img_size):
    # 回傳 normalized center (x,y) in [0,1]
    grid_y, grid_x = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')  # (H,W)
    cx = (tf.cast(grid_x, tf.float32) + 0.5) * stride / img_size
    cy = (tf.cast(grid_y, tf.float32) + 0.5) * stride / img_size
    # (H*W,2)
    centers = tf.stack([tf.reshape(cx, [-1]), tf.reshape(cy, [-1])], axis=-1)
    return centers  # normalized

def dfl_expectation(dfl_logits, reg_max):
    # dfl_logits: (..., 4*reg_max) ，先拆四邊，每邊 softmax 做期望值 → (...,4)
    # 輸出仍是「距離 bins」的期望（尚未乘 stride/img_size）
    four = []
    for i in range(4):
        x = dfl_logits[..., i*reg_max:(i+1)*reg_max]
        p = tf.nn.softmax(x, axis=-1)  # (..., reg_max)
        bins = tf.cast(tf.range(reg_max), tf.float32)  # 0..reg_max-1
        e = tf.reduce_sum(p * bins, axis=-1, keepdims=True)  # (...,1)
        four.append(e)
    return tf.concat(four, axis=-1)

def dfl_ce_loss(dfl_logits, target_d, reg_max):
    # target_d: (...,4) in [0, reg_max]（連續值）
    # 兩鄰點分配：floor/ceil + 線性權重
    loss = 0.0
    for i in range(4):
        logits = dfl_logits[..., i*reg_max:(i+1)*reg_max]  # (..., reg_max)
        t = target_d[..., i:i+1]                           # (..., 1)
        t0 = tf.floor(t)
        t1 = tf.minimum(t0 + 1.0, reg_max - 1.0)
        w1 = t - t0
        w0 = 1.0 - w1
        t0 = tf.cast(t0, tf.int32)
        t1 = tf.cast(t1, tf.int32)
        ce0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t0[...,0], logits=logits)
        ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t1[...,0], logits=logits)
        loss += tf.reduce_mean(w0[...,0] * ce0 + w1[...,0] * ce1)
    return loss / 4.0

# ---------- 主 Loss 類別 ----------

class v8PoseLossTF:
    """
    Simplified TF implementation aligned with Ultralytics v8 pose head.
    Inputs:
      - pred = (feats_list, kpts_list)
          feats_list: [ (B, no, H, W) * 3 ], no = nc + 4*reg_max
          kpts_list : [ (B, nk*kv, H, W) * 3 ], nk*kv = num_kpt * kpt_vals
      - batch:
          dict with keys:
            'batch_idx': (Ngt,1) int32
            'cls'      : (Ngt,1) int32
            'bboxes'   : (Ngt,4) float32 (xywh normalized)
            'keypoints': (Ngt, nk, kv) float32 (x,y normalized in [0,1], v in [0,1])
    """
    def __init__(self, strides=(8,16,32), nc=1, reg_max=16, kpt_shape=(17,3),
                 img_size=640, topk=10,
                 lambda_box=7.5, lambda_dfl=1.5, lambda_cls=0.5, lambda_kpts=4.0, lambda_kobj=1.0):
        self.strides = strides
        self.nc = nc
        self.reg_max = reg_max
        self.nk, self.kv = int(kpt_shape[0]), int(kpt_shape[1])
        self.img_size = float(img_size)
        self.topk = topk
        self.lb_box = lambda_box
        self.lb_dfl = lambda_dfl
        self.lb_cls = lambda_cls
        self.lb_kpts = lambda_kpts
        self.lb_kobj = lambda_kobj

    def _bchw_to_bnc(self, t):
        # (B,C,H,W) → (B, H*W, C)
        B, C, H, W = t.shape
        t = tf.transpose(t, [0,2,3,1])
        return tf.reshape(t, [B, H*W, C])

    def _decode_level(self, feats_bchw, kpts_bchw, stride):
        # feats_bchw: (B, no, H, W)
        # kpts_bchw : (B, nk*kv, H, W)
        no = self.nc + 4*self.reg_max
        B = tf.shape(feats_bchw)[0]
        H = tf.shape(feats_bchw)[2]
        W = tf.shape(feats_bchw)[3]

        feats_bnc = self._bchw_to_bnc(feats_bchw)  # (B, N, no)
        kpts_bnc  = self._bchw_to_bnc(kpts_bchw)   # (B, N, nk*kv)
        N = tf.shape(feats_bnc)[1]
        
        centers   = make_anchors(tf.shape(feats_bchw)[2], tf.shape(feats_bchw)[3],
                             stride=stride, img_size=self.img_size)  # (N,2); normalized
        centers   = tf.expand_dims(centers, 0)      # (1,N,2)

        if config.USE_DFL:
            # 原本 DFL 路徑：cls_logits + dfl_logits -> expectation -> ltrb -> xyxy
            cls_logits = feats_bnc[..., :self.nc]
            dfl_logits = feats_bnc[..., self.nc:self.nc+4*self.reg_max]  # (B,N,4*R)
            d_bins     = dfl_expectation(dfl_logits, self.reg_max)       # (B,N,4)
            dist_norm  = d_bins * (stride / self.img_size)
            l,t,r,b    = tf.split(dist_norm, 4, axis=-1)
            cx, cy     = centers[...,0:1], centers[...,1:2]
            x1 = cx - l; y1 = cy - t; x2 = cx + r; y2 = cy + b
            pred_xyxy  = tf.concat([x1,y1,x2,y2], axis=-1)
        else:
            # ★ 非 DFL 路徑：no = nc + 4，取出 4 個框，走 sigmoid 當 normalized xywh
            cls_logits = feats_bnc[..., :self.nc]          # (B,N,nc)
            box_xywh   = feats_bnc[..., self.nc:self.nc+4]  # (B,N,4) in [0,1]

            box_xywh = tf.nn.relu(box_xywh)                  # (B,N,4)  非負距離

            # 轉 xyxy
            x, y, w, h = tf.split(box_xywh, 4, axis=-1)
            cx, cy = centers[...,0:1], centers[...,1:2]
            x1 = cx - x; y1 = cy - y; x2 = cx + w; y2 = cy + h
            pred_xyxy = tf.concat([x1,y1,x2,y2], axis=-1)
            dfl_logits = None   # 明確空值

        k = tf.reshape(kpts_bnc, [B, N, self.nk, self.kv])      # (B,N,nk,kv)
        kxy = k[...,:2] * 2.0                         # 和 Ultralytics 一樣的 scale
        kxy = tf.stack([
            kxy[...,0] + centers[...,0:1] - 0.5,
            kxy[...,1] + centers[...,1:2] - 0.5
        ], axis=-1)                                   # (B,N,nk,2)
        if self.kv > 2:
            kvis_logits = k[...,2:3]                  # logits；BCE 用 logits
            kpts = tf.concat([kxy, kvis_logits], axis=-1)
        else:
            kpts = kxy

        return cls_logits, dfl_logits, pred_xyxy, kpts  # (B,N,nc), (B,N,4*R), (B,N,4), (B,N,nk,kv)

    def _gather_by_index(self, x, idx):
        # x: (B,N,...)  idx: (P,2) [img_idx, point_idx]
        # 回傳 (P, ...)
        b = idx[:,0]
        n = idx[:,1]
        return tf.gather_nd(x, tf.stack([b, n], axis=1))

    def __call__(self, pred, batch):
        feats_list, kpts_list = pred  # list len=3
        assert len(feats_list) == len(kpts_list) == len(self.strides)

        # 逐層 decode
        all_cls, all_dfl, all_xyxy, all_kpts, all_img_idx, all_loc_idx, all_stride = [],[],[],[],[],[],[]
        for li, (feats, kpts, s) in enumerate(zip(feats_list, kpts_list, self.strides)):
            cls_logits, dfl_logits, xyxy, kpt = self._decode_level(feats, kpts, s)  # (B,N,...)
            B = tf.shape(xyxy)[0]; N = tf.shape(xyxy)[1]
            img_idx = tf.reshape(tf.repeat(tf.range(B), repeats=N), [-1,1])  # (B*N,1)
            loc_idx = tf.reshape(tf.tile(tf.range(N), [B]), [-1,1])          # (B*N,1)
            stridev = tf.reshape(tf.fill([B*N,1], float(s)), [-1,1])

            all_cls.append(tf.reshape(cls_logits, [B*N, self.nc]))
            if config.USE_DFL and dfl_logits is not None:
                all_dfl.append(tf.reshape(dfl_logits, [B*N, 4*self.reg_max]))
            all_xyxy.append(tf.reshape(xyxy,     [B*N, 4]))
            all_kpts.append(tf.reshape(kpt,      [B*N, self.nk, self.kv]))
            all_img_idx.append(img_idx)
            all_loc_idx.append(loc_idx)
            all_stride.append(stridev)

        cls_logits = tf.concat(all_cls, axis=0)         # (P, nc)

        if config.USE_DFL and len(all_dfl) > 0:
            dfl_logits = tf.concat(all_dfl, axis=0) # (P, 4*reg_max)
        else:
            dfl_logits = None

        pred_xyxy  = tf.concat(all_xyxy, axis=0)        # (P, 4)
        pred_kpts  = tf.concat(all_kpts, axis=0)        # (P, nk, kv)
        img_idx    = tf.concat(all_img_idx, axis=0)     # (P,1)
        loc_idx    = tf.concat(all_loc_idx, axis=0)     # (P,1)

        # 讀 targets
        gt_img  = tf.cast(batch['batch_idx'][:,0], tf.int32)  # (G,)
        gt_cls  = tf.cast(batch['cls'][:,0], tf.int32)
        gt_xywh = batch['bboxes']
        gt_xyxy = bbox_xywh_to_xyxy(gt_xywh)
        gt_kpts = batch['keypoints']

        # 從「預測」推 batch 數，避免空 GT 時 reduce_max 爆掉
        # img_idx 形狀是 (P,1)，前面你已經組好了
        img_idx_flat = tf.squeeze(img_idx, axis=1)                    # (P,)
        B_total = tf.cast(tf.reduce_max(img_idx_flat) + 1, tf.int32)  # ✅ 安全

        pos_indices = []

        # 若真的連預測都沒有（理論上不會），或 B_total<=0，直接回零 loss
        cond_no_pred = tf.logical_or(tf.equal(tf.size(img_idx_flat), 0),
                                    tf.less_equal(B_total, 0))
        if cond_no_pred:
            zero = tf.constant(0., tf.float32)
            return zero, {'lbox': zero, 'lcls': zero, 'ldfl': zero,
                        'lkpts': zero, 'lkobj': zero, 'npos': tf.constant(0.)}

        # ------- assigner：每張圖分別，把該圖的 pred 與該圖的 gt 做 IoU，選 top-k -------
        for b in tf.range(B_total):                                   # ✅ 用 int32 上限
            p_mask = tf.where(tf.equal(img_idx_flat, b))[:, 0]        # indices for this image
            # 這張圖若沒有 GT，直接跳過，不會報錯
            g_mask = tf.where(tf.equal(gt_img, b))[:, 0]

            if tf.size(p_mask) == 0 or tf.size(g_mask) == 0:
                continue

            p_boxes = tf.gather(pred_xyxy, p_mask)
            g_boxes = tf.gather(gt_xyxy, g_mask)
            iou = bbox_iou_xyxy(p_boxes, g_boxes)         # (P_b, G_b)

            topk = tf.minimum(self.topk, tf.shape(p_boxes)[0])
            iou_T = tf.transpose(iou, [1, 0])             # (G_b, P_b)
            _, top_idx = tf.math.top_k(iou_T, k=topk)     # (G_b, topk)

            p_sel = tf.gather(p_mask, tf.reshape(top_idx, [-1]))
            g_sel = tf.repeat(g_mask, repeats=topk)
            pos_indices.append(tf.stack([p_sel, g_sel], axis=1))  # (G_b*topk, 2)


        if len(pos_indices) == 0:
            zero = tf.constant(0., tf.float32)
            return zero, {'lbox': zero, 'lcls': zero, 'ldfl': zero,
                        'lkpts': zero, 'lkobj': zero, 'npos': tf.constant(0.)}
        pos_indices = tf.concat(pos_indices, axis=0)

        # ------- 收集 positive 上的 pred / gt -------
        p_sel = pos_indices[:,0]
        g_sel = pos_indices[:,1]
        pred_boxes_pos = tf.gather(pred_xyxy, p_sel)         # (P_pos,4)
        pred_cls_pos   = tf.gather(cls_logits, p_sel)        # (P_pos,nc)
        if config.USE_DFL and dfl_logits is not None:
            pred_dfl_pos = tf.gather(dfl_logits, p_sel)
        else:
            pred_dfl_pos = None
        pred_kpts_pos  = tf.gather(pred_kpts, p_sel)         # (P_pos,nk,kv)

        gt_boxes_pos   = tf.gather(gt_xyxy, g_sel)           # (P_pos,4)
        gt_cls_pos     = tf.gather(gt_cls, g_sel)            # (P_pos,)
        gt_kpts_pos    = tf.gather(gt_kpts, g_sel)           # (P_pos,nk,kv)

        npos = tf.cast(tf.shape(p_sel)[0], tf.float32) + 1e-9

        # ------- 各項損失 -------
        # 1) box ciou
        ciou = bbox_ciou_xyxy(pred_boxes_pos, gt_boxes_pos)  # (P_pos,1)
        l_box = tf.reduce_sum(1.0 - ciou) / npos

        # 2) cls BCE
        gt_onehot = tf.one_hot(gt_cls_pos, depth=self.nc, dtype=tf.float32)
        l_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_onehot, logits=pred_cls_pos)) / npos

        # 3) dfl CE（為了做 target_d，我們把 gt boxes 轉回 ltrb 相對 pred center）
        #   先還原 pred center（注意：assigner用的是 pred box，這裡用 pred box center 當近似）
        px1, py1, px2, py2 = tf.split(pred_boxes_pos, 4, axis=-1)
        pcx = (px1 + px2)/2.0; pcy = (py1 + py2)/2.0
        gx1, gy1, gx2, gy2 = tf.split(gt_boxes_pos, 4, axis=-1)
        gl = tf.maximum(pcx - gx1, 0.0)
        gt = tf.maximum(pcy - gy1, 0.0)
        gr = tf.maximum(gx2 - pcx, 0.0)
        gb = tf.maximum(gy2 - pcy, 0.0)
        g_ltrb_norm = tf.concat([gl,gt,gr,gb], axis=-1)         # normalized in [0,1]
        # 轉為「bins」標籤（除以 stride/img_size 等效乘 img_size/stride）
        # 我們沒有每個點的 stride（不同層），簡化：用「就近的 stride」估計。
        # 更精準可在 pos_indices 時一併攜帶 stride；這裡先取 img_size/平均stride 作近似。
        avg_s = tf.reduce_mean(tf.cast(self.strides, tf.float32))
        target_d = g_ltrb_norm * (self.img_size / avg_s)
        target_d = tf.clip_by_value(target_d, 0.0, float(self.reg_max - 1e-3))
        if config.USE_DFL:
            # 用每個 positive 的 stride 做 bins 會更準；你先前可用平均 stride 簡化
            l_dfl = dfl_ce_loss(pred_dfl_pos, target_d, self.reg_max)
        else:
            l_dfl = tf.constant(0.0, tf.float32)

        # 4) keypoints L1 + kobj BCE
        #   只在 visible>0.5 上計算 (x,y) L1
        if self.kv >= 3:
            gt_vis = gt_kpts_pos[...,2:3]                      # (P_pos,nk,1)
        else:
            gt_vis = tf.ones_like(gt_kpts_pos[...,:1])         # 全可見
        vis_mask = tf.cast(gt_vis > 0.5, tf.float32)           # (P_pos,nk,1)
        # L1 on (x,y)
        kp_l1 = tf.abs(pred_kpts_pos[...,:2] - gt_kpts_pos[...,:2]) * vis_mask
        l_kpts = tf.reduce_sum(kp_l1) / (tf.reduce_sum(vis_mask) + 1e-9)
        # kobj BCE on visibility (若 kv<3，這項設為 0)
        if self.kv >= 3:
            pred_vis_logit = pred_kpts_pos[..., 2:3]

            # pred_vis_logit = tf.math.log(pred_kpts_pos[...,2:3] + 1e-7) - tf.math.log(1.0 - pred_kpts_pos[...,2:3] + 1e-7)
            l_kobj = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_vis, logits=pred_vis_logit))
        else:
            l_kobj = tf.constant(0.0, tf.float32)

        total = (self.lb_box * l_box +
                 (self.lb_dfl * l_dfl if config.USE_DFL else 0.0) +
                 self.lb_cls * l_cls +
                 self.lb_kpts * l_kpts +
                 self.lb_kobj * l_kobj)

        logs = {'lbox': l_box, 'lcls': l_cls, 'ldfl': l_dfl, 'lkpts': l_kpts, 'lkobj': l_kobj, 'npos': npos}
        return total, logs

