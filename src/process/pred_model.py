
import config
import tensorflow as tf

from itertools import permutations


'''
預測 Teacher 模型結果進行比較
'''

def ensure_BNC_static(y, expected_C: int):
    y = tf.convert_to_tensor(y)
    tf.debugging.assert_rank(y, 3, message="expect rank-3 (B,?,?)")
    s = y.shape  # 靜態 shape

    # 情況 A：已是 (B, N, C)
    if s.rank == 3 and s[-1] == expected_C:
        return y

    # 情況 B：是 (B, C, N) -> 轉成 (B, N, C)
    if s.rank == 3 and s[1] == expected_C:
        return tf.transpose(y, [0, 2, 1])

    # 情況 C：靜態資訊不足，直接報錯，避免圖中殘留 tf.cond
    raise ValueError(f"ensure_BNC_static: cannot infer layout from shape {s}, expected_C={expected_C}. "
                     f"Expected either (B,N,{expected_C}) or (B,{expected_C},N).")

def align_student_to_domain(y_s_raw, num_cls, num_kpt, kpt_vals,
                            batch_imgs,  # 取 W,H
                            target_domain_is_pixel=False):
    """
    y_s_raw: (B,N,C) KD 頭的原始輸出（logits for box/kxy/cls/ksc）
    回傳：
      s_box, s_cls_logit, s_kxy, s_ksc_logit  # 其中 box/kxy 已轉到目標數域
    """
    H = tf.shape(batch_imgs)[1]
    W = tf.shape(batch_imgs)[2]

    s_box_logit, s_cls_logit, s_kxy_logit, s_ksc_logit = split_BNC(
        y_s_raw, num_cls, num_kpt, kpt_vals
    )

    # 將 box/kxy logits -> 0~1
    s_box_u = tf.nn.sigmoid(s_box_logit)  # 0~1
    s_kxy_u = tf.nn.sigmoid(s_kxy_logit)  # 0~1

    # 再依需求轉到像素
    def _to_px():
        return (box_unit_to_pixel(s_box_u, W, H),
                kxy_unit_to_pixel(s_kxy_u, W, H))
    def _stay_unit():
        return (s_box_u, s_kxy_u)

    s_box, s_kxy = tf.cond(
        tf.convert_to_tensor(target_domain_is_pixel),
        _to_px, _stay_unit
    )

    return s_box, s_cls_logit, s_kxy, s_ksc_logit



def split_BNC(y, num_cls, num_kpt, kpt_vals):
    # y: (B, N, 4 + num_cls + num_kpt*kpt_vals)
    box = y[..., 0:4]                         # (B,N,4)  內容是 xywh
    cls = y[..., 4:4+num_cls]                 # (B,N,C)
    kpt = y[..., 4+num_cls:]                  # (B,N,K*V)
    kpt = tf.reshape(kpt, [tf.shape(y)[0], tf.shape(y)[1], num_kpt, kpt_vals])  # (B,N,K,V)
    kxy = kpt[..., :2]                        # (B,N,K,2)
    ksc = kpt[..., 2:3] if kpt_vals >= 3 else None   # (B,N,K,1) or None
    return box, cls, kxy, ksc

def pack_BNC(box, cls, kxy, ksc):
    # box: (B,N,4), cls: (B,N,C), kxy: (B,N,K,2), ksc: (B,N,K,1 or None)
    if ksc is not None:
        kpt = tf.concat([kxy, ksc], axis=-1)  # (B,N,K,3)
    else:
        kpt = kxy                             # (B,N,K,2)
    B = tf.shape(box)[0]
    N = tf.shape(box)[1]
    K = tf.shape(kpt)[2]
    V = tf.shape(kpt)[3]
    kpt_flat = tf.reshape(kpt, [B, N, K*V])   # (B,N,K*V)
    return tf.concat([box, cls, kpt_flat], axis=-1)  # (B,N,4+C+K*V)

# ---- Pixel <-> Unit (0-1) ----
def box_pixel_to_unit(box_px, W, H):
    # xywh in pixels -> [0,1]
    scale = tf.stack([W, H, W, H], axis=0)         # (4,)
    return box_px / tf.cast(scale, box_px.dtype)

def box_unit_to_pixel(box_u, W, H):
    scale = tf.stack([W, H, W, H], axis=0)
    return box_u * tf.cast(scale, box_u.dtype)

def kxy_pixel_to_unit(kxy_px, W, H):
    scale = tf.reshape(tf.stack([W, H], axis=0), [1,1,1,2])  # broadcast
    return kxy_px / tf.cast(scale, kxy_px.dtype)

def kxy_unit_to_pixel(kxy_u, W, H):
    scale = tf.reshape(tf.stack([W, H], axis=0), [1,1,1,2])
    return kxy_u * tf.cast(scale, kxy_u.dtype)

# ---- 數域偵測（TF 友善；避免 strings，回傳 tf.bool）----
def detect_is_pixel_domain(box, kxy, thr=0.01, eps=1e-6):
    # 超過 [0,1] 視為像素域；抓「有多少比例的元素越界」
    box_out = tf.reduce_mean(tf.cast(tf.logical_or(box < -eps, box > 1.0+eps), tf.float32))
    kxy_out = tf.reduce_mean(tf.cast(tf.logical_or(kxy < -eps, kxy > 1.0+eps), tf.float32))
    frac_out = tf.maximum(box_out, kxy_out)  # 單一 scalar
    return frac_out > thr  # tf.bool

def normalize_teacher_pred(y, expected_C, num_cls, num_kpt, kpt_vals,
                           batch_imgs,  # (B,H,W,C) 用來取得 W,H
                           target_domain='unit',  # 'unit' or 'pixel' or 'auto'
                           return_detected=False):
    """
    回傳：
      y_out: (B,N,C) 已轉成 target_domain 的數域
      is_pixel_detected: tf.bool（可選）
    """
    y = ensure_BNC_static(y, expected_C)
    box, cls, kxy, ksc = split_BNC(y, num_cls, num_kpt, kpt_vals)

    # 取得當前 batch 的 W, H
    H = tf.shape(batch_imgs)[1]
    W = tf.shape(batch_imgs)[2]

    # 偵測 teacher 原始數域
    is_pixel = detect_is_pixel_domain(box, kxy)  # tf.bool

    # 決定要輸出的目標數域
    if target_domain == 'auto':
        # 若偵測為像素域，就輸出像素；否則輸出 0-1
        out_is_pixel = is_pixel
    elif target_domain == 'pixel':
        out_is_pixel = tf.constant(True)
    else:  # 'unit'
        out_is_pixel = tf.constant(False)

    def _to_unit():
        b = tf.identity(box)
        k = tf.identity(kxy)
        b = tf.cond(is_pixel, lambda: box_pixel_to_unit(b, W, H), lambda: b)
        k = tf.cond(is_pixel, lambda: kxy_pixel_to_unit(k, W, H), lambda: k)
        return pack_BNC(b, cls, k, ksc)

    def _to_pixel():
        b = tf.identity(box)
        k = tf.identity(kxy)
        b = tf.cond(is_pixel, lambda: b, lambda: box_unit_to_pixel(b, W, H))
        k = tf.cond(is_pixel, lambda: k, lambda: kxy_unit_to_pixel(k, W, H))
        return pack_BNC(b, cls, k, ksc)

    y_out = tf.cond(out_is_pixel, _to_pixel, _to_unit)

    if return_detected:
        return y_out, is_pixel
    else:
        return y_out
    

def choose_student_split_order(student_infer, teacher, sample_one,
                               N3, N4, N5, expected_C,
                               num_cls, num_kpt, kpt_vals,
                               target_domain='unit'):   # 'unit' or 'pixel'
    """
    自動偵測學生模型 P3,P4,P5 輸出塊的最佳排列順序。
    會先把 Teacher/Student 都映到相同數域，再比較 MAE。
    """
    print("\n--- Aligning Student/Teacher Output Order ---")

    # --- Teacher / Student
    y_te = teacher(sample_one, training=False)
    y_st = student_infer(sample_one, training=False)

    if isinstance(y_st, (list, tuple)):
        y_st = y_st[1]
    if y_st.shape.rank != 3:
        raise ValueError(f"[EXPORT] Student single output must be rank-3, got {y_st.shape}")

    # 先確保 (B,N,C) 排列
    y_te = ensure_BNC_static(y_te, expected_C)
    y_st = ensure_BNC_static(y_st, expected_C)

    t_box, t_cls, t_kxy, t_ksc = split_BNC(y_te, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
    s_box, s_cls, s_kxy, s_ksc = split_BNC(y_st, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)

    s_full = pack_BNC(s_box, s_cls, s_kxy, s_ksc if s_ksc is not None else None)
    y_te_prob = pack_BNC(t_box, t_cls, t_kxy, t_ksc if t_ksc is not None else None)

    # 依 lens 做順序搜尋
    lens = [N3, N4, N5]
    teacher_orders = {"forward": [N3, N4, N5], "reverse": [N5, N4, N3]}
    y_te_orders = {
        "forward": y_te_prob,
        "reverse": tf.concat(
            [y_te_prob[:, N3+N4:, :], y_te_prob[:, N3:N3+N4, :], y_te_prob[:, :N3, :]],
            axis=1
        ),
    }

    # 若學生 (B,N,C) 的 N 次序不明，窮舉排列名稱
    best_perm, best_order, best_mae = None, None, float("inf")
    for perm in permutations(lens, 3):
        s0, s1, s2 = perm
        split_student = tf.split(s_full, [s0, s1, s2], axis=1)
        for name, to_order in teacher_orders.items():
            reorder_map = {s0: split_student[0], s1: split_student[1], s2: split_student[2]}
            y_st_aligned = tf.concat([reorder_map[l] for l in to_order], axis=1)
            mae = tf.reduce_mean(tf.abs(y_st_aligned - y_te_orders[name])).numpy()
            if mae < best_mae:
                best_mae, best_perm, best_order = mae, perm, name

    if best_perm is None:
        raise RuntimeError("[ALIGN] failed to decide student N-order.")

    split_index_by_len = {best_perm[0]: 0, best_perm[1]: 1, best_perm[2]: 2}
    reorder_idx = [split_index_by_len[l] for l in teacher_orders[best_order]]
    print(f"✅ Alignment complete: lens_perm={best_perm}, teacher_order={best_order}, MAE={best_mae:.6e}")
    return best_perm, reorder_idx