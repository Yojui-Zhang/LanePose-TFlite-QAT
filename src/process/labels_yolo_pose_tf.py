
"""
Utility to parse YOLOv8-pose style label .txt lines into arrays.
Each line:
cls cx cy w h [x1 y1 v1 ... xK yK vK]
All coordinates normalized to 0..1 relative to image W,H.
"""
import numpy as np
import tensorflow as tf

def parse_label_lines(lines, num_kpt:int, kpt_vals:int=3):
    arr = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        vals = [float(x) for x in parts]
        cls_id = int(vals[0])
        # pad or trim kpts to expected length
        needed = 5 + num_kpt*kpt_vals
        if len(vals) < needed:
            vals = vals + [0.0]*(needed - len(vals))
        elif len(vals) > needed:
            vals = vals[:needed]
        vals[0] = float(cls_id)
        arr.append(vals)
    if not arr:
        return np.zeros((0, 5 + num_kpt*kpt_vals), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)

def letterbox_adjust_yolo_pose(arr: np.ndarray,
                               orig_h: float, orig_w: float,
                               scale: float, pad_x: float, pad_y: float,
                               new_size: int,
                               num_kpt: int, kpt_vals: int) -> np.ndarray:
    """
    arr: (N, 5 + K*kpt_vals) YOLO normalized to original W/H
    return: YOLO normalized to letterboxed (new_size,new_size)
    """
    if arr is None or arr.size == 0:
        return arr

    out = arr.astype(np.float32, copy=True)

    # --- bbox: (cx,cy,w,h) normalized -> pixel(orig) -> letterbox pixel -> normalized(new_size) ---
    cx = out[:, 1] * orig_w
    cy = out[:, 2] * orig_h
    bw = out[:, 3] * orig_w
    bh = out[:, 4] * orig_h

    cx = cx * scale + pad_x
    cy = cy * scale + pad_y
    bw = bw * scale
    bh = bh * scale

    out[:, 1] = cx / new_size
    out[:, 2] = cy / new_size
    out[:, 3] = bw / new_size
    out[:, 4] = bh / new_size

    # --- keypoints: (x,y,v...) normalized -> pixel(orig) -> letterbox -> normalized ---
    if num_kpt > 0 and kpt_vals >= 2:
        k = out[:, 5:].reshape(-1, num_kpt, kpt_vals)

        kx = k[:, :, 0] * orig_w
        ky = k[:, :, 1] * orig_h

        kx = kx * scale + pad_x
        ky = ky * scale + pad_y

        if kpt_vals >= 3:
            v = k[:, :, 2]
            valid = (v > 0.0)
        else:
            valid = np.ones_like(kx, dtype=bool)

        k[:, :, 0] = np.where(valid, kx / new_size, 0.0)
        k[:, :, 1] = np.where(valid, ky / new_size, 0.0)

        out[:, 5:] = k.reshape(-1, num_kpt * kpt_vals)

    # 安全裁切到 [0,1]
    out[:, 1:5] = np.clip(out[:, 1:5], 0.0, 1.0)
    if num_kpt > 0 and kpt_vals >= 2:
        k2 = out[:, 5:].reshape(-1, num_kpt, kpt_vals)
        k2[:, :, 0] = np.clip(k2[:, :, 0], 0.0, 1.0)
        k2[:, :, 1] = np.clip(k2[:, :, 1], 0.0, 1.0)
        out[:, 5:] = k2.reshape(-1, num_kpt * kpt_vals)

    return out



@tf.function
def parse_label_text_tf(text: tf.Tensor, num_kpt: int, kpt_vals: int = 3) -> tf.Tensor:
    """
    YOLO pose label txt (whole file) -> float32 tensor [N, D]
    line format: cls cx cy w h [kpt...]
    all normalized in original image space
    """
    num_kpt_t  = tf.cast(num_kpt, tf.int32)
    kpt_vals_t = tf.cast(kpt_vals, tf.int32)
    needed = 5 + num_kpt_t * kpt_vals_t  # D

    text = tf.strings.strip(text)

    def _empty():
        out = tf.zeros([0, needed], dtype=tf.float32)
        out.set_shape([0, None])
        out = tf.ensure_shape(out, [0, None])
        return out

    def _nonempty():
        # split lines
        lines = tf.strings.split(text, sep="\n")  # RaggedTensor [N]
        lines = tf.ragged.boolean_mask(lines, tf.strings.length(lines) > 0)
        # split tokens
        toks = tf.strings.split(lines, sep=" ")   # RaggedTensor [N, (var)]
        toks_dense = toks.to_tensor(default_value="0")  # [N, maxlen]

        # ensure width == needed (pad or trim)
        toks_dense = toks_dense[:, :needed]
        cur = tf.shape(toks_dense)[1]
        pad = tf.maximum(needed - cur, 0)
        toks_dense = tf.pad(toks_dense, [[0, 0], [0, pad]], constant_values="0")

        vals = tf.strings.to_number(toks_dense, out_type=tf.float32)  # [N, needed]
        # force cls to int then back to float (match your numpy behavior)
        cls = tf.cast(tf.cast(vals[:, 0], tf.int32), tf.float32)
        vals = tf.concat([cls[:, None], vals[:, 1:]], axis=1)

        vals.set_shape([None, None])
        vals = tf.ensure_shape(vals, [None, None])
        return vals

    out = tf.cond(tf.equal(tf.strings.length(text), 0), _empty, _nonempty)
    # finalize static last-dim
    out = tf.ensure_shape(out, [None, None])
    return out


@tf.function
def letterbox_adjust_yolo_pose_tf(
    labels: tf.Tensor,
    meta: tf.Tensor,
    new_size: int,
    num_kpt: int,
    kpt_vals: int = 3
) -> tf.Tensor:
    """
    Map labels from original normalized coords -> letterbox normalized coords
    meta = [orig_h, orig_w, scale, pad_x, pad_y]
    """
    labels = tf.cast(labels, tf.float32)
    meta   = tf.cast(meta, tf.float32)
    new_s  = tf.cast(new_size, tf.float32)

    orig_h, orig_w, scale, pad_x, pad_y = tf.unstack(meta, num=5)

    # guard empty
    n = tf.shape(labels)[0]
    d = tf.shape(labels)[1]

    def _empty():
        out = tf.zeros([0, d], tf.float32)
        out.set_shape([0, None])
        return out

    def _nonempty():
        # bbox
        cx = labels[:, 1] * orig_w
        cy = labels[:, 2] * orig_h
        bw = labels[:, 3] * orig_w
        bh = labels[:, 4] * orig_h

        cx = cx * scale + pad_x
        cy = cy * scale + pad_y
        bw = bw * scale
        bh = bh * scale

        bbox = tf.stack([cx / new_s, cy / new_s, bw / new_s, bh / new_s], axis=-1)
        out = tf.concat([labels[:, :1], bbox, labels[:, 5:]], axis=1)

        # keypoints
        if num_kpt > 0:
            nk = tf.cast(num_kpt, tf.int32)
            kv = tf.cast(kpt_vals, tf.int32)

            k = out[:, 5:]
            k = tf.reshape(k, [-1, nk, kv])  # [N, K, V]

            kx = k[:, :, 0] * orig_w
            ky = k[:, :, 1] * orig_h

            kx = kx * scale + pad_x
            ky = ky * scale + pad_y

            if kpt_vals >= 3:
                v = k[:, :, 2]
                valid = v > 0
            else:
                v = None
                valid = tf.ones_like(kx, dtype=tf.bool)

            kx_n = tf.where(valid, kx / new_s, tf.zeros_like(kx))
            ky_n = tf.where(valid, ky / new_s, tf.zeros_like(ky))

            kx_e = kx_n[..., None]
            ky_e = ky_n[..., None]

            if kpt_vals == 2:
                k_new = tf.concat([kx_e, ky_e], axis=-1)
            elif kpt_vals == 3:
                k_new = tf.concat([kx_e, ky_e, k[:, :, 2:3]], axis=-1)
            else:
                # keep extra dims >=3
                k_new = tf.concat([kx_e, ky_e, k[:, :, 2:3], k[:, :, 3:]], axis=-1)

            out = tf.concat([out[:, :5], tf.reshape(k_new, [-1, nk * kv])], axis=1)

        # clip to [0,1]
        out_bbox = tf.clip_by_value(out[:, 1:5], 0.0, 1.0)
        out = tf.concat([out[:, :1], out_bbox, out[:, 5:]], axis=1)

        if num_kpt > 0 and kpt_vals >= 2:
            nk = tf.cast(num_kpt, tf.int32)
            kv = tf.cast(kpt_vals, tf.int32)
            k2 = tf.reshape(out[:, 5:], [-1, nk, kv])
            kx2 = tf.clip_by_value(k2[:, :, 0], 0.0, 1.0)
            ky2 = tf.clip_by_value(k2[:, :, 1], 0.0, 1.0)
            if kpt_vals == 2:
                k2n = tf.concat([kx2[..., None], ky2[..., None]], axis=-1)
            else:
                k2n = tf.concat([kx2[..., None], ky2[..., None], k2[:, :, 2:]], axis=-1)
            out = tf.concat([out[:, :5], tf.reshape(k2n, [-1, nk * kv])], axis=1)

        return out

    return tf.cond(tf.equal(n, 0), _empty, _nonempty)


@tf.function
def parse_label_file_tf(
    lbl_path: tf.Tensor,
    meta: tf.Tensor,
    new_size: int,
    num_kpt: int,
    kpt_vals: int = 3
) -> tf.Tensor:
    """
    read_file(lbl_path) -> parse -> letterbox adjust
    """
    txt = tf.io.read_file(lbl_path)
    labels = parse_label_text_tf(txt, num_kpt=num_kpt, kpt_vals=kpt_vals)
    labels = letterbox_adjust_yolo_pose_tf(
        labels, meta,
        new_size=new_size, num_kpt=num_kpt, kpt_vals=kpt_vals
    )
    # set static last dim if possible
    needed = 5 + int(num_kpt) * int(kpt_vals)
    labels = tf.ensure_shape(labels, [None, needed])
    return labels
