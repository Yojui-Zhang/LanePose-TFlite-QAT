import tensorflow as tf
import numpy as np
import re

# ==============================================================================
# Numpy Parsing Logic (For Class Weights / Offline processing)
# ==============================================================================
def parse_label_lines_numpy(lines, num_kpt: int, kpt_vals: int = 3):
    """
    解析 YOLO Pose 格式的文字行 (Numpy 版本)。
    Format: cls cx cy w h [x1 y1 v1 ... xK yK vK]
    """
    arr = []
    needed = 5 + num_kpt * kpt_vals
    
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        
        parts = ln.split()
        vals = [float(x) for x in parts]
        
        # Pad or Trim
        if len(vals) < needed:
            vals = vals + [0.0] * (needed - len(vals))
        elif len(vals) > needed:
            vals = vals[:needed]
            
        arr.append(vals)
        
    if not arr:
        return np.zeros((0, needed), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)

# ==============================================================================
# TensorFlow Parsing Logic (For tf.data Pipeline)
# ==============================================================================

def img_path_to_label_path_tf(img_path: tf.Tensor) -> tf.Tensor:
    """將圖片路徑轉換為標籤路徑 (/images/ -> /labels/, .jpg -> .txt)"""
    lbl_path = tf.strings.regex_replace(img_path, r"/images/", "/labels/")
    # 替換副檔名為 .txt
    lbl_path = tf.strings.regex_replace(lbl_path, r"\.[^/.]+$", ".txt")
    return lbl_path

@tf.function
def letterbox_adjust_labels_tf(labels, meta, new_size, num_kpt, kpt_vals=3):
    """
    將 Normalized 座標 (原圖) 轉換為 Letterbox 後的 Normalized 座標。
    meta: [orig_h, orig_w, scale, pad_x, pad_y]
    """
    labels = tf.cast(labels, tf.float32)
    meta = tf.cast(meta, tf.float32)
    new_s = tf.cast(new_size, tf.float32)
    
    orig_h, orig_w, scale, pad_x, pad_y = tf.unstack(meta, num=5)
    
    # Check Empty
    n = tf.shape(labels)[0]
    
    def _empty():
        d = tf.shape(labels)[1]
        out = tf.zeros([0, d], tf.float32)
        out.set_shape([0, None])
        return out

    def _nonempty():
        # 1. 還原回原圖 Pixel 座標
        cx = labels[:, 1] * orig_w
        cy = labels[:, 2] * orig_h
        bw = labels[:, 3] * orig_w
        bh = labels[:, 4] * orig_h

        # 2. 應用 Letterbox 變換 (Scale + Pad)
        cx = cx * scale + pad_x
        cy = cy * scale + pad_y
        bw = bw * scale
        bh = bh * scale

        # 3. Normalize 到新尺寸 (new_size)
        bbox = tf.stack([cx / new_s, cy / new_s, bw / new_s, bh / new_s], axis=-1)
        
        # 組合 BBox 部分
        out_base = tf.concat([labels[:, :1], bbox], axis=1)

        # 4. 處理 Keypoints
        if num_kpt > 0:
            k = labels[:, 5:]
            nk = tf.cast(num_kpt, tf.int32)
            kv = tf.cast(kpt_vals, tf.int32)
            
            k = tf.reshape(k, [-1, nk, kv]) # [N, K, V]
            
            kx = k[:, :, 0] * orig_w
            ky = k[:, :, 1] * orig_h
            
            kx = kx * scale + pad_x
            ky = ky * scale + pad_y
            
            # Visibility mask logic
            if kpt_vals >= 3:
                v = k[:, :, 2]
                valid = v > 0
            else:
                valid = tf.ones_like(kx, dtype=tf.bool)
                
            # 若不可見設為 0，可見則 Normalize
            kx_n = tf.where(valid, kx / new_s, tf.zeros_like(kx))
            ky_n = tf.where(valid, ky / new_s, tf.zeros_like(ky))
            
            # 重新組合 KPT
            k_parts = [kx_n[..., None], ky_n[..., None]]
            if kpt_vals >= 3:
                k_parts.append(k[:, :, 2:]) # 保留 V
                
            k_new = tf.concat(k_parts, axis=-1)
            out_kpt = tf.reshape(k_new, [-1, nk * kv])
            
            out_full = tf.concat([out_base, out_kpt], axis=1)
        else:
            out_full = out_base
            
        # Clip BBox to [0, 1]
        final_box = tf.clip_by_value(out_full[:, 1:5], 0.0, 1.0)
        final_out = tf.concat([out_full[:, :1], final_box, out_full[:, 5:]], axis=1)
        
        return final_out

    return tf.cond(tf.equal(n, 0), _empty, _nonempty)

@tf.function
def parse_label_text_tf(text, num_kpt, kpt_vals=3):
    """
    將 TXT 內容 (String Tensor) 解析為 Float Tensor。
    """
    text = tf.strings.strip(text)
    needed = 5 + num_kpt * kpt_vals

    def _empty():
        out = tf.zeros([0, needed], dtype=tf.float32)
        out.set_shape([0, None])
        return tf.ensure_shape(out, [0, None])

    def _nonempty():
        # Split lines
        lines = tf.strings.split(text, sep="\n")
        lines = tf.ragged.boolean_mask(lines, tf.strings.length(lines) > 0)
        
        # Split tokens
        toks = tf.strings.split(lines, sep=" ")
        toks_dense = toks.to_tensor(default_value="0")
        
        # Ensure correct width
        toks_dense = toks_dense[:, :needed]
        cur_w = tf.shape(toks_dense)[1]
        pad_w = tf.maximum(needed - cur_w, 0)
        toks_dense = tf.pad(toks_dense, [[0, 0], [0, pad_w]], constant_values="0")
        
        vals = tf.strings.to_number(toks_dense, out_type=tf.float32)
        
        # Ensure Class ID is integer-like (floor) but keep float dtype
        cls_id = tf.math.floor(vals[:, 0])
        vals = tf.concat([cls_id[:, None], vals[:, 1:]], axis=1)
        
        return tf.ensure_shape(vals, [None, None])

    return tf.cond(tf.equal(tf.strings.length(text), 0), _empty, _nonempty)

@tf.function
def parse_label_file_tf(lbl_path, meta, new_size, num_kpt, kpt_vals=3):
    """Read -> Parse -> Letterbox Adjust"""
    txt = tf.io.read_file(lbl_path)
    labels = parse_label_text_tf(txt, num_kpt, kpt_vals)
    labels = letterbox_adjust_labels_tf(labels, meta, new_size, num_kpt, kpt_vals)
    
    # Set static shape for compile-time optimization
    needed = 5 + num_kpt * kpt_vals
    labels = tf.ensure_shape(labels, [None, needed])
    return labels