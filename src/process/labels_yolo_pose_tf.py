
"""
Utility to parse YOLOv8-pose style label .txt lines into arrays.
Each line:
cls cx cy w h [x1 y1 v1 ... xK yK vK]
All coordinates normalized to 0..1 relative to image W,H.
"""
import numpy as np

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
