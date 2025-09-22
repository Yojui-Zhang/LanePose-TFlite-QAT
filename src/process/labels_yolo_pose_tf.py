
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
