from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import numpy as np
import tensorflow as tf
import cv2

from QAT_Refactored.utils.visualization import draw_on_image, save_pred_and_plot


def main() -> None:
    B = 1
    H = W = 640
    num_cls = 7
    num_kpt = 17
    kpt_vals = 3
    C = 4 + num_cls + num_kpt * kpt_vals

    # dummy image
    imgs = tf.ones((B, H, W, 3), dtype=tf.float32) * 0.2

    # dummy preds: (B,C,N)
    N = 10
    y = np.zeros((B, C, N), dtype=np.float32)

    # one detection with high score
    # box: cx,cy,w,h in [0,1]
    y[0, 0:4, 0] = np.array([0.5, 0.5, 0.4, 0.3], dtype=np.float32)

    # cls logits (make cls=2 confident)
    y[0, 4 + 2, 0] = 8.0

    # kpts: simple diagonal, vis=1
    kpts = np.zeros((num_kpt, kpt_vals), dtype=np.float32)
    for i in range(num_kpt):
        kpts[i, 0] = 0.2 + 0.6 * (i / max(1, num_kpt - 1))
        kpts[i, 1] = 0.2 + 0.6 * (i / max(1, num_kpt - 1))
        kpts[i, 2] = 1.0
    y[0, 4 + num_cls : 4 + num_cls + num_kpt * kpt_vals, 0] = kpts.reshape(-1)

    preds = tf.convert_to_tensor(y)

    # RGB-order sanity check:
    # input image is RGB red; after draw_on_image(BGR) + cvtColor(BGR2RGB), it must stay red.
    red_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    red_rgb[..., 0] = 255
    drawn_bgr = draw_on_image(red_rgb, boxes=np.zeros((0, 4), dtype=np.float32))
    shown_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    mean_rgb = shown_rgb.mean(axis=(0, 1))
    assert mean_rgb[0] > 250.0 and mean_rgb[2] < 5.0, f"RGB order mismatch: mean_rgb={mean_rgb}"

    save_pred_and_plot(
        imgs,
        preds,
        output_dir="./_vis_smoke",
        step=0,
        num_cls=num_cls,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        total_C=C,
        conf_thres=0.25,
        force_draw_topk_if_empty=1,
    )
    print("verify_visualization_smoke: OK")


if __name__ == "__main__":
    main()
