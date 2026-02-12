from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf
import numpy as np

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.losses.pose_loss import PoseLabelLoss, get_anchors


def main() -> None:
    cfg = AppConfig()
    loss_fn = PoseLabelLoss(cfg)
    anchors = get_anchors(cfg.IMGSZ, strides=[8, 16, 32], grid_cell_offset=0.5)

    B = 2
    N = int(anchors.shape[0])
    C = cfg.total_output_channels
    M = 64

    # y_pred logits：刻意讓 cls logits 接近 0 => sigmoid 約 0.5（最容易被負樣本主導的情境）
    y_pred = tf.zeros((B, C, N), dtype=tf.float32)

    # labels：每張圖給 1 個有效 GT，其他 padding
    bboxes = np.zeros((B, M, 4), dtype=np.float32)
    cls = np.zeros((B, M, 1), dtype=np.int32)
    keypoints = np.zeros((B, M, cfg.NUM_KPT * cfg.KPT_VALS), dtype=np.float32)
    valid_mask = np.zeros((B, M), dtype=bool)

    for b in range(B):
        bboxes[b, 0] = np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32)
        cls[b, 0, 0] = 0
        valid_mask[b, 0] = True

    batch = {
        "bboxes": tf.constant(bboxes),
        "cls": tf.constant(cls),
        "keypoints": tf.constant(keypoints),
        "valid_mask": tf.constant(valid_mask),
    }

    outputs = loss_fn(y_pred, batch, anchors)
    total, l_box, l_cls, l_kpt = outputs[:4]
    tf.debugging.assert_all_finite(total, "total loss must be finite")
    tf.debugging.assert_all_finite(l_cls, "cls loss must be finite")

    print("total:", float(total.numpy()), "l_box:", float(l_box.numpy()), "l_cls:", float(l_cls.numpy()), "l_kpt:", float(l_kpt.numpy()))


if __name__ == "__main__":
    main()
