from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf

from QAT_Refactored.losses.pose_loss import get_anchors, TaskAlignedAssigner


def test_positive_assignment_units() -> None:
    imgsz = 640
    num_cls = 7

    anchors = get_anchors(imgsz=imgsz, strides=[8, 16, 32])  # (N,4) normalized
    n = int(anchors.shape[0])

    # 1 個 GT：置中且面積夠大，理論上必有 anchors center 落在 bbox 內
    gt_box = tf.constant([[0.5, 0.5, 0.4, 0.4]], dtype=tf.float32)  # (1,4)
    gt_cls = tf.constant([0], dtype=tf.int32)                      # (1,)
    valid_mask = tf.constant([True], dtype=tf.bool)                # (1,)

    # 讓 IoU=1、cls_score=1，避免因 pred_box/cls 太爛導致 align_metric=0
    pred_box = tf.repeat(gt_box, repeats=n, axis=0)                # (N,4)
    pred_cls_prob = tf.one_hot(tf.fill([n], 0), depth=num_cls, dtype=tf.float32)  # (N,C)

    assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
    _, _, _, pos_mask = assigner.assign(pred_box, pred_cls_prob, anchors, gt_box, gt_cls, valid_mask)

    pos_cnt = int(tf.reduce_sum(tf.cast(pos_mask, tf.int32)).numpy())
    assert pos_cnt > 0, f"pos_cnt must be > 0, got {pos_cnt}"
    print("[OK] pos_cnt =", pos_cnt)


if __name__ == "__main__":
    test_positive_assignment_units()
