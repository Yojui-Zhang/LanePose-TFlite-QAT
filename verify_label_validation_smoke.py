from __future__ import annotations

import tensorflow as tf

from QAT_Refactored.losses.pose_loss import build_batch_dict_from_padded_labels


def main() -> None:
    B = 1
    M = 4
    num_cls = 7
    num_kpt = 17
    kpt_vals = 3
    needed = 5 + num_kpt * kpt_vals

    labels = tf.zeros((B, M, needed), dtype=tf.float32)

    # 1) 合法 GT
    cls = tf.constant([[[2.0]]], tf.float32)
    box = tf.constant([[[0.5, 0.5, 0.2, 0.2]]], tf.float32)
    kpt = tf.zeros((B, 1, num_kpt * kpt_vals), tf.float32)
    one = tf.concat([cls, box, kpt], axis=-1)
    labels = tf.tensor_scatter_nd_update(labels, indices=[[0, 0]], updates=tf.reshape(one, [-1, needed]))

    d = build_batch_dict_from_padded_labels(labels, num_cls=num_cls, num_kpt=num_kpt, kpt_vals=kpt_vals)
    tf.print("OK valid_mask:", d["valid_mask"])

    # 2) 非法 class id（應觸發 assert）
    labels_bad = tf.identity(labels)
    cls_bad = tf.constant([[[99.0]]], tf.float32)
    one_bad = tf.concat([cls_bad, box, kpt], axis=-1)
    labels_bad = tf.tensor_scatter_nd_update(labels_bad, indices=[[0, 0]], updates=tf.reshape(one_bad, [-1, needed]))

    try:
        _ = build_batch_dict_from_padded_labels(labels_bad, num_cls=num_cls, num_kpt=num_kpt, kpt_vals=kpt_vals)
        raise RuntimeError("expected assertion but did not happen")
    except tf.errors.InvalidArgumentError as e:
        print("Caught expected InvalidArgumentError:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()

