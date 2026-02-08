# Source File: verify_layout.py
from __future__ import annotations

import os
import sys
import inspect

sys.path.append(os.getcwd())

from QAT_Refactored.utils.env_setup import setup_environment

setup_environment()

import tensorflow as tf

from QAT_Refactored.utils import tensor_layout as tl


def main() -> None:
    B, C, N = 2, 13, 8400

    print(f"[verify_layout] tensor_layout = {tl.__file__}")
    print(f"[verify_layout] ensure_bnc_tf = {inspect.signature(tl.ensure_bnc_tf)}")
    print(f"[verify_layout] ensure_bcn_tf = {inspect.signature(tl.ensure_bcn_tf)}")

    x_bcn = tf.random.uniform([B, C, N], dtype=tf.float32)
    x_bnc = tf.random.uniform([B, N, C], dtype=tf.float32)

    y1 = tl.ensure_bnc_tf(x_bcn, C)
    y2 = tl.ensure_bnc_tf(x_bnc, C)

    tf.debugging.assert_equal(tf.shape(y1), tf.constant([B, N, C], tf.int32))
    tf.debugging.assert_equal(tf.shape(y2), tf.constant([B, N, C], tf.int32))

    rt = tl.ensure_bcn_tf(y1, C)

    ne = tf.not_equal(rt, x_bcn)
    if tf.reduce_any(ne):
        diff = tf.math.abs(rt - x_bcn)
        max_abs = tf.reduce_max(diff)
        bad = tf.where(ne)
        first = bad[0]
        i0, i1, i2 = tf.unstack(first, axis=0)
        v_rt = rt[i0, i1, i2]
        v_x = x_bcn[i0, i1, i2]
        raise AssertionError(
            f"Roundtrip mismatch: max_abs={max_abs.numpy():.9g}, "
            f"first_idx={first.numpy().tolist()}, rt={float(v_rt.numpy()):.9g}, x={float(v_x.numpy()):.9g}"
        )

    print("verify_layout: OK")


if __name__ == "__main__":
    main()
