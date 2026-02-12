# Source File: verify_distill_layout.py
from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.losses.distill_loss import DistillLossPose


def main() -> None:
    cfg = AppConfig()
    loss_fn = DistillLossPose(cfg)

    B = 2
    C = cfg.total_output_channels
    N = cfg.get_total_anchors()

    t_bcn = tf.random.uniform([B, C, N], dtype=tf.float32)
    s_bcn = tf.random.uniform([B, C, N], dtype=tf.float32)

    t_bnc = tf.transpose(t_bcn, [0, 2, 1])
    s_bnc = tf.transpose(s_bcn, [0, 2, 1])

    l1 = loss_fn(t_bcn, s_bcn)
    l2 = loss_fn(t_bnc, s_bnc)

    tf.debugging.assert_near(l1, l2, atol=1e-6, rtol=1e-6)
    print("verify_distill_layout: OK")


if __name__ == "__main__":
    main()
