from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd())

from QAT_Refactored.utils.env_setup import setup_environment
setup_environment()

import tensorflow as tf
from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.models.builder import build_student_qat
from QAT_Refactored.core.engine import Trainer


def main() -> None:
    cfg = AppConfig()
    cfg.BATCH_SIZE = 2

    model = build_student_qat(cfg)
    _ = model(tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3)))  # build

    trainer = Trainer(cfg=cfg, student=model, teacher=None)
    opt = tf.keras.optimizers.Adam(1e-4)

    B = 2
    M = 64
    needed = 5 + cfg.NUM_KPT * cfg.KPT_VALS

    imgs = tf.random.uniform((B, cfg.IMGSZ, cfg.IMGSZ, 3), dtype=tf.float32)

    labels = tf.zeros((B, M, needed), dtype=tf.float32)
    cls = tf.zeros((B, 1, 1), dtype=tf.float32)
    bbox = tf.constant([[[0.5, 0.5, 0.2, 0.2]]], dtype=tf.float32)
    bbox = tf.tile(bbox, [B, 1, 1])
    kpt = tf.zeros((B, 1, cfg.NUM_KPT * cfg.KPT_VALS), dtype=tf.float32)
    one = tf.concat([cls, bbox, kpt], axis=-1)  # (B,1,needed)

    labels = tf.tensor_scatter_nd_update(
        labels,
        indices=tf.constant([[0, 0], [1, 0]], dtype=tf.int32),
        updates=tf.reshape(one, [-1, needed]),
    )

    cw = tf.ones((cfg.NUM_CLS,), dtype=tf.float32)

    # 1) normal training=True
    logs0 = trainer.train_step(imgs, labels, opt, cw, freeze_bn=False)
    tf.print("freeze_bn=False logs:", logs0)

    # 2) freeze mode training=False (BN/observer freeze)
    logs1 = trainer.train_step(imgs, labels, opt, cw, freeze_bn=True)
    tf.print("freeze_bn=True logs:", logs1)

    tf.nest.map_structure(lambda t: tf.debugging.assert_all_finite(t, "non-finite"), logs0)
    tf.nest.map_structure(lambda t: tf.debugging.assert_all_finite(t, "non-finite"), logs1)

    print("verify_freeze_bn_smoke: OK")


if __name__ == "__main__":
    main()

