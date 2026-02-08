from __future__ import annotations

import os
import sys

sys.path.append(os.getcwd())

from QAT_Refactored.utils.env_setup import setup_environment
setup_environment()

import tensorflow as tf  # noqa: F401

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.models.builder import build_student_qat
from QAT_Refactored.core.engine import Trainer  # 依你的實際類名調整
# 若 Trainer 建構需要 teacher/ema/cfg 以外依賴，照你的 engine.py 參數補齊即可。

def main() -> None:
    cfg = AppConfig()

    model = build_student_qat(cfg)
    _ = model(tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3)))  # build weights

    trainer = Trainer(cfg=cfg, student=model, teacher=None)  # 依你的 engine 實作調整
    opt = tf.keras.optimizers.Adam(1e-4)

    B = 2
    M = 64
    needed = 5 + cfg.NUM_KPT * cfg.KPT_VALS

    imgs = tf.random.uniform((B, cfg.IMGSZ, cfg.IMGSZ, 3), dtype=tf.float32)

    # labels: (B, M, needed) => [cls, cx, cy, w, h, kpts...]
    labels = tf.zeros((B, M, needed), dtype=tf.float32)

    # 塞一筆有效 GT：w>0 => valid_mask True
    cls = tf.zeros((B, 1, 1), dtype=tf.float32)
    bbox = tf.constant([[[0.5, 0.5, 0.2, 0.2]]], dtype=tf.float32)  # cx,cy,w,h
    bbox = tf.tile(bbox, [B, 1, 1])

    kpt = tf.zeros((B, 1, cfg.NUM_KPT * cfg.KPT_VALS), dtype=tf.float32)
    one = tf.concat([cls, bbox, kpt], axis=-1)  # (B,1,needed)

    labels = tf.tensor_scatter_nd_update(
        labels,
        indices=tf.constant([[0, 0], [1, 0]], dtype=tf.int32),
        updates=tf.reshape(one, [-1, needed]),
    )

    class_weights = tf.ones((cfg.NUM_CLS,), dtype=tf.float32)

    logs_tr = trainer.train_step(imgs, labels, opt, class_weights)
    logs_va = trainer.val_step(imgs, labels, class_weights)

    tf.print("train logs:", logs_tr)
    tf.print("val logs:", logs_va)

if __name__ == "__main__":
    main()

