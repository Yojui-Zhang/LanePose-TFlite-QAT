from __future__ import annotations

from pathlib import Path

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.data.ultralytics_bridge import build_ultralytics_pose_data


def main() -> None:
    cfg = AppConfig()
    cfg.DATA_BACKEND = "ultralytics"
    cfg.DATA_YAML = Path("./dataset/lanepose-carkeypoint.yaml")
    cfg.BATCH_SIZE = 2
    cfg.IMGSZ = 640
    cfg.ULTRA_WORKERS = 0
    cfg.ULTRA_CACHE = False
    cfg.TRAIN_DROP_REMAINDER = False

    cfg.validate()

    bundle = build_ultralytics_pose_data(cfg)
    imgs, labels = next(iter(bundle.train_ds))

    tf.debugging.assert_rank(imgs, 4)
    tf.debugging.assert_rank(labels, 3)
    tf.debugging.assert_equal(tf.shape(imgs)[0], 2)
    tf.debugging.assert_equal(tf.shape(labels)[0], 2)
    tf.debugging.assert_equal(tf.shape(imgs)[1], cfg.IMGSZ)
    tf.debugging.assert_equal(tf.shape(imgs)[2], cfg.IMGSZ)

    if bundle.val_ds is not None and bundle.val_steps > 0:
        val_imgs, val_labels = next(iter(bundle.val_ds))
        tf.debugging.assert_rank(val_imgs, 4)
        tf.debugging.assert_rank(val_labels, 3)
        tf.debugging.assert_equal(tf.shape(val_imgs)[1], cfg.IMGSZ)
        tf.debugging.assert_equal(tf.shape(val_imgs)[2], cfg.IMGSZ)

    print(
        "verify_ultralytics_bridge_smoke: OK",
        f"train_steps={bundle.steps_per_epoch}",
        f"val_steps={bundle.val_steps}",
    )


if __name__ == "__main__":
    main()
