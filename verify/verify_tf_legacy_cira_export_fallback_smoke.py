from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from pathlib import Path

import numpy as np
import tensorflow as tf

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.core.exporter import Exporter
from QAT_Refactored.models.builder import build_student_qat


def _rep_dataset():
    for _ in range(2):
        yield [np.random.rand(1, 128, 128, 3).astype(np.float32)]


def main() -> None:
    out_dir = Path("verify/_tmp/tf_legacy_cira_export")
    cfg = AppConfig(
        IMGSZ=128,
        NUM_CLS=7,
        NUM_KPT=15,
        KPT_VALS=3,
        TRAIN_ENGINE="tf-legacy",
        DATA_BACKEND="native",
        TF_LEGACY_BACKBONE="cira-lite",
        TF_CIRA_WIDTH_MULT=0.3,
        TF_CIRA_USE_ATTENTION=True,
        TF_CIRA_USE_DEFORM=True,
        TFLITE_QUANT_MODE="int8",
        OUTPUT_DIR=out_dir,
        BATCH_SIZE=1,
    )
    cfg.validate()

    model = build_student_qat(cfg)
    _ = model(tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3), dtype=tf.float32), training=False)

    exporter = Exporter(cfg)
    saved_model_dir = out_dir / "saved_model"
    tflite_path = out_dir / "model_int8.tflite"
    exporter.export_saved_model(model, saved_model_dir)
    exporter.convert_to_tflite(saved_model_dir, tflite_path, _rep_dataset)

    if not tflite_path.exists():
        raise FileNotFoundError(f"Expected export not found: {tflite_path}")
    print("verify_tf_legacy_cira_export_fallback_smoke: OK")


if __name__ == "__main__":
    main()
