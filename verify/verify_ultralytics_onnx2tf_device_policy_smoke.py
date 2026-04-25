from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

import ultralytics

import train_QAT
from QAT_Refactored.config.config import AppConfig


def main() -> None:
    marker = Path("./runs/_verify_onnx2tf_policy/fake_saved_model")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    captured: dict[str, str | None] = {}
    original_yolo = ultralytics.YOLO

    class _FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, **kwargs):
            captured["during_export"] = os.environ.get("ULTRALYTICS_ONNX2TF_DEVICE")
            return str(marker)

    previous_env = os.environ.get("ULTRALYTICS_ONNX2TF_DEVICE")
    ultralytics.YOLO = _FakeYOLO
    try:
        cfg = AppConfig()
        cfg.TFLITE_QUANT_MODE = "fp32"
        cfg.ULTRA_ONNX2TF_DEVICE = "cpu"
        cfg.ULTRA_TASK = "pose"
        cfg.DATA_YAML = Path("./dataset/lanepose-carkeypoint.yaml")
        cfg.validate()

        export_path, aliases = train_QAT._export_ultralytics_tflite(Path("./yolo11n.pt"), cfg)
    finally:
        ultralytics.YOLO = original_yolo

    assert export_path == marker
    assert aliases == []
    assert captured.get("during_export") == "cpu"
    assert os.environ.get("ULTRALYTICS_ONNX2TF_DEVICE") == previous_env
    print("verify_ultralytics_onnx2tf_device_policy_smoke: OK")


if __name__ == "__main__":
    main()
