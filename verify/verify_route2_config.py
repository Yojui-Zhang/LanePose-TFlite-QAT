from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from pathlib import Path

from QAT_Refactored.core.ultralytics_route2 import (
    UltralyticsExportConfig,
    UltralyticsRoute2Runner,
    UltralyticsTrainConfig,
)


def main() -> None:
    train_cfg = UltralyticsTrainConfig(
        ultralytics_root=Path("./ultralytics"),
        model="cfg/models/v8/yolov8-pose.yaml",
        data="./dataset/lanepose-carkeypoint.yaml",
        epochs=1,
        batch=2,
        imgsz=(640, 640),
        device="cpu",
        workers=0,
        amp=False,
        fliplr=0.0,
        cos_lr=True,
        project=Path("./runs/pose"),
        name="verify_route2_cfg",
        resume=False,
        exist_ok=True,
        cache=False,
        task="pose",
    )
    export_cfg = UltralyticsExportConfig(
        do_export=False,
        format="tflite",
        int8=False,
        half=False,
        nms=False,
        data=None,
        imgsz=(640, 640),
    )
    export_cfg_checked = UltralyticsExportConfig(
        do_export=True,
        format="tflite",
        int8=True,
        half=False,
        nms=False,
        data="./dataset/lanepose-carkeypoint.yaml",
        imgsz=(640, 640),
        fraction=0.5,
    )
    export_cfg_checked.validate()

    runner = UltralyticsRoute2Runner(train_cfg=train_cfg, export_cfg=export_cfg)
    model = runner._build_model()  # smoke only
    assert model is not None
    print("verify_route2_config: OK")


if __name__ == "__main__":
    main()
