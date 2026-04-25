from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from QAT_Refactored.core.backbone_selector import BackboneModelSet, resolve_backbone_model


def main() -> None:
    custom_model, mode = resolve_backbone_model(
        backbone="custom",
        task=None,
        custom_model="yolo11n.pt",
        models=BackboneModelSet(),
    )
    assert mode == "custom"
    assert custom_model == "yolo11n.pt"

    yolo_detect, mode = resolve_backbone_model(
        backbone="yolo",
        task="detect",
        custom_model="ignored.pt",
        models=BackboneModelSet(),
    )
    assert mode == "yolo"
    assert yolo_detect.endswith("cfg/models/v8/yolov8.yaml")

    cira_detect, mode = resolve_backbone_model(
        backbone="cira",
        task="detect",
        custom_model="ignored.pt",
        models=BackboneModelSet(),
    )
    assert mode == "cira"
    assert cira_detect.endswith("ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml")

    try:
        resolve_backbone_model(
            backbone="yolo",
            task=None,
            custom_model="ignored.pt",
            models=BackboneModelSet(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when task is missing for yolo/cira backbone")

    print("verify_qat_backbone_switch_smoke: OK")


if __name__ == "__main__":
    main()
