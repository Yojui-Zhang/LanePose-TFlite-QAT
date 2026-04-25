from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from train_pose import (
    _build_arg_parser,
    _build_kd_deploy_overrides,
    _infer_task_from_data_yaml,
    _parse_imgsz,
    _resolve_backbone_model_from_args,
)


def main() -> None:
    parser = _build_arg_parser()

    kd_args = parser.parse_args(
        [
            "--data",
            "./dataset/KITTI.yaml",
            "--task",
            "detect",
            "--qat-loss-mode",
            "kd-deploy",
            "--qat-backbone",
            "cira",
            "--imgsz",
            "640",
            "640",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--workers",
            "0",
        ]
    )
    imgsz = _parse_imgsz(kd_args.imgsz)
    overrides, _ = _build_kd_deploy_overrides(kd_args, imgsz)
    assert overrides["ULTRA_BACKBONE"] == "cira"
    assert str(overrides["ULTRA_MODEL"]).endswith(
        "ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml"
    )

    yolo_args = parser.parse_args(
        [
            "--data",
            "./dataset/lanepose-carkeypoint.yaml",
            "--qat-backbone",
            "yolo",
        ]
    )
    inferred_task = _infer_task_from_data_yaml(yolo_args.data)
    model_source, backbone_mode = _resolve_backbone_model_from_args(yolo_args, inferred_task)
    assert inferred_task == "pose"
    assert backbone_mode == "yolo"
    assert model_source.endswith("cfg/models/v8/yolov8-pose.yaml")

    print("verify_train_pose_backbone_switch_smoke: OK")


if __name__ == "__main__":
    main()
