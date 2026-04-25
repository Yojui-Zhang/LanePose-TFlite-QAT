from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_BACKBONES = {"custom", "yolo", "cira"}
_SUPPORTED_TASKS = {"pose", "detect"}


@dataclass(frozen=True)
class BackboneModelSet:
    yolo_pose: str = "cfg/models/v8/yolov8-pose.yaml"
    yolo_detect: str = "cfg/models/v8/yolov8.yaml"
    cira_pose: str = "ultralytics/cfg/models/Yojui/yolov8_CIRA-Pose.yaml"
    cira_detect: str = "ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml"


def normalize_backbone(backbone: str) -> str:
    mode = str(backbone).strip().lower()
    if mode not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone must be one of {sorted(_SUPPORTED_BACKBONES)}, got {backbone!r}"
        )
    return mode


def normalize_task(task: str | None) -> str:
    if task is None:
        raise ValueError(
            "task is required when selecting yolo/cira backbone; "
            "set --task (pose/detect) or provide ULTRA_TASK."
        )
    norm = str(task).strip().lower()
    if norm not in _SUPPORTED_TASKS:
        raise ValueError(
            f"task must be one of {sorted(_SUPPORTED_TASKS)} for backbone switching, got {task!r}"
        )
    return norm


def resolve_backbone_model(
    *,
    backbone: str,
    task: str | None,
    custom_model: str,
    models: BackboneModelSet,
) -> tuple[str, str]:
    """
    Resolve model source for QAT/Ultralytics routes.

    Returns:
      (resolved_model_source, normalized_backbone_mode)
    """
    mode = normalize_backbone(backbone)
    if mode == "custom":
        source = str(custom_model).strip()
        if not source:
            raise ValueError("custom_model must be non-empty when backbone='custom'")
        return source, mode

    requested_task = normalize_task(task)
    if mode == "yolo":
        return (
            models.yolo_pose if requested_task == "pose" else models.yolo_detect,
            mode,
        )

    return (
        models.cira_pose if requested_task == "pose" else models.cira_detect,
        mode,
    )
