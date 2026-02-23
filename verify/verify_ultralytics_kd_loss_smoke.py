from __future__ import annotations

import copy

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import torch

from QAT_Refactored.core.loss_balancer import LossBalanceConfig
from QAT_Refactored.core.ultralytics_kd import KDLossConfig, KDDetectionModel, KDPoseModel


def _build_balance_cfg() -> LossBalanceConfig:
    return LossBalanceConfig(
        strategy="grad_norm",
        shared_param_group="head",
        ema_decay=0.9,
        update_interval=1,
        warmup_steps=0,
        deploy_ramp_steps=0,
        min_weight=0.2,
        max_weight=5.0,
        max_step_change=1.2,
        adapt_power=0.5,
        renorm_sum=2.0,
        eps=1e-6,
    )


def _build_kd_cfg() -> KDLossConfig:
    return KDLossConfig(
        temperature=1.0,
        aux_kd_head_label_loss=False,
        balance=_build_balance_cfg(),
        log_interval_steps=1,
    )


def main() -> None:
    pose_model = KDPoseModel(
        "ultralytics/cfg/models/11/yolo11-pose.yaml",
        nc=1,
        ch=3,
        data_kpt_shape=(17, 3),
        verbose=False,
    )
    pose_teacher = copy.deepcopy(pose_model).eval()

    pose_model.__dict__["kd_teacher"] = pose_teacher
    pose_model.__dict__["kd_loss_config"] = _build_kd_cfg()

    pose_model.train()
    pose_criterion = pose_model.init_criterion()

    imgs = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    pose_preds = pose_model(imgs)

    keypoints = torch.zeros((1, 17, 3), dtype=torch.float32)
    keypoints[..., 2] = 1.0
    pose_batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0], dtype=torch.int64),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "keypoints": keypoints,
    }
    pose_total, pose_items = pose_criterion(pose_preds, pose_batch)
    assert torch.isfinite(pose_total).all(), "Pose KD total loss has NaN/Inf"
    assert torch.isfinite(pose_items).all(), "Pose deploy loss items have NaN/Inf"
    assert pose_criterion.balancer.last_alpha_kd >= 0.0, "Pose alpha_kd must be >= 0"

    detect_model = KDDetectionModel(
        "ultralytics/cfg/models/11/yolo11.yaml",
        nc=1,
        ch=3,
        verbose=False,
    )
    detect_teacher = copy.deepcopy(detect_model).eval()
    detect_model.__dict__["kd_teacher"] = detect_teacher
    detect_model.__dict__["kd_loss_config"] = _build_kd_cfg()

    detect_model.train()
    detect_criterion = detect_model.init_criterion()
    detect_preds = detect_model(imgs)
    detect_batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0], dtype=torch.int64),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
    }
    detect_total, detect_items = detect_criterion(detect_preds, detect_batch)
    assert torch.isfinite(detect_total).all(), "Detect KD total loss has NaN/Inf"
    assert torch.isfinite(detect_items).all(), "Detect deploy loss items have NaN/Inf"
    assert detect_criterion.balancer.last_alpha_kd >= 0.0, "Detect alpha_kd must be >= 0"

    print(
        "verify_ultralytics_kd_loss_smoke: OK",
        f"pose_total={float(pose_total.detach()):.6f}",
        f"detect_total={float(detect_total.detach()):.6f}",
    )


if __name__ == "__main__":
    main()
