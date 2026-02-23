from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import DetectionModel, PoseModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss, v8PoseLoss

from QAT_Refactored.core.loss_balancer import LossBalanceConfig, LossBalancer

# Keep Ultralytics local settings writable inside workspace.
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))


@dataclass(frozen=True)
class KDLossConfig:
    temperature: float = 1.0
    aux_kd_head_label_loss: bool = False
    balance: LossBalanceConfig = field(default_factory=LossBalanceConfig)
    log_interval_steps: int = 50


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cudnn_status_alloc_failed" in msg


def _as_torch_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _extract_raw_detect_outputs(preds: Any) -> list[torch.Tensor]:
    """Normalize Detect head outputs to raw training tensors: [P3, P4, P5]."""
    feats = preds[1] if isinstance(preds, tuple) else preds
    if isinstance(feats, list) and all(torch.is_tensor(x) for x in feats):
        return feats
    raise TypeError(f"Unsupported detect prediction structure for KD: {type(preds).__name__}")



def _extract_raw_pose_outputs(preds: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Normalize Pose head outputs to raw training tensors: (feats, kpt)."""
    if isinstance(preds, tuple) and len(preds) == 2:
        first, second = preds
        if isinstance(first, list) and torch.is_tensor(second):
            return first, second
        if isinstance(second, tuple) and len(second) == 2 and isinstance(second[0], list) and torch.is_tensor(second[1]):
            return second[0], second[1]
    raise TypeError(f"Unsupported pose prediction structure for KD: {type(preds).__name__}")



def _align_feature_maps(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher feature shape to student feature shape conservatively."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError(f"Expected 4D feature maps, got student={student.shape}, teacher={teacher.shape}")

    if teacher.shape[2:] != student.shape[2:]:
        teacher = F.interpolate(teacher, size=student.shape[2:], mode="bilinear", align_corners=False)

    if teacher.shape[1] != student.shape[1]:
        c = min(student.shape[1], teacher.shape[1])
        student = student[:, :c]
        teacher = teacher[:, :c]

    return student, teacher



def _align_keypoints(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align keypoint tensor shape from teacher to student."""
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError(f"Expected 3D keypoint tensors, got student={student.shape}, teacher={teacher.shape}")

    if teacher.shape[-1] != student.shape[-1]:
        teacher = F.interpolate(teacher, size=student.shape[-1], mode="linear", align_corners=False)

    if teacher.shape[1] != student.shape[1]:
        c = min(student.shape[1], teacher.shape[1])
        student = student[:, :c]
        teacher = teacher[:, :c]

    return student, teacher


def _compute_feature_kd_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
    *,
    reg_max: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    kd_dfl = torch.zeros((), device=device)
    kd_cls = torch.zeros((), device=device)
    levels = min(len(student_feats), len(teacher_feats))
    dfl_channels = int(reg_max * 4)

    for i in range(levels):
        s_map = student_feats[i]
        t_map = teacher_feats[i].to(device=s_map.device, dtype=s_map.dtype, non_blocking=True)
        s_map, t_map = _align_feature_maps(s_map, t_map)
        c = min(s_map.shape[1], t_map.shape[1])
        d = min(dfl_channels, c)
        cls_c = max(c - d, 0)

        if d > 0:
            kd_dfl = kd_dfl + F.smooth_l1_loss(s_map[:, :d], t_map[:, :d], reduction="mean")
        if cls_c > 0:
            s_cls = s_map[:, d : d + cls_c]
            t_cls = t_map[:, d : d + cls_c]
            temp = max(float(temperature), 1e-6)
            s_logits = (s_cls / temp).permute(0, 2, 3, 1).reshape(-1, cls_c)
            t_logits = (t_cls / temp).permute(0, 2, 3, 1).reshape(-1, cls_c)
            kd_cls = kd_cls + F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            ) * (temp**2)

    if levels > 0:
        kd_dfl = kd_dfl / float(levels)
        kd_cls = kd_cls / float(levels)

    return kd_dfl + kd_cls


def _write_kd_scalars_to_tensorboard(
    *,
    step: int,
    supervised_loss: float,
    kd_loss: float,
    alpha_kd: float,
    grad_norm_ratio_sup_over_kd: Optional[float],
) -> None:
    try:
        from ultralytics.utils.callbacks import tensorboard as tb_callbacks
    except Exception:
        return

    writer = getattr(tb_callbacks, "WRITER", None)
    if writer is None:
        return

    writer.add_scalar("train/supervised_loss", supervised_loss, step)
    writer.add_scalar("train/kd_loss", kd_loss, step)
    writer.add_scalar("train/alpha_kd", alpha_kd, step)
    if grad_norm_ratio_sup_over_kd is not None and math.isfinite(float(grad_norm_ratio_sup_over_kd)):
        writer.add_scalar("train/grad_norm_ratio_sup_over_kd", float(grad_norm_ratio_sup_over_kd), step)


class KDPoseLoss(v8PoseLoss):
    """Ultralytics pose loss + optional KD term with dynamic KD alpha balancing."""

    def __init__(self, model: PoseModel):
        super().__init__(model)
        cfg = model.__dict__.get("kd_loss_config", KDLossConfig())
        if not isinstance(cfg, KDLossConfig):
            raise TypeError("model.kd_loss_config must be KDLossConfig")

        teacher = model.__dict__.get("kd_teacher", None)
        if teacher is not None and not isinstance(teacher, torch.nn.Module):
            raise TypeError("model.kd_teacher must be torch.nn.Module or None")

        self.kd_cfg = cfg
        self.teacher = teacher
        self.balancer = LossBalancer(cfg.balance)
        self._shared_params = self._select_shared_params(model)
        self._log_interval = max(1, int(cfg.log_interval_steps))
        self._teacher_device = _as_torch_device(self.device)
        self._teacher_dtype: Optional[torch.dtype] = None
        fixed_alpha = cfg.balance.fixed_kd_weight
        logging.info(
            "[KD] Balance enabled: strategy=%s shared=%s fixed_alpha=%s params=%d",
            cfg.balance.strategy,
            cfg.balance.shared_param_group,
            "None" if fixed_alpha is None else f"{float(fixed_alpha):.6f}",
            len(self._shared_params),
        )
        if self.teacher is not None:
            self._prepare_teacher_runtime(prefer_device=self.device)

    def _prepare_teacher_runtime(self, *, prefer_device: Any) -> None:
        if self.teacher is None:
            return
        target = _as_torch_device(prefer_device)
        try:
            self.teacher.to(target)
            self._teacher_device = target
        except RuntimeError as exc:
            if _is_oom_error(exc) and target.type == "cuda":
                logging.warning(
                    "[KD] Teacher move to %s failed due to OOM, fallback to CPU.",
                    target,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.teacher.to("cpu")
                self._teacher_device = torch.device("cpu")
            else:
                raise

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        teacher_param = next(self.teacher.parameters(), None)
        self._teacher_dtype = teacher_param.dtype if teacher_param is not None else torch.float32

    def _select_shared_params(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        group = self.kd_cfg.balance.shared_param_group
        params: list[torch.nn.Parameter] = []
        if group == "head":
            try:
                params = [p for p in model.model[-1].parameters() if p.requires_grad]
            except Exception:
                params = []
        if not params:
            params = [p for p in model.parameters() if p.requires_grad]
        return params

    def _compute_kd_loss(self, preds: Any, batch: dict[str, Any]) -> torch.Tensor:
        if self.teacher is None:
            return torch.zeros((), device=self.device)

        imgs = batch["img"].to(self.device, non_blocking=True)
        s_feats, s_kpt = _extract_raw_pose_outputs(preds)
        teacher_imgs = imgs.to(
            device=self._teacher_device,
            dtype=(self._teacher_dtype or imgs.dtype),
            non_blocking=True,
        )
        try:
            with torch.no_grad():
                t_preds = self.teacher(teacher_imgs)
        except RuntimeError as exc:
            if _is_oom_error(exc) and self._teacher_device.type == "cuda":
                logging.warning(
                    "[KD] Teacher forward OOM on %s, fallback to CPU for KD branch.",
                    self._teacher_device,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._prepare_teacher_runtime(prefer_device=torch.device("cpu"))
                teacher_imgs = imgs.to(
                    device=self._teacher_device,
                    dtype=(self._teacher_dtype or imgs.dtype),
                    non_blocking=True,
                )
                with torch.no_grad():
                    t_preds = self.teacher(teacher_imgs)
            else:
                raise
        t_feats, t_kpt = _extract_raw_pose_outputs(t_preds)
        kd = _compute_feature_kd_loss(
            student_feats=s_feats,
            teacher_feats=t_feats,
            reg_max=self.reg_max,
            temperature=float(self.kd_cfg.temperature),
            device=self.device,
        )

        t_kpt = t_kpt.to(device=s_kpt.device, dtype=s_kpt.dtype, non_blocking=True)
        s_kpt, t_kpt = _align_keypoints(s_kpt, t_kpt)
        kd_kpt = F.smooth_l1_loss(s_kpt, t_kpt, reduction="mean")

        kd = kd + kd_kpt
        # Keep KD scalar in the same batch-scaled convention as Ultralytics deploy loss.
        return kd * float(imgs.shape[0])

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        deploy_total, deploy_items = super().__call__(preds, batch)
        if deploy_total.ndim > 0:
            deploy_total = deploy_total.sum()

        kd_total = torch.zeros_like(deploy_total)
        if self.teacher is not None:
            kd_total = self._compute_kd_loss(preds, batch)
        elif self.kd_cfg.aux_kd_head_label_loss:
            # Fallback keeps KD branch effective even when teacher is not available.
            kd_total = deploy_total

        total, alpha_kd = self.balancer.build_total(
            deploy_loss=deploy_total,
            kd_loss=kd_total,
            shared_params=self._shared_params,
        )
        step = self.balancer.step
        if step % self._log_interval == 0:
            sup_loss = float(self.balancer.last_supervised_loss)
            kd_loss = float(self.balancer.last_kd_loss)
            grad_ratio = self.balancer.last_grad_norm_ratio_sup_over_kd
            _write_kd_scalars_to_tensorboard(
                step=step,
                supervised_loss=sup_loss,
                kd_loss=kd_loss,
                alpha_kd=float(alpha_kd),
                grad_norm_ratio_sup_over_kd=grad_ratio,
            )
            logging.info(
                "[KD] step=%d supervised_loss=%.6f kd_loss=%.6f alpha_kd=%.6f grad_ratio_sup_over_kd=%s",
                step,
                sup_loss,
                kd_loss,
                float(alpha_kd),
                (
                    "None"
                    if grad_ratio is None or not math.isfinite(float(grad_ratio))
                    else f"{float(grad_ratio):.6f}"
                ),
            )
        return total, deploy_items


class KDDetectLoss(v8DetectionLoss):
    """Ultralytics detect loss + optional KD term with dynamic KD alpha balancing."""

    def __init__(self, model: DetectionModel):
        super().__init__(model)
        cfg = model.__dict__.get("kd_loss_config", KDLossConfig())
        if not isinstance(cfg, KDLossConfig):
            raise TypeError("model.kd_loss_config must be KDLossConfig")

        teacher = model.__dict__.get("kd_teacher", None)
        if teacher is not None and not isinstance(teacher, torch.nn.Module):
            raise TypeError("model.kd_teacher must be torch.nn.Module or None")

        self.kd_cfg = cfg
        self.teacher = teacher
        self.balancer = LossBalancer(cfg.balance)
        self._shared_params = self._select_shared_params(model)
        self._log_interval = max(1, int(cfg.log_interval_steps))
        self._teacher_device = _as_torch_device(self.device)
        self._teacher_dtype: Optional[torch.dtype] = None
        fixed_alpha = cfg.balance.fixed_kd_weight
        logging.info(
            "[KD] Balance enabled: strategy=%s shared=%s fixed_alpha=%s params=%d",
            cfg.balance.strategy,
            cfg.balance.shared_param_group,
            "None" if fixed_alpha is None else f"{float(fixed_alpha):.6f}",
            len(self._shared_params),
        )
        if self.teacher is not None:
            self._prepare_teacher_runtime(prefer_device=self.device)

    def _prepare_teacher_runtime(self, *, prefer_device: Any) -> None:
        if self.teacher is None:
            return
        target = _as_torch_device(prefer_device)
        try:
            self.teacher.to(target)
            self._teacher_device = target
        except RuntimeError as exc:
            if _is_oom_error(exc) and target.type == "cuda":
                logging.warning(
                    "[KD] Teacher move to %s failed due to OOM, fallback to CPU.",
                    target,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.teacher.to("cpu")
                self._teacher_device = torch.device("cpu")
            else:
                raise

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        teacher_param = next(self.teacher.parameters(), None)
        self._teacher_dtype = teacher_param.dtype if teacher_param is not None else torch.float32

    def _select_shared_params(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        group = self.kd_cfg.balance.shared_param_group
        params: list[torch.nn.Parameter] = []
        if group == "head":
            try:
                params = [p for p in model.model[-1].parameters() if p.requires_grad]
            except Exception:
                params = []
        if not params:
            params = [p for p in model.parameters() if p.requires_grad]
        return params

    def _compute_kd_loss(self, preds: Any, batch: dict[str, Any]) -> torch.Tensor:
        if self.teacher is None:
            return torch.zeros((), device=self.device)

        imgs = batch["img"].to(self.device, non_blocking=True)
        s_feats = _extract_raw_detect_outputs(preds)
        teacher_imgs = imgs.to(
            device=self._teacher_device,
            dtype=(self._teacher_dtype or imgs.dtype),
            non_blocking=True,
        )
        try:
            with torch.no_grad():
                t_preds = self.teacher(teacher_imgs)
        except RuntimeError as exc:
            if _is_oom_error(exc) and self._teacher_device.type == "cuda":
                logging.warning(
                    "[KD] Teacher forward OOM on %s, fallback to CPU for KD branch.",
                    self._teacher_device,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._prepare_teacher_runtime(prefer_device=torch.device("cpu"))
                teacher_imgs = imgs.to(
                    device=self._teacher_device,
                    dtype=(self._teacher_dtype or imgs.dtype),
                    non_blocking=True,
                )
                with torch.no_grad():
                    t_preds = self.teacher(teacher_imgs)
            else:
                raise
        t_feats = _extract_raw_detect_outputs(t_preds)

        kd = _compute_feature_kd_loss(
            student_feats=s_feats,
            teacher_feats=t_feats,
            reg_max=self.reg_max,
            temperature=float(self.kd_cfg.temperature),
            device=self.device,
        )
        # Keep KD scalar in the same batch-scaled convention as Ultralytics deploy loss.
        return kd * float(imgs.shape[0])

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        deploy_total, deploy_items = super().__call__(preds, batch)
        if deploy_total.ndim > 0:
            deploy_total = deploy_total.sum()

        kd_total = torch.zeros_like(deploy_total)
        if self.teacher is not None:
            kd_total = self._compute_kd_loss(preds, batch)
        elif self.kd_cfg.aux_kd_head_label_loss:
            # Fallback keeps KD branch effective even when teacher is not available.
            kd_total = deploy_total

        total, alpha_kd = self.balancer.build_total(
            deploy_loss=deploy_total,
            kd_loss=kd_total,
            shared_params=self._shared_params,
        )
        step = self.balancer.step
        if step % self._log_interval == 0:
            sup_loss = float(self.balancer.last_supervised_loss)
            kd_loss = float(self.balancer.last_kd_loss)
            grad_ratio = self.balancer.last_grad_norm_ratio_sup_over_kd
            _write_kd_scalars_to_tensorboard(
                step=step,
                supervised_loss=sup_loss,
                kd_loss=kd_loss,
                alpha_kd=float(alpha_kd),
                grad_norm_ratio_sup_over_kd=grad_ratio,
            )
            logging.info(
                "[KD] step=%d supervised_loss=%.6f kd_loss=%.6f alpha_kd=%.6f grad_ratio_sup_over_kd=%s",
                step,
                sup_loss,
                kd_loss,
                float(alpha_kd),
                (
                    "None"
                    if grad_ratio is None or not math.isfinite(float(grad_ratio))
                    else f"{float(grad_ratio):.6f}"
                ),
            )
        return total, deploy_items


class KDDetectionModel(DetectionModel):
    """DetectionModel wired with KDDetectLoss criterion."""

    def init_criterion(self):
        if not hasattr(self, "args"):
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG

            self.args = get_cfg(DEFAULT_CFG)
        return KDDetectLoss(self)


class KDPoseModel(PoseModel):
    """PoseModel wired with KDPoseLoss criterion."""

    def init_criterion(self):
        if not hasattr(self, "args"):
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG

            self.args = get_cfg(DEFAULT_CFG)
        return KDPoseLoss(self)


class KDPoseTrainer(PoseTrainer):
    """Ultralytics PoseTrainer extension that injects KD settings into the model."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[dict[str, Any]] = None,
        _callbacks=None,
        teacher_model: Optional[torch.nn.Module] = None,
        kd_cfg: Optional[KDLossConfig] = None,
    ):
        self._teacher_model = teacher_model
        self._kd_cfg = kd_cfg or KDLossConfig()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KDPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)

        # Use __dict__ to avoid registering teacher as a trainable submodule.
        model.__dict__["kd_teacher"] = self._teacher_model
        model.__dict__["kd_loss_config"] = self._kd_cfg
        return model


class KDDetectTrainer(DetectionTrainer):
    """Ultralytics DetectionTrainer extension that injects KD settings into the model."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[dict[str, Any]] = None,
        _callbacks=None,
        teacher_model: Optional[torch.nn.Module] = None,
        kd_cfg: Optional[KDLossConfig] = None,
    ):
        self._teacher_model = teacher_model
        self._kd_cfg = kd_cfg or KDLossConfig()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KDDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)

        # Use __dict__ to avoid registering teacher as a trainable submodule.
        model.__dict__["kd_teacher"] = self._teacher_model
        model.__dict__["kd_loss_config"] = self._kd_cfg
        return model



def _resolve_teacher_pt_path(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".pt":
        return path

    if path.is_dir():
        candidates = [
            path / "weights" / "best.pt",
            path / "best.pt",
            path / "weights" / "last.pt",
            path / "last.pt",
        ]
        for cand in candidates:
            if cand.exists():
                return cand

        pts = sorted(path.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]

    raise FileNotFoundError(f"Teacher .pt not found from path: {path}")



def load_teacher_model(teacher_path: Path) -> torch.nn.Module:
    """Load teacher model as an Ultralytics-compatible PyTorch module."""
    resolved = _resolve_teacher_pt_path(teacher_path)
    model, _ = attempt_load_one_weight(str(resolved), device="cpu", inplace=True, fuse=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logging.info("[KD] Teacher loaded from %s", resolved)
    return model
