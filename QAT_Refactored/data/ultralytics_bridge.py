from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import tensorflow as tf

from QAT_Refactored.config.config import AppConfig

try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover - runtime optional dependency
    torch = None
    torch_f = None


# Keep Ultralytics local settings writable inside workspace.
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))


@dataclass
class UltralyticsTFDataBundle:
    """Container for Ultralytics-backed data adapted to TensorFlow consumers."""

    train_ds: Any
    val_ds: Optional[Any]
    steps_per_epoch: int
    val_steps: int
    num_train: int
    num_val: int
    rep_dataset_gen: Any
    dataset_info: Dict[str, Any]


class _TorchLoaderToTFAdapter:
    """Adapter that yields TensorFlow tensors from an Ultralytics Torch dataloader."""

    def __init__(self, loader: Any, cfg: AppConfig) -> None:
        self.loader = loader
        self.cfg = cfg

    def __iter__(self) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
        for batch in self.loader:
            yield _convert_ultralytics_batch_to_tf(batch, self.cfg)

    def __len__(self) -> int:
        return len(self.loader)



def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/tensor-like values to numpy arrays without assuming a specific backend."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _prepare_images_torch_fast(batch: Dict[str, Any], cfg: AppConfig) -> Optional[np.ndarray]:
    """
    Fast path for Ultralytics torch batches:
    keep resize/transpose in torch, then perform one CPU materialization.
    """
    if torch is None:
        return None

    imgs_t = batch.get("img", None)
    if imgs_t is None or not torch.is_tensor(imgs_t):
        return None
    if imgs_t.ndim != 4:
        raise ValueError(f"Ultralytics batch['img'] must be rank-4, got shape={tuple(imgs_t.shape)}")

    # Ultralytics dataloader commonly emits BCHW.
    if imgs_t.shape[1] != 3 and imgs_t.shape[-1] == 3:
        imgs_t = imgs_t.permute(0, 3, 1, 2)
    if imgs_t.shape[1] != 3:
        raise ValueError(f"Unsupported image layout for torch batch: shape={tuple(imgs_t.shape)}")

    if not torch.is_floating_point(imgs_t):
        imgs_t = imgs_t.float().div_(255.0)
    else:
        imgs_t = imgs_t.float()
        # Preserve legacy normalization behavior when dataloader emits [0,255] float tensors.
        if float(imgs_t.max().item()) > 1.0:
            imgs_t = imgs_t / 255.0

    target = int(cfg.IMGSZ)
    if imgs_t.shape[2] != target or imgs_t.shape[3] != target:
        if torch_f is None:
            return None
        imgs_t = torch_f.interpolate(imgs_t, size=(target, target), mode="bilinear", align_corners=False)

    imgs_t = imgs_t.permute(0, 2, 3, 1).contiguous()
    return imgs_t.detach().cpu().numpy().astype(np.float32, copy=False)



def _prepare_images(batch: Dict[str, Any], cfg: AppConfig) -> np.ndarray:
    imgs_torch = _prepare_images_torch_fast(batch, cfg)
    if imgs_torch is not None:
        return imgs_torch

    imgs = _to_numpy(batch["img"])
    if imgs.ndim != 4:
        raise ValueError(f"Ultralytics batch['img'] must be rank-4, got shape={imgs.shape}")

    # Ultralytics format is usually BCHW uint8.
    if imgs.shape[1] == 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))

    imgs = imgs.astype(np.float32, copy=False)
    if imgs.max(initial=0.0) > 1.0:
        imgs /= 255.0

    target = int(cfg.IMGSZ)
    if imgs.shape[1] != target or imgs.shape[2] != target:
        imgs = tf.image.resize(
            imgs,
            size=(target, target),
            method="bilinear",
            antialias=False,
        ).numpy()
        imgs = imgs.astype(np.float32, copy=False)
    return imgs



def _normalize_keypoints(
    keypoints: Optional[np.ndarray],
    num_targets: int,
    num_kpt: int,
    kpt_vals: int,
) -> Optional[np.ndarray]:
    if num_kpt <= 0:
        return None

    if keypoints is None:
        return np.zeros((num_targets, num_kpt, kpt_vals), dtype=np.float32)

    kp = keypoints.astype(np.float32, copy=False)

    if kp.ndim == 2:
        # Handles empty edge cases like (0, 3) and keeps a consistent rank.
        if num_targets == 0:
            return np.zeros((0, num_kpt, kpt_vals), dtype=np.float32)
        kp = kp.reshape(num_targets, -1, kp.shape[-1])

    if kp.ndim != 3:
        raise ValueError(f"Unexpected keypoints shape from Ultralytics batch: {kp.shape}")

    n, k, v = kp.shape
    if n != num_targets:
        raise ValueError(
            "Ultralytics keypoint row count mismatch: "
            f"num_targets={num_targets}, keypoints_rows={n}"
        )

    out = np.zeros((num_targets, num_kpt, kpt_vals), dtype=np.float32)
    k_use = min(num_kpt, k)
    v_use = min(kpt_vals, v)
    if k_use > 0 and v_use > 0:
        out[:, :k_use, :v_use] = kp[:, :k_use, :v_use]
    return out



def _prepare_padded_labels(batch: Dict[str, Any], cfg: AppConfig, batch_size: int) -> np.ndarray:
    num_kpt = int(cfg.NUM_KPT)
    kpt_vals = int(cfg.KPT_VALS)
    feature_dim = 5 + num_kpt * kpt_vals

    cls = _to_numpy(batch["cls"]).reshape(-1).astype(np.float32, copy=False)
    bboxes = _to_numpy(batch["bboxes"]).astype(np.float32, copy=False)
    batch_idx = _to_numpy(batch["batch_idx"]).reshape(-1).astype(np.int32, copy=False)

    num_targets = int(batch_idx.shape[0])
    if num_targets != int(cls.shape[0]) or num_targets != int(bboxes.shape[0]):
        raise ValueError(
            "Ultralytics target tensor length mismatch: "
            f"batch_idx={batch_idx.shape[0]}, cls={cls.shape[0]}, bboxes={bboxes.shape[0]}"
        )

    keypoints_np: Optional[np.ndarray] = None
    if "keypoints" in batch:
        keypoints_np = _to_numpy(batch["keypoints"])
    keypoints_np = _normalize_keypoints(keypoints_np, num_targets, num_kpt, kpt_vals)

    labels = np.zeros((batch_size, int(cfg.MAX_OBJS), feature_dim), dtype=np.float32)
    if num_targets == 0:
        return labels

    # Preserve per-image target order while avoiding per-target Python loops.
    order = np.argsort(batch_idx, kind="stable")
    sorted_batch = batch_idx[order]

    in_range = (sorted_batch >= 0) & (sorted_batch < batch_size)
    if not np.all(in_range):
        order = order[in_range]
        sorted_batch = sorted_batch[in_range]

    if sorted_batch.size == 0:
        return labels

    counts = np.bincount(sorted_batch, minlength=batch_size).astype(np.int32, copy=False)
    starts = np.zeros_like(counts)
    if counts.size > 1:
        starts[1:] = np.cumsum(counts[:-1], dtype=np.int32)
    local_idx = np.arange(sorted_batch.shape[0], dtype=np.int32) - starts[sorted_batch]

    keep = local_idx < int(cfg.MAX_OBJS)
    dropped = int(local_idx.shape[0] - np.count_nonzero(keep))

    src = order[keep]
    img_slot = sorted_batch[keep]
    obj_slot = local_idx[keep]

    labels[img_slot, obj_slot, 0] = cls[src]
    labels[img_slot, obj_slot, 1:5] = np.clip(bboxes[src], 0.0, 1.0)

    if keypoints_np is not None and num_kpt > 0:
        labels[img_slot, obj_slot, 5:] = keypoints_np[src].reshape(src.shape[0], -1)

    if dropped > 0:
        logging.warning(
            "[Data] Dropped %d targets due to MAX_OBJS=%d limit.",
            dropped,
            int(cfg.MAX_OBJS),
        )

    return labels



def _convert_ultralytics_batch_to_tf(batch: Dict[str, Any], cfg: AppConfig) -> tuple[tf.Tensor, tf.Tensor]:
    imgs_np = _prepare_images(batch, cfg)
    batch_size = int(imgs_np.shape[0])
    labels_np = _prepare_padded_labels(batch, cfg, batch_size)

    imgs_tf = tf.convert_to_tensor(imgs_np, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels_np, dtype=tf.float32)
    return imgs_tf, labels_tf



def _build_rep_dataset_gen(source_loader: Any, cfg: AppConfig) -> Any:
    """Create representative dataset generator from Ultralytics dataloader batches."""

    def gen() -> Iterator[list[np.ndarray]]:
        emitted = 0
        for batch in source_loader:
            imgs = _prepare_images(batch, cfg)
            for i in range(imgs.shape[0]):
                yield [np.expand_dims(imgs[i], axis=0).astype(np.float32, copy=False)]
                emitted += 1
                if emitted >= 100:
                    return

    return gen



def _make_ultralytics_cfg(cfg: AppConfig, task: str) -> Any:
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    overrides = {
        "task": str(task),
        "data": str(cfg.DATA_YAML),
        "imgsz": int(cfg.IMGSZ),
        "batch": int(cfg.BATCH_SIZE),
        "workers": int(cfg.ULTRA_WORKERS),
        "cache": bool(cfg.ULTRA_CACHE),
        "rect": bool(cfg.ULTRA_RECT),
        "fraction": float(cfg.ULTRA_FRACTION),
        "close_mosaic": int(cfg.ULTRA_CLOSE_MOSAIC),
        "fliplr": float(cfg.ULTRA_FLIPLR),
        "flipud": float(cfg.ULTRA_FLIPUD),
        "hsv_h": float(cfg.ULTRA_HSV_H),
        "hsv_s": float(cfg.ULTRA_HSV_S),
        "hsv_v": float(cfg.ULTRA_HSV_V),
        "mosaic": float(cfg.ULTRA_MOSAIC),
        "mixup": float(cfg.ULTRA_MIXUP),
        "copy_paste": float(cfg.ULTRA_COPY_PASTE),
        "erasing": float(cfg.ULTRA_ERASING),
    }
    return get_cfg(DEFAULT_CFG, overrides=overrides)



def build_ultralytics_pose_data(cfg: AppConfig) -> UltralyticsTFDataBundle:
    """Build Ultralytics pose/detect dataloaders and adapt them to TensorFlow tensors."""
    if cfg.DATA_YAML is None:
        raise ValueError("DATA_YAML is required for Ultralytics data backend.")

    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data_info = check_det_dataset(str(cfg.DATA_YAML), autodownload=False)
    requested_task = str(getattr(cfg, "ULTRA_TASK", "pose")).strip().lower()
    if requested_task not in {"pose", "detect"}:
        raise ValueError(
            "Ultralytics TensorFlow bridge currently supports ULTRA_TASK in {'pose', 'detect'}, "
            f"got {cfg.ULTRA_TASK!r}."
        )

    kpt_shape = data_info.get("kpt_shape")
    has_keypoints = isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 2

    effective_task = requested_task
    if requested_task == "pose" and not has_keypoints:
        logging.warning(
            "[Data] ULTRA_TASK='pose' but dataset has no kpt_shape; auto-fallback to detect mode."
        )
        effective_task = "detect"
    cfg.ULTRA_TASK = effective_task
    yolo_cfg = _make_ultralytics_cfg(cfg, effective_task)

    dataset_nc = int(data_info.get("nc", cfg.NUM_CLS))
    if dataset_nc != int(cfg.NUM_CLS):
        logging.warning(
            "[Data] Overriding NUM_CLS from %d to dataset nc=%d for Ultralytics parity.",
            int(cfg.NUM_CLS),
            dataset_nc,
        )
        cfg.NUM_CLS = dataset_nc

    if effective_task == "pose" and has_keypoints:
        kpt_num = int(kpt_shape[0])
        kpt_vals = int(kpt_shape[1])
        if kpt_num != int(cfg.NUM_KPT) or kpt_vals != int(cfg.KPT_VALS):
            logging.warning(
                "[Data] Overriding kpt shape from (%d,%d) to dataset kpt_shape=(%d,%d).",
                int(cfg.NUM_KPT),
                int(cfg.KPT_VALS),
                kpt_num,
                kpt_vals,
            )
            cfg.NUM_KPT = kpt_num
            cfg.KPT_VALS = kpt_vals
    elif effective_task == "detect" and int(cfg.NUM_KPT) != 0:
        logging.warning(
            "[Data] ULTRA_TASK='detect' overrides NUM_KPT from %d to 0 for detection labels.",
            int(cfg.NUM_KPT),
        )
        cfg.NUM_KPT = 0

    stride = max(int(s) for s in cfg.STRIDES) if cfg.STRIDES else 32

    train_dataset = build_yolo_dataset(
        yolo_cfg,
        data_info["train"],
        batch=int(cfg.BATCH_SIZE),
        data=data_info,
        mode="train",
        rect=bool(cfg.ULTRA_RECT),
        stride=stride,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch=int(cfg.BATCH_SIZE),
        workers=int(cfg.ULTRA_WORKERS),
        shuffle=True,
        rank=-1,
        drop_last=bool(cfg.TRAIN_DROP_REMAINDER),
    )

    val_ds: Optional[_TorchLoaderToTFAdapter] = None
    val_loader: Optional[Any] = None
    val_steps = 0
    num_val = 0

    val_path = data_info.get("val") or data_info.get("test")
    if val_path:
        val_dataset = build_yolo_dataset(
            yolo_cfg,
            val_path,
            batch=int(cfg.BATCH_SIZE),
            data=data_info,
            mode="val",
            rect=True,
            stride=stride,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch=int(cfg.BATCH_SIZE),
            workers=(int(cfg.ULTRA_WORKERS) * 2 if int(cfg.ULTRA_WORKERS) > 0 else 0),
            shuffle=False,
            rank=-1,
            drop_last=False,
        )
        val_ds = _TorchLoaderToTFAdapter(val_loader, cfg)
        val_steps = len(val_loader)
        num_val = len(val_dataset)

    train_ds = _TorchLoaderToTFAdapter(train_loader, cfg)
    rep_source = val_loader if val_loader is not None else train_loader

    return UltralyticsTFDataBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        steps_per_epoch=len(train_loader),
        val_steps=val_steps,
        num_train=len(train_dataset),
        num_val=num_val,
        rep_dataset_gen=_build_rep_dataset_gen(rep_source, cfg),
        dataset_info=data_info,
    )
