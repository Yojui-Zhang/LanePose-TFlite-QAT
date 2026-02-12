from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any, Optional, Sequence

# Keep Ultralytics local settings writable inside workspace (no HOME permission dependency).
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

from ultralytics import YOLO


_SUPPORTED_TASKS = {"detect", "segment", "classify", "pose", "obb"}


def _normalize_imgsz(imgsz: tuple[int, int]) -> int | tuple[int, int]:
    if imgsz[0] == imgsz[1]:
        return imgsz[0]
    return imgsz


def _resolve_model_source(model: str, ultralytics_root: Path) -> str:
    """
    Resolve local model source in a version-tolerant way.
    Accepts:
    - explicit file path (.pt/.yaml)
    - built-in model name (e.g. yolo11n-pose.pt)
    - cfg relative path (e.g. v8/yolov8-pose.yaml)
    """
    raw = Path(model)
    if raw.exists():
        return str(raw)

    candidates: list[Path] = []
    if not raw.is_absolute():
        candidates.extend(
            [
                ultralytics_root / model,
                ultralytics_root / "cfg" / "models" / model,
                ultralytics_root / "ultralytics" / "cfg" / "models" / model,
            ]
        )
    for cand in candidates:
        if cand.exists():
            return str(cand)

    # For hub / named pretrained weights, let Ultralytics resolve remotely.
    if raw.suffix == ".pt":
        return model

    if raw.suffix in {".yaml", ".yml"}:
        raise FileNotFoundError(
            "Model yaml not found. Checked: "
            + ", ".join(str(p) for p in [raw, *candidates])
        )

    return model


def _create_int8_export_aliases(export_path: Path) -> list[str]:
    """
    Create compatibility aliases for INT8 exports.
    Ultralytics currently returns '*_int8.tflite' in this codebase, while some
    downstream scripts/tools expect '*_integer_quant.tflite' style names.
    """
    if not export_path.exists() or export_path.suffix.lower() != ".tflite":
        return []

    suffix = "_int8.tflite"
    if not export_path.name.endswith(suffix):
        return []

    stem_prefix = export_path.name[: -len(suffix)]
    alias_names = [
        f"{stem_prefix}_integer_quant.tflite",
        f"{stem_prefix}_full_integer_quant.tflite",
    ]

    aliases: list[str] = []
    for alias_name in alias_names:
        alias_path = export_path.with_name(alias_name)
        if alias_path != export_path and not alias_path.exists():
            copy2(export_path, alias_path)
        aliases.append(str(alias_path))
    return aliases


@dataclass(frozen=True)
class UltralyticsTrainConfig:
    ultralytics_root: Path
    model: str
    data: str
    epochs: int
    batch: int
    imgsz: tuple[int, int]
    device: str
    workers: int
    amp: bool
    fliplr: float
    cos_lr: bool
    project: Path
    name: str
    resume: bool
    exist_ok: bool
    cache: bool
    task: Optional[str] = None
    seed: int = 0
    deterministic: bool = True
    close_mosaic: int = 0
    optimizer: Optional[str] = None
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None

    def validate(self) -> None:
        errors: list[str] = []

        if self.epochs <= 0:
            errors.append(f"epochs must be > 0, got {self.epochs}")
        if self.batch <= 0:
            errors.append(f"batch must be > 0, got {self.batch}")
        if self.workers < 0:
            errors.append(f"workers must be >= 0, got {self.workers}")
        if self.seed < 0:
            errors.append(f"seed must be >= 0, got {self.seed}")
        if self.close_mosaic < 0:
            errors.append(f"close_mosaic must be >= 0, got {self.close_mosaic}")
        if self.imgsz[0] <= 0 or self.imgsz[1] <= 0:
            errors.append(f"imgsz must be positive, got {self.imgsz}")
        if not (0.0 <= self.fliplr <= 1.0):
            errors.append(f"fliplr must be in [0,1], got {self.fliplr}")
        if self.task is not None and self.task not in _SUPPORTED_TASKS:
            errors.append(
                f"task must be one of {sorted(_SUPPORTED_TASKS)}, got {self.task}"
            )
        if self.optimizer is not None and not str(self.optimizer).strip():
            errors.append("optimizer must be non-empty when provided")
        if self.lr0 is not None and self.lr0 <= 0.0:
            errors.append(f"lr0 must be > 0 when provided, got {self.lr0}")
        if self.lrf is not None and self.lrf <= 0.0:
            errors.append(f"lrf must be > 0 when provided, got {self.lrf}")
        if self.momentum is not None and not (0.0 <= self.momentum <= 1.0):
            errors.append(f"momentum must be in [0,1] when provided, got {self.momentum}")
        if self.weight_decay is not None and self.weight_decay < 0.0:
            errors.append(
                f"weight_decay must be >= 0 when provided, got {self.weight_decay}"
            )

        data_path = Path(self.data)
        if not data_path.exists():
            errors.append(f"data yaml not found: {data_path}")

        if errors:
            raise ValueError("Invalid UltralyticsTrainConfig:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class UltralyticsExportConfig:
    do_export: bool
    format: str
    int8: bool
    half: bool
    nms: bool
    data: Optional[str]
    imgsz: tuple[int, int]
    fraction: float = 1.0

    def validate(self) -> None:
        if not self.do_export:
            return
        if self.format != "tflite":
            raise ValueError(f"Only tflite export is supported in Route 2, got {self.format}")
        if self.data is not None and not Path(self.data).exists():
            raise ValueError(f"export data yaml not found: {self.data}")
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"export fraction must be in (0,1], got {self.fraction}")
        if self.imgsz[0] <= 0 or self.imgsz[1] <= 0:
            raise ValueError(f"imgsz must be positive, got {self.imgsz}")


class UltralyticsRoute2Runner:
    """
    Route 2 workflow:
    1) Train/Fine-tune directly in Ultralytics (.pt source).
    2) Optional TFLite export from trained checkpoint.
    """

    def __init__(self, train_cfg: UltralyticsTrainConfig, export_cfg: UltralyticsExportConfig):
        self.train_cfg = train_cfg
        self.export_cfg = export_cfg
        self.last_export_aliases: list[str] = []
        self.train_cfg.validate()
        self.export_cfg.validate()

    def _build_model(self) -> YOLO:
        source = _resolve_model_source(self.train_cfg.model, self.train_cfg.ultralytics_root)
        kwargs: dict[str, Any] = {}
        if self.train_cfg.task is not None:
            kwargs["task"] = self.train_cfg.task
        return YOLO(source, **kwargs)

    def _train_model(self, model: YOLO) -> Any:
        train_kwargs: dict[str, Any] = {
            "data": self.train_cfg.data,
            "epochs": self.train_cfg.epochs,
            "batch": self.train_cfg.batch,
            "imgsz": _normalize_imgsz(self.train_cfg.imgsz),
            "device": self.train_cfg.device,
            "amp": self.train_cfg.amp,
            "workers": self.train_cfg.workers,
            "fliplr": self.train_cfg.fliplr,
            "cos_lr": self.train_cfg.cos_lr,
            "project": str(self.train_cfg.project),
            "name": self.train_cfg.name,
            "resume": self.train_cfg.resume,
            "exist_ok": self.train_cfg.exist_ok,
            "cache": self.train_cfg.cache,
            "seed": self.train_cfg.seed,
            "deterministic": self.train_cfg.deterministic,
            "close_mosaic": self.train_cfg.close_mosaic,
        }
        if self.train_cfg.optimizer is not None:
            train_kwargs["optimizer"] = str(self.train_cfg.optimizer)
        if self.train_cfg.lr0 is not None:
            train_kwargs["lr0"] = float(self.train_cfg.lr0)
        if self.train_cfg.lrf is not None:
            train_kwargs["lrf"] = float(self.train_cfg.lrf)
        if self.train_cfg.momentum is not None:
            train_kwargs["momentum"] = float(self.train_cfg.momentum)
        if self.train_cfg.weight_decay is not None:
            train_kwargs["weight_decay"] = float(self.train_cfg.weight_decay)

        return model.train(
            **train_kwargs,
        )

    def run(self, *, skip_train: bool = False) -> dict[str, Any]:
        model = self._build_model()
        info: dict[str, Any] = {"train_result": None}
        if not skip_train:
            info["train_result"] = self._train_model(model)

        if self.export_cfg.do_export:
            info["export_path"] = self.export(model)
            if self.last_export_aliases:
                info["export_aliases"] = list(self.last_export_aliases)
        return info

    def train(self) -> dict[str, Any]:
        # Backward-compatible behavior.
        return self.run(skip_train=False)

    def export(self, model: YOLO) -> str:
        export_path = model.export(
            format=self.export_cfg.format,
            imgsz=_normalize_imgsz(self.export_cfg.imgsz),
            int8=self.export_cfg.int8,
            half=self.export_cfg.half,
            nms=self.export_cfg.nms,
            data=self.export_cfg.data or self.train_cfg.data,
            fraction=float(self.export_cfg.fraction),
        )
        export_path_obj = Path(str(export_path))
        self.last_export_aliases = (
            _create_int8_export_aliases(export_path_obj) if self.export_cfg.int8 else []
        )
        return str(export_path_obj)
