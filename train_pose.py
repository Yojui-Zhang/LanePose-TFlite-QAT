from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from QAT_Refactored.core.ultralytics_route2 import (
    UltralyticsExportConfig,
    UltralyticsRoute2Runner,
    UltralyticsTrainConfig,
)


def _parse_imgsz(values: Sequence[int]) -> tuple[int, int]:
    if len(values) == 1:
        side = int(values[0])
        if side <= 0:
            raise ValueError(f"imgsz must be > 0, got {side}")
        return (side, side)
    if len(values) == 2:
        h, w = int(values[0]), int(values[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"imgsz values must be > 0, got {(h, w)}")
        return (h, w)
    raise ValueError("imgsz accepts one value (square) or two values (h w)")


def _normalize_data_entries(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        s = str(value).strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    raise ValueError(f"Invalid {field_name} in data yaml: expected str/list, got {type(value).__name__}")


def _read_qat_patterns_from_data_yaml(data_yaml: str) -> tuple[list[str], list[str], dict[str, Any]]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to parse --data yaml for kd-deploy mode.") from exc

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset yaml must be a mapping, got {type(raw).__name__}")

    root: Path | None = None
    path_value = raw.get("path")
    if path_value:
        root = Path(str(path_value))
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()

    def _resolve_entry(entry: str) -> list[str]:
        src = Path(entry)
        if not src.is_absolute():
            base = root if root is not None else yaml_path.parent
            src = base / src
        src = src.expanduser()

        if src.suffix.lower() == ".txt" and src.exists():
            out: list[str] = []
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (src.parent / p).resolve()
                out.append(str(p))
            return out

        return [str(src)]

    train_entries = _normalize_data_entries(raw.get("train"), "train")
    val_entries = _normalize_data_entries(raw.get("val"), "val")

    train_patterns: list[str] = []
    for item in train_entries:
        train_patterns.extend(_resolve_entry(item))

    val_patterns: list[str] = []
    for item in val_entries:
        val_patterns.extend(_resolve_entry(item))

    return train_patterns, val_patterns, raw


def _build_kd_deploy_overrides(args: argparse.Namespace, imgsz: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    if imgsz[0] != imgsz[1]:
        raise ValueError("kd-deploy mode requires square --imgsz because train_QAT uses single IMGSZ.")

    train_patterns, val_patterns, data_yaml = _read_qat_patterns_from_data_yaml(args.data)
    if not train_patterns:
        raise ValueError("No training patterns found in --data yaml for kd-deploy mode.")

    inferred_task = "pose" if isinstance(data_yaml.get("kpt_shape"), (list, tuple)) else "detect"
    requested_task = str(args.task).strip().lower() if args.task else inferred_task
    if requested_task not in {"pose", "detect"}:
        raise ValueError(
            "kd-deploy mode currently supports --task pose or --task detect, "
            f"got {requested_task!r}."
        )

    balance_strategy = str(args.qat_balance_strategy).strip().lower()
    if balance_strategy not in {"grad_norm", "dwa", "ratio"}:
        raise ValueError(f"Unsupported --qat-balance-strategy: {balance_strategy}")

    balance_shared_group = str(args.qat_balance_shared_group).strip().lower()
    if balance_shared_group not in {"head", "all"}:
        raise ValueError(f"Unsupported --qat-balance-shared-group: {balance_shared_group}")

    balance_ema_decay = float(args.qat_balance_ema_decay)
    if not (0.0 <= balance_ema_decay < 1.0):
        raise ValueError(f"qat-balance-ema-decay must be in [0,1), got {balance_ema_decay}")

    balance_update_interval = int(args.qat_balance_update_interval)
    if balance_update_interval < 1:
        raise ValueError(f"qat-balance-update-interval must be >= 1, got {balance_update_interval}")

    balance_warmup_steps = int(args.qat_balance_warmup_steps)
    if balance_warmup_steps < 0:
        raise ValueError(f"qat-balance-warmup-steps must be >= 0, got {balance_warmup_steps}")

    balance_deploy_ramp_steps = int(args.qat_balance_deploy_ramp_steps)
    if balance_deploy_ramp_steps < 0:
        raise ValueError(
            f"qat-balance-deploy-ramp-steps must be >= 0, got {balance_deploy_ramp_steps}"
        )

    balance_min = float(args.qat_balance_min)
    balance_max = float(args.qat_balance_max)
    if balance_min <= 0.0:
        raise ValueError(f"qat-balance-min must be > 0, got {balance_min}")
    if balance_max < balance_min:
        raise ValueError(f"qat-balance-max must be >= qat-balance-min, got {balance_max} < {balance_min}")

    balance_max_step_change = float(args.qat_balance_max_step_change)
    if balance_max_step_change < 1.0:
        raise ValueError(
            f"qat-balance-max-step-change must be >= 1.0, got {balance_max_step_change}"
        )

    balance_adapt_power = float(args.qat_balance_adapt_power)
    if balance_adapt_power <= 0.0:
        raise ValueError(f"qat-balance-adapt-power must be > 0, got {balance_adapt_power}")

    balance_renorm_sum = float(args.qat_balance_renorm_sum)
    if balance_renorm_sum <= 0.0:
        raise ValueError(f"qat-balance-renorm-sum must be > 0, got {balance_renorm_sum}")

    balance_eps = float(args.qat_balance_eps)
    if balance_eps <= 0.0:
        raise ValueError(f"qat-balance-eps must be > 0, got {balance_eps}")

    teacher_dir: Path | None = None
    if args.qat_teacher_exported_dir:
        teacher_dir = Path(args.qat_teacher_exported_dir)
        if not teacher_dir.exists():
            raise FileNotFoundError(f"Teacher exported dir not found: {teacher_dir}")

    export_data: Path | None = None
    if args.export_data:
        export_data = Path(args.export_data)
        if not export_data.exists():
            raise FileNotFoundError(f"Export data yaml not found: {export_data}")

    export_fraction = float(args.export_fraction)
    if not (0.0 < export_fraction <= 1.0):
        raise ValueError(f"export-fraction must be in (0,1], got {export_fraction}")

    seed = int(args.seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    close_mosaic = int(args.close_mosaic)
    if close_mosaic < 0:
        raise ValueError(f"close-mosaic must be >= 0, got {close_mosaic}")

    optimizer: str | None = None
    if args.optimizer is not None:
        optimizer = str(args.optimizer).strip()
        if not optimizer:
            raise ValueError("optimizer must be non-empty when provided")

    lr0: float | None = None
    if args.lr0 is not None:
        lr0 = float(args.lr0)
        if lr0 <= 0.0:
            raise ValueError(f"lr0 must be > 0, got {lr0}")

    lrf: float | None = None
    if args.lrf is not None:
        lrf = float(args.lrf)
        if lrf <= 0.0:
            raise ValueError(f"lrf must be > 0, got {lrf}")

    momentum: float | None = None
    if args.momentum is not None:
        momentum = float(args.momentum)
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"momentum must be in [0,1], got {momentum}")

    weight_decay: float | None = None
    if args.weight_decay is not None:
        weight_decay = float(args.weight_decay)
        if weight_decay < 0.0:
            raise ValueError(f"weight-decay must be >= 0, got {weight_decay}")

    aux_kd_head_label_loss = bool(args.qat_aux_kd_head_label_loss)
    if teacher_dir is None and not aux_kd_head_label_loss:
        # Keep KD branch active when no teacher is provided.
        aux_kd_head_label_loss = True

    # Distillation requires a teacher path in AppConfig validation.
    # For teacher-free KD mode, fall back to label supervision and keep
    # KD branch active via AUX_KD_HEAD_LABEL_LOSS.
    train_supervision = "distill" if teacher_dir is not None else "label"

    quant_mode = "int8" if args.export_int8 else "fp16" if args.export_half else "fp32"

    overrides: dict[str, Any] = {
        "IMGSZ": int(imgsz[0]),
        "BATCH_SIZE": int(args.batch),
        "EPOCHS": int(args.epochs),
        "TRAIN_ENGINE": "ultralytics",
        "QAT_LOSS_MODE": "kd-deploy",
        "DATA_BACKEND": "ultralytics",
        "DATA_YAML": str(Path(args.data).resolve()),
        "ULTRA_MODEL": str(args.model),
        "ULTRA_TASK": requested_task,
        "ULTRA_DEVICE": str(args.device),
        "ULTRA_NAME": f"{args.name}_qat",
        "ULTRA_EXIST_OK": (not args.strict_run),
        "ULTRA_RESUME": bool(args.resume),
        "ULTRA_COS_LR": (not args.no_cos_lr),
        "ULTRA_AMP": (not args.no_amp),
        "ULTRA_SEED": seed,
        "ULTRA_DETERMINISTIC": (not args.non_deterministic),
        "ULTRA_EXPORT_DATE": os.environ.get("ULTRALYTICS_EXPORT_DATE"),
        "ULTRA_EXPORT_DATA": export_data,
        "ULTRA_EXPORT_FRACTION": export_fraction,
        "ULTRA_NMS_EXPORT": bool(args.export_nms),
        "ULTRA_WORKERS": int(args.workers),
        "ULTRA_CACHE": bool(args.cache),
        "ULTRA_CLOSE_MOSAIC": close_mosaic,
        "ULTRA_OPTIMIZER": optimizer,
        "ULTRA_LR0": lr0,
        "ULTRA_LRF": lrf,
        "ULTRA_MOMENTUM": momentum,
        "ULTRA_WEIGHT_DECAY": weight_decay,
        "ULTRA_FLIPLR": float(args.fliplr),
        "TRAIN_PATTERNS": train_patterns,
        "VAL_PATTERNS": val_patterns,
        "TRAIN_SUPERVISION": train_supervision,
        "EXPORTED_TEACHER_DIR": teacher_dir,
        "AUX_KD_HEAD_LABEL_LOSS": aux_kd_head_label_loss,
        "KD_BALANCE_STRATEGY": balance_strategy,
        "KD_BALANCE_SHARED_PARAM_GROUP": balance_shared_group,
        "KD_BALANCE_EMA_DECAY": balance_ema_decay,
        "KD_BALANCE_UPDATE_INTERVAL": balance_update_interval,
        "KD_BALANCE_WARMUP_STEPS": balance_warmup_steps,
        "KD_BALANCE_DEPLOY_RAMP_STEPS": balance_deploy_ramp_steps,
        "KD_BALANCE_MIN_WEIGHT": balance_min,
        "KD_BALANCE_MAX_WEIGHT": balance_max,
        "KD_BALANCE_MAX_STEP_CHANGE": balance_max_step_change,
        "KD_BALANCE_ADAPT_POWER": balance_adapt_power,
        "KD_BALANCE_RENORM_SUM": balance_renorm_sum,
        "KD_BALANCE_EPS": balance_eps,
        "OUTPUT_DIR": (Path(args.project) / f"{args.name}_qat"),
        "TFLITE_QUANT_MODE": quant_mode,
        "USE_AMP": False,
    }

    nc = data_yaml.get("nc")
    if nc is not None:
        overrides["NUM_CLS"] = int(nc)

    kpt_shape = data_yaml.get("kpt_shape")
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 2:
        overrides["NUM_KPT"] = int(kpt_shape[0])
        overrides["KPT_VALS"] = int(kpt_shape[1])

    balance_info = {
        "strategy": balance_strategy,
        "shared_group": balance_shared_group,
        "ema_decay": balance_ema_decay,
        "update_interval": balance_update_interval,
        "warmup_steps": balance_warmup_steps,
        "deploy_ramp_steps": balance_deploy_ramp_steps,
        "min_weight": balance_min,
        "max_weight": balance_max,
    }
    return overrides, balance_info


def _run_kd_deploy_mode(args: argparse.Namespace, imgsz: tuple[int, int]) -> dict[str, Any]:
    from train_QAT import run_train_qat

    overrides, balance_info = _build_kd_deploy_overrides(args, imgsz)
    run_train_qat(config_overrides=overrides)
    return {
        "mode": "kd-deploy",
        "balance": balance_info,
        "output_dir": str(overrides["OUTPUT_DIR"]),
        "quant_mode": str(overrides["TFLITE_QUANT_MODE"]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route 2 trainer: Ultralytics train/fine-tune from .pt/.yaml with optional TFLite export.",
    )
    parser.add_argument(
        "--ultralytics-root",
        type=str,
        default="./ultralytics",
        help="Local Ultralytics source root.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cfg/models/v8/yolov8-pose.yaml",
        help="Model source: .pt, .yaml, or built-in model name.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./dataset/lanepose-carkeypoint.yaml",
        help="Dataset yaml path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["detect", "segment", "classify", "pose", "obb"],
        help="Optional explicit task override.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640, 640],
        help="Image size: one value (square) or two values (h w).",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic mode for Ultralytics training.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=0,
        help="Disable mosaic in the last N epochs (Ultralytics close_mosaic).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Ultralytics optimizer override (e.g. SGD/Adam/AdamW), default keeps upstream behavior.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate override.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=None,
        help="Final learning rate factor override.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Momentum override.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override.",
    )
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--project", type=str, default="./runs/pose")
    parser.add_argument("--name", type=str, default="route2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run export.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--no-cos-lr", action="store_true", help="Disable cosine LR.")
    parser.add_argument("--strict-run", action="store_true", help="Set exist_ok=False.")

    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--export-int8", action="store_true")
    parser.add_argument("--export-half", action="store_true")
    parser.add_argument("--export-nms", action="store_true")
    parser.add_argument(
        "--export-data",
        type=str,
        default=None,
        help="Calibration/export dataset yaml; default uses --data.",
    )
    parser.add_argument(
        "--export-fraction",
        type=float,
        default=1.0,
        help="Fraction of export calibration dataset used by Ultralytics INT8 export.",
    )
    parser.add_argument(
        "--qat-loss-mode",
        type=str,
        default="original",
        choices=["original", "kd-deploy"],
        help="Loss strategy for train_pose direction: original Ultralytics or TensorFlow KD+deploy.",
    )
    parser.add_argument(
        "--qat-balance-strategy",
        type=str,
        default="grad_norm",
        choices=["grad_norm", "dwa", "ratio"],
        help="Dynamic deploy/KD balancing strategy in kd-deploy mode.",
    )
    parser.add_argument(
        "--qat-balance-shared-group",
        type=str,
        default="head",
        choices=["head", "all"],
        help="Shared parameter group used by grad-norm balancing.",
    )
    parser.add_argument(
        "--qat-balance-ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for dynamic balance statistics, in [0,1).",
    )
    parser.add_argument(
        "--qat-balance-update-interval",
        type=int,
        default=10,
        help="Update interval (steps) for dynamic balance weights.",
    )
    parser.add_argument(
        "--qat-balance-warmup-steps",
        type=int,
        default=0,
        help="Number of initial steps to keep balance weights unchanged.",
    )
    parser.add_argument(
        "--qat-balance-deploy-ramp-steps",
        type=int,
        default=1000,
        help="Linear ramp steps before deploy weight fully takes effect.",
    )
    parser.add_argument(
        "--qat-balance-min",
        type=float,
        default=0.2,
        help="Lower bound for dynamic deploy/KD weights.",
    )
    parser.add_argument(
        "--qat-balance-max",
        type=float,
        default=5.0,
        help="Upper bound for dynamic deploy/KD weights.",
    )
    parser.add_argument(
        "--qat-balance-max-step-change",
        type=float,
        default=1.2,
        help="Maximum multiplicative change per update step.",
    )
    parser.add_argument(
        "--qat-balance-adapt-power",
        type=float,
        default=0.5,
        help="Adaptation exponent used by dynamic balancing updates.",
    )
    parser.add_argument(
        "--qat-balance-renorm-sum",
        type=float,
        default=2.0,
        help="Target sum for deploy/KD weights after each update.",
    )
    parser.add_argument(
        "--qat-balance-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for dynamic balancing.",
    )
    parser.add_argument(
        "--qat-teacher-exported-dir",
        type=str,
        default=None,
        help="Teacher SavedModel/Keras path for kd-deploy distillation mode.",
    )
    parser.add_argument(
        "--qat-aux-kd-head-label-loss",
        action="store_true",
        help="Enable auxiliary KD-head label loss when teacher is absent in kd-deploy mode.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    imgsz = _parse_imgsz(args.imgsz)

    if args.qat_loss_mode == "kd-deploy":
        if args.skip_train:
            parser.error("--skip-train is not supported with --qat-loss-mode=kd-deploy")
        out = _run_kd_deploy_mode(args, imgsz)
        print("[Route2] training completed.")
        print(f"[Route2] mode: {out['mode']}")
        balance = out["balance"]
        print(
            "[Route2] balance:",
            f"strategy={balance['strategy']}",
            f"shared_group={balance['shared_group']}",
            f"ema_decay={balance['ema_decay']}",
            f"update_interval={balance['update_interval']}",
            f"warmup_steps={balance['warmup_steps']}",
            f"deploy_ramp_steps={balance['deploy_ramp_steps']}",
            f"min={balance['min_weight']}",
            f"max={balance['max_weight']}",
        )
        print(f"[Route2] quant_mode: {out['quant_mode']}")
        print(f"[Route2] output_dir: {out['output_dir']}")
        return

    train_cfg = UltralyticsTrainConfig(
        ultralytics_root=Path(args.ultralytics_root),
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=args.device,
        workers=args.workers,
        amp=not args.no_amp,
        fliplr=args.fliplr,
        cos_lr=not args.no_cos_lr,
        project=Path(args.project),
        name=args.name,
        resume=args.resume,
        exist_ok=not args.strict_run,
        cache=args.cache,
        task=args.task,
        seed=int(args.seed),
        deterministic=not args.non_deterministic,
        close_mosaic=int(args.close_mosaic),
        optimizer=(str(args.optimizer).strip() if args.optimizer is not None else None),
        lr0=(float(args.lr0) if args.lr0 is not None else None),
        lrf=(float(args.lrf) if args.lrf is not None else None),
        momentum=(float(args.momentum) if args.momentum is not None else None),
        weight_decay=(float(args.weight_decay) if args.weight_decay is not None else None),
    )
    export_cfg = UltralyticsExportConfig(
        do_export=args.export_tflite,
        format="tflite",
        int8=args.export_int8,
        half=args.export_half,
        nms=args.export_nms,
        data=args.export_data,
        imgsz=imgsz,
        fraction=float(args.export_fraction),
    )

    runner = UltralyticsRoute2Runner(train_cfg=train_cfg, export_cfg=export_cfg)
    if args.skip_train and not args.export_tflite:
        parser.error("--skip-train requires --export-tflite")
    out = runner.run(skip_train=args.skip_train)

    if args.skip_train:
        print("[Route2] skip_train mode completed.")
    else:
        print("[Route2] training completed.")
    if "export_path" in out:
        print(f"[Route2] tflite: {out['export_path']}")
    if "export_aliases" in out:
        for alias in out["export_aliases"]:
            print(f"[Route2] alias: {alias}")


if __name__ == "__main__":
    main()
