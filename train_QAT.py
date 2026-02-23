# Source File: train_QAT.py
import os
import sys
import math
import random

# Force TF2 Keras mode BEFORE any imports - MUST BE FIRST
if 'TF_USE_LEGACY_KERAS' not in os.environ:
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'tensorflow'

import logging
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np

# ==============================================================================
# 1. Environment Setup (CRITICAL: Must be first)
# ==============================================================================
from QAT_Refactored.utils.env_setup import setup_environment, check_tf_version
setup_environment()

# ==============================================================================
# 2. Lazy Imports (Safe after env setup)
# ==============================================================================
import tensorflow as tf
from QAT_Refactored.config.config import cfg, AppConfig
from QAT_Refactored.utils.device import enable_gpu_mem_growth, setup_mixed_precision
from QAT_Refactored.utils.system import install_interrupt_handlers
from QAT_Refactored.utils.loading import try_load_keras_model
from QAT_Refactored.utils.checks import validate_data_access
from QAT_Refactored.data.pipeline import DataPipeline
from QAT_Refactored.data.ultralytics_bridge import build_ultralytics_pose_data
from QAT_Refactored.models.builder import build_student_qat
from QAT_Refactored.core.engine import Trainer
from QAT_Refactored.core.exporter import Exporter

def initialize_system(config: AppConfig) -> None:
    """Performs all hardware and configuration checks."""
    logging.info("="*60)
    logging.info("[Init] System Initialization")
    logging.info("="*60)
    
    check_tf_version()
    set_global_reproducibility(config)
    enable_gpu_mem_growth()
    
    logging.info("[Init] Validating configuration...")
    config.validate()
    
    logging.info("[Init] Validating data access...")
    validate_data_access(config)

    enforce_qat_precision_policy(config)
    setup_mixed_precision(config.USE_AMP)
    logging.info("[Init] System Ready.\n")

def set_global_reproducibility(config: AppConfig) -> None:
    """Sets global RNG states and deterministic runtime switches."""
    logging.info(f"[Init] Setting global seed to {config.SEED}")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.keras.utils.set_random_seed(config.SEED)

    if os.environ.get("TF_DETERMINISTIC_OPS", "0") == "1":
        logging.warning(
            "[Init] Detected TF_DETERMINISTIC_OPS=1 in environment. "
            "QAT FakeQuant gradients on GPU are not deterministic-safe; forcing TF_DETERMINISTIC_OPS=0."
        )
        os.environ["TF_DETERMINISTIC_OPS"] = "0"

    if config.DETERMINISTIC:
        logging.warning(
            "[Init] DETERMINISTIC=True requested, but QAT FakeQuant gradients are not fully "
            "deterministic-safe on GPU. Forcing DETERMINISTIC=False to avoid runtime UNIMPLEMENTED errors."
        )
        config.DETERMINISTIC = False
        os.environ["TF_DETERMINISTIC_OPS"] = "0"

def enforce_qat_precision_policy(config: AppConfig) -> None:
    """
    TFMOT QAT fake-quant ops require float32 inputs. Mixed-float16 causes
    FakeQuantWithMinMaxVars type mismatch during quantize_apply.
    """
    if config.USE_AMP:
        logging.warning("[Init] USE_AMP=True is incompatible with TFMOT QAT. Forcing USE_AMP=False.")
        config.USE_AMP = False

def prepare_data(
    config: AppConfig,
) -> Tuple[Any, Optional[Any], int, int, Optional[np.ndarray], Any]:
    """Build datasets with Ultralytics parity backend (preferred) and native fallback."""
    data_backend = str(config.DATA_BACKEND).lower()

    if data_backend == "ultralytics" and config.DATA_YAML is not None:
        logging.info("[Data] Building Ultralytics parity pipeline...")
        bundle = build_ultralytics_pose_data(config)
        class_weights = None
        if config.USE_CLASS_WEIGHTS:
            logging.warning(
                "[Data] Class weights are disabled for Ultralytics backend parity "
                "(Ultralytics default training does not use external class weights)."
            )

        logging.info(
            "[Data] Ultralytics Train: %d imgs (%d steps/epoch)",
            bundle.num_train,
            bundle.steps_per_epoch,
        )
        logging.info(
            "[Data] Ultralytics Val:   %d imgs (%d steps/epoch)",
            bundle.num_val,
            bundle.val_steps,
        )
        return (
            bundle.train_ds,
            bundle.val_ds,
            bundle.steps_per_epoch,
            bundle.val_steps,
            class_weights,
            bundle.rep_dataset_gen,
        )

    # Native fallback path
    logging.info("[Data] Building native TensorFlow pipeline...")
    pipeline = DataPipeline(config)

    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = pipeline.compute_class_weights()
    else:
        logging.info("[Data] Class weights disabled.")

    ds_train, ds_val, num_train, num_val = pipeline.get_train_val_datasets()

    if config.BATCH_SIZE <= 0:
        raise ValueError(f"Batch size must be > 0. Got {config.BATCH_SIZE}")

    if config.TRAIN_DROP_REMAINDER:
        steps_per_epoch = max(1, num_train // config.BATCH_SIZE)
    else:
        steps_per_epoch = max(1, math.ceil(num_train / config.BATCH_SIZE))
    val_steps = max(1, math.ceil(num_val / config.BATCH_SIZE)) if num_val > 0 else 0

    logging.info(f"[Data] Train: {num_train} imgs ({steps_per_epoch} steps/epoch)")
    logging.info(f"[Data] Val:   {num_val} imgs ({val_steps} steps/epoch)")
    return ds_train, ds_val, steps_per_epoch, val_steps, class_weights, pipeline.get_rep_dataset_gen()

def construct_models(config: AppConfig) -> Tuple[tf.keras.Model, Optional[tf.keras.Model]]:
    """Builds Student and optionally loads Teacher."""
    logging.info("\n[Model] Constructing Architecture...")
    
    # A. Load Teacher (Distillation)
    teacher_model = None
    if config.TRAIN_SUPERVISION == 'distill':
        if not config.EXPORTED_TEACHER_DIR:
             raise ValueError("Distillation mode requires EXPORTED_TEACHER_DIR")
             
        logging.info(f"[Model] Loading teacher from {config.EXPORTED_TEACHER_DIR}...")
        teacher_model, _ = try_load_keras_model(config.EXPORTED_TEACHER_DIR)
        teacher_model.trainable = False
        logging.info("[Model] Teacher loaded.")

    # B. Build Student (QAT)
    qat_model = build_student_qat(config)
    
    # C. Resume Weights
    if config.RESUME_WEIGHTS and config.RESUME_WEIGHTS.exists():
        logging.info(f"[Model] Resuming weights from: {config.RESUME_WEIGHTS}")
        try:
            # skip_mismatch=True allows loading partial weights if architecture slightly changed
            qat_model.load_weights(str(config.RESUME_WEIGHTS), by_name=True, skip_mismatch=True)
            logging.info("[Model] Weights restored.")
        except Exception as e:
            logging.warning(f"[Model] Failed to load resume weights: {e}")

    qat_model.summary(line_length=100, print_fn=logging.info)
    return qat_model, teacher_model

def run_export(trainer: Trainer, rep_dataset_gen: Any, config: AppConfig) -> None:
    """Handles the export to SavedModel and TFLite."""
    logging.info("\n" + "="*60)
    logging.info("[Export] Starting Export Process")
    logging.info("="*60)
    
    exporter = Exporter(config)
    run_dir = trainer.models_dir
    saved_model_path = run_dir / "saved_model"
    tflite_path = run_dir / f"model_quant_{config.TFLITE_QUANT_MODE}.tflite"

    # Load Best EMA Weights
    best_weight_path = run_dir / "best_ema.weights.h5"
    if best_weight_path.exists():
        logging.info(f"[Export] Loading best EMA weights: {best_weight_path}")
        try:
            trainer.student.load_weights(str(best_weight_path))
        except Exception as e:
            logging.warning(f"[Export] Failed to load best EMA weights: {e}")
            logging.warning("[Export] Proceeding with current weights.")
    else:
        logging.warning("[Export] Best EMA weights not found. Using current weights.")

    # Export SavedModel
    exporter.export_saved_model(trainer.student, saved_model_path)

    # Convert to TFLite
    logging.info("[Export] Generating Representative Dataset...")
    rep_gen = rep_dataset_gen

    exporter.convert_to_tflite(
        saved_model_path=saved_model_path,
        output_path=tflite_path,
        rep_dataset_gen=rep_gen
    )
    
    logging.info(f"[Export] TFLite saved to: {tflite_path}")


def _normalize_quant_mode(mode: str) -> str:
    norm = str(mode).lower()
    if norm in {"fp32", "float32"}:
        return "fp32"
    if norm in {"fp16", "float16"}:
        return "fp16"
    if norm == "int8":
        return "int8"
    raise ValueError(f"Unsupported TFLITE_QUANT_MODE: {mode}")


def _build_ultralytics_train_overrides(config: AppConfig) -> dict[str, Any]:
    if config.DATA_YAML is None:
        raise ValueError("DATA_YAML is required for TRAIN_ENGINE=ultralytics")

    run_dir = Path(config.OUTPUT_DIR)
    project = run_dir.parent if run_dir.parent != Path("") else Path(".")

    overrides: dict[str, Any] = {
        "model": str(config.ULTRA_MODEL),
        "data": str(config.DATA_YAML),
        "task": str(config.ULTRA_TASK),
        "epochs": int(config.EPOCHS),
        "batch": int(config.BATCH_SIZE),
        "imgsz": int(config.IMGSZ),
        "device": str(config.ULTRA_DEVICE),
        "workers": int(config.ULTRA_WORKERS),
        "cache": bool(config.ULTRA_CACHE),
        "project": str(project),
        "name": str(run_dir.name or config.ULTRA_NAME),
        "exist_ok": bool(config.ULTRA_EXIST_OK),
        "resume": bool(config.ULTRA_RESUME),
        "cos_lr": bool(config.ULTRA_COS_LR),
        "amp": bool(config.ULTRA_AMP),
        "seed": int(config.ULTRA_SEED),
        "deterministic": bool(config.ULTRA_DETERMINISTIC),
        "fliplr": float(config.ULTRA_FLIPLR),
        "flipud": float(config.ULTRA_FLIPUD),
        "hsv_h": float(config.ULTRA_HSV_H),
        "hsv_s": float(config.ULTRA_HSV_S),
        "hsv_v": float(config.ULTRA_HSV_V),
        "mosaic": float(config.ULTRA_MOSAIC),
        "mixup": float(config.ULTRA_MIXUP),
        "copy_paste": float(config.ULTRA_COPY_PASTE),
        "erasing": float(config.ULTRA_ERASING),
        "fraction": float(config.ULTRA_FRACTION),
        "close_mosaic": int(config.ULTRA_CLOSE_MOSAIC),
    }
    if config.ULTRA_OPTIMIZER is not None:
        overrides["optimizer"] = str(config.ULTRA_OPTIMIZER)
    if config.ULTRA_LR0 is not None:
        overrides["lr0"] = float(config.ULTRA_LR0)
    if config.ULTRA_LRF is not None:
        overrides["lrf"] = float(config.ULTRA_LRF)
    if config.ULTRA_MOMENTUM is not None:
        overrides["momentum"] = float(config.ULTRA_MOMENTUM)
    if config.ULTRA_WEIGHT_DECAY is not None:
        overrides["weight_decay"] = float(config.ULTRA_WEIGHT_DECAY)
    return overrides


def _export_ultralytics_tflite(best_pt: Path, config: AppConfig) -> tuple[Path, list[str]]:
    from ultralytics import YOLO

    from QAT_Refactored.core.ultralytics_route2 import _create_int8_export_aliases

    quant_mode = _normalize_quant_mode(config.TFLITE_QUANT_MODE)
    export_data = config.ULTRA_EXPORT_DATA or config.DATA_YAML
    export_kwargs: dict[str, Any] = {
        "format": "tflite",
        "imgsz": int(config.IMGSZ),
        "data": str(export_data),
        "fraction": float(config.ULTRA_EXPORT_FRACTION),
        "nms": bool(config.ULTRA_NMS_EXPORT),
    }
    if quant_mode == "int8":
        export_kwargs["int8"] = True
    elif quant_mode == "fp16":
        export_kwargs["half"] = True

    prev_export_date = os.environ.get("ULTRALYTICS_EXPORT_DATE")
    if config.ULTRA_EXPORT_DATE:
        os.environ["ULTRALYTICS_EXPORT_DATE"] = str(config.ULTRA_EXPORT_DATE)
    else:
        os.environ.pop("ULTRALYTICS_EXPORT_DATE", None)
    try:
        export_model = YOLO(str(best_pt), task=str(config.ULTRA_TASK))
        export_path = Path(str(export_model.export(**export_kwargs)))
    finally:
        if prev_export_date is None:
            os.environ.pop("ULTRALYTICS_EXPORT_DATE", None)
        else:
            os.environ["ULTRALYTICS_EXPORT_DATE"] = prev_export_date

    aliases = _create_int8_export_aliases(export_path) if quant_mode == "int8" else []
    return export_path, aliases


def _resolve_best_or_last_ckpt(best_path: str, last_path: str) -> Path:
    best_pt = Path(best_path)
    if not best_pt.exists():
        best_pt = Path(last_path)
    if not best_pt.exists():
        raise FileNotFoundError(f"No trained checkpoint found at {best_path} or {last_path}")
    return best_pt


def _run_train_qat_ultralytics_original(config: AppConfig) -> None:
    from ultralytics import YOLO

    logging.info("=" * 60)
    logging.info("[train_QAT] Ultralytics OFFICIAL API mode (strict parity)")
    logging.info("=" * 60)

    if str(config.TRAIN_SUPERVISION).lower() == "distill":
        logging.warning(
            "[train_QAT] QAT_LOSS_MODE='original' ignores distillation. "
            "Switch QAT_LOSS_MODE='kd-deploy' to enable KD."
        )

    yolo = YOLO(str(config.ULTRA_MODEL), task=str(config.ULTRA_TASK))
    yolo.train(**_build_ultralytics_train_overrides(config))
    trainer = yolo.trainer

    best_pt = _resolve_best_or_last_ckpt(str(trainer.best), str(trainer.last))
    export_path, aliases = _export_ultralytics_tflite(best_pt, config)

    logging.info("[train_QAT] Train checkpoint: %s", best_pt)
    logging.info("[train_QAT] TFLite export: %s", export_path)
    for alias in aliases:
        logging.info("[train_QAT] TFLite alias: %s", alias)
    logging.info("[train_QAT] Output dir: %s", trainer.save_dir)


def _run_train_qat_ultralytics_kd(config: AppConfig) -> None:
    from QAT_Refactored.core.loss_balancer import LossBalanceConfig
    from QAT_Refactored.core.ultralytics_kd import (
        KDLossConfig,
        KDDetectTrainer,
        KDPoseTrainer,
        load_teacher_model,
    )

    logging.info("=" * 60)
    logging.info("[train_QAT] Ultralytics KD+deploy mode")
    logging.info("=" * 60)

    teacher_model = None
    if config.TRAIN_SUPERVISION == "distill":
        if not config.EXPORTED_TEACHER_DIR:
            raise ValueError("TRAIN_SUPERVISION='distill' requires EXPORTED_TEACHER_DIR")
        try:
            teacher_model = load_teacher_model(Path(config.EXPORTED_TEACHER_DIR))
        except FileNotFoundError:
            if config.AUX_KD_HEAD_LABEL_LOSS:
                logging.warning(
                    "[KD] Teacher checkpoint not found under %s. "
                    "Falling back to AUX_KD_HEAD_LABEL_LOSS behavior.",
                    config.EXPORTED_TEACHER_DIR,
                )
                teacher_model = None
            else:
                raise

    balance_cfg = LossBalanceConfig(
        strategy=str(config.KD_BALANCE_STRATEGY),
        shared_param_group=str(config.KD_BALANCE_SHARED_PARAM_GROUP),
        ema_decay=float(config.KD_BALANCE_EMA_DECAY),
        update_interval=int(config.KD_BALANCE_UPDATE_INTERVAL),
        warmup_steps=int(config.KD_BALANCE_WARMUP_STEPS),
        deploy_ramp_steps=int(config.KD_BALANCE_DEPLOY_RAMP_STEPS),
        min_weight=float(config.KD_BALANCE_MIN_WEIGHT),
        max_weight=float(config.KD_BALANCE_MAX_WEIGHT),
        max_step_change=float(config.KD_BALANCE_MAX_STEP_CHANGE),
        adapt_power=float(config.KD_BALANCE_ADAPT_POWER),
        renorm_sum=float(config.KD_BALANCE_RENORM_SUM),
        eps=float(config.KD_BALANCE_EPS),
        fixed_kd_weight=(
            None
            if config.KD_BALANCE_FIXED_KD_WEIGHT is None
            else float(config.KD_BALANCE_FIXED_KD_WEIGHT)
        ),
    )
    kd_cfg = KDLossConfig(
        temperature=1.0,
        aux_kd_head_label_loss=bool(config.AUX_KD_HEAD_LABEL_LOSS),
        balance=balance_cfg,
        log_interval_steps=int(config.KD_BALANCE_LOG_INTERVAL),
    )
    task = str(config.ULTRA_TASK).strip().lower()
    if task == "pose":
        trainer_cls = KDPoseTrainer
    elif task == "detect":
        trainer_cls = KDDetectTrainer
    else:
        raise ValueError(
            "QAT_LOSS_MODE='kd-deploy' currently supports ULTRA_TASK in {'pose', 'detect'}, "
            f"got {config.ULTRA_TASK!r}."
        )

    trainer = trainer_cls(
        overrides=_build_ultralytics_train_overrides(config),
        teacher_model=teacher_model,
        kd_cfg=kd_cfg,
    )
    trainer.train()

    best_pt = _resolve_best_or_last_ckpt(str(trainer.best), str(trainer.last))
    export_path, aliases = _export_ultralytics_tflite(best_pt, config)

    logging.info("[train_QAT] Train checkpoint: %s", best_pt)
    logging.info("[train_QAT] TFLite export: %s", export_path)
    for alias in aliases:
        logging.info("[train_QAT] TFLite alias: %s", alias)
    logging.info("[train_QAT] Output dir: %s", trainer.save_dir)


def _run_train_qat_ultralytics(config: AppConfig) -> None:
    mode = str(config.QAT_LOSS_MODE).lower()
    if mode == "original":
        _run_train_qat_ultralytics_original(config)
        return
    if mode == "kd-deploy":
        _run_train_qat_ultralytics_kd(config)
        return
    raise ValueError(f"Unsupported QAT_LOSS_MODE: {config.QAT_LOSS_MODE}")

def _apply_config_overrides(config: AppConfig, overrides: dict[str, Any]) -> None:
    """Apply runtime config overrides for scripted integrations (e.g., train_pose switch)."""
    path_fields = {
        "DATA_ROOT",
        "OUTPUT_DIR",
        "EXPORTED_TEACHER_DIR",
        "RESUME_WEIGHTS",
        "DATA_YAML",
        "ULTRA_EXPORT_DATA",
    }
    list_fields = {"TRAIN_PATTERNS", "VAL_PATTERNS"}
    str_fields = {"VAL_PATTERN"}

    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown AppConfig field override: {key}")

        if key in path_fields:
            if value is None:
                setattr(config, key, None)
            else:
                setattr(config, key, Path(value))
            continue

        if key in list_fields:
            if value is None:
                setattr(config, key, [])
            elif isinstance(value, (list, tuple)):
                setattr(config, key, [str(v) for v in value])
            else:
                raise TypeError(f"{key} must be list/tuple, got {type(value).__name__}")
            continue

        if key in str_fields:
            setattr(config, key, None if value is None else str(value))
            continue

        setattr(config, key, value)

    # Recompute derived fields (e.g., EXPORT_INPUT_SHAPE) and ensure output directory exists.
    config.__post_init__()


def run_train_qat(config_overrides: Optional[dict[str, Any]] = None) -> None:
    """
    Programmatic entrypoint for TensorFlow QAT training.
    Can be called from train_pose.py when switching to KD+deploy loss mode.
    """
    if config_overrides:
        _apply_config_overrides(cfg, config_overrides)

    train_engine = str(cfg.TRAIN_ENGINE).lower()
    if train_engine == "ultralytics":
        initialize_system(cfg)
        _run_train_qat_ultralytics(cfg)
        return

    # 1. System Init
    initialize_system(cfg)

    # 2. Data
    ds_train, ds_val, steps, val_steps, cls_weights, rep_dataset_gen = prepare_data(cfg)

    # 3. Models
    student_model, teacher_model = construct_models(cfg)

    # 4. Training
    logging.info("\n[Train] Initializing Trainer...")
    trainer = Trainer(cfg, student_model=student_model, teacher_model=teacher_model)
    install_interrupt_handlers(trainer)

    trainer.run(
        train_ds=ds_train,
        val_ds=ds_val,
        steps_per_epoch=steps,
        val_steps=val_steps,
        class_weights=cls_weights
    )

    # 5. Export
    # Create a fresh pipeline instance for export generation (stateless safe)
    run_export(trainer, rep_dataset_gen, cfg)

    logging.info("\n[train_QAT] All processes completed successfully.")
    logging.info(f"[Output] Results: {cfg.OUTPUT_DIR}")


def main() -> None:
    try:
        run_train_qat()
    except KeyboardInterrupt:
        logging.warning("\n[train_QAT] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"\n[CRITICAL ERROR] {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
