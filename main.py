# Source File: main.py

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

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
from QAT_Refactored.models.builder import build_student_qat
from QAT_Refactored.core.engine import Trainer
from QAT_Refactored.core.exporter import Exporter

def initialize_system(config: AppConfig) -> None:
    """Performs all hardware and configuration checks."""
    logging.info("="*60)
    logging.info("[Init] System Initialization")
    logging.info("="*60)
    
    check_tf_version()
    enable_gpu_mem_growth()
    
    logging.info("[Init] Validating configuration...")
    config.validate()
    
    logging.info("[Init] Validating data access...")
    validate_data_access(config)

    setup_mixed_precision(config.USE_AMP)
    logging.info("[Init] System Ready.\n")

def prepare_data(config: AppConfig) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], int, int, Optional[float]]:
    """Builds data pipeline and calculates steps."""
    logging.info("[Data] Building Pipeline...")
    pipeline = DataPipeline(config)
    
    # Calculate Class Weights (Optional)
    class_weights = pipeline.compute_class_weights()
    # class_weights = None 

    ds_train, ds_val, num_train, num_val = pipeline.get_train_val_datasets()
    
    # Defensive math for steps
    if config.BATCH_SIZE <= 0:
        raise ValueError(f"Batch size must be > 0. Got {config.BATCH_SIZE}")

    steps_per_epoch = max(1, num_train // config.BATCH_SIZE)
    val_steps = max(1, num_val // config.BATCH_SIZE) if num_val > 0 else 0
    
    logging.info(f"[Data] Train: {num_train} imgs ({steps_per_epoch} steps/epoch)")
    logging.info(f"[Data] Val:   {num_val} imgs ({val_steps} steps/epoch)")
    
    return ds_train, ds_val, steps_per_epoch, val_steps, class_weights

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

def run_export(trainer: Trainer, pipeline: DataPipeline, config: AppConfig) -> None:
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
    rep_gen = pipeline.get_rep_dataset_gen()
    
    exporter.convert_to_tflite(
        saved_model_path=saved_model_path,
        output_path=tflite_path,
        rep_dataset_gen=rep_gen
    )
    
    logging.info(f"[Export] TFLite saved to: {tflite_path}")

def main():
    try:
        # 1. System Init
        initialize_system(cfg)
        
        # 2. Data
        ds_train, ds_val, steps, val_steps, cls_weights = prepare_data(cfg)
        
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
        run_export(trainer, DataPipeline(cfg), cfg)
        
        logging.info("\n[Main] All processes completed successfully.")
        logging.info(f"[Output] Results: {cfg.OUTPUT_DIR}")

    except KeyboardInterrupt:
        logging.warning("\n[Main] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"\n[CRITICAL ERROR] {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()