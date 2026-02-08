# File: QAT_Refactored/utils/checks.py
import sys
import os
import logging
import tensorflow as tf
from pathlib import Path
from typing import List, Optional

# Project Imports
from QAT_Refactored.config.config import AppConfig

def check_system_requirements() -> None:
    """
    Verifies critical system dependencies (TensorFlow, GPU, Environment).
    Must be called AFTER setup_environment().
    """
    # 1. Verify Legacy Keras Flag
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        logging.critical("[Check] TF_USE_LEGACY_KERAS is NOT set to '1'. QAT will fail!")
        sys.exit(1)
        
    # 2. Verify TensorFlow Version
    try:
        logging.info(f"[Check] TensorFlow Version: {tf.__version__}")
    except ImportError:
        logging.critical("[Check] TensorFlow is not installed.")
        sys.exit(1)

    # 3. GPU Availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logging.warning("[Check] ⚠️ No GPU detected. Training will be extremely slow.")
    else:
        # Check VRAM (optional, but good for debugging)
        try:
            details = [tf.config.experimental.get_device_details(g) for g in gpus]
            names = [d.get('device_name', 'Unknown') for d in details]
            logging.info(f"[Check] GPUs Detected: {len(gpus)} -> {names}")
        except Exception:
            logging.info(f"[Check] GPUs Detected: {len(gpus)}")

def validate_data_access(cfg: AppConfig) -> None:
    """
    Performs a 'Fail-Fast' check on data directories.
    Ensures the DATA_ROOT and pattern matches actually exist.
    """
    logging.info("[Check] Validating Data Access...")
    
    # 1. Check Root Existence
    root = Path(cfg.DATA_ROOT)
    if not root.exists():
        logging.critical(f"[Check] DATA_ROOT not found: {root.absolute()}")
        logging.critical("Please check 'config.py' or mount your dataset correctly.")
        sys.exit(1) # Strict Fail

    # 2. Check Train Patterns (Sample Check)
    # We check if the glob returns at least ONE file to prevent zero-data training.
    import glob
    total_files = 0
    patterns = cfg.TRAIN_PATTERNS
    
    if not patterns:
        logging.warning("[Check] No TRAIN_PATTERNS defined in config.")
        
    for p in patterns:
        # Resolve path relative to CWD if not absolute
        # Note: Glob patterns might contain wildcards, so we can't use Path.exists()
        matched = glob.glob(str(p))
        if not matched:
            logging.warning(f"[Check] Pattern matched 0 files: {p}")
        else:
            total_files += len(matched)
            
    if total_files == 0:
        logging.critical(f"[Check] No training images found in any defined patterns under {root}.")
        sys.exit(1)
        
    logging.info(f"[Check] Data validation passed. Found ~{total_files} training files.")