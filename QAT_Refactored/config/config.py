import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import logging
import sys

@dataclass
class AppConfig:
    """
    Production-Grade Configuration for YOLOv8-Pose QAT.
    Acts as the single source of truth for both Training (Python) and Inference (C++).
    """
    
    # ===================================================
    # [Critical] Architecture Constants (Must Match C++)
    # ===================================================
    IMGSZ: int = 640
    NUM_CLS: int = 7
    NUM_KPT: int = 15
    KPT_VALS: int = 3  # (x, y, v)
    
    # Strides determine the feature map sizes. 
    # YOLOv8 default: P3(8), P4(16), P5(32)
    STRIDES: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    # Output channel counts for heads
    REG_MAX: int = 16  # DFL channels
    
    # ===================================================
    # Model Capacity & Export
    # ===================================================
    WIDTH_MULT: float = 3.0
    DEPTH_MULT: float = 1.0
    MAX_OBJS: int = 64
    
    # TFLite Export Settings
    TFLITE_QUANT_MODE: str = "int8"   # Options: int8, fp16, fp32
    EXPORT_INPUT_SHAPE: Tuple[int, int, int, int] = field(init=False)
    
    # ===================================================
    # Training Hyperparameters
    # ===================================================
    SEED: int = 42
    VAL_SPLIT: float = 0.05
    BATCH_SIZE: int = 32 # Reduced for testing
    EPOCHS: int = 10 # Reduced for testing
    
    # Optimizer
    BASE_LR: float = 0.001
    END_LR: float = 0.0001
    MOMENTUM: float = 0.937
    NESTEROV: bool = True
    CLIPNORM: float = 1.0
    WARMUP_STEPS: int = 100
    
    # Loss Weights
    W_BOX: float = 7.0
    W_CLS: float = 1.0
    W_KPT_XY: float = 12.0
    W_KPT_V: float = 1.0
    KD_LOSS_WEIGHT: float = 1.0
    DEPLOY_LOSS_WEIGHT: float = 1.0
    
    # ===================================================
    # Paths & System
    # ===================================================
    DATA_ROOT: Path = field(default_factory=lambda: Path("../../../Dataset/"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("./output"))
    
    # Dynamic Paths (Resolved in __post_init__)
    TRAIN_PATTERNS: List[str] = field(default_factory=list)
    VAL_PATTERN: Optional[str] = None
    EXPORTED_TEACHER_DIR: Optional[Path] = None
    RESUME_WEIGHTS: Optional[Path] = None

    # Flags
    USE_AMP: bool = False
    TRAIN_SUPERVISION: str = 'label'  # 'label' or 'distill'
    LETTERBOX_PAD_VALUE: float = 114.0 / 255.0

    def __post_init__(self):
        """Initialization logic."""
        # 1. Enforce Fixed Input Shape for TFLite (Critical for C++ safety)
        self.EXPORT_INPUT_SHAPE = (1, self.IMGSZ, self.IMGSZ, 3)
        
        # 2. Path Validation
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.TRAIN_PATTERNS:
            # Default fallback for testing
            self.TRAIN_PATTERNS = [str(self.DATA_ROOT / "acc_dataset/images/*.jpg")]
            
        logging.info(f"[Config] System initialized with IMGSZ={self.IMGSZ}")
        logging.info(f"[Config] C++ Alignment Check: Total Anchors = {self.get_total_anchors()}")

    def validate(self):
        """
        Explicit validation method called by main.py and verify_install.py.
        Performs sanity checks and raises errors if config is invalid.
        """
        errors = []
        
        # 1. Check Data Root
        if not self.DATA_ROOT.exists():
            errors.append(f"DATA_ROOT does not exist: {self.DATA_ROOT}")
            
        # 2. Check Input Size Divisibility
        max_stride = max(self.STRIDES)
        if self.IMGSZ % max_stride != 0:
            errors.append(f"IMGSZ ({self.IMGSZ}) must be divisible by max stride ({max_stride})")
            
        # 3. Check Batch Size
        if self.BATCH_SIZE < 1:
            errors.append(f"BATCH_SIZE must be >= 1, got {self.BATCH_SIZE}")
            
        # 4. Check Teacher Path (if needed)
        if self.TRAIN_SUPERVISION == 'distill':
            if not self.EXPORTED_TEACHER_DIR:
                errors.append("Distillation mode requires EXPORTED_TEACHER_DIR to be set.")
            elif not Path(self.EXPORTED_TEACHER_DIR).exists():
                errors.append(f"Teacher model path not found: {self.EXPORTED_TEACHER_DIR}")

        if errors:
            logging.critical("="*40)
            logging.critical("CONFIGURATION ERROR(S):")
            for e in errors:
                logging.critical(f"  - {e}")
            logging.critical("="*40)
            raise ValueError("Invalid Configuration. See logs above.")
        else:
            logging.info("[Config] Validation Passed.")

    @property
    def total_output_channels(self) -> int:
        """
        Calculates the total channels in the final output tensor.
        Structure: [Box(4) | Cls(NUM_CLS) | Kpts(NUM_KPT * KPT_VALS)]
        """
        return 4 + self.NUM_CLS + (self.NUM_KPT * self.KPT_VALS)

    def get_total_anchors(self) -> int:
        """
        Calculates the exact number of anchors (NUM_BOXES in C++).
        Formula: Sum((IMGSZ / stride)^2) for all strides.
        Example: 640 / 8 = 80 -> 6400 anchors.
        """
        total = 0
        for s in self.STRIDES:
            grid_size = self.IMGSZ // s
            total += grid_size * grid_size
        return total

    def generate_cpp_header_snippet(self) -> str:
        """
        Generates the C++ config lines for TFlite.h to ensure synchronization.
        """
        code = (
            f"// [Generated by QAT-Re Config]\n"
            f"#define INPUT_WIDTH {self.IMGSZ}\n"
            f"#define INPUT_HEIGHT {self.IMGSZ}\n"
            f"#define NUM_CLASS {self.NUM_CLS}\n"
            f"#define NUM_BOXES {self.get_total_anchors()} // Critical: Must match Python export\n"
            f"#define Keypoint_NUM {self.NUM_KPT}\n"
        )
        return code

# Global Instance
cfg = AppConfig()