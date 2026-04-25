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
    SEED: int = 0
    # TFMOT fake-quant gradients on GPU do not fully support deterministic kernels.
    DETERMINISTIC: bool = False
    VAL_SPLIT: float = 0.1
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    
    # Optimizer
    BASE_LR: float = 0.01
    END_LR: float = 0.0001
    MOMENTUM: float = 0.937
    WEIGHT_DECAY: float = 0.0005
    NESTEROV: bool = True
    CLIPNORM: Optional[float] = None
    WARMUP_EPOCHS: float = 3.0
    # If >= 0 this value overrides WARMUP_EPOCHS.
    WARMUP_STEPS: int = -1

    # Why: QAT/小 batch 下 BN moving stats 容易漂；提供顯式 step 控制 freeze 時機。
    # -1: 沿用舊邏輯（最後 10%）
    FREEZE_BN_FROM_STEP: int = -1
    FREEZE_BN_RATIO: float = 0.90


    # Loss Weights
    W_BOX: float = 7.5
    W_CLS: float = 0.5
    W_KPT_XY: float = 12.0
    W_KPT_V: float = 1.0
    KD_LOSS_WEIGHT: float = 1.0
    DEPLOY_LOSS_WEIGHT: float = 1.0
    AUX_KD_HEAD_LABEL_LOSS: bool = False
    KD_BALANCE_STRATEGY: str = "grad_norm"  # grad_norm | dwa | ratio
    KD_BALANCE_SHARED_PARAM_GROUP: str = "head"  # head | all
    KD_BALANCE_EMA_DECAY: float = 0.95
    KD_BALANCE_UPDATE_INTERVAL: int = 10
    KD_BALANCE_WARMUP_STEPS: int = 0
    KD_BALANCE_DEPLOY_RAMP_STEPS: int = 1000
    KD_BALANCE_MIN_WEIGHT: float = 0.2
    KD_BALANCE_MAX_WEIGHT: float = 5.0
    KD_BALANCE_MAX_STEP_CHANGE: float = 1.2
    KD_BALANCE_ADAPT_POWER: float = 0.5
    KD_BALANCE_RENORM_SUM: float = 2.0
    KD_BALANCE_EPS: float = 1e-6
    KD_BALANCE_FIXED_KD_WEIGHT: Optional[float] = None
    KD_BALANCE_LOG_INTERVAL: int = 50

    # KD distillation specifics (Ultralytics kd-deploy mode)
    KD_TEMPERATURE: float = 1.0
    KD_CLS_DISTILL: str = "bce"          # bce | softmax_kl
    KD_DFL_DISTILL: str = "kldiv"        # kldiv | smoothl1
    KD_FG_THRESHOLD: float = 0.0         # [0,1], 0 disables
    KD_FG_TOPK: int = 0                  # 0 disables
    KD_FG_MIN_POS: int = 0               # 0 disables
    KD_FG_APPLY_TO: str = "cls"          # cls | dfl | both
    ULTRA_KD_LOSS_COMPOSITION: str = "dynamic_kd_deploy"  # dynamic_kd_deploy | fixed_kd_deploy | pure_kd
    
    # ===================================================
    # Loss Function Selection
    # ===================================================
    LOSS_TYPE: str = 'ultralytics'  # 'ultralytics' for stability - QAT requires tf-nightly
    
     # ===================================================
     # Paths & System
     # ===================================================
    DATA_ROOT: Path = field(default_factory=lambda: Path("../../../../Dataset"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("./output"))
    
    # Dynamic Paths (Resolved in __post_init__)
    TRAIN_PATTERNS: List[str] = field(default_factory=list)
    VAL_PATTERNS: List[str] = field(default_factory=list)
    VAL_PATTERN: Optional[str] = None
    EXPORTED_TEACHER_DIR: Optional[Path] = None
    RESUME_WEIGHTS: Optional[Path] = None

    # Flags
    # TFMOT FakeQuant requires float32 tensors. Keep AMP off for QAT path.
    USE_AMP: bool = False
    TRAIN_SUPERVISION: str = 'label'  # 'label' or 'distill'
    TRAIN_ENGINE: str = "tf-legacy"  # "ultralytics" or "tf-legacy"
    TF_LEGACY_BACKBONE: str = "yolo-repvgg"  # yolo-repvgg | cira-lite
    TF_CIRA_WIDTH_MULT: float = 0.3
    TF_CIRA_USE_ATTENTION: bool = True
    TF_CIRA_USE_DEFORM: bool = True
    QAT_LOSS_MODE: str = "kd-deploy"  # "original" (strict Ultralytics parity) or "kd-deploy"
    DATA_BACKEND: str = "ultralytics"  # "ultralytics" or "native" "cira"
    DATA_YAML: Optional[Path] = field(default_factory=lambda: Path("./dataset/lanepose-carkeypoint.yaml"))
    ULTRA_BACKBONE: str = "custom"  # custom | yolo | cira
    ULTRA_MODEL: str = "yolo11n-pose.pt"
    ULTRA_MODEL_YOLO_POSE: str = "cfg/models/v8/yolov8-pose.yaml"
    ULTRA_MODEL_YOLO_DETECT: str = "cfg/models/v8/yolov8.yaml"
    ULTRA_MODEL_CIRA_POSE: str = "ultralytics/cfg/models/Yojui/yolov8_CIRA-Pose.yaml"
    ULTRA_MODEL_CIRA_DETECT: str = "ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml"
    ULTRA_TASK: str = "pose"
    ULTRA_DEVICE: str = "0"
    ULTRA_NAME: str = "train_qat"
    ULTRA_EXIST_OK: bool = True
    ULTRA_RESUME: bool = False
    ULTRA_COS_LR: bool = True
    ULTRA_AMP: bool = True
    ULTRA_SEED: int = 0
    ULTRA_DETERMINISTIC: bool = True
    ULTRA_EXPORT_DATE: Optional[str] = None  # set fixed ISO datetime for bitwise-stable export metadata
    ULTRA_EXPORT_DATA: Optional[Path] = None
    ULTRA_EXPORT_FRACTION: float = 1.0
    # Device policy for TensorFlow onnx2tf conversion during Ultralytics export.
    # cpu: force CPU to avoid PyTorch/TF CUDA context conflicts in a shared process.
    # gpu: keep legacy behavior.
    # auto: try GPU first, fallback to CPU on CUDA handle/runtime failures.
    ULTRA_ONNX2TF_DEVICE: str = "cpu"
    ULTRA_NMS_EXPORT: bool = False
    ULTRA_WORKERS: int = 4
    ULTRA_CACHE: bool = False
    ULTRA_RECT: bool = False
    ULTRA_FRACTION: float = 1.0
    ULTRA_CLOSE_MOSAIC: int = 0
    ULTRA_OPTIMIZER: Optional[str] = None
    ULTRA_LR0: Optional[float] = None
    ULTRA_LRF: Optional[float] = None
    ULTRA_MOMENTUM: Optional[float] = None
    ULTRA_WEIGHT_DECAY: Optional[float] = None
    ULTRA_FLIPLR: float = 0.0
    ULTRA_FLIPUD: float = 0.0
    ULTRA_HSV_H: float = 0.015
    ULTRA_HSV_S: float = 0.7
    ULTRA_HSV_V: float = 0.4
    ULTRA_MOSAIC: float = 1.0
    ULTRA_MIXUP: float = 0.0
    ULTRA_COPY_PASTE: float = 0.0
    ULTRA_ERASING: float = 0.4
    USE_CLASS_WEIGHTS: bool = False
    TRAIN_DROP_REMAINDER: bool = False
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

        if not self.VAL_PATTERNS and not self.VAL_PATTERN and self.VAL_SPLIT <= 0.0:
            # Ultralytics train_pose often validates on an explicit val set.
            # If user has not configured one, default to train patterns to keep parity behavior.
            self.VAL_PATTERNS = list(self.TRAIN_PATTERNS)
            
        logging.info(f"[Config] System initialized with IMGSZ={self.IMGSZ}")
        logging.info(f"[Config] C++ Alignment Check: Total Anchors = {self.get_total_anchors()}")

    def validate(self):
        """
        Explicit validation method called by train_QAT.py and verify/verify_install.py.
        Performs sanity checks and raises errors if config is invalid.
        """
        errors = []
        data_backend = str(self.DATA_BACKEND).lower()
        train_engine = str(self.TRAIN_ENGINE).lower()
        
        # 1. Check Data Root
        if not (data_backend == "ultralytics" and self.DATA_YAML is not None) and not self.DATA_ROOT.exists():
            errors.append(f"DATA_ROOT does not exist: {self.DATA_ROOT}")
            
        # 2. Check Input Size Divisibility
        max_stride = max(self.STRIDES)
        if self.IMGSZ % max_stride != 0:
            errors.append(f"IMGSZ ({self.IMGSZ}) must be divisible by max stride ({max_stride})")
            
        # 3. Check Batch Size
        if self.BATCH_SIZE < 1:
            errors.append(f"BATCH_SIZE must be >= 1, got {self.BATCH_SIZE}")

        if self.WEIGHT_DECAY < 0:
            errors.append(f"WEIGHT_DECAY must be >= 0, got {self.WEIGHT_DECAY}")

        if self.WARMUP_EPOCHS < 0:
            errors.append(f"WARMUP_EPOCHS must be >= 0, got {self.WARMUP_EPOCHS}")
        if self.KD_BALANCE_STRATEGY not in {"grad_norm", "dwa", "ratio"}:
            errors.append(
                f"KD_BALANCE_STRATEGY must be one of ['dwa', 'grad_norm', 'ratio'], got {self.KD_BALANCE_STRATEGY}"
            )
        if self.KD_BALANCE_SHARED_PARAM_GROUP not in {"head", "all"}:
            errors.append(
                "KD_BALANCE_SHARED_PARAM_GROUP must be 'head' or 'all', "
                f"got {self.KD_BALANCE_SHARED_PARAM_GROUP}"
            )
        if not (0.0 <= self.KD_BALANCE_EMA_DECAY < 1.0):
            errors.append(f"KD_BALANCE_EMA_DECAY must be in [0,1), got {self.KD_BALANCE_EMA_DECAY}")
        if self.KD_BALANCE_UPDATE_INTERVAL < 1:
            errors.append(
                f"KD_BALANCE_UPDATE_INTERVAL must be >= 1, got {self.KD_BALANCE_UPDATE_INTERVAL}"
            )
        if self.KD_BALANCE_WARMUP_STEPS < 0:
            errors.append(
                f"KD_BALANCE_WARMUP_STEPS must be >= 0, got {self.KD_BALANCE_WARMUP_STEPS}"
            )
        if self.KD_BALANCE_DEPLOY_RAMP_STEPS < 0:
            errors.append(
                "KD_BALANCE_DEPLOY_RAMP_STEPS must be >= 0, "
                f"got {self.KD_BALANCE_DEPLOY_RAMP_STEPS}"
            )
        if self.KD_BALANCE_MIN_WEIGHT < 0.0:
            errors.append(f"KD_BALANCE_MIN_WEIGHT must be >= 0, got {self.KD_BALANCE_MIN_WEIGHT}")
        if self.KD_BALANCE_MAX_WEIGHT < self.KD_BALANCE_MIN_WEIGHT:
            errors.append(
                "KD_BALANCE_MAX_WEIGHT must be >= KD_BALANCE_MIN_WEIGHT, "
                f"got {self.KD_BALANCE_MAX_WEIGHT} < {self.KD_BALANCE_MIN_WEIGHT}"
            )
        if self.KD_BALANCE_MAX_STEP_CHANGE < 1.0:
            errors.append(
                f"KD_BALANCE_MAX_STEP_CHANGE must be >= 1, got {self.KD_BALANCE_MAX_STEP_CHANGE}"
            )
        if self.KD_BALANCE_ADAPT_POWER <= 0.0:
            errors.append(f"KD_BALANCE_ADAPT_POWER must be > 0, got {self.KD_BALANCE_ADAPT_POWER}")
        if self.KD_BALANCE_RENORM_SUM <= 0.0:
            errors.append(f"KD_BALANCE_RENORM_SUM must be > 0, got {self.KD_BALANCE_RENORM_SUM}")
        if self.KD_BALANCE_EPS <= 0.0:
            errors.append(f"KD_BALANCE_EPS must be > 0, got {self.KD_BALANCE_EPS}")
        if self.KD_BALANCE_FIXED_KD_WEIGHT is not None and self.KD_BALANCE_FIXED_KD_WEIGHT < 0.0:
            errors.append(
                "KD_BALANCE_FIXED_KD_WEIGHT must be >= 0 when set, "
                f"got {self.KD_BALANCE_FIXED_KD_WEIGHT}"
            )
        if self.KD_BALANCE_LOG_INTERVAL < 1:
            errors.append(f"KD_BALANCE_LOG_INTERVAL must be >= 1, got {self.KD_BALANCE_LOG_INTERVAL}")

        # 4. Check Teacher Path (if needed)
        if self.TRAIN_SUPERVISION == 'distill':
            if not self.EXPORTED_TEACHER_DIR:
                errors.append("Distillation mode requires EXPORTED_TEACHER_DIR to be set.")
            elif not Path(self.EXPORTED_TEACHER_DIR).exists():
                errors.append(f"Teacher model path not found: {self.EXPORTED_TEACHER_DIR}")

        # 5. Data backend validation
        if train_engine not in {"ultralytics", "tf-legacy"}:
            errors.append(f"TRAIN_ENGINE must be 'ultralytics' or 'tf-legacy', got {self.TRAIN_ENGINE}")
        if str(self.TF_LEGACY_BACKBONE).lower() not in {"yolo-repvgg", "cira-lite"}:
            errors.append(
                "TF_LEGACY_BACKBONE must be one of ['cira-lite', 'yolo-repvgg'], "
                f"got {self.TF_LEGACY_BACKBONE}"
            )
        if float(self.TF_CIRA_WIDTH_MULT) <= 0.0:
            errors.append(f"TF_CIRA_WIDTH_MULT must be > 0, got {self.TF_CIRA_WIDTH_MULT}")
        if str(self.QAT_LOSS_MODE).lower() not in {"original", "kd-deploy"}:
            errors.append(
                f"QAT_LOSS_MODE must be 'original' or 'kd-deploy', got {self.QAT_LOSS_MODE}"
            )
        if data_backend not in {"ultralytics", "native"}:
            errors.append(f"DATA_BACKEND must be 'ultralytics' or 'native', got {self.DATA_BACKEND}")
        if data_backend == "ultralytics" and self.DATA_YAML is not None and not Path(self.DATA_YAML).exists():
            errors.append(f"DATA_YAML not found: {self.DATA_YAML}")
        if str(self.ULTRA_TASK).lower() not in {"detect", "segment", "classify", "pose", "obb"}:
            errors.append(
                "ULTRA_TASK must be one of ['classify', 'detect', 'obb', 'pose', 'segment'], "
                f"got {self.ULTRA_TASK}"
            )
        if str(self.ULTRA_BACKBONE).lower() not in {"custom", "yolo", "cira"}:
            errors.append(
                "ULTRA_BACKBONE must be one of ['custom', 'yolo', 'cira'], "
                f"got {self.ULTRA_BACKBONE}"
            )
        if str(self.ULTRA_BACKBONE).lower() in {"yolo", "cira"} and str(self.ULTRA_TASK).lower() not in {
            "pose",
            "detect",
        }:
            errors.append(
                "ULTRA_BACKBONE in {'yolo', 'cira'} requires ULTRA_TASK in {'pose', 'detect'}, "
                f"got {self.ULTRA_TASK}"
            )
        if not str(self.ULTRA_MODEL).strip():
            errors.append("ULTRA_MODEL must be non-empty")
        if not str(self.ULTRA_MODEL_YOLO_POSE).strip():
            errors.append("ULTRA_MODEL_YOLO_POSE must be non-empty")
        if not str(self.ULTRA_MODEL_YOLO_DETECT).strip():
            errors.append("ULTRA_MODEL_YOLO_DETECT must be non-empty")
        if not str(self.ULTRA_MODEL_CIRA_POSE).strip():
            errors.append("ULTRA_MODEL_CIRA_POSE must be non-empty")
        if not str(self.ULTRA_MODEL_CIRA_DETECT).strip():
            errors.append("ULTRA_MODEL_CIRA_DETECT must be non-empty")
        if self.ULTRA_EXPORT_DATA is not None and not Path(self.ULTRA_EXPORT_DATA).exists():
            errors.append(f"ULTRA_EXPORT_DATA not found: {self.ULTRA_EXPORT_DATA}")
        if str(self.ULTRA_ONNX2TF_DEVICE).lower() not in {"cpu", "gpu", "auto"}:
            errors.append(
                "ULTRA_ONNX2TF_DEVICE must be one of ['auto', 'cpu', 'gpu'], "
                f"got {self.ULTRA_ONNX2TF_DEVICE}"
            )
        if self.ULTRA_WORKERS < 0:
            errors.append(f"ULTRA_WORKERS must be >= 0, got {self.ULTRA_WORKERS}")
        if self.ULTRA_SEED < 0:
            errors.append(f"ULTRA_SEED must be >= 0, got {self.ULTRA_SEED}")
        if not (0.0 < self.ULTRA_EXPORT_FRACTION <= 1.0):
            errors.append(
                f"ULTRA_EXPORT_FRACTION must be in (0,1], got {self.ULTRA_EXPORT_FRACTION}"
            )
        if not (0.0 < self.ULTRA_FRACTION <= 1.0):
            errors.append(f"ULTRA_FRACTION must be in (0,1], got {self.ULTRA_FRACTION}")
        if self.ULTRA_CLOSE_MOSAIC < 0:
            errors.append(f"ULTRA_CLOSE_MOSAIC must be >= 0, got {self.ULTRA_CLOSE_MOSAIC}")
        if self.ULTRA_OPTIMIZER is not None and not str(self.ULTRA_OPTIMIZER).strip():
            errors.append("ULTRA_OPTIMIZER must be non-empty when provided")
        if self.ULTRA_LR0 is not None and self.ULTRA_LR0 <= 0.0:
            errors.append(f"ULTRA_LR0 must be > 0, got {self.ULTRA_LR0}")
        if self.ULTRA_LRF is not None and self.ULTRA_LRF <= 0.0:
            errors.append(f"ULTRA_LRF must be > 0, got {self.ULTRA_LRF}")
        if self.ULTRA_MOMENTUM is not None and not (0.0 <= self.ULTRA_MOMENTUM <= 1.0):
            errors.append(f"ULTRA_MOMENTUM must be in [0,1], got {self.ULTRA_MOMENTUM}")
        if self.ULTRA_WEIGHT_DECAY is not None and self.ULTRA_WEIGHT_DECAY < 0.0:
            errors.append(
                f"ULTRA_WEIGHT_DECAY must be >= 0, got {self.ULTRA_WEIGHT_DECAY}"
            )
        if not (0.0 <= self.ULTRA_FLIPLR <= 1.0):
            errors.append(f"ULTRA_FLIPLR must be in [0,1], got {self.ULTRA_FLIPLR}")
        if not (0.0 <= self.ULTRA_FLIPUD <= 1.0):
            errors.append(f"ULTRA_FLIPUD must be in [0,1], got {self.ULTRA_FLIPUD}")
        if not (0.0 <= self.ULTRA_MOSAIC <= 1.0):
            errors.append(f"ULTRA_MOSAIC must be in [0,1], got {self.ULTRA_MOSAIC}")
        if not (0.0 <= self.ULTRA_MIXUP <= 1.0):
            errors.append(f"ULTRA_MIXUP must be in [0,1], got {self.ULTRA_MIXUP}")
        if not (0.0 <= self.ULTRA_COPY_PASTE <= 1.0):
            errors.append(f"ULTRA_COPY_PASTE must be in [0,1], got {self.ULTRA_COPY_PASTE}")
        if not (0.0 <= self.ULTRA_ERASING <= 1.0):
            errors.append(f"ULTRA_ERASING must be in [0,1], got {self.ULTRA_ERASING}")

        # 6. Check Loss Type
        valid_loss_types = {"ultralytics", "qat"}
        if str(self.LOSS_TYPE).lower() not in valid_loss_types:
            errors.append(
                f"LOSS_TYPE must be one of {sorted(valid_loss_types)}, got {self.LOSS_TYPE}"
            )

        # 7. Check Quant Mode
        valid_quant_modes = {"int8", "fp16", "fp32", "float32", "float16"}
        if str(self.TFLITE_QUANT_MODE).lower() not in valid_quant_modes:
            errors.append(
                "TFLITE_QUANT_MODE must be one of "
                f"{sorted(valid_quant_modes)}, got {self.TFLITE_QUANT_MODE}"
            )

        # 8. KD-specific validation (only meaningful in kd-deploy)
        if str(getattr(self, "QAT_LOSS_MODE", "")).lower() == "kd-deploy":
            if str(self.ULTRA_KD_LOSS_COMPOSITION).lower() not in {"dynamic_kd_deploy", "fixed_kd_deploy", "pure_kd"}:
                errors.append(
                    "ULTRA_KD_LOSS_COMPOSITION must be in "
                    "{dynamic_kd_deploy, fixed_kd_deploy, pure_kd}, "
                    f"got {self.ULTRA_KD_LOSS_COMPOSITION}"
                )
            if float(self.KD_TEMPERATURE) <= 0.0:
                errors.append(f"KD_TEMPERATURE must be > 0, got {self.KD_TEMPERATURE}")
            if str(self.KD_CLS_DISTILL) not in {"bce", "softmax_kl"}:
                errors.append(f"KD_CLS_DISTILL must be in {{bce, softmax_kl}}, got {self.KD_CLS_DISTILL}")
            if str(self.KD_DFL_DISTILL) not in {"kldiv", "smoothl1"}:
                errors.append(f"KD_DFL_DISTILL must be in {{kldiv, smoothl1}}, got {self.KD_DFL_DISTILL}")
            thr = float(self.KD_FG_THRESHOLD)
            if not (0.0 <= thr <= 1.0):
                errors.append(f"KD_FG_THRESHOLD must be in [0,1], got {self.KD_FG_THRESHOLD}")
            if int(self.KD_FG_TOPK) < 0:
                errors.append(f"KD_FG_TOPK must be >= 0, got {self.KD_FG_TOPK}")
            if int(self.KD_FG_MIN_POS) < 0:
                errors.append(f"KD_FG_MIN_POS must be >= 0, got {self.KD_FG_MIN_POS}")
            if str(self.KD_FG_APPLY_TO) not in {"cls", "dfl", "both"}:
                errors.append(f"KD_FG_APPLY_TO must be in {{cls, dfl, both}}, got {self.KD_FG_APPLY_TO}")

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
