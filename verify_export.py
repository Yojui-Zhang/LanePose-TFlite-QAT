# File: verify_export.py (For verification only)
import sys
import os

# ==============================================================================
# 1. Environment Setup (CRITICAL: Must be BEFORE import tensorflow)
# ==============================================================================
# 強制專案根目錄加入路徑，確保能找到 QAT_Refactored
sys.path.append(os.getcwd())

from QAT_Refactored.utils.env_setup import setup_environment
setup_environment()

# ==============================================================================
# 2. Lazy Imports
# ==============================================================================
import tensorflow as tf
from QAT_Refactored.config.config import cfg
from QAT_Refactored.models.builder import build_student_qat 
from QAT_Refactored.core.exporter import Exporter

def run_test():
    print("[Test] Initializing Model...")
    # Build a dummy model structure
    model = build_student_qat(cfg) 
    
    # Initialize weights
    print("[Test] Running Dummy Inference (Build)...")
    fake_input = tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3))
    model(fake_input)
    
    print("[Test] Running Exporter...")
    exporter = Exporter(cfg)
    
    # Mock representative dataset
    def rep_gen():
        for _ in range(5):
            yield [tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3))]

    exporter.export(model, rep_dataset=rep_gen)
    
    print("\n[Test] SUCCESS. TFLite generated.")

if __name__ == "__main__":
    run_test()