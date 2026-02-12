# File: verify_export.py (For verification only)
try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

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
