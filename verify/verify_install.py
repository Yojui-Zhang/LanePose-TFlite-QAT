# File: verify_install.py
import sys

import logging
from pathlib import Path

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

from QAT_Refactored.utils.checks import check_system_requirements, validate_data_access
from QAT_Refactored.config.config import cfg

def run_verification():
    print("="*60)
    print("      QAT-Re1 Environment Verification Tool")
    print("="*60)
    
    # 1. System Checks
    print("\n[1/3] Checking System Dependencies...")
    check_system_requirements()
    print("PASS.")
    
    # 2. Config Validation
    print("\n[2/3] Checking Configuration Structure...")
    try:
        cfg.validate()
        print(f"Output Dir: {cfg.OUTPUT_DIR}")
        print("PASS.")
    except Exception as e:
        logging.critical(f"Config Validation Failed: {e}")
        sys.exit(1)

    # 3. Data Integrity
    print("\n[3/3] Checking Dataset Access...")
    validate_data_access(cfg)
    print("PASS.")
    
    print("\n" + "="*60)
    print("✅  VERIFICATION SUCCESSFUL. You are ready to train.")
    print("="*60)

if __name__ == "__main__":
    run_verification()
