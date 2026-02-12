from __future__ import annotations

import os
import sys
from pathlib import Path


def setup_project_root() -> Path:
    """Ensure project root is importable no matter where verify script is launched."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def setup_runtime_env() -> Path:
    """
    Standard verify bootstrap:
    1) enforce legacy keras flags
    2) add project root to sys.path
    3) run project env setup
    """
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    root = setup_project_root()

    from QAT_Refactored.utils.env_setup import setup_environment

    setup_environment()
    return root
