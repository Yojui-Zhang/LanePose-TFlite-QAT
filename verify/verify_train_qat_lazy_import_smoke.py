from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import sys
from importlib.abc import MetaPathFinder


class _BlockTFMOTFinder(MetaPathFinder):
    """Force import failure for tfmot to verify train_QAT lazy imports."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow_model_optimization" or fullname.startswith(
            "tensorflow_model_optimization."
        ):
            raise ModuleNotFoundError("blocked by verify_train_qat_lazy_import_smoke")
        return None


def main() -> None:
    finder = _BlockTFMOTFinder()
    blocked = [name for name in tuple(sys.modules) if name.startswith("tensorflow_model_optimization")]
    for name in blocked:
        sys.modules.pop(name, None)

    sys.meta_path.insert(0, finder)
    try:
        from train_QAT import run_train_qat
    finally:
        sys.meta_path.remove(finder)

    assert callable(run_train_qat)
    print("verify_train_qat_lazy_import_smoke: OK")


if __name__ == "__main__":
    main()
