from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf  # noqa: E402

from QAT_Refactored.config.config import AppConfig  # noqa: E402


def _resolve_model_path(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"TFLite model not found: {p}")
        return p

    candidates = sorted(
        Path("output").rglob("model_quant_*.tflite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No exported TFLite found under output/**/model_quant_*.tflite")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TFLite contract for TFlite.h parser.")
    parser.add_argument("--model", type=str, default=None, help="Path to .tflite model")
    args = parser.parse_args()

    cfg = AppConfig()
    model_path = _resolve_model_path(args.model)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    ins = interpreter.get_input_details()
    outs = interpreter.get_output_details()

    assert len(ins) == 1, f"Expected 1 input, got {len(ins)}"
    assert len(outs) == 1, f"Expected 1 output, got {len(outs)}"

    expected_in = tuple(int(v) for v in cfg.EXPORT_INPUT_SHAPE)
    expected_out = (1, int(cfg.total_output_channels), int(cfg.get_total_anchors()))

    in_shape = tuple(int(v) for v in ins[0]["shape"])
    out_shape = tuple(int(v) for v in outs[0]["shape"])
    in_dtype = ins[0]["dtype"]
    out_dtype = outs[0]["dtype"]

    assert in_shape == expected_in, f"Input shape mismatch: expected={expected_in}, got={in_shape}"
    assert out_shape == expected_out, (
        "Output shape mismatch for TFlite.h parser: "
        f"expected={expected_out} (B,C,N), got={out_shape}"
    )
    assert in_dtype == np.float32 and out_dtype == np.float32, (
        f"DType mismatch for typed_tensor<float>: input={in_dtype}, output={out_dtype}"
    )

    dummy = np.zeros(expected_in, dtype=np.float32)
    interpreter.set_tensor(ins[0]["index"], dummy)
    interpreter.invoke()
    y = interpreter.get_tensor(outs[0]["index"])

    assert np.all(np.isfinite(y)), "Output contains NaN/Inf"

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    assert ymin >= -1e-3 and ymax <= 1.0 + 1e-3, (
        f"Output out of sigmoid domain: min={ymin:.6f}, max={ymax:.6f}"
    )

    print(f"[verify_tflite_contract] model: {model_path}")
    print(f"[verify_tflite_contract] input: {in_shape} {in_dtype}")
    print(f"[verify_tflite_contract] output: {out_shape} {out_dtype}")
    print(f"[verify_tflite_contract] output range: [{ymin:.6f}, {ymax:.6f}]")
    print("[verify_tflite_contract] OK")


if __name__ == "__main__":
    main()
