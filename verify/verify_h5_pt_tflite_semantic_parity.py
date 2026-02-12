from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _autodiscover_h5_tflite() -> Path | None:
    return _latest(list(Path("output").rglob("model_quant_*.tflite")))


def _autodiscover_pt_tflite() -> Path | None:
    candidates = list(Path("runs").rglob("best_*.tflite"))
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda p: (
            0 if "saved_model" in str(p) else 1,
            -int(p.stat().st_mtime),
        ),
    )
    return scored[0]


def _make_interpreter(model_path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _op_names(interpreter: tf.lite.Interpreter) -> list[str]:
    ops: list[dict[str, Any]] = interpreter._get_ops_details()  # noqa: SLF001
    return [str(op.get("op_name", "")).upper() for op in ops]


def _counter_overlap_ratio(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    inter = 0
    union = 0
    for k in keys:
        av = int(a.get(k, 0))
        bv = int(b.get(k, 0))
        inter += min(av, bv)
        union += max(av, bv)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _quantize_input(sample: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    q_scale, q_zero = detail.get("quantization", (0.0, 0))
    if np.issubdtype(dtype, np.floating):
        return sample.astype(dtype)
    if q_scale and q_scale > 0:
        q = np.round(sample / q_scale + q_zero)
    else:
        q = sample
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max)
    return q.astype(dtype)


def _dequantize_output(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    q_scale, q_zero = detail.get("quantization", (0.0, 0))
    if np.issubdtype(dtype, np.floating) or not (q_scale and q_scale > 0):
        return value.astype(np.float32)
    return (value.astype(np.float32) - float(q_zero)) * float(q_scale)


def _infer_once(interpreter: tf.lite.Interpreter, sample_f32: np.ndarray) -> np.ndarray:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    x = _quantize_input(sample_f32, in_detail)
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out_detail["index"])
    return _dequantize_output(y, out_detail)


def _shape_tuple(detail: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in np.array(detail["shape"]).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic parity check between train_QAT(h5-route) and Ultralytics(pt-route) exported TFLite models.",
    )
    parser.add_argument("--h5-tflite", type=str, default=None, help="Path to h5-route exported .tflite")
    parser.add_argument("--pt-tflite", type=str, default=None, help="Path to pt-route exported .tflite")
    parser.add_argument("--samples", type=int, default=3, help="Number of random samples for output comparison.")
    parser.add_argument("--max-abs-threshold", type=float, default=0.10, help="Max abs diff threshold.")
    parser.add_argument("--mean-abs-threshold", type=float, default=0.02, help="Mean abs diff threshold.")
    parser.add_argument("--min-op-overlap", type=float, default=0.70, help="Minimum op histogram overlap ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_tflite = Path(args.h5_tflite) if args.h5_tflite else _autodiscover_h5_tflite()
    pt_tflite = Path(args.pt_tflite) if args.pt_tflite else _autodiscover_pt_tflite()

    if h5_tflite is None or pt_tflite is None:
        print("verify_h5_pt_tflite_semantic_parity: SKIP (missing h5/pt tflite path)")
        return
    if not h5_tflite.exists() or not pt_tflite.exists():
        print("verify_h5_pt_tflite_semantic_parity: SKIP (input model path does not exist)")
        return

    interp_h5 = _make_interpreter(h5_tflite)
    interp_pt = _make_interpreter(pt_tflite)

    in_h5 = interp_h5.get_input_details()[0]
    in_pt = interp_pt.get_input_details()[0]
    out_h5 = interp_h5.get_output_details()[0]
    out_pt = interp_pt.get_output_details()[0]

    in_shape_h5 = _shape_tuple(in_h5)
    in_shape_pt = _shape_tuple(in_pt)
    out_shape_h5 = _shape_tuple(out_h5)
    out_shape_pt = _shape_tuple(out_pt)
    if in_shape_h5 != in_shape_pt:
        raise AssertionError(f"Input shape mismatch: h5={in_shape_h5}, pt={in_shape_pt}")
    if out_shape_h5 != out_shape_pt:
        raise AssertionError(f"Output shape mismatch: h5={out_shape_h5}, pt={out_shape_pt}")

    ops_h5 = Counter(_op_names(interp_h5))
    ops_pt = Counter(_op_names(interp_pt))
    op_overlap = _counter_overlap_ratio(ops_h5, ops_pt)
    if op_overlap < float(args.min_op_overlap):
        raise AssertionError(
            f"Operator histogram overlap too low: overlap={op_overlap:.4f} < min={args.min_op_overlap:.4f}"
        )

    rng = np.random.default_rng(seed=0)
    max_abs = 0.0
    mean_abs = 0.0
    num = max(int(args.samples), 1)
    sample_shape = in_shape_h5
    for _ in range(num):
        sample = rng.standard_normal(sample_shape, dtype=np.float32)
        y_h5 = _infer_once(interp_h5, sample)
        y_pt = _infer_once(interp_pt, sample)
        diff = np.abs(y_h5 - y_pt)
        max_abs = max(max_abs, float(diff.max()))
        mean_abs += float(diff.mean())
    mean_abs /= float(num)

    if max_abs > float(args.max_abs_threshold) or mean_abs > float(args.mean_abs_threshold):
        raise AssertionError(
            "Semantic parity failed: "
            f"max_abs={max_abs:.6f} (th={args.max_abs_threshold}), "
            f"mean_abs={mean_abs:.6f} (th={args.mean_abs_threshold})"
        )

    h5_hash = _sha256(h5_tflite)
    pt_hash = _sha256(pt_tflite)
    strict = os.getenv("PARITY_STRICT_BYTES", "0") == "1"
    if strict and h5_hash != pt_hash:
        raise AssertionError(
            "Strict bytes parity failed:\n"
            f"- h5-route: {h5_tflite} sha256={h5_hash}\n"
            f"- pt-route: {pt_tflite} sha256={pt_hash}\n"
            f"- op_overlap={op_overlap:.6f} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )

    print("verify_h5_pt_tflite_semantic_parity: OK")
    print(f"h5_tflite={h5_tflite} sha256={h5_hash}")
    print(f"pt_tflite={pt_tflite} sha256={pt_hash}")
    print(f"op_overlap={op_overlap:.6f} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")
    if h5_hash != pt_hash:
        print("note=raw_tflite_bytes_differ_but_architecture_and_outputs_are_semantically_similar")


if __name__ == "__main__":
    main()
