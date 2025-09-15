#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare four model routes with MAE:
  1) Teacher SavedModel (TF)  vs Teacher TFLite
  2) Student SavedModel (TF)  vs Student TFLite
  3) Teacher TFLite         vs Student TFLite
  4) Teacher SavedModel (TF) vs Student SavedModel (TF)

Features
- Accept a single image file or a directory of images (jpg/png/jpeg/bmp/webp)
- Auto-detect input size from each model (batch=1 assumed)
- Unified preprocessing (resize + optional normalization) applied to all models
- Proper INT8 quantize/dequantize for TFLite inputs/outputs
- Handles multiple outputs (concats in a stable key/index order)
- Prints per-pair overall MAE and per-image MAE
- Optional CSV export

Usage example
-------------
python3 compare_tf_tflite_mae.py \
  --teacher_savedmodel /path/to/teacher_savedmodel \
  --teacher_tflite /path/to/teacher.tflite \
  --student_savedmodel /path/to/student_savedmodel \
  --student_tflite /path/to/student.tflite \
  --images /path/to/img_or_dir \
  --norm 0to1 --csv out.csv

Notes
- Batch size is assumed to be 1 for all models.
- If your model expects NCHW instead of NHWC, pass --nchw to transpose accordingly.
- If your model uses different preprocessing (mean/std), set --norm imagenet or --mean/--std.
- If output shapes differ, by default it will error. You can use --allow-mismatch to compare the common prefix after flattening.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# TensorFlow is required for both SavedModel and TFLite
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(path: Path):
    p = Path(path)
    if p.is_dir():
        files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"No images found at {path}")
    return files


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def resize_image(img: np.ndarray, size_hw: tuple, method: str = "bilinear"):
    h, w = size_hw
    pil = Image.fromarray(img)
    resample = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }.get(method, Image.BILINEAR)
    out = pil.resize((w, h), resample=resample)
    return np.array(out)


def normalize(img: np.ndarray, norm: str, mean=None, std=None) -> np.ndarray:
    x = img.astype(np.float32)
    if norm == "none":
        return x
    if norm == "0to1":
        return x / 255.0
    if norm == "imagenet":
        # RGB mean/std in 0-1 scale
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x / 255.0 - mean) / std
    if norm == "custom":
        if mean is None or std is None:
            raise ValueError("custom norm requires --mean and --std")
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        if mean.size == 1:
            mean = np.full(3, float(mean))
        if std.size == 1:
            std = np.full(3, float(std))
        return (x - mean) / std
    raise ValueError(f"Unknown norm: {norm}")


# ---------- SavedModel runner ----------
class SavedModelRunner:
    def __init__(self, model_dir: str, nchw: bool = False):
        self.model = tf.saved_model.load(model_dir)
        self.nchw = nchw
        # Try to get serving signature; else fallback to calling model
        self._infer = None
        if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
            self._infer = self.model.signatures["serving_default"]
        # Determine input tensor
        if self._infer is not None:
            self.input_key = list(self._infer.structured_input_signature[1].keys())[0]
            inp = self._infer.structured_input_signature[1][self.input_key]
            self.inp_dtype = inp.dtype
            self.inp_shape = tuple(int(x) if x is not None else -1 for x in inp.shape)
        else:
            # Best-effort: try to find an attribute that gives input shape
            try:
                # Many Keras models keep .inputs
                self.inp_shape = tuple(int(x) if x is not None else -1 for x in self.model.inputs[0].shape)
                self.inp_dtype = self.model.inputs[0].dtype
            except Exception as e:
                raise RuntimeError("Cannot determine input spec for SavedModel") from e

        # NHWC or NCHW check
        if len(self.inp_shape) != 4:
            raise ValueError(f"Expected 4D input [B,H,W,C] or [B,C,H,W], got {self.inp_shape}")

        # Dummy run to probe outputs ordering
        dummy = np.zeros((1,) + self.inp_shape[1:], dtype=np.float32)
        out = self._run_raw(dummy)
        self.out_keys = list(out.keys()) if isinstance(out, dict) else None

    def input_hw_c(self):
        # return (H, W, C) no matter NCHW/NHWC
        b, a, b2, c = self.inp_shape[0], self.inp_shape[1], self.inp_shape[2], self.inp_shape[3]
        if self.nchw or (self.inp_shape[1] in (1,3) and self.inp_shape[3] not in (1,3)):
            # assume NCHW
            c = self.inp_shape[1]
            h = self.inp_shape[2]
            w = self.inp_shape[3]
        else:
            # NHWC
            h = self.inp_shape[1]
            w = self.inp_shape[2]
            c = self.inp_shape[3]
        return int(h), int(w), int(c)

    def _run_raw(self, x_np: np.ndarray):
        # x_np should already be in the model's native layout
        if self._infer is not None:
            # using serving_default signature
            out = self._infer(tf.convert_to_tensor(x_np))
            # Tensor dict -> numpy
            return {k: v.numpy() for k, v in out.items()}
        else:
            out = self.model(tf.convert_to_tensor(x_np))
            if isinstance(out, dict):
                return {k: v.numpy() for k, v in out.items()}
            return {"output_0": out.numpy()}

    def forward(self, img_np: np.ndarray) -> np.ndarray:
        # img_np: HWC float32 after normalization
        h, w, c = img_np.shape
        H, W, C = self.input_hw_c()
        if C == 1 and c == 3:
            # convert to grayscale
            img_np = img_np.mean(axis=2, keepdims=True)
        elif C == 3 and c == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        x = img_np[None, ...]  # add batch
        # transpose if model expects NCHW
        if self.nchw or (self.inp_shape[1] in (1,3) and self.inp_shape[3] not in (1,3)):
            x = np.transpose(x, (0, 3, 1, 2))
        out = self._run_raw(x)
        # flatten & concat in a stable order
        keys = self.out_keys or sorted(out.keys())
        flat = [out[k].reshape(-1) for k in keys]
        return np.concatenate(flat, axis=0)


# ---------- TFLite runner ----------
class TFLiteRunner:
    def __init__(self, model_path: str, nchw: bool = False):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.nchw = nchw
        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()
        self.inp_idx = inp["index"]
        self.inp_dtype = inp["dtype"]
        self.inp_q = inp.get("quantization", (0.0, 0))
        self.inp_shape = tuple(int(x) for x in inp["shape"])
        self.out_details = out

    def input_hw_c(self):
        shape = self.inp_shape
        if len(shape) != 4:
            raise ValueError(f"TFLite expects 4D input, got {shape}")
        # Heuristic: if second dim is 1 or 3, assume NCHW; else NHWC
        if self.nchw or (shape[1] in (1,3) and shape[3] not in (1,3)):
            c = shape[1]
            h = shape[2]
            w = shape[3]
        else:
            h = shape[1]
            w = shape[2]
            c = shape[3]
        return int(h), int(w), int(c)

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        if self.inp_dtype == np.int8:
            scale, zero = self.inp_q
            if scale == 0:
                raise ValueError("Input scale is 0; TFLite model not properly quantized?")
            q = np.round(x / scale + zero)
            return np.clip(q, -128, 127).astype(np.int8)
        elif self.inp_dtype == np.uint8:
            scale, zero = self.inp_q
            q = np.round(x / scale + zero)
            return np.clip(q, 0, 255).astype(np.uint8)
        else:
            return x.astype(self.inp_dtype)

    @staticmethod
    def _dequantize(arr: np.ndarray, detail: dict) -> np.ndarray:
        dtype = detail["dtype"]
        if dtype in (np.int8, np.uint8):
            scale, zero = detail.get("quantization", (1.0, 0))
            return (arr.astype(np.float32) - zero) * scale
        return arr.astype(np.float32)

    def forward(self, img_np: np.ndarray) -> np.ndarray:
        # img_np: HWC float32 after normalization
        h, w, c = img_np.shape
        H, W, C = self.input_hw_c()
        if C == 1 and c == 3:
            img_np = img_np.mean(axis=2, keepdims=True)
        elif C == 3 and c == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        x = img_np[None, ...]
        # transpose if model expects NCHW
        if self.nchw or (self.inp_shape[1] in (1,3) and self.inp_shape[3] not in (1,3)):
            x = np.transpose(x, (0, 3, 1, 2))
        # quantize to model dtype
        xq = self._quantize(x)
        self.interp.set_tensor(self.inp_idx, xq)
        self.interp.invoke()
        outs = []
        for od in self.out_details:
            yq = self.interp.get_tensor(od["index"])  # possibly quantized
            y = self._dequantize(yq, od)
            outs.append(y.reshape(-1))
        return np.concatenate(outs, axis=0)


# ---------- Utilities ----------

def mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def mae_allow_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        raise ValueError("No overlapping elements to compare")
    return float(np.mean(np.abs(a[:n] - b[:n])))


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Compare TF/TFLite models via MAE")
    ap.add_argument("--teacher_savedmodel", type=str, required=True)
    ap.add_argument("--teacher_tflite", type=str, required=True)
    ap.add_argument("--student_savedmodel", type=str, required=True)
    ap.add_argument("--student_tflite", type=str, required=True)
    ap.add_argument("--images", type=str, required=True, help="Image file or directory")
    ap.add_argument("--norm", type=str, default="0to1", choices=["none", "0to1", "imagenet", "custom"], help="Preprocess normalization")
    ap.add_argument("--mean", type=float, nargs="*", default=None, help="Mean for custom norm (scalar or 3 values)")
    ap.add_argument("--std", type=float, nargs="*", default=None, help="Std for custom norm (scalar or 3 values)")
    ap.add_argument("--resize", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"], help="Resize method")
    ap.add_argument("--nchw", action="store_true", help="Force NCHW layout for models that expect it")
    ap.add_argument("--allow-mismatch", action="store_true", help="Allow output-size mismatch (compare common prefix)")
    ap.add_argument("--csv", type=str, default=None, help="Optional path to save per-image MAE results")
    args = ap.parse_args()

    # Load images
    img_paths = list_images(Path(args.images))

    # Init runners
    print("\nLoading models...")
    tr_tf = SavedModelRunner(args.teacher_savedmodel, nchw=args.nchw)
    tr_tfl = TFLiteRunner(args.teacher_tflite, nchw=args.nchw)
    st_tf = SavedModelRunner(args.student_savedmodel, nchw=args.nchw)
    st_tfl = TFLiteRunner(args.student_tflite, nchw=args.nchw)

    # Determine a unified input size (H,W) — use teacher TF as the reference
    H_ref, W_ref, C_ref = tr_tf.input_hw_c()
    print(f"Reference input size (Teacher TF): H={H_ref}, W={W_ref}, C={C_ref}")

    # Results accumulators
    pairs = [
        ("Teacher TF vs Teacher TFLite", tr_tf, tr_tfl),
        ("Student TF vs Student TFLite", st_tf, st_tfl),
        ("Teacher TFLite vs Student TFLite", tr_tfl, st_tfl),
        ("Teacher TF vs Student TF", tr_tf, st_tf),
    ]
    totals = {name: [] for name, _, _ in pairs}

    # CSV header
    csv_lines = []
    if args.csv:
        csv_lines.append(
            "image,mae_teacherTF_teacherTFLite,mae_studentTF_studentTFLite,mae_teacherTFLite_studentTFLite,mae_teacherTF_studentTF"
        )

    # Iterate images
    print(f"\nRunning {len(img_paths)} image(s)...")
    for i, p in enumerate(img_paths):
        img = load_image(p)
        img_r = resize_image(img, (H_ref, W_ref), method=args.resize)
        x = normalize(img_r, args.norm, args.mean, args.std)

        # Run all
        outs = {}
        for name, left, right in pairs:
            if name not in outs:
                outs[name] = [None, None]

            # left output
            yL = left.forward(x)
            # right output
            yR = right.forward(x)

            # MAE
            if args.allow_mismatch:
                val = mae_allow_mismatch(yL, yR)
            else:
                val = mae(yL, yR)
            totals[name].append(val)

        # CSV row
        if args.csv:
            row = [str(p)] + [f"{totals[n][-1]:.6f}" for n, _, _ in pairs]
            csv_lines.append(",".join(row))

        if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
            print(f"Processed {i+1}/{len(img_paths)} images")

    # Print summary
    print("\n==== MAE SUMMARY (lower is better) ====")
    for name, _, _ in pairs:
        arr = np.array(totals[name], dtype=np.float32)
        print(f"{name:32s}  mean={arr.mean():.6f}  std={arr.std():.6f}  min={arr.min():.6f}  max={arr.max():.6f}")

    # Save CSV if requested
    if args.csv:

        dir_path = Path(args.csv).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = Path(args.csv)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))
        print(f"\nPer-image MAEs saved to: {args.csv}")


if __name__ == "__main__":
    main()


# python3 compare_models.py \
#   --teacher_savedmodel ./Teacher_models/lanepose20250807_s_model_640_640_6c_v1_saved_model \
#   --teacher_tflite ./Teacher_models/lanepose20250807_s_model_640_640_6c_v1_integer_quant.tflite \
#   --student_savedmodel ./student_models/qat_saved_model_interrupted \
#   --student_tflite ./student_models/best_qat_int8_interrupted.tflite \
#   --images ./test.jpg \
#   --norm 0to1 \
#   --csv ./output/mae_results.csv
