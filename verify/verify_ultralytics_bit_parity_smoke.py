from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()
os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

import yaml
import numpy as np
import tensorflow as tf
import torch
from ultralytics import YOLO

from train_QAT import run_train_qat


ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / "verify" / "_tmp" / "bit_parity"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_entries(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out
    raise TypeError(f"Unsupported yaml entry type: {type(value).__name__}")


def _resolve_path(entry: str, *, root: Path | None, yaml_dir: Path) -> Path:
    p = Path(entry)
    if p.is_absolute():
        return p
    base = root if root is not None else yaml_dir
    return (base / p).resolve()


def _iter_image_paths(source: Path) -> Iterable[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield Path(line)
        return

    if source.is_file():
        yield source
        return

    if source.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img in sorted(source.glob(ext)):
                yield img


def _build_smoke_yaml() -> Path:
    src_yaml = ROOT / "dataset" / "lanepose-carkeypoint.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(src_yaml)

    raw = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("dataset yaml must be a mapping")

    data_root: Path | None = None
    path_entry = raw.get("path")
    if isinstance(path_entry, str) and path_entry.strip():
        data_root = _resolve_path(path_entry, root=None, yaml_dir=src_yaml.parent)

    train_entries = _normalize_entries(raw.get("train"))
    if not train_entries:
        raise ValueError("dataset yaml has empty train entry")

    samples: list[Path] = []
    for entry in train_entries:
        resolved = _resolve_path(entry, root=data_root, yaml_dir=src_yaml.parent)
        for img in _iter_image_paths(resolved):
            img = img.resolve()
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            if label.exists():
                samples.append(img)
            if len(samples) >= 32:
                break
        if len(samples) >= 32:
            break

    if len(samples) < 8:
        raise RuntimeError(f"Not enough images with labels for smoke test. Found={len(samples)}")

    smoke_dir = TMP_DIR / "data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_txt = smoke_dir / "train.txt"
    val_txt = smoke_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in samples[:16]) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in samples[:8]) + "\n", encoding="utf-8")

    kpt_shape = raw.get("kpt_shape", [15, 3])
    flip_idx = raw.get("flip_idx", list(range(int(kpt_shape[0]))))
    nc = int(raw.get("nc", 7))
    names = raw.get("names", [str(i) for i in range(nc)])
    if not isinstance(names, list):
        names = [str(i) for i in range(nc)]

    smoke_yaml = smoke_dir / "dataset.yaml"
    smoke_yaml.write_text(
        "\n".join(
            [
                f"train: {train_txt}",
                f"val: {val_txt}",
                f"kpt_shape: [{int(kpt_shape[0])}, {int(kpt_shape[1])}]",
                "flip_idx: [" + ", ".join(str(int(x)) for x in flip_idx) + "]",
                f"nc: {nc}",
                "names: [" + ", ".join(f"'{str(n)}'" for n in names) + "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return smoke_yaml


def _select_seed_model() -> Path:
    candidates = [
        ROOT / "runs" / "pose" / "acc-dataset-YOLOn-20260209" / "weights" / "best.pt",
        ROOT / "runs" / "pose" / "yolov8n" / "weights" / "best.pt",
        ROOT / "yolo11n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No seed model found for parity smoke.")


def _load_state_dict(pt_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return model.state_dict()


def _state_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[bool, str]:
    if set(a.keys()) != set(b.keys()):
        return False, "state_dict keys mismatch"
    for k in sorted(a.keys()):
        ta = a[k]
        tb = b[k]
        if ta.shape != tb.shape:
            return False, f"shape mismatch at {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}"
        if ta.dtype != tb.dtype:
            return False, f"dtype mismatch at {k}: {ta.dtype} vs {tb.dtype}"
        if not torch.equal(ta, tb):
            max_abs = (ta.float() - tb.float()).abs().max().item()
            return False, f"tensor mismatch at {k}, max_abs={max_abs}"
    return True, "ok"


def _tflite_outputs_equal(model_a: Path, model_b: Path) -> tuple[bool, float]:
    ia = tf.lite.Interpreter(model_path=str(model_a))
    ib = tf.lite.Interpreter(model_path=str(model_b))
    ia.allocate_tensors()
    ib.allocate_tensors()
    in_a = ia.get_input_details()[0]
    in_b = ib.get_input_details()[0]
    out_a = ia.get_output_details()[0]
    out_b = ib.get_output_details()[0]

    x = np.random.RandomState(0).randn(*in_a["shape"]).astype(np.float32)
    ia.set_tensor(in_a["index"], x)
    ia.invoke()
    ya = ia.get_tensor(out_a["index"])
    ib.set_tensor(in_b["index"], x)
    ib.invoke()
    yb = ib.get_tensor(out_b["index"])

    max_abs = float(np.max(np.abs(ya - yb)))
    return bool(np.array_equal(ya, yb)), max_abs


def main() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    fixed_export_date = "2026-01-01T00:00:00"
    os.environ["ULTRALYTICS_EXPORT_DATE"] = fixed_export_date

    smoke_yaml = _build_smoke_yaml()
    seed_model = _select_seed_model()

    common_train = {
        "data": str(smoke_yaml),
        "task": "pose",
        "epochs": 1,
        "batch": 2,
        "imgsz": 64,
        "device": "cpu",
        "workers": 0,
        "cache": False,
        "resume": False,
        "exist_ok": True,
        "seed": 0,
        "deterministic": True,
        "amp": False,
        "cos_lr": True,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "fraction": 1.0,
        "close_mosaic": 0,
    }

    official_project = TMP_DIR / "official"
    official = YOLO(str(seed_model), task="pose")
    official.train(project=str(official_project), name="run", **common_train)
    official_best = Path(str(official.trainer.best))
    official_tflite = Path(
        str(
            YOLO(str(official_best), task="pose").export(
                format="tflite",
                imgsz=64,
                int8=True,
                data=str(smoke_yaml),
                nms=False,
            )
        )
    )

    qat_output = TMP_DIR / "qat" / "run"
    run_train_qat(
        config_overrides={
            "TRAIN_ENGINE": "ultralytics",
            "QAT_LOSS_MODE": "original",
            "TRAIN_SUPERVISION": "label",
            "AUX_KD_HEAD_LABEL_LOSS": False,
            "KD_LOSS_WEIGHT": 0.0,
            "DEPLOY_LOSS_WEIGHT": 1.0,
            "TFLITE_QUANT_MODE": "int8",
            "DATA_BACKEND": "ultralytics",
            "DATA_YAML": str(smoke_yaml),
            "ULTRA_MODEL": str(seed_model),
            "ULTRA_DEVICE": "cpu",
            "ULTRA_WORKERS": 0,
            "ULTRA_CACHE": False,
            "ULTRA_RESUME": False,
            "ULTRA_EXIST_OK": True,
            "ULTRA_SEED": 0,
            "ULTRA_DETERMINISTIC": True,
            "ULTRA_EXPORT_DATE": fixed_export_date,
            "ULTRA_AMP": False,
            "ULTRA_COS_LR": True,
            "ULTRA_FLIPLR": 0.0,
            "ULTRA_FLIPUD": 0.0,
            "ULTRA_HSV_H": 0.015,
            "ULTRA_HSV_S": 0.7,
            "ULTRA_HSV_V": 0.4,
            "ULTRA_MOSAIC": 0.0,
            "ULTRA_MIXUP": 0.0,
            "ULTRA_COPY_PASTE": 0.0,
            "ULTRA_ERASING": 0.0,
            "ULTRA_FRACTION": 1.0,
            "ULTRA_CLOSE_MOSAIC": 0,
            "IMGSZ": 64,
            "BATCH_SIZE": 2,
            "EPOCHS": 1,
            "OUTPUT_DIR": str(qat_output),
        }
    )
    qat_tflite = qat_output / "weights" / "best_saved_model" / "best_int8.tflite"
    if not qat_tflite.exists():
        raise FileNotFoundError(qat_tflite)

    official_hash = _sha256(official_tflite)
    qat_hash = _sha256(qat_tflite)
    if official_hash != qat_hash:
        strict_bytes = os.getenv("PARITY_STRICT_BYTES", "0") == "1"
        official_sd = _load_state_dict(official_best)
        qat_best = qat_output / "weights" / "best.pt"
        qat_sd = _load_state_dict(qat_best)
        same_state, state_msg = _state_dicts_equal(official_sd, qat_sd)
        same_out, max_abs = _tflite_outputs_equal(official_tflite, qat_tflite)

        if strict_bytes:
            raise AssertionError(
                "Bit parity failed (strict raw bytes mode):\n"
                f"- official: {official_tflite} sha256={official_hash}\n"
                f"- train_QAT: {qat_tflite} sha256={qat_hash}\n"
                f"- state_dict_equal: {same_state} ({state_msg})\n"
                f"- tflite_output_equal: {same_out} (max_abs={max_abs})"
            )

        if not same_state or not same_out:
            raise AssertionError(
                "Parity failed:\n"
                f"- official: {official_tflite} sha256={official_hash}\n"
                f"- train_QAT: {qat_tflite} sha256={qat_hash}\n"
                f"- state_dict_equal: {same_state} ({state_msg})\n"
                f"- tflite_output_equal: {same_out} (max_abs={max_abs})"
            )

        print("verify_ultralytics_bit_parity_smoke: OK (functional parity)")
        print(f"official_sha256={official_hash}")
        print(f"train_qat_sha256={qat_hash}")
        print("note=raw_tflite_bytes_differ_but_weights_and_outputs_are_exactly_equal")
        return

    print("verify_ultralytics_bit_parity_smoke: OK")
    print(f"sha256={official_hash}")


if __name__ == "__main__":
    main()
