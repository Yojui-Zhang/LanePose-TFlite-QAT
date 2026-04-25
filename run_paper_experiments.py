from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
import shutil
from datetime import datetime

import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from ultralytics import YOLO

import numpy as np
import yaml




QAT_ROOT = Path(__file__).resolve().parent
WORK_ROOT = QAT_ROOT.parent
DEFAULT_DATA_ROOT = WORK_ROOT / "Data"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OOM_HINTS = (
    "out of memory",
    "cuda error: out of memory",
    "resourceexhaustederror",
    "cudnn_status_alloc_failed",
    "\nkilled\n",
)

_KD_LINE_RE = re.compile(
    r"\[KD\]\s+step=(?P<step>\d+)\s+"
    r"supervised_loss=(?P<sup>[0-9eE\.\+\-]+)\s+"
    r"kd_loss=(?P<kd>[0-9eE\.\+\-]+)\s+"
    r"alpha_kd=(?P<alpha>[0-9eE\.\+\-]+)\s+"
    r"grad_ratio_sup_over_kd=(?P<ratio>None|[0-9eE\.\+\-]+)"
)


@dataclass(frozen=True)
class TrainSpec:
    study: str
    dataset: str
    variant: str
    mode: str
    model: Path
    data_yaml: Path
    task: str
    device: str
    seed: int
    epochs: int
    batch: int
    imgsz: int
    workers: int
    close_mosaic: int
    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    export_fraction: float
    qat_kd_weight: Optional[float]

    # KD distillation specifics (forwarded to train_pose.py; only used when mode == "kd-deploy")
    qat_kd_temperature: float
    qat_kd_cls_distill: str
    qat_kd_dfl_distill: str
    qat_kd_fg_threshold: float
    qat_kd_fg_topk: int
    qat_kd_fg_min_pos: int
    qat_kd_fg_apply_to: str

    qat_balance_log_interval: int
    qat_balance_min: Optional[float]
    qat_balance_max: Optional[float]
    qat_balance_warmup_steps: Optional[int]
    qat_balance_max_step_change: Optional[float]
    qat_balance_adapt_power: Optional[float]
    qat_balance_strategy: Optional[str]
    qat_balance_shared_group: Optional[str]
    qat_balance_deploy_ramp_steps: Optional[int]
    qat_balance_update_interval: Optional[int]
    project: Path
    name: str
    teacher_dir: Optional[Path] = None
    kd_loss_composition: Optional[str] = None

    @property
    def run_dir(self) -> Path:
        if self.mode == "kd-deploy":
            return self.project / f"{self.name}_qat"
        return self.project / self.name


def _parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for token in _parse_csv_list(raw):
        value = int(token)
        if value < 0:
            raise ValueError(f"seed must be >= 0, got {value}")
        seeds.append(value)
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _resolve_dataset_entries(value: Any, yaml_path: Path, root: Optional[Path]) -> list[str]:
    entries: list[str] = []
    if value is None:
        return entries
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, (list, tuple)):
        entries = [str(v) for v in value if str(v).strip()]
    else:
        raise ValueError(
            f"Invalid dataset yaml entry type in {yaml_path}: {type(value).__name__}"
        )

    resolved: list[str] = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            base = root if root is not None else yaml_path.parent
            path = (base / path).resolve()
        resolved.append(str(path))
    return resolved


def _expand_image_sources(entry: str) -> list[Path]:
    path = Path(entry)
    if any(ch in entry for ch in "*?[]"):
        return sorted(p.resolve() for p in path.parent.glob(path.name) if p.suffix.lower() in IMAGE_EXTS)

    if path.suffix.lower() == ".txt" and path.exists():
        out: list[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            if p.suffix.lower() in IMAGE_EXTS and p.exists():
                out.append(p)
        return out

    if path.is_dir():
        files = [p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        return sorted(files)

    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path.resolve()]

    return []


def _build_smoke_yaml(source_yaml: Path, smoke_root: Path, tag: str, max_items: int = 12) -> Path:
    raw = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset yaml must be a mapping: {source_yaml}")

    root = None
    if raw.get("path"):
        root = Path(str(raw["path"]))
        if not root.is_absolute():
            root = (source_yaml.parent / root).resolve()

    train_entries = _resolve_dataset_entries(raw.get("train"), source_yaml, root)
    val_entries = _resolve_dataset_entries(raw.get("val"), source_yaml, root)

    train_images: list[Path] = []
    for entry in train_entries:
        train_images.extend(_expand_image_sources(entry))

    val_images: list[Path] = []
    for entry in (val_entries or train_entries):
        val_images.extend(_expand_image_sources(entry))

    train_images = sorted(dict.fromkeys(train_images))[:max_items]
    val_images = sorted(dict.fromkeys(val_images))[:max_items]
    if not train_images or not val_images:
        raise FileNotFoundError(f"Unable to build smoke subset from {source_yaml}")

    out_dir = smoke_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n", encoding="utf-8")

    output: dict[str, Any] = {
        "train": str(train_txt),
        "val": str(val_txt),
        "nc": raw.get("nc"),
        "names": raw.get("names"),
    }
    if raw.get("kpt_shape") is not None:
        output["kpt_shape"] = raw.get("kpt_shape")
    if raw.get("flip_idx") is not None:
        output["flip_idx"] = raw.get("flip_idx")

    smoke_yaml = out_dir / f"{tag}.yaml"
    smoke_yaml.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    return smoke_yaml


def _teacher_pt_exists(path: Path) -> bool:
    if path.is_file() and path.suffix.lower() == ".pt":
        return True
    if not path.is_dir():
        return False
    candidates = [
        path / "weights" / "best.pt",
        path / "best.pt",
        path / "weights" / "last.pt",
        path / "last.pt",
    ]
    if any(c.exists() for c in candidates):
        return True
    return any(True for _ in path.rglob("*.pt"))


def _build_train_cmd(spec: TrainSpec, batch: int, workers: int) -> list[str]:
    cmd = [
        sys.executable,
        "train_pose.py",
        "--model",
        str(spec.model),
        "--data",
        str(spec.data_yaml),
        "--task",
        spec.task,
        "--epochs",
        str(spec.epochs),
        "--batch",
        str(batch),
        "--imgsz",
        str(spec.imgsz),
        str(spec.imgsz),
        "--device",
        spec.device,
        "--workers",
        str(workers),
        "--project",
        str(spec.project),
        "--name",
        spec.name,
        "--qat-loss-mode",
        spec.mode,
        "--seed",
        str(spec.seed),
        "--close-mosaic",
        str(spec.close_mosaic),
        "--optimizer",
        spec.optimizer,
        "--lr0",
        str(spec.lr0),
        "--lrf",
        str(spec.lrf),
        "--momentum",
        str(spec.momentum),
        "--weight-decay",
        str(spec.weight_decay),
        "--export-data",
        str(spec.data_yaml),
        "--export-fraction",
        str(spec.export_fraction),
    ]
    if spec.mode == "kd-deploy" and spec.teacher_dir is not None:
        cmd.extend(["--qat-teacher-exported-dir", str(spec.teacher_dir)])
    if spec.mode == "kd-deploy":
        cmd.extend(["--qat-balance-log-interval", str(spec.qat_balance_log_interval)])
        if spec.kd_loss_composition is not None:
            cmd.extend(["--qat-kd-loss-composition", str(spec.kd_loss_composition)])

        # KD specifics (new)
        cmd.extend(["--qat-kd-temperature", str(spec.qat_kd_temperature)])
        cmd.extend(["--qat-kd-cls-distill", str(spec.qat_kd_cls_distill)])
        cmd.extend(["--qat-kd-dfl-distill", str(spec.qat_kd_dfl_distill)])
        cmd.extend(["--qat-kd-fg-threshold", str(spec.qat_kd_fg_threshold)])
        cmd.extend(["--qat-kd-fg-topk", str(spec.qat_kd_fg_topk)])
        cmd.extend(["--qat-kd-fg-min-pos", str(spec.qat_kd_fg_min_pos)])
        cmd.extend(["--qat-kd-fg-apply-to", str(spec.qat_kd_fg_apply_to)])

        if spec.qat_kd_weight is not None:
            cmd.extend(["--qat-kd-weight", str(spec.qat_kd_weight)])
        if spec.qat_balance_min is not None:
            cmd.extend(["--qat-balance-min", str(spec.qat_balance_min)])
        if spec.qat_balance_max is not None:
            cmd.extend(["--qat-balance-max", str(spec.qat_balance_max)])
        if spec.qat_balance_warmup_steps is not None:
            cmd.extend(["--qat-balance-warmup-steps", str(spec.qat_balance_warmup_steps)])
        if spec.qat_balance_max_step_change is not None:
            cmd.extend(["--qat-balance-max-step-change", str(spec.qat_balance_max_step_change)])
        if spec.qat_balance_adapt_power is not None:
            cmd.extend(["--qat-balance-adapt-power", str(spec.qat_balance_adapt_power)])
        if spec.qat_balance_strategy is not None:
            cmd.extend(["--qat-balance-strategy", str(spec.qat_balance_strategy)])
        if spec.qat_balance_shared_group is not None:
            cmd.extend(["--qat-balance-shared-group", str(spec.qat_balance_shared_group)])
        if spec.qat_balance_deploy_ramp_steps is not None:
            cmd.extend(["--qat-balance-deploy-ramp-steps", str(spec.qat_balance_deploy_ramp_steps)])
        if spec.qat_balance_update_interval is not None:
            cmd.extend(["--qat-balance-update-interval", str(spec.qat_balance_update_interval)])


    return cmd


def _run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(
            cmd,
            cwd=str(QAT_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(result.returncode)


def _tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:].lower()


def _looks_like_oom(log_tail: str) -> bool:
    return any(token in log_tail for token in OOM_HINTS)


def _run_train_with_retry(
    spec: TrainSpec,
    *,
    dry_run: bool,
    max_attempts: int = 3,
) -> tuple[int, int]:
    batch = spec.batch
    workers = spec.workers
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("YOLO_CONFIG_DIR", str((QAT_ROOT / ".ultralytics").resolve()))

    for attempt in range(1, max_attempts + 1):
        cmd = _build_train_cmd(spec, batch=batch, workers=workers)
        log_path = spec.project / "_logs" / f"{spec.name}.attempt{attempt}.log"
        logging.info(
            "[Train] %s/%s attempt=%d batch=%d workers=%d",
            spec.dataset,
            spec.variant,
            attempt,
            batch,
            workers,
        )
        if dry_run:
            logging.info("[DryRun] %s", " ".join(cmd))
            return batch, workers

        rc = _run_logged(cmd, log_path, env)
        if rc == 0:
            return batch, workers

        tail = _tail_text(log_path)
        if attempt >= max_attempts or not _looks_like_oom(tail):
            raise RuntimeError(
                f"Training failed ({spec.name}) rc={rc}. See log: {log_path}"
            )

        new_batch = max(8, batch // 2) if batch > 8 else batch
        batch = new_batch
        workers = 0
        logging.warning(
            "[Train] OOM-like failure detected, retrying with batch=%d workers=%d",
            batch,
            workers,
        )

    raise RuntimeError(f"Training retries exhausted: {spec.name}")


def _resolve_best_weight(run_dir: Path) -> Path:
    candidates = [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _export_tflite(
    *,
    model_pt: Path,
    task: str,
    data_yaml: Path,
    imgsz: int,
    int8: bool,
    fraction: float,
) -> Path:
    from ultralytics import YOLO
    
    # 1. 載入模型
    model = YOLO(str(model_pt), task=task)

    # -------------------------------------------------------------
    # [關鍵修改] 匯出前，強制將 CIRA 模組切換為 "Fallback" 模式
    # 這樣會把 deform_conv2d 替換為標準 conv2d，解決 ONNX/TFLite 報錯
    # -------------------------------------------------------------
    print(f"[CIRA Export] Enabling Force Fallback for TFLite compatibility...")
    for m in model.model.modules():
        if hasattr(m, 'force_fallback'):
            m.force_fallback = True  # 強制切換為標準卷積
            
    # -------------------------------------------------------------

    kwargs: dict[str, Any] = {
        "format": "tflite",
        "imgsz": int(imgsz),
        "data": str(data_yaml),
        "fraction": float(fraction),
        "nms": False,
    }
    if int8:
        kwargs["int8"] = True
        
    # 執行匯出 (現在不會報錯了，因為它變成了純 CNN)
    export_path = Path(str(model.export(**kwargs)))
    
    if not export_path.exists():
        raise FileNotFoundError(f"TFLite export path missing: {export_path}")
    return export_path


def _export_with_retry(
    *,
    model_pt: Path,
    task: str,
    data_yaml: Path,
    imgsz: int,
    int8: bool,
    fraction: float,
) -> Path:
    current_fraction = fraction
    attempts = 3 if int8 else 1
    for attempt in range(1, attempts + 1):
        try:
            return _export_tflite(
                model_pt=model_pt,
                task=task,
                data_yaml=data_yaml,
                imgsz=imgsz,
                int8=int8,
                fraction=current_fraction,
            )
        except Exception as exc:
            if attempt >= attempts:
                raise
            msg = str(exc).lower()
            if not _looks_like_oom(msg):
                raise
            current_fraction = max(0.1, current_fraction * 0.5)
            logging.warning(
                "[Export] INT8 export retry due to OOM-like error, fraction=%.3f",
                current_fraction,
            )
    raise RuntimeError("unexpected export retry flow")


def _dequantize_output(out: np.ndarray, quant: tuple[float, int]) -> np.ndarray:
    scale, zero = quant
    if out.dtype == np.float32 or scale == 0:
        return out.astype(np.float32)
    return (out.astype(np.float32) - float(zero)) * float(scale)


def _prepare_input(input_details: dict[str, Any], src: np.ndarray) -> np.ndarray:
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return src.astype(np.float32)

    scale, zero = input_details["quantization"]
    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max

    # Why: 部分匯出器會產生 scale=0 的非 float input（量測 latency/contract 不應因此整段變 NaN）。
    if scale == 0:
        q = np.round(src)
        return np.clip(q, qmin, qmax).astype(dtype)

    q = np.round(src / float(scale) + float(zero))
    return np.clip(q, qmin, qmax).astype(dtype)


def _new_tflite_interpreter(model_path: Path, *, num_threads: int | None = None):
    """
    Why: 量測/驗證不應綁死 TensorFlow；若環境只有 tflite_runtime 也能跑，避免整段回報 NaN。
    """
    kwargs: dict[str, Any] = {"model_path": str(model_path)}
    if num_threads is not None:
        kwargs["num_threads"] = int(num_threads)
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(**kwargs)
    except ModuleNotFoundError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            return Interpreter(**kwargs)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing TensorFlow and tflite_runtime; cannot run TFLite contract/latency measures."
            ) from exc
        
def _sanitize_tflite_shape(shape: Any) -> list[int]:
    out: list[int] = []
    for v in list(shape):
        iv = int(v)
        out.append(iv if iv > 0 else 1)
    return out

def _set_all_inputs_zero(interpreter, input_details: list[dict[str, Any]]) -> None:
    if not input_details:
        raise ValueError("TFLite model has no inputs.")
    for inp in input_details:
        shape = _sanitize_tflite_shape(inp["shape"])
        x = np.zeros(shape, dtype=np.float32)
        interpreter.set_tensor(inp["index"], _prepare_input(inp, x))

def _set_inputs_for_latency(interpreter, input_details: list[dict[str, Any]], *, seed: int) -> None:
    if not input_details:
        raise ValueError("TFLite model has no inputs.")
    rng = np.random.default_rng(seed)
    for i, inp in enumerate(input_details):
        shape = _sanitize_tflite_shape(inp["shape"])
        x = rng.random(shape, dtype=np.float32) if i == 0 else np.zeros(shape, dtype=np.float32)
        interpreter.set_tensor(inp["index"], _prepare_input(inp, x))

def _read_all_outputs_fp(interpreter, output_details: list[dict[str, Any]]) -> list[np.ndarray]:
    if not output_details:
        raise ValueError("TFLite model has no outputs.")
    outs: list[np.ndarray] = []
    for out in output_details:
        raw = interpreter.get_tensor(out["index"])
        outs.append(_dequantize_output(raw, out["quantization"]))
    return outs


def _tflite_latency_ms(model_path: Path, *, warmup: int = 8, runs: int = 40, seed: int = 0) -> float:
    interpreter = _new_tflite_interpreter(model_path, num_threads=1)
    interpreter.allocate_tensors()
    ins = interpreter.get_input_details()

    for _ in range(max(1, warmup)):
        _set_inputs_for_latency(interpreter, ins, seed=seed)
        interpreter.invoke()

    elapsed_ms: list[float] = []
    for _ in range(max(1, runs)):
        _set_inputs_for_latency(interpreter, ins, seed=seed)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000.0)
    return float(np.mean(elapsed_ms))


def _contract_check(model_path: Path) -> None:
    interpreter = _new_tflite_interpreter(model_path)
    interpreter.allocate_tensors()
    ins = interpreter.get_input_details()
    outs = interpreter.get_output_details()

    if not ins:
        raise ValueError(f"No input tensors found for {model_path}")
    if not outs:
        raise ValueError(f"No output tensors found for {model_path}")
    
    _set_all_inputs_zero(interpreter, ins)
    interpreter.invoke()

    out_fps = _read_all_outputs_fp(interpreter, outs)
    for idx, out_fp in enumerate(out_fps):
        if not np.all(np.isfinite(out_fp)):
            raise ValueError(f"Non-finite output detected in {model_path} (out#{idx})")
 

def _contract_jitter(fp32_model: Path, int8_model: Path, *, samples: int = 16, seed: int = 0) -> float:
    fp32_itp = _new_tflite_interpreter(fp32_model)
    int8_itp = _new_tflite_interpreter(int8_model)
    fp32_itp.allocate_tensors()
    int8_itp.allocate_tensors()

    fp32_ins = fp32_itp.get_input_details()
    int8_ins = int8_itp.get_input_details()
    fp32_outs = fp32_itp.get_output_details()
    int8_outs = int8_itp.get_output_details()

    if not fp32_ins or not int8_ins:
        raise ValueError("Missing input tensors for contract jitter.")
    if not fp32_outs or not int8_outs:
        raise ValueError("Missing output tensors for contract jitter.")
    if len(fp32_outs) != len(int8_outs):
        raise ValueError(
            f"Output count mismatch: fp32={len(fp32_outs)} int8={len(int8_outs)}"
        )

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(max(1, samples)):
        # Why: 用同一組輸入比較 fp32/int8 輸出差異，避免 input 隨機性污染 jitter 指標。
        for i, (fp_in, int8_in) in enumerate(zip(fp32_ins, int8_ins, strict=False)):
            shape = _sanitize_tflite_shape(fp_in["shape"])
            x = rng.random(shape, dtype=np.float32) if i == 0 else np.zeros(shape, dtype=np.float32)
            fp32_itp.set_tensor(fp_in["index"], _prepare_input(fp_in, x))
            int8_itp.set_tensor(int8_in["index"], _prepare_input(int8_in, x))

        fp32_itp.invoke()
        int8_itp.invoke()

        y_fp32_list = _read_all_outputs_fp(fp32_itp, fp32_outs)
        y_int8_list = _read_all_outputs_fp(int8_itp, int8_outs)
        total = 0.0
        denom = 0
        for y_fp32, y_int8 in zip(y_fp32_list, y_int8_list, strict=True):
            if y_fp32.shape != y_int8.shape:
                raise ValueError(f"Output shape mismatch: fp32={y_fp32.shape} int8={y_int8.shape}")
            w = int(y_fp32.size)
            total += float(np.mean(np.abs(y_fp32 - y_int8))) * w
            denom += w
        diffs.append(total / max(1, denom))

    return float(np.mean(diffs))

def _extract_ultralytics_map(metrics: Any, task: str) -> tuple[float, float]:
    # Why: Ultralytics 在 detect/pose 的 metric 容器不同，統一抽取 (map50, map50-95)。
    mobj = None
    if task == "pose" and hasattr(metrics, "pose"):
        mobj = getattr(metrics, "pose", None)
    if mobj is None and hasattr(metrics, "box"):
        mobj = getattr(metrics, "box", None)
    if mobj is None:
        return (float("nan"), float("nan"))
    map50 = float(getattr(mobj, "map50", float("nan")))
    map5095 = float(getattr(mobj, "map", float("nan")))
    return (map50, map5095)

def _val_tflite_map(
    *,
    model_path: Path,
    task: str,
    data_yaml: Path,
    imgsz: int,
    split: str,
    out_dir: Path,
    name: str,
) -> tuple[float, float]:
    model = YOLO(str(model_path), task=task)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=int(imgsz),
        split=str(split),
        device="cpu",
        batch=1,
        workers=0,
        verbose=False,
        plots=False,
        save=False,
        project=str(out_dir),
        name=str(name),
        exist_ok=True,
    )
    return _extract_ultralytics_map(metrics, task=task)

def _read_map_metric(run_dir: Path, task: str) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return float("nan")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return float("nan")

    row = {str(k).strip(): str(v).strip() for k, v in rows[-1].items()}
    if task == "pose":
        candidates = ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)"]
    else:
        candidates = ["metrics/mAP50-95(B)", "metrics/mAP50-95(P)"]

    for key in candidates:
        if key in row and row[key]:
            try:
                return float(row[key])
            except ValueError:
                continue
    return float("nan")


def _find_latest_attempt_log(spec: TrainSpec) -> Path | None:
    log_dir = spec.project / "_logs"
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob(f"{spec.name}.attempt*.log"))
    if not candidates:
        return None
    # Why: retry 可能產生 attempt2/3；以 mtime 選最新可對應實際成功那次
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_kd_stats_from_log(
    *,
    log_path: Path | None,
    max_weight: float,
) -> dict[str, float]:
    if log_path is None or (not log_path.exists()):
        return {
            "kd_log_steps": float("nan"),
            "alpha_kd_min": float("nan"),
            "alpha_kd_median": float("nan"),
            "alpha_kd_max": float("nan"),
            "alpha_kd_sat_ratio": float("nan"),
            "grad_ratio_median": float("nan"),
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    alphas: list[float] = []
    ratios: list[float] = []
    for m in _KD_LINE_RE.finditer(text):
        try:
            alphas.append(float(m.group("alpha")))
        except Exception:
            continue
        r = m.group("ratio")
        if r != "None":
            try:
                ratios.append(float(r))
            except Exception:
                pass

    if not alphas:
        return {
            "kd_log_steps": 0.0,
            "alpha_kd_min": float("nan"),
            "alpha_kd_median": float("nan"),
            "alpha_kd_max": float("nan"),
            "alpha_kd_sat_ratio": float("nan"),
            "grad_ratio_median": float("nan"),
        }

    a = np.asarray(alphas, dtype=np.float64)
    max_w = float(max_weight)
    sat_thr = max_w * (1.0 - 1e-6)
    sat_ratio = float(np.mean(a >= sat_thr))
    grad_med = float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else float("nan")
    return {
        "kd_log_steps": float(a.size),
        "alpha_kd_min": float(np.min(a)),
        "alpha_kd_median": float(np.median(a)),
        "alpha_kd_max": float(np.max(a)),
        "alpha_kd_sat_ratio": sat_ratio,
        "grad_ratio_median": grad_med,
    }


def _now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stable_field_order(fieldnames: set[str]) -> list[str]:
    base = [
        "run_ts",
        "run_id",
        "study",
        "dataset",
        "variant",
        "mode",
        "seed",
        "run_dir",
        "best_pt",
        "fp32_tflite",
        "int8_tflite",
        "artifact_dir",
        "artifact_best_pt",
        "artifact_fp32_tflite",
        "artifact_int8_tflite",
        "artifact_err",
        "map50_95",
        "kd_log_steps",
        "alpha_kd_min",
        "alpha_kd_median",
        "alpha_kd_max",
        "alpha_kd_sat_ratio",
        "grad_ratio_median",
        "lat_fp32_ms",
        "lat_int8_ms",
        "contract_jitter",
        "tflite_map_ok",
        "tflite_map_err",
        "map50_fp32_tflite",
        "map50_95_fp32_tflite",
        "map50_int8_tflite",
        "map50_95_int8_tflite",
        "export_ok",
        "contract_ok",
        "latency_ok",
        "export_err",
        "contract_err",
        "latency_err",
    ]
    ordered: list[str] = [k for k in base if k in fieldnames]
    tail = sorted(k for k in fieldnames if k not in set(base))
    ordered.extend(tail)
    return ordered


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fields: set[str] = set()
    for r in rows:
        fields.update(r.keys())
    fieldnames = _stable_field_order(fields)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def _as_float(value: Any) -> float:
    # Why: 欄位擴充後舊列可能是 ""，直接 float("") 會炸；統一轉成 NaN。
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _append_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Why: history CSV 欄位若隨版本增加，直接 append 會造成欄位錯位；採用「讀舊 + 合併 + 重寫」保證一致。
    existing_rows = _read_csv_rows(out_path) if out_path.exists() else []
    merged = existing_rows + rows
    _write_csv(merged, out_path)
def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    study = str(row.get("study", ""))
    dataset = str(row.get("dataset", ""))
    variant = str(row.get("variant", ""))
    mode = str(row.get("mode", ""))
    try:
        seed = int(row.get("seed", 0))
    except Exception:
        seed = 0
    return (study, dataset, variant, mode, seed)


def _select_newer_row(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    # Why: 以 run_ts 決定最新；若缺失則一律以 new 覆蓋，避免舊資料壓回去。
    old_ts = str(old.get("run_ts", ""))
    new_ts = str(new.get("run_ts", ""))
    if not old_ts:
        return new
    if not new_ts:
        return old
    return new if new_ts >= old_ts else old


def _upsert_latest(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    # preserve old first
    for r in existing:
        index[_row_key(r)] = r
    for r in incoming:
        k = _row_key(r)
        if k in index:
            index[k] = _select_newer_row(index[k], r)
        else:
            index[k] = r
    # stable ordering for determinism
    out = list(index.values())
    out.sort(key=lambda r: (
        str(r.get("study","")),
        str(r.get("dataset","")),
        str(r.get("variant","")),
        str(r.get("mode","")),
        int(r.get("seed",0) or 0),
    ))
    return out


def _make_run_id(spec: TrainSpec, run_ts: str) -> str:
    safe = run_ts.replace(":", "").replace("-", "").replace("+", "").replace("T", "_")
    return f"{spec.name}_{safe}"


def _snapshot_artifacts(
    *,
    report_root: Path,
    spec: TrainSpec,
    run_id: str,
    best_pt: Path,
    fp32_tflite: Optional[Path],
    int8_tflite: Optional[Path],
) -> dict[str, str]:
    # Why: 避免搬移資料後 run_dir 路徑失效；保留論文可追溯 artifacts。
    artifacts_root = report_root / "artifacts"
    dst_dir = artifacts_root / spec.study / spec.dataset / spec.variant / spec.mode / f"seed{spec.seed}" / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, str] = {"artifact_dir": str(dst_dir.relative_to(report_root))}

    dst_best = dst_dir / "best.pt"
    shutil.copy2(best_pt, dst_best)
    out["artifact_best_pt"] = str(dst_best.relative_to(report_root))

    if fp32_tflite is not None and fp32_tflite.exists():
        dst_fp32 = dst_dir / "model_fp32.tflite"
        shutil.copy2(fp32_tflite, dst_fp32)
        out["artifact_fp32_tflite"] = str(dst_fp32.relative_to(report_root))

    if int8_tflite is not None and int8_tflite.exists():
        dst_int8 = dst_dir / "model_int8.tflite"
        shutil.copy2(int8_tflite, dst_int8)
        out["artifact_int8_tflite"] = str(dst_int8.relative_to(report_root))

    return out


def _compute_deltas(rows: list[dict[str, Any]], study: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["study"] != study:
            continue
        key = (str(row["dataset"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    out: list[dict[str, Any]] = []
    if study == "A":
        baseline = "yolo"
        for (dataset, seed), variants in sorted(grouped.items()):
            if baseline not in variants:
                continue
            rhs = variants[baseline]
            for lhs_name, lhs in sorted(variants.items()):
                if lhs_name == baseline:
                    continue
                out.append(
                    {
                        "study": study,
                        "dataset": dataset,
                        "seed": seed,
                        "lhs": lhs_name,
                        "rhs": baseline,
                        "delta_map50_95": _as_float(lhs.get("map50_95")) - _as_float(rhs.get("map50_95")),
                        "delta_int8_map50": _as_float(lhs.get("map50_int8_tflite")) - _as_float(rhs.get("map50_int8_tflite")),
                        "delta_int8_map50_95": _as_float(lhs.get("map50_95_int8_tflite")) - _as_float(rhs.get("map50_95_int8_tflite")),
                        "delta_int8_latency_ms": _as_float(lhs.get("lat_int8_ms")) - _as_float(rhs.get("lat_int8_ms")),
                        "delta_contract_jitter": _as_float(lhs.get("contract_jitter")) - _as_float(rhs.get("contract_jitter")),
                    }
                )
        return out
    elif study == "B":
        left, right = "kd_deploy", "deploy_only"
    else:
        return out

    for (dataset, seed), variants in sorted(grouped.items()):
        if left not in variants or right not in variants:
            continue
        lhs = variants[left]
        rhs = variants[right]
        out.append(
            {
                "study": study,
                "dataset": dataset,
                "seed": seed,
                "lhs": left,
                "rhs": right,
                "delta_map50_95": _as_float(lhs.get("map50_95")) - _as_float(rhs.get("map50_95")),
                "delta_int8_map50": _as_float(lhs.get("map50_int8_tflite")) - _as_float(rhs.get("map50_int8_tflite")),
                "delta_int8_map50_95": _as_float(lhs.get("map50_95_int8_tflite")) - _as_float(rhs.get("map50_95_int8_tflite")),
                "delta_int8_latency_ms": _as_float(lhs.get("lat_int8_ms")) - _as_float(rhs.get("lat_int8_ms")),
                "delta_contract_jitter": _as_float(lhs.get("contract_jitter")) - _as_float(rhs.get("contract_jitter")),

            }
        )
    return out


def _compute_deltas_with_baseline(
    rows: list[dict[str, Any]],
    study: str,
    *,
    baseline_variant: str,
) -> list[dict[str, Any]]:
    """
    Why: Study-B variants are now CLI-gated. Baseline may be deploy_only/kd_only/kd_deploy,
    so delta computation must not assume a fixed pair.
    """
    if study != "B":
        return _compute_deltas(rows, study)

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["study"] != "B":
            continue
        key = (str(row["dataset"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    out: list[dict[str, Any]] = []
    for (dataset, seed), variants in sorted(grouped.items()):
        if baseline_variant not in variants:
            continue
        rhs = variants[baseline_variant]
        for lhs_name, lhs in sorted(variants.items()):
            if lhs_name == baseline_variant:
                continue
            out.append(
                {
                    "study": "B",
                    "dataset": dataset,
                    "seed": seed,
                    "lhs": lhs_name,
                    "rhs": baseline_variant,
                    "delta_map50_95": _as_float(lhs.get("map50_95")) - _as_float(rhs.get("map50_95")),
                    "delta_int8_map50": _as_float(lhs.get("map50_int8_tflite")) - _as_float(rhs.get("map50_int8_tflite")),
                    "delta_int8_map50_95": _as_float(lhs.get("map50_95_int8_tflite")) - _as_float(rhs.get("map50_95_int8_tflite")),
                    "delta_int8_latency_ms": _as_float(lhs.get("lat_int8_ms")) - _as_float(rhs.get("lat_int8_ms")),
                    "delta_contract_jitter": _as_float(lhs.get("contract_jitter")) - _as_float(rhs.get("contract_jitter")),
                }
            )
    return out


def _normalize_study_b_variant_name(name: str) -> str:
    raw = str(name).strip()
    if raw == "kd_only":
        return "KdDepoly_half"
    return raw


def _build_specs(
    *,
    datasets: list[str],
    studies: list[str],
    seeds: list[int],
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    close_mosaic: int,
    optimizer: str,
    lr0: float,
    lrf: float,
    momentum: float,
    weight_decay: float,
    export_fraction: float,
    data_root: Path,
    device_acc: str,
    device_kitti: str,
    acc_data: Path,
    kitti_data: Path,
    acc_teacher: Path,
    kitti_teacher: Path,
    kitti_student_model: Path,
    acc_student_model: Path,
    kitti_mobilenetv3_model: Optional[Path],
    kitti_ghostnetv2_model: Optional[Path],
    kitti_shufflenetv2_model: Optional[Path],
    kitti_cira_lite_model: Optional[Path],
    qat_kd_weight: Optional[float],

    qat_kd_temperature: float,
    qat_kd_cls_distill: str,
    qat_kd_dfl_distill: str,
    qat_kd_fg_threshold: float,
    qat_kd_fg_topk: int,
    qat_kd_fg_min_pos: int,
    qat_kd_fg_apply_to: str,
    include_a_cira: bool,
    include_a_kitti_cira_lite: bool,
    include_a_kitti_mobilenetv3: bool,
    include_a_kitti_ghostnetv2: bool,
    include_a_kitti_shufflenetv2: bool,
    include_b_deploy_only: bool,
    include_b_kd_only: bool,
    include_b_pure_kd: bool,
    include_b_kd_deploy: bool,
    b_kd_only_weight: float,

    qat_balance_log_interval: int,
    qat_balance_min: Optional[float],
    qat_balance_max: Optional[float],
    qat_balance_warmup_steps: Optional[int],
    qat_balance_max_step_change: Optional[float],
    qat_balance_adapt_power: Optional[float],
    qat_balance_strategy: Optional[str],
    qat_balance_shared_group: Optional[str],
    qat_balance_deploy_ramp_steps: Optional[int],
    qat_balance_update_interval: Optional[int],
) -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    model_cfg = {
        "acc": {
            "task": "pose",
            "data_yaml": acc_data,
            "device": device_acc,
            "yolo": QAT_ROOT / "ultralytics/cfg/models/v8/yolov8-pose.yaml",
            "cira": QAT_ROOT / "ultralytics/cfg/models/Yojui/yolov8_CIRA-Pose.yaml",
        },
        "kitti": {
            "task": "detect",
            "data_yaml": kitti_data,
            "device": device_kitti,
            "yolo": QAT_ROOT / "ultralytics/cfg/models/v8/yolov8.yaml",
            "cira": QAT_ROOT / "ultralytics/cfg/models/Yojui/yolov8_CIRA-Detect.yaml",
             **(
                 {"cira-lite": kitti_cira_lite_model}
                 if kitti_cira_lite_model is not None
                 else {}
             ),
             **(
                 {"mobilenetv3": kitti_mobilenetv3_model}
                 if kitti_mobilenetv3_model is not None
                 else {}
             ),
             **(
                 {"ghostnetv2": kitti_ghostnetv2_model}
                 if kitti_ghostnetv2_model is not None
                 else {}
             ),
             **(
                 {"shufflenetv2": kitti_shufflenetv2_model}
                 if kitti_shufflenetv2_model is not None
                 else {}
             ),
        },
    }

    for dataset in datasets:
        cfg = model_cfg[dataset]
        for seed in seeds:
            common = dict(
                dataset=dataset,
                task=str(cfg["task"]),
                data_yaml=Path(cfg["data_yaml"]),
                device=str(cfg["device"]),
                seed=int(seed),
                epochs=int(epochs),
                batch=int(batch),
                imgsz=int(imgsz),
                workers=int(workers),
                close_mosaic=int(close_mosaic),
                optimizer=str(optimizer),
                lr0=float(lr0),
                lrf=float(lrf),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                export_fraction=float(export_fraction),
                qat_kd_weight=qat_kd_weight,

                qat_kd_temperature=float(qat_kd_temperature),
                qat_kd_cls_distill=str(qat_kd_cls_distill),
                qat_kd_dfl_distill=str(qat_kd_dfl_distill),
                qat_kd_fg_threshold=float(qat_kd_fg_threshold),
                qat_kd_fg_topk=int(qat_kd_fg_topk),
                qat_kd_fg_min_pos=int(qat_kd_fg_min_pos),
                qat_kd_fg_apply_to=str(qat_kd_fg_apply_to),

                qat_balance_log_interval=int(qat_balance_log_interval),
                qat_balance_min=qat_balance_min,
                qat_balance_max=qat_balance_max,
                qat_balance_warmup_steps=qat_balance_warmup_steps,
                qat_balance_max_step_change=qat_balance_max_step_change,
                qat_balance_adapt_power=qat_balance_adapt_power,
                qat_balance_strategy=qat_balance_strategy,
                qat_balance_shared_group=qat_balance_shared_group,
                qat_balance_deploy_ramp_steps=qat_balance_deploy_ramp_steps,
                qat_balance_update_interval=qat_balance_update_interval,
            )

            if "A" in studies:
                # Study A: yolo baseline is always included. Other variants require explicit include flags.
                if dataset == "kitti":
                    variant_keys = ["yolo"]
                    if include_a_cira and "cira" in cfg:
                        variant_keys.append("cira")
                    if include_a_kitti_cira_lite and "cira-lite" in cfg:
                        variant_keys.append("cira-lite")
                    if include_a_kitti_mobilenetv3 and "mobilenetv3" in cfg:
                        variant_keys.append("mobilenetv3")
                    if include_a_kitti_ghostnetv2 and "ghostnetv2" in cfg:
                        variant_keys.append("ghostnetv2")
                    if include_a_kitti_shufflenetv2 and "shufflenetv2" in cfg:
                        variant_keys.append("shufflenetv2")
                else:
                    variant_keys = ["yolo"]
                    if include_a_cira and "cira" in cfg:
                        variant_keys.append("cira")

                for variant_key in variant_keys:
                    specs.append(
                        TrainSpec(
                            study="A",
                            variant=variant_key,
                            mode="original",
                            model=Path(cfg[variant_key]).resolve(),
                            project=data_root / "paper_runs" / dataset / "A_model_compare",
                            name=f"A_{dataset}_{variant_key}_seed{seed}",
                            teacher_dir=None,
                            **common,
                        )
                    )

            if "B" in studies:
                student_model = acc_student_model if dataset == "acc" else kitti_student_model
                teacher_dir = acc_teacher if dataset == "acc" else kitti_teacher

                if include_b_deploy_only:
                    specs.append(
                        TrainSpec(
                            study="B",
                            variant="deploy_only",
                            mode="original",
                            model=student_model,
                            project=data_root / "paper_runs" / dataset / "B_kd_vs_deploy",
                            name=f"B_{dataset}_deploy_only_seed{seed}",
                            teacher_dir=None,
                            **common,
                        )
                    )

                if include_b_kd_only:
                    kd_only_common = dict(common)
                    kd_only_common["qat_kd_weight"] = float(b_kd_only_weight)
                    # Why: 固定 alpha 的 run，避免動態 balance 參數造成解讀混亂
                    kd_only_common["qat_balance_min"] = 1.0
                    kd_only_common["qat_balance_max"] = 1.0
                    kd_only_common["qat_balance_warmup_steps"] = 0
                    kd_only_common["qat_balance_deploy_ramp_steps"] = 0
                    kd_only_common["qat_balance_update_interval"] = 1
                    specs.append(
                        TrainSpec(
                            study="B",
                            variant="KdDepoly_half",
                            mode="kd-deploy",
                            model=student_model,
                            project=data_root / "paper_runs" / dataset / "B_kd_vs_deploy",
                            name=f"B_{dataset}_KdDepoly_half_seed{seed}",
                            teacher_dir=teacher_dir,
                            kd_loss_composition="fixed_kd_deploy",
                            **kd_only_common,
                        )
                    )

                if include_b_pure_kd:
                    pure_kd_common = dict(common)
                    pure_kd_common["qat_kd_weight"] = None
                    specs.append(
                        TrainSpec(
                            study="B",
                            variant="pure_kd",
                            mode="kd-deploy",
                            model=student_model,
                            project=data_root / "paper_runs" / dataset / "B_kd_vs_deploy",
                            name=f"B_{dataset}_pure_kd_seed{seed}",
                            teacher_dir=teacher_dir,
                            kd_loss_composition="pure_kd",
                            **pure_kd_common,
                        )
                    )

                if include_b_kd_deploy:
                    kd_deploy_common = dict(common)
                    kd_deploy_common["qat_kd_weight"] = None
                    if kd_deploy_common["qat_balance_max"] is None:
                        kd_deploy_common["qat_balance_max"] = 1.25
                    else:
                        kd_deploy_common["qat_balance_max"] = min(float(kd_deploy_common["qat_balance_max"]), 1.25)
                    if kd_deploy_common["qat_balance_warmup_steps"] is None:
                        kd_deploy_common["qat_balance_warmup_steps"] = 4000
                    else:
                        kd_deploy_common["qat_balance_warmup_steps"] = max(int(kd_deploy_common["qat_balance_warmup_steps"]), 4000)
                    if kd_deploy_common["qat_balance_deploy_ramp_steps"] is None:
                        kd_deploy_common["qat_balance_deploy_ramp_steps"] = 1600
                    else:
                        kd_deploy_common["qat_balance_deploy_ramp_steps"] = max(
                            int(kd_deploy_common["qat_balance_deploy_ramp_steps"]), 1600
                        )
                    if kd_deploy_common["qat_balance_update_interval"] is None:
                        kd_deploy_common["qat_balance_update_interval"] = 20
                    else:
                        kd_deploy_common["qat_balance_update_interval"] = max(
                            int(kd_deploy_common["qat_balance_update_interval"]), 20
                        )
                    specs.append(
                        TrainSpec(
                            study="B",
                            variant="kd_deploy",
                            mode="kd-deploy",
                            model=student_model,
                            project=data_root / "paper_runs" / dataset / "B_kd_vs_deploy",
                            name=f"B_{dataset}_kd_deploy_seed{seed}",
                            teacher_dir=teacher_dir,
                            kd_loss_composition="dynamic_kd_deploy",
                            **kd_deploy_common,
                        )
                    )
    return specs


def _require_paths(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n- " + "\n- ".join(missing))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click paper experiment runner for A/B studies with FP32+INT8 exports and summary report."
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--export-fraction", type=float, default=0.25)
    parser.add_argument(
        "--eval-tflite-map",
        action="store_true",
        help="Evaluate exported TFLite (FP32/INT8) via Ultralytics val() and write mAP into CSV.",
    )
    parser.add_argument(
        "--tflite-map-split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split used for TFLite mAP evaluation.",
    )
    parser.add_argument(
        "--qat-kd-weight",
        type=float,
        default=None,
        help="Optional fixed alpha_kd passed to kd-deploy runs.",
    )

    # Study-B extra variant
    parser.add_argument(
        "--include-b-kd-only",
        dest="include_b_kd_only",
        action="store_true",
        help="Include Study-B variant 'KdDepoly_half' (fixed alpha KD+deploy; deprecated kd_only alias).",
    )
    # Backward-compatible alias
    parser.add_argument("--include-kd-only", dest="include_b_kd_only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-b-pure-kd",
        action="store_true",
        help="Include Study-B variant 'pure_kd' (reserved for true pure-KD mode).",
    )
    parser.add_argument(
        "--include-b-deploy-only",
        action="store_true",
        help="Include Study-B variant 'deploy_only' (original loss, no KD).",
    )
    parser.add_argument(
        "--include-b-kd-deploy",
        action="store_true",
        help="Include Study-B variant 'kd_deploy' (dynamic KD+deploy balancing).",
    )
    parser.add_argument(
        "--b-kd-only-weight",
        type=float,
        default=1.0,
        help="Fixed alpha for Study-B 'KdDepoly_half' variant (default=1.0).",
    )
    parser.add_argument(
        "--b-delta-baseline",
        type=str,
        default="deploy_only",
        choices=["deploy_only", "kd_only", "KdDepoly_half", "pure_kd", "kd_deploy"],
        help="Baseline variant for Study-B delta report.",
    )

    # KD distillation specifics (forward to train_pose.py; used in kd-deploy runs)
    parser.add_argument("--qat-kd-temperature", type=float, default=1.0)
    parser.add_argument("--qat-kd-cls-distill", type=str, default="bce", choices=["bce", "softmax_kl"])
    parser.add_argument("--qat-kd-dfl-distill", type=str, default="kldiv", choices=["kldiv", "smoothl1"])
    parser.add_argument("--qat-kd-fg-threshold", type=float, default=0.0)
    parser.add_argument("--qat-kd-fg-topk", type=int, default=0)
    parser.add_argument("--qat-kd-fg-min-pos", type=int, default=0)
    parser.add_argument("--qat-kd-fg-apply-to", type=str, default="cls", choices=["cls", "dfl", "both"])

    parser.add_argument(
        "--qat-balance-log-interval",
        type=int,
        default=50,
        help="Step interval for KD scalars logging in kd-deploy runs.",
    )
    parser.add_argument(
        "--qat-balance-min",
        type=float,
        default=0.2,
        help="Lower bound for alpha_kd (must be >= 0).",
    )
    parser.add_argument(
        "--qat-balance-max",
        type=float,
        default=5.0,
        help="Upper bound for alpha_kd (must be >= qat-balance-min).",
    )
    parser.add_argument(
        "--qat-balance-warmup-steps",
        type=int,
        default=0,
        help="Number of initial steps to keep balance weights unchanged.",
    )
    parser.add_argument(
        "--qat-balance-max-step-change",
        type=float,
        default=1.2,
        help="Maximum multiplicative change per update step (must be >= 1).",
    )
    parser.add_argument(
        "--qat-balance-adapt-power",
        type=float,
        default=0.5,
        help="Smoothing factor for dynamic balancing updates, must be in (0,1].",
    )
    parser.add_argument(
        "--qat-balance-strategy",
        type=str,
        default="grad_norm",
        choices=["grad_norm", "dwa", "ratio"],
        help="Balancing strategy in kd-deploy mode.",
    )
    parser.add_argument(
        "--qat-balance-shared-group",
        type=str,
        default="head",
        choices=["head", "all"],
        help="Shared parameter group used by grad-norm balancing.",
    )
    parser.add_argument(
        "--qat-balance-deploy-ramp-steps",
        type=int,
        default=1000,
        help="Ramp up steps for deployment loss (must be >= 0).",
    )
    parser.add_argument(
        "--qat-balance-update-interval",
        type=int,
        default=10,
        help="Interval (steps) for balance updates (must be >= 1).",
    )
    parser.add_argument("--device-acc", type=str, default="0")
    parser.add_argument("--device-kitti", type=str, default="1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--datasets", type=str, default="acc,kitti")
    parser.add_argument("--studies", type=str, default="A,B")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--acc-teacher",
        type=str,
        default=str(
            QAT_ROOT
            / "Teacher-model/acc-dataset/carkeypoint-20251207-Rep-AutoLabel-192GFlops"
        ),
    )
    parser.add_argument("--kitti-teacher", type=str, default="")
    parser.add_argument(
        "--kitti-teacher-model",
        type=str,
        default=str(QAT_ROOT / "ultralytics/cfg/models/Yojui/yolov8_CIRA-Detect.yaml"),
    )
    parser.add_argument("--teacher-epochs", type=int, default=0)
    parser.add_argument(
        "--acc-student-model",
        type=str,
        default=str(QAT_ROOT / "ultralytics/cfg/models/v8/yolov8-pose.yaml"),
    )
    parser.add_argument(
        "--kitti-student-model",
        type=str,
        default=str(QAT_ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"),
    )
    parser.add_argument(
        "--kitti-cira-lite-model",
        type=str,
        default=str(QAT_ROOT / "ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml"),
    )
    parser.add_argument(
        "--include-a-cira",
        action="store_true",
        help="Include Study-A CIRA variant (acc and kitti).",
    )
    parser.add_argument(
        "--include-a-kitti-cira-lite",
        action="store_true",
        help="Include Study-A KITTI cira-lite variant.",
    )
    parser.add_argument(
        "--include-a-kitti-mobilenetv3",
        action="store_true",
        help="Include Study-A KITTI mobilenetv3 variant.",
    )
    parser.add_argument(
        "--include-a-kitti-ghostnetv2",
        action="store_true",
        help="Include Study-A KITTI ghostnetv2 variant.",
    )
    parser.add_argument(
        "--include-a-kitti-shufflenetv2",
        action="store_true",
        help="Include Study-A KITTI shufflenetv2 variant.",
    )
    # Study A additional baselines (optional model paths)
    parser.add_argument("--kitti-mobilenetv3-model", type=str, default="")
    parser.add_argument("--kitti-ghostnetv2-model", type=str, default="")
    parser.add_argument("--kitti-shufflenetv2-model", type=str, default="")


    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.qat_kd_weight is not None and float(args.qat_kd_weight) < 0.0:
        raise ValueError(f"--qat-kd-weight must be >= 0, got {args.qat_kd_weight}")
    if int(args.qat_balance_log_interval) < 1:
        raise ValueError(
            f"--qat-balance-log-interval must be >= 1, got {args.qat_balance_log_interval}"
        )

    if args.qat_balance_min is not None and float(args.qat_balance_min) < 0.0:
        raise ValueError(f"--qat-balance-min must be >= 0, got {args.qat_balance_min}")
    if args.qat_balance_max is not None and float(args.qat_balance_max) < 0.0:
        raise ValueError(f"--qat-balance-max must be >= 0, got {args.qat_balance_max}")
    if (
        args.qat_balance_min is not None
        and args.qat_balance_max is not None
        and float(args.qat_balance_min) > float(args.qat_balance_max)
    ):
        raise ValueError(
            f"--qat-balance-min must be <= --qat-balance-max, got {args.qat_balance_min} > {args.qat_balance_max}"
        )
    if args.qat_balance_warmup_steps is not None and int(args.qat_balance_warmup_steps) < 0:
        raise ValueError(
            f"--qat-balance-warmup-steps must be >= 0, got {args.qat_balance_warmup_steps}"
        )
    if args.qat_balance_max_step_change is not None and float(args.qat_balance_max_step_change) < 1.0:
        raise ValueError(
            f"--qat-balance-max-step-change must be >= 1.0, got {args.qat_balance_max_step_change}"
        )
    if args.qat_balance_adapt_power is not None and not (0.0 < float(args.qat_balance_adapt_power) <= 1.0):
        raise ValueError(
            f"--qat-balance-adapt-power must be in (0,1], got {args.qat_balance_adapt_power}"
        )

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    report_root = data_root / "paper_reports"
    report_root.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv_list(args.datasets)
    studies = _parse_csv_list(args.studies)
    seeds = _parse_seeds(args.seeds)

    # KD args validation (defensive; keeps paper runner fail-fast)
    if float(args.qat_kd_temperature) <= 0.0:
        raise ValueError(f"--qat-kd-temperature must be > 0, got {args.qat_kd_temperature}")
    thr = float(args.qat_kd_fg_threshold)
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f"--qat-kd-fg-threshold must be in [0,1], got {args.qat_kd_fg_threshold}")
    if int(args.qat_kd_fg_topk) < 0:
        raise ValueError(f"--qat-kd-fg-topk must be >= 0, got {args.qat_kd_fg_topk}")
    if int(args.qat_kd_fg_min_pos) < 0:
        raise ValueError(f"--qat-kd-fg-min-pos must be >= 0, got {args.qat_kd_fg_min_pos}")
    if args.include_b_kd_only and float(args.b_kd_only_weight) < 0.0:
        raise ValueError(f"--b-kd-only-weight must be >= 0, got {args.b_kd_only_weight}")

    for dataset in datasets:
        if dataset not in {"acc", "kitti"}:
            raise ValueError(f"Unsupported dataset: {dataset}")
    for study in studies:
        if study not in {"A", "B"}:
            raise ValueError(f"Unsupported study: {study}")

    if args.include_a_kitti_mobilenetv3 and not str(args.kitti_mobilenetv3_model).strip():
        raise ValueError("--include-a-kitti-mobilenetv3 requires --kitti-mobilenetv3-model")
    if args.include_a_kitti_ghostnetv2 and not str(args.kitti_ghostnetv2_model).strip():
        raise ValueError("--include-a-kitti-ghostnetv2 requires --kitti-ghostnetv2-model")
    if args.include_a_kitti_shufflenetv2 and not str(args.kitti_shufflenetv2_model).strip():
        raise ValueError("--include-a-kitti-shufflenetv2 requires --kitti-shufflenetv2-model")

    selected_b_variants = [
        bool(args.include_b_deploy_only),
        bool(args.include_b_kd_only),
        bool(args.include_b_pure_kd),
        bool(args.include_b_kd_deploy),
    ]
    if "B" in studies and not any(selected_b_variants):
        raise ValueError(
            "Study-B requested but no variant selected. Add one or more of "
            "--include-b-deploy-only / --include-b-kd-only / --include-b-pure-kd / --include-b-kd-deploy."
        )

    if args.smoke:
        logging.info("[Smoke] Applying smoke defaults (epochs=1, batch=2, workers=0, single seed)")
        args.epochs = 1
        args.batch = 2
        args.workers = 0
        seeds = [seeds[0]]
        args.export_fraction = min(float(args.export_fraction), 0.25)

    acc_yaml = QAT_ROOT / "dataset/lanepose-carkeypoint.yaml"
    kitti_yaml = QAT_ROOT / "dataset/KITTI.yaml"
    kitti_teacher_provided = bool(args.kitti_teacher)
    args.b_delta_baseline = _normalize_study_b_variant_name(str(args.b_delta_baseline))
    needs_kd_teacher = "B" in studies and (
        bool(args.include_b_kd_only) or bool(args.include_b_pure_kd) or bool(args.include_b_kd_deploy)
    )

    required_paths: list[Path] = [
        QAT_ROOT / "train_pose.py",
        acc_yaml,
        kitti_yaml,
        Path(args.acc_student_model),
        Path(args.kitti_student_model),
    ]
    if "B" in studies and "acc" in datasets and needs_kd_teacher:
        required_paths.append(Path(args.acc_teacher))
    if "B" in studies and "kitti" in datasets and needs_kd_teacher:
        required_paths.append(Path(args.kitti_teacher_model))
    if args.include_a_kitti_cira_lite and str(args.kitti_cira_lite_model).strip():
        required_paths.append(Path(args.kitti_cira_lite_model))
    if args.include_a_kitti_mobilenetv3 and str(args.kitti_mobilenetv3_model).strip():
        required_paths.append(Path(args.kitti_mobilenetv3_model))
    if args.include_a_kitti_ghostnetv2 and str(args.kitti_ghostnetv2_model).strip():
        required_paths.append(Path(args.kitti_ghostnetv2_model))
    if args.include_a_kitti_shufflenetv2 and str(args.kitti_shufflenetv2_model).strip():
        required_paths.append(Path(args.kitti_shufflenetv2_model))

    _require_paths(required_paths)

    if args.smoke:
        smoke_root = data_root / "_smoke_data"
        acc_yaml = _build_smoke_yaml(acc_yaml, smoke_root, "acc_pose_smoke")
        kitti_yaml = _build_smoke_yaml(kitti_yaml, smoke_root, "kitti_detect_smoke")
        logging.info("[Smoke] acc yaml: %s", acc_yaml)
        logging.info("[Smoke] kitti yaml: %s", kitti_yaml)

    acc_teacher = Path(args.acc_teacher).resolve()
    if "B" in studies and "acc" in datasets and needs_kd_teacher and not _teacher_pt_exists(acc_teacher):
        raise FileNotFoundError(
            f"ACC teacher path exists but no .pt found under: {acc_teacher}"
        )

    kitti_teacher = Path(args.kitti_teacher).resolve() if kitti_teacher_provided else None
    if "B" in studies and "kitti" in datasets and needs_kd_teacher and (
        kitti_teacher is None or not _teacher_pt_exists(kitti_teacher)
    ):
        teacher_epochs = int(args.teacher_epochs) if int(args.teacher_epochs) > 0 else int(args.epochs)
        teacher_spec = TrainSpec(
            study="B",
            dataset="kitti",
            variant="teacher",
            mode="original",
            model=Path(args.kitti_teacher_model).resolve(),
            data_yaml=kitti_yaml,
            task="detect",
            device=str(args.device_kitti),
            seed=seeds[0],
            epochs=teacher_epochs,
            batch=int(args.batch),
            imgsz=int(args.imgsz),
            workers=int(args.workers),
            close_mosaic=int(args.close_mosaic),
            optimizer=str(args.optimizer),
            lr0=float(args.lr0),
            lrf=float(args.lrf),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            export_fraction=float(args.export_fraction),
            qat_kd_weight=None,
            qat_kd_temperature=float(args.qat_kd_temperature),
            qat_kd_cls_distill=str(args.qat_kd_cls_distill),
            qat_kd_dfl_distill=str(args.qat_kd_dfl_distill),
            qat_kd_fg_threshold=float(args.qat_kd_fg_threshold),
            qat_kd_fg_topk=int(args.qat_kd_fg_topk),
            qat_kd_fg_min_pos=int(args.qat_kd_fg_min_pos),
            qat_kd_fg_apply_to=str(args.qat_kd_fg_apply_to),
            qat_balance_log_interval=int(args.qat_balance_log_interval),
            qat_balance_min=(None if args.qat_balance_min is None else float(args.qat_balance_min)),
            qat_balance_max=(None if args.qat_balance_max is None else float(args.qat_balance_max)),
            qat_balance_warmup_steps=(
                None if args.qat_balance_warmup_steps is None else int(args.qat_balance_warmup_steps)
            ),
            qat_balance_max_step_change=(
                None if args.qat_balance_max_step_change is None else float(args.qat_balance_max_step_change)
            ),
            qat_balance_adapt_power=(
                None if args.qat_balance_adapt_power is None else float(args.qat_balance_adapt_power)
            ),
            qat_balance_strategy=args.qat_balance_strategy,
            qat_balance_shared_group=args.qat_balance_shared_group,
            qat_balance_deploy_ramp_steps=(
                None if args.qat_balance_deploy_ramp_steps is None else int(args.qat_balance_deploy_ramp_steps)
            ),
            qat_balance_update_interval=(
                None if args.qat_balance_update_interval is None else int(args.qat_balance_update_interval)
            ),
            project=data_root / "paper_runs" / "kitti" / "teacher",
            name=f"B_kitti_teacher_seed{seeds[0]}",
            teacher_dir=None,
        )
        if args.skip_existing and (teacher_spec.run_dir / "weights" / "best.pt").exists():
            logging.info("[Teacher] Reuse existing KITTI teacher: %s", teacher_spec.run_dir)
        else:
            _run_train_with_retry(teacher_spec, dry_run=args.dry_run)
        kitti_teacher = teacher_spec.run_dir

    if "B" in studies and "kitti" in datasets and needs_kd_teacher and kitti_teacher is None:
        raise ValueError("KITTI teacher is required for study B")
    if (
        needs_kd_teacher
        and
        kitti_teacher is not None
        and (not args.dry_run or kitti_teacher_provided)
        and not _teacher_pt_exists(kitti_teacher)
    ):
        raise FileNotFoundError(f"KITTI teacher .pt not found: {kitti_teacher}")

    specs = _build_specs(
        datasets=datasets,
        studies=studies,
        seeds=seeds,
        epochs=int(args.epochs),
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        workers=int(args.workers),
        close_mosaic=int(args.close_mosaic),
        optimizer=str(args.optimizer),
        lr0=float(args.lr0),
        lrf=float(args.lrf),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        export_fraction=float(args.export_fraction),
        data_root=data_root,
        device_acc=str(args.device_acc),
        device_kitti=str(args.device_kitti),
        acc_data=acc_yaml,
        kitti_data=kitti_yaml,
        acc_teacher=acc_teacher,
        kitti_teacher=(kitti_teacher or Path(args.kitti_student_model).resolve()),
        kitti_student_model=Path(args.kitti_student_model).resolve(),
        acc_student_model=Path(args.acc_student_model).resolve(),
        kitti_mobilenetv3_model=(
            None
            if not str(args.kitti_mobilenetv3_model).strip()
            else Path(args.kitti_mobilenetv3_model).resolve()
        ),
        kitti_ghostnetv2_model=(
            None
            if not str(args.kitti_ghostnetv2_model).strip()
            else Path(args.kitti_ghostnetv2_model).resolve()
        ),
        kitti_shufflenetv2_model=(
            None
            if not str(args.kitti_shufflenetv2_model).strip()
            else Path(args.kitti_shufflenetv2_model).resolve()
        ),
        kitti_cira_lite_model=(
            None
            if not str(args.kitti_cira_lite_model).strip()
            else Path(args.kitti_cira_lite_model).resolve()
        ),
        qat_kd_weight=(
            None if args.qat_kd_weight is None else float(args.qat_kd_weight)
        ),

        qat_kd_temperature=float(args.qat_kd_temperature),
        qat_kd_cls_distill=str(args.qat_kd_cls_distill),
        qat_kd_dfl_distill=str(args.qat_kd_dfl_distill),
        qat_kd_fg_threshold=float(args.qat_kd_fg_threshold),
        qat_kd_fg_topk=int(args.qat_kd_fg_topk),
        qat_kd_fg_min_pos=int(args.qat_kd_fg_min_pos),
        qat_kd_fg_apply_to=str(args.qat_kd_fg_apply_to),
        include_a_cira=bool(args.include_a_cira),
        include_a_kitti_cira_lite=bool(args.include_a_kitti_cira_lite),
        include_a_kitti_mobilenetv3=bool(args.include_a_kitti_mobilenetv3),
        include_a_kitti_ghostnetv2=bool(args.include_a_kitti_ghostnetv2),
        include_a_kitti_shufflenetv2=bool(args.include_a_kitti_shufflenetv2),
        include_b_deploy_only=bool(args.include_b_deploy_only),
        include_b_kd_only=bool(args.include_b_kd_only),
        include_b_pure_kd=bool(args.include_b_pure_kd),
        include_b_kd_deploy=bool(args.include_b_kd_deploy),
        b_kd_only_weight=float(args.b_kd_only_weight),

        qat_balance_log_interval=int(args.qat_balance_log_interval),
        qat_balance_min=(None if args.qat_balance_min is None else float(args.qat_balance_min)),
        qat_balance_max=(None if args.qat_balance_max is None else float(args.qat_balance_max)),
        qat_balance_warmup_steps=(None if args.qat_balance_warmup_steps is None else int(args.qat_balance_warmup_steps)),
        qat_balance_max_step_change=(None if args.qat_balance_max_step_change is None else float(args.qat_balance_max_step_change)),
        qat_balance_adapt_power=(None if args.qat_balance_adapt_power is None else float(args.qat_balance_adapt_power)),
        qat_balance_strategy=args.qat_balance_strategy,
        qat_balance_shared_group=args.qat_balance_shared_group,
        qat_balance_deploy_ramp_steps=(None if args.qat_balance_deploy_ramp_steps is None else int(args.qat_balance_deploy_ramp_steps)),
        qat_balance_update_interval=(None if args.qat_balance_update_interval is None else int(args.qat_balance_update_interval)),

    )

    rows: list[dict[str, Any]] = []
    executed = 0
    for spec in specs:
        if args.max_runs is not None and executed >= int(args.max_runs):
            break
        executed += 1

        spec.project.mkdir(parents=True, exist_ok=True)
        
        # --- Logic: Check for existing checkpoint to support Resume ---
        checkpoint_found = False
        try:
            # 嘗試解析 best.pt (若不存在會拋出 FileNotFoundError)
            _resolve_best_weight(spec.run_dir)
            checkpoint_found = True
        except FileNotFoundError:
            checkpoint_found = False

        if args.skip_existing and checkpoint_found:
            logging.info(f"[Skip] Found existing checkpoint for {spec.name}, skipping training phase.")
        else:
            _run_train_with_retry(spec, dry_run=args.dry_run)

        if args.dry_run:
            continue

        run_dir = spec.run_dir
        # 如果訓練階段失敗，這裡會報錯；如果跳過訓練，這裡應該能找到權重
        best_pt = _resolve_best_weight(run_dir)
        exports_dir = run_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        run_ts = _now_ts()
        run_id = _make_run_id(spec, run_ts)

        # --- Logic: Fault-tolerant Export Block ---
        # Initialize metrics with NaN to handle failures gracefully
        fp32_copy = None
        int8_copy = None
        lat_fp32_ms = float("nan")
        lat_int8_ms = float("nan")
        contract_jitter = float("nan")
        export_ok = False
        contract_ok = False
        latency_ok = False
        export_err = ""
        contract_err = ""
        latency_err = ""
        tflite_map_ok = False
        tflite_map_err = "" if args.eval_tflite_map else "disabled (--eval-tflite-map not set)"
        map50_fp32_tflite = float("nan")
        map50_95_fp32_tflite = float("nan")
        map50_int8_tflite = float("nan")
        map50_95_int8_tflite = float("nan")
        artifact_dir = ""
        artifact_best_pt = ""
        artifact_fp32_tflite = ""
        artifact_int8_tflite = ""
        artifact_err = ""

        try:
            fp32_export = _export_with_retry(
                model_pt=best_pt,
                task=spec.task,
                data_yaml=spec.data_yaml,
                imgsz=spec.imgsz,
                int8=False,
                fraction=spec.export_fraction,
            )
            int8_export = _export_with_retry(
                model_pt=best_pt,
                task=spec.task,
                data_yaml=spec.data_yaml,
                imgsz=spec.imgsz,
                int8=True,
                fraction=spec.export_fraction,
            )

            fp32_copy = exports_dir / "model_fp32.tflite"
            int8_copy = exports_dir / "model_int8.tflite"
            fp32_copy.write_bytes(fp32_export.read_bytes())
            int8_copy.write_bytes(int8_export.read_bytes())

            export_ok = True
            # latency：不應被 contract 失敗牽連
            try:
                lat_fp32_ms = _tflite_latency_ms(fp32_copy)
                lat_int8_ms = _tflite_latency_ms(int8_copy)
                latency_ok = True
            except Exception as e:
                latency_err = repr(e)
                logging.error(f"[Latency Failed] {spec.name}@{spec.imgsz}: {e}")
            # contract + jitter：多輸出相容
            try:
                _contract_check(fp32_copy)
                _contract_check(int8_copy)
                contract_jitter = _contract_jitter(fp32_copy, int8_copy)
                contract_ok = True
            except Exception as e:
                contract_err = repr(e)
                logging.error(f"[Contract Failed] {spec.name}@{spec.imgsz}: {e}")
 
            # tflite mAP：不應被 contract/latency 失敗牽連
            if args.eval_tflite_map and export_ok:
                try:
                    out_dir = exports_dir / "tflite_val"
                    map50_fp32_tflite, map50_95_fp32_tflite = _val_tflite_map(
                        model_path=fp32_copy,
                        task=spec.task,
                        data_yaml=spec.data_yaml,
                        imgsz=spec.imgsz,
                        split=args.tflite_map_split,
                        out_dir=out_dir,
                        name="fp32",
                    )
                    map50_int8_tflite, map50_95_int8_tflite = _val_tflite_map(
                        model_path=int8_copy,
                        task=spec.task,
                        data_yaml=spec.data_yaml,
                        imgsz=spec.imgsz,
                        split=args.tflite_map_split,
                        out_dir=out_dir,
                        name="int8",
                    )
                    tflite_map_ok = True
                except Exception as e:
                    tflite_map_err = repr(e)
                    logging.error(f"[TFLite mAP Failed] {spec.name}@{spec.imgsz}: {e}")
 

        except Exception as e:
            # Capture specific export failures (e.g., CIRA ONNX issues) without stopping the whole script
            export_err = repr(e)
            logging.error(f"[Export Failed] {spec.name}@{spec.imgsz}: {e}")
            logging.warning(f"[Export Failed] Skipping latency measure for {spec.name}, recording as NaN.")
        try:
            snap = _snapshot_artifacts(
                report_root=report_root,
                spec=spec,
                run_id=run_id,
                best_pt=best_pt,
                fp32_tflite=(fp32_copy if export_ok else None),
                int8_tflite=(int8_copy if export_ok else None),
            )
            artifact_dir = snap.get("artifact_dir", "")
            artifact_best_pt = snap.get("artifact_best_pt", "")
            artifact_fp32_tflite = snap.get("artifact_fp32_tflite", "")
            artifact_int8_tflite = snap.get("artifact_int8_tflite", "")
        except Exception as e:
            artifact_err = repr(e)
            logging.error(f"[Snapshot Failed] {spec.name}: {e}")

        map50_95 = _read_map_metric(run_dir, task=spec.task)


        kd_stats: dict[str, float] = {
            "kd_log_steps": float("nan"),
            "alpha_kd_min": float("nan"),
            "alpha_kd_median": float("nan"),
            "alpha_kd_max": float("nan"),
            "alpha_kd_sat_ratio": float("nan"),
            "grad_ratio_median": float("nan"),
        }
        if spec.mode == "kd-deploy":
            log_path = _find_latest_attempt_log(spec)
            max_w = float(spec.qat_balance_max) if spec.qat_balance_max is not None else 5.0
            kd_stats = _parse_kd_stats_from_log(log_path=log_path, max_weight=max_w)
 

        rows.append(
            {
                "run_ts": run_ts,
                "run_id": run_id,
                "study": spec.study,
                "dataset": spec.dataset,
                "variant": spec.variant,
                "mode": spec.mode,
                "seed": spec.seed,
                "run_dir": str(run_dir),
                "best_pt": str(best_pt),
                "fp32_tflite": str(fp32_copy) if export_ok else "EXPORT_FAILED",
                "int8_tflite": str(int8_copy) if export_ok else "EXPORT_FAILED",
                "artifact_dir": artifact_dir,
                "artifact_best_pt": artifact_best_pt,
                "artifact_fp32_tflite": artifact_fp32_tflite,
                "artifact_int8_tflite": artifact_int8_tflite,
                "artifact_err": artifact_err,
                "map50_95": map50_95,
                **kd_stats,
                "lat_fp32_ms": lat_fp32_ms,
                "lat_int8_ms": lat_int8_ms,
                "contract_jitter": contract_jitter,
                "tflite_map_ok": tflite_map_ok,
                "tflite_map_err": tflite_map_err,
                "map50_fp32_tflite": map50_fp32_tflite,
                "map50_95_fp32_tflite": map50_95_fp32_tflite,
                "map50_int8_tflite": map50_int8_tflite,
                "map50_95_int8_tflite": map50_95_int8_tflite,
                "export_ok": export_ok,
                "contract_ok": contract_ok,
                "latency_ok": latency_ok,
                "export_err": export_err,
                "contract_err": contract_err,
                "latency_err": latency_err,
            }
        )

    history_csv = report_root / "all_runs_history.csv"
    _append_csv(rows, history_csv)

    all_runs_csv = report_root / "all_runs.csv"
    merged_rows = _upsert_latest(_read_csv_rows(all_runs_csv), rows)
    _write_csv(merged_rows, all_runs_csv)

    delta_a = _compute_deltas(merged_rows, "A")
    delta_b = _compute_deltas_with_baseline(
        merged_rows,
        "B",
        baseline_variant=str(args.b_delta_baseline),
    )
    _write_csv(delta_a, report_root / "delta_A.csv")
    _write_csv(delta_b, report_root / "delta_B.csv")

    report_md = report_root / "report.md"
    report_lines = [
        "# Paper Experiment Summary",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total executed runs (this invocation): {len(rows)}",
        f"- Total latest keys (all_runs.csv): {len(merged_rows)}",
        f"- all_runs.csv: {all_runs_csv}",
        f"- all_runs_history.csv: {history_csv}",
        "",
        "## Delta A (Variant - YOLO)",
    ]
    for row in delta_a:
        report_lines.append(
            f"- {row['dataset']} seed={row['seed']} lhs={row['lhs']}: "
            f"delta_map50_95={row['delta_map50_95']:.6f}, "
            f"delta_int8_map50={row['delta_int8_map50']:.6f}, "
            f"delta_int8_map50_95={row['delta_int8_map50_95']:.6f}, "
            f"delta_int8_latency_ms={row['delta_int8_latency_ms']:.6f}, "
            f"delta_contract_jitter={row['delta_contract_jitter']:.6f}"
        )
    report_lines.append("")
    report_lines.append(f"## Delta B (lhs - {args.b_delta_baseline})")
    for row in delta_b:
        report_lines.append(
            f"- {row['dataset']} seed={row['seed']}: "
            f"delta_map50_95={row['delta_map50_95']:.6f}, "
            f"delta_int8_map50={row['delta_int8_map50']:.6f}, "
            f"delta_int8_map50_95={row['delta_int8_map50_95']:.6f}, "
            f"delta_int8_latency_ms={row['delta_int8_latency_ms']:.6f}, "
            f"delta_contract_jitter={row['delta_contract_jitter']:.6f}"
        )
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.dry_run:
        logging.info("[Done] dry-run completed. No training executed.")
    else:
        logging.info("[Done] completed. Summary at %s", report_root)


if __name__ == "__main__":
    main()
