from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time

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
    project: Path
    name: str
    teacher_dir: Optional[Path] = None

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
    if scale == 0:
        raise ValueError("Invalid quantization scale=0 for non-float input")
    q = np.round(src / float(scale) + float(zero))
    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max
    return np.clip(q, qmin, qmax).astype(dtype)


def _tflite_latency_ms(model_path: Path, *, warmup: int = 8, runs: int = 40, seed: int = 0) -> float:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    shape = [int(v) if int(v) > 0 else 1 for v in inp["shape"]]
    rng = np.random.default_rng(seed)
    x = rng.random(shape, dtype=np.float32)

    for _ in range(max(1, warmup)):
        interpreter.set_tensor(inp["index"], _prepare_input(inp, x))
        interpreter.invoke()

    elapsed_ms: list[float] = []
    for _ in range(max(1, runs)):
        interpreter.set_tensor(inp["index"], _prepare_input(inp, x))
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000.0)
    return float(np.mean(elapsed_ms))


def _contract_check(model_path: Path) -> None:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    ins = interpreter.get_input_details()
    outs = interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError(
            f"Expected 1 input/1 output, got in={len(ins)} out={len(outs)} for {model_path}"
        )
    shape = [int(v) if int(v) > 0 else 1 for v in ins[0]["shape"]]
    x = np.zeros(shape, dtype=np.float32)
    interpreter.set_tensor(ins[0]["index"], _prepare_input(ins[0], x))
    interpreter.invoke()
    out = interpreter.get_tensor(outs[0]["index"])
    out_fp = _dequantize_output(out, outs[0]["quantization"])
    if not np.all(np.isfinite(out_fp)):
        raise ValueError(f"Non-finite output detected in {model_path}")


def _contract_jitter(fp32_model: Path, int8_model: Path, *, samples: int = 16, seed: int = 0) -> float:
    import tensorflow as tf

    fp32_itp = tf.lite.Interpreter(model_path=str(fp32_model))
    int8_itp = tf.lite.Interpreter(model_path=str(int8_model))
    fp32_itp.allocate_tensors()
    int8_itp.allocate_tensors()

    fp32_in = fp32_itp.get_input_details()[0]
    int8_in = int8_itp.get_input_details()[0]
    fp32_out = fp32_itp.get_output_details()[0]
    int8_out = int8_itp.get_output_details()[0]
    shape = [int(v) if int(v) > 0 else 1 for v in fp32_in["shape"]]

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(max(1, samples)):
        x = rng.random(shape, dtype=np.float32)
        fp32_itp.set_tensor(fp32_in["index"], _prepare_input(fp32_in, x))
        int8_itp.set_tensor(int8_in["index"], _prepare_input(int8_in, x))
        fp32_itp.invoke()
        int8_itp.invoke()

        y_fp32 = _dequantize_output(
            fp32_itp.get_tensor(fp32_out["index"]),
            fp32_out["quantization"],
        )
        y_int8 = _dequantize_output(
            int8_itp.get_tensor(int8_out["index"]),
            int8_out["quantization"],
        )
        diffs.append(float(np.mean(np.abs(y_fp32 - y_int8))))
    return float(np.mean(diffs))


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


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_deltas(rows: list[dict[str, Any]], study: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["study"] != study:
            continue
        key = (str(row["dataset"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    out: list[dict[str, Any]] = []
    if study == "A":
        left, right = "cira", "yolo"
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
                "delta_map50_95": float(lhs["map50_95"]) - float(rhs["map50_95"]),
                "delta_int8_latency_ms": float(lhs["lat_int8_ms"]) - float(rhs["lat_int8_ms"]),
                "delta_contract_jitter": float(lhs["contract_jitter"]) - float(rhs["contract_jitter"]),
            }
        )
    return out


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
            )

            if "A" in studies:
                for variant_key in ("yolo", "cira"):
                    specs.append(
                        TrainSpec(
                            study="A",
                            variant=variant_key,
                            mode="original",
                            model=Path(cfg[variant_key]),
                            project=data_root / "paper_runs" / dataset / "A_model_compare",
                            name=f"A_{dataset}_{variant_key}_seed{seed}",
                            teacher_dir=None,
                            **common,
                        )
                    )

            if "B" in studies:
                student_model = acc_student_model if dataset == "acc" else kitti_student_model
                teacher_dir = acc_teacher if dataset == "acc" else kitti_teacher
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
                specs.append(
                    TrainSpec(
                        study="B",
                        variant="kd_deploy",
                        mode="kd-deploy",
                        model=student_model,
                        project=data_root / "paper_runs" / dataset / "B_kd_vs_deploy",
                        name=f"B_{dataset}_kd_deploy_seed{seed}",
                        teacher_dir=teacher_dir,
                        **common,
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
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv_list(args.datasets)
    studies = _parse_csv_list(args.studies)
    seeds = _parse_seeds(args.seeds)
    for dataset in datasets:
        if dataset not in {"acc", "kitti"}:
            raise ValueError(f"Unsupported dataset: {dataset}")
    for study in studies:
        if study not in {"A", "B"}:
            raise ValueError(f"Unsupported study: {study}")

    if args.smoke:
        logging.info("[Smoke] Applying smoke defaults (epochs=1, batch=2, workers=0, single seed)")
        args.epochs = 1
        args.batch = 2
        args.workers = 0
        seeds = [seeds[0]]
        args.export_fraction = min(float(args.export_fraction), 0.25)

    acc_yaml = QAT_ROOT / "dataset/lanepose-carkeypoint.yaml"
    kitti_yaml = QAT_ROOT / "dataset/KITTI.yaml"
    _require_paths(
        [
            QAT_ROOT / "train_pose.py",
            acc_yaml,
            kitti_yaml,
            Path(args.acc_teacher),
            Path(args.kitti_teacher_model),
            Path(args.acc_student_model),
            Path(args.kitti_student_model),
        ]
    )

    if args.smoke:
        smoke_root = data_root / "_smoke_data"
        acc_yaml = _build_smoke_yaml(acc_yaml, smoke_root, "acc_pose_smoke")
        kitti_yaml = _build_smoke_yaml(kitti_yaml, smoke_root, "kitti_detect_smoke")
        logging.info("[Smoke] acc yaml: %s", acc_yaml)
        logging.info("[Smoke] kitti yaml: %s", kitti_yaml)

    acc_teacher = Path(args.acc_teacher).resolve()
    if not _teacher_pt_exists(acc_teacher):
        raise FileNotFoundError(
            f"ACC teacher path exists but no .pt found under: {acc_teacher}"
        )

    kitti_teacher_provided = bool(args.kitti_teacher)
    kitti_teacher = Path(args.kitti_teacher).resolve() if kitti_teacher_provided else None
    if "B" in studies and "kitti" in datasets and (
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
            project=data_root / "paper_runs" / "kitti" / "teacher",
            name=f"B_kitti_teacher_seed{seeds[0]}",
            teacher_dir=None,
        )
        if args.skip_existing and (teacher_spec.run_dir / "weights" / "best.pt").exists():
            logging.info("[Teacher] Reuse existing KITTI teacher: %s", teacher_spec.run_dir)
        else:
            _run_train_with_retry(teacher_spec, dry_run=args.dry_run)
        kitti_teacher = teacher_spec.run_dir

    if "B" in studies and "kitti" in datasets and kitti_teacher is None:
        raise ValueError("KITTI teacher is required for study B")
    if (
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
        kitti_teacher=(kitti_teacher or Path(args.acc_teacher).resolve()),
        kitti_student_model=Path(args.kitti_student_model).resolve(),
        acc_student_model=Path(args.acc_student_model).resolve(),
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

        # --- Logic: Fault-tolerant Export Block ---
        # Initialize metrics with NaN to handle failures gracefully
        fp32_copy = None
        int8_copy = None
        lat_fp32 = float("nan")
        lat_int8 = float("nan")
        jitter = float("nan")
        export_success = False

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

            _contract_check(fp32_copy)
            _contract_check(int8_copy)
            lat_fp32 = _tflite_latency_ms(fp32_copy, seed=spec.seed)
            lat_int8 = _tflite_latency_ms(int8_copy, seed=spec.seed)
            jitter = _contract_jitter(fp32_copy, int8_copy, seed=spec.seed)
            export_success = True

        except Exception as e:
            # Capture specific export failures (e.g., CIRA ONNX issues) without stopping the whole script
            logging.error(f"[Export Failed] Model: {spec.name} | Error: {e}")
            logging.warning(f"[Export Failed] Skipping latency measure for {spec.name}, recording as NaN.")

        map50_95 = _read_map_metric(run_dir, task=spec.task)

        rows.append(
            {
                "study": spec.study,
                "dataset": spec.dataset,
                "variant": spec.variant,
                "mode": spec.mode,
                "seed": spec.seed,
                "run_dir": str(run_dir),
                "best_pt": str(best_pt),
                "fp32_tflite": str(fp32_copy) if export_success else "EXPORT_FAILED",
                "int8_tflite": str(int8_copy) if export_success else "EXPORT_FAILED",
                "map50_95": map50_95,
                "lat_fp32_ms": lat_fp32,
                "lat_int8_ms": lat_int8,
                "contract_jitter": jitter,
            }
        )

    report_root = data_root / "paper_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    all_runs_csv = report_root / "all_runs.csv"
    _write_csv(rows, all_runs_csv)

    delta_a = _compute_deltas(rows, "A")
    delta_b = _compute_deltas(rows, "B")
    _write_csv(delta_a, report_root / "delta_A.csv")
    _write_csv(delta_b, report_root / "delta_B.csv")

    report_md = report_root / "report.md"
    report_lines = [
        "# Paper Experiment Summary",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total executed runs: {len(rows)}",
        f"- all_runs.csv: {all_runs_csv}",
        "",
        "## Delta A (CIRA - YOLO)",
    ]
    for row in delta_a:
        report_lines.append(
            f"- {row['dataset']} seed={row['seed']}: "
            f"delta_map50_95={row['delta_map50_95']:.6f}, "
            f"delta_int8_latency_ms={row['delta_int8_latency_ms']:.6f}, "
            f"delta_contract_jitter={row['delta_contract_jitter']:.6f}"
        )
    report_lines.append("")
    report_lines.append("## Delta B (KD+deploy - deploy-only)")
    for row in delta_b:
        report_lines.append(
            f"- {row['dataset']} seed={row['seed']}: "
            f"delta_map50_95={row['delta_map50_95']:.6f}, "
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
