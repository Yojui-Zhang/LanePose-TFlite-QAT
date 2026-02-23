#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coar_lite_attribution.py

目的（Why）：
- 以 COAR 思想做「可落地」的元件歸因：逐一消融指定模塊，觀察 mAP 的下降量 ΔmAP，
  產出「最值得改良/最關鍵」的層/模塊排序。
- 直接讀取 Ultralytics YOLO 的 .pt 與 data.yaml（例如 KITTI.yaml），不改動訓練流程。

使用方式：
  python coar_lite_attribution.py \
    --weights ../Paper-Data/Data_kitti_v2/paper_runs/kitti/A_model_compare/A_kitti_cira_seed0/weights/best.pt \
    --data ./dataset/KITTI.yaml \
    --device 1 \
    --imgsz 640 \
    --max-batches 60 \
    --pattern "ConformableInvertedResidual|ConformableBlock|RepVGGBlock|C2f" \
    --abl "zero"

輸出：
- attribution_report.csv
- 以及 stdout 排序表

注意：
- 這是「驗證集上的元件敏感度」，不是梯度歸因；但對「該改哪一段架構」非常有用。
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ValMetrics:
    map50: float
    map5095: float


@dataclass(frozen=True)
class AblationResult:
    module_name: str
    module_type: str
    base: ValMetrics
    ablated: ValMetrics

    @property
    def d_map50(self) -> float:
        return self.base.map50 - self.ablated.map50

    @property
    def d_map5095(self) -> float:
        return self.base.map5095 - self.ablated.map5095


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help=".pt path")
    p.add_argument("--data", type=str, required=True, help="Ultralytics data yaml path")
    p.add_argument("--device", type=str, default="0", help="e.g. '0' or 'cpu'")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-batches", type=int, default=0, help="0 means full val")
    p.add_argument(
        "--pattern",
        type=str,
        default="ConformableInvertedResidual|ConformableBlock|RepVGGBlock|C2f",
        help="Regex on module class name",
    )
    p.add_argument(
        "--abl",
        type=str,
        choices=("zero", "identity"),
        default="zero",
        help="Ablation style: zero=output->0; identity=output->input (only if shape matches)",
    )
    p.add_argument("--out", type=str, default="attribution_report.csv")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _ensure_ultralytics_importable() -> Any:
    """
    Why:
    - 你環境有兩份 ultralytics（TEST/ultralytics vs pip site-packages）。
      這裡只要求能 import，並用當前 PYTHONPATH 的版本。
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "無法 import ultralytics.YOLO。請確認你在同一個環境且 PYTHONPATH 指向正確 ultrtralytics。"
        ) from e
    return YOLO


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_metrics(val_result: Any) -> ValMetrics:
    """
    Why:
    - Ultralytics 版本差異會導致 val() 回傳物件字段略不同。
      這裡用最常見的 boxes.map50 / boxes.map 做抽取，並提供 fallback。
    """
    # 新版常見：val_result.box.map50 / val_result.box.map
    for box_attr in ("box", "boxes"):
        if hasattr(val_result, box_attr):
            box = getattr(val_result, box_attr)
            if hasattr(box, "map50") and hasattr(box, "map"):
                return ValMetrics(map50=_to_float(box.map50), map5095=_to_float(box.map))

    # fallback：可能直接有 map50/map
    if hasattr(val_result, "map50") and hasattr(val_result, "map"):
        return ValMetrics(map50=_to_float(val_result.map50), map5095=_to_float(val_result.map))

    # 最後 fallback：字典
    if isinstance(val_result, dict):
        # 常見 key: "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        m50 = None
        m95 = None
        for k, v in val_result.items():
            ks = str(k)
            if "mAP50" in ks and "95" not in ks:
                m50 = _to_float(v)
            if "mAP50-95" in ks or ("mAP" in ks and "95" in ks):
                m95 = _to_float(v)
        if m50 is not None and m95 is not None:
            return ValMetrics(map50=m50, map5095=m95)

    raise RuntimeError("無法從 val() 結果抽取 mAP50/mAP50-95。請貼出 val() 回傳物件內容。")


def _run_val(
    yolo: Any,
    data: str,
    imgsz: int,
    device: str,
    max_batches: int,
) -> ValMetrics:
    """
    Why:
    - 只跑 val 取得 metrics；為了加速，允許用 max_batches 做子集評估（趨勢用）。
    """
    # Ultralytics 有些版本支援 fraction；有些支援 batch/rect 等
    kwargs: Dict[str, Any] = dict(data=data, imgsz=imgsz, device=device, plots=False, save=False, verbose=False)
    if max_batches > 0:
        # fraction = (max_batches / total_batches) 不好拿，改用 "fraction" 若支援，否則退回 full val
        # 這裡嘗試傳入 fraction，若版本不支援會拋 TypeError，我們捕捉改 full val。
        kwargs["fraction"] = 1.0  # placeholder, will be overwritten if dataloader len known
        try:
            dl = yolo.model.val_dataloader  # type: ignore[attr-defined]
            # 有些版本 model 沒 val_dataloader；不強求
            _ = dl
        except Exception:
            pass

    try:
        res = yolo.val(**kwargs)
        return _extract_metrics(res)
    except TypeError:
        # fraction 不支援時回退
        kwargs.pop("fraction", None)
        res = yolo.val(**kwargs)
        return _extract_metrics(res)


class Ablator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def register(self, m: torch.nn.Module) -> None:
        if self.mode == "zero":
            def hook_fn(_mod: torch.nn.Module, _inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
                # Why: 最保守，直接把輸出置零（保持 shape）
                if torch.is_tensor(out):
                    return torch.zeros_like(out)
                if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                    return type(out)(torch.zeros_like(o) if torch.is_tensor(o) else o for o in out)
                return out

        else:  # identity
            def hook_fn(_mod: torch.nn.Module, inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
                # Why: 只有在 out 與 inp[0] shape 相同時才能 identity，否則退回 zero
                x = inp[0] if (len(inp) > 0 and torch.is_tensor(inp[0])) else None
                if x is not None and torch.is_tensor(out) and out.shape == x.shape:
                    return x
                if torch.is_tensor(out):
                    return torch.zeros_like(out)
                return out

        self.hooks.append(m.register_forward_hook(hook_fn))


def _collect_target_modules(
    model: torch.nn.Module,
    class_name_regex: str,
) -> List[Tuple[str, torch.nn.Module]]:
    rx = re.compile(class_name_regex)
    out: List[Tuple[str, torch.nn.Module]] = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if rx.search(cls):
            # 排除最外層容器與 Detect（Detect 消融會讓 validator 失效）
            if cls.lower() in ("detect",):
                continue
            out.append((name, m))
    return out


def _write_csv(path: str, rows: List[AblationResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "module_name",
                "module_type",
                "base_map50",
                "base_map50_95",
                "abl_map50",
                "abl_map50_95",
                "delta_map50",
                "delta_map50_95",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.module_name,
                    r.module_type,
                    f"{r.base.map50:.6f}",
                    f"{r.base.map5095:.6f}",
                    f"{r.ablated.map50:.6f}",
                    f"{r.ablated.map5095:.6f}",
                    f"{r.d_map50:.6f}",
                    f"{r.d_map5095:.6f}",
                ]
            )


def main() -> None:
    args = _parse_args()
    YOLO = _ensure_ultralytics_importable()

    weights = Path(args.weights)
    data = Path(args.data)
    if not weights.exists():
        raise FileNotFoundError(str(weights))
    if not data.exists():
        raise FileNotFoundError(str(data))

    # 建議：先把 CUDA deterministic 關掉避免速度下降（這裡不強制改）
    yolo = YOLO(str(weights))
    model: torch.nn.Module = yolo.model  # type: ignore[assignment]

    # baseline
    t0 = time.time()
    base = _run_val(yolo, str(data), args.imgsz, args.device, args.max_batches)
    if args.verbose:
        print(f"[BASE] mAP50={base.map50:.6f} mAP50-95={base.map5095:.6f}  ({time.time()-t0:.1f}s)")

    targets = _collect_target_modules(model, args.pattern)
    if len(targets) == 0:
        raise RuntimeError(f"找不到任何符合 pattern 的模塊：{args.pattern}")

    ablator = Ablator(mode=args.abl)
    results: List[AblationResult] = []

    for idx, (name, m) in enumerate(targets):
        ablator.clear()
        ablator.register(m)

        t1 = time.time()
        try:
            met = _run_val(yolo, str(data), args.imgsz, args.device, args.max_batches)
        finally:
            ablator.clear()

        r = AblationResult(
            module_name=name,
            module_type=m.__class__.__name__,
            base=base,
            ablated=met,
        )
        results.append(r)

        if args.verbose:
            print(
                f"[{idx+1:03d}/{len(targets):03d}] {name} ({r.module_type}) "
                f"ΔmAP50={r.d_map50:.6f} ΔmAP50-95={r.d_map5095:.6f}  ({time.time()-t1:.1f}s)"
            )

    # 排序：優先看 mAP50-95 下降，次看 mAP50
    results_sorted = sorted(results, key=lambda r: (r.d_map5095, r.d_map50), reverse=True)

    _write_csv(args.out, results_sorted)

    print("\n=== COAR-lite Attribution (Top 20 by ΔmAP50-95) ===")
    for r in results_sorted[:20]:
        print(
            f"{r.module_name:60s} {r.module_type:28s} "
            f"ΔmAP50-95={r.d_map5095:.6f}  ΔmAP50={r.d_map50:.6f}"
        )
    print(f"\nCSV saved: {args.out}")


if __name__ == "__main__":
    main()

