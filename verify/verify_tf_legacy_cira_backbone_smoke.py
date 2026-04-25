from __future__ import annotations

import re
from pathlib import Path
import os

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf
import yaml

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.core.exporter import Exporter
from QAT_Refactored.models.layers import DeformableDepthwiseConv2D
from QAT_Refactored.models.builder import build_student_qat


_CIRA_LITE_YAML = Path("ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml")
_CIRA_POSE_YAML = Path("ultralytics/cfg/models/Yojui/yolov8_CIRA-Pose.yaml")


def _load_expected_ir_config_from_yaml(path: Path) -> list[tuple[str, bool, float]]:
    if not path.exists():
        raise FileNotFoundError(f"CIRA reference YAML not found: {path}")

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = list(spec.get("backbone", []) or []) + list(spec.get("head", []) or [])

    expected: list[tuple[str, bool, float]] = []
    for node in nodes:
        if not isinstance(node, list) or len(node) < 4:
            continue
        module_name = str(node[2])
        if module_name != "ConformableInvertedResidual":
            continue
        args = node[3] if isinstance(node[3], list) else []
        if len(args) < 9:
            raise AssertionError(
                f"Unexpected ConformableInvertedResidual args in {path}: {args!r}"
            )
        mode = str(args[4])
        deform_enabled = bool(args[5])
        use_mask = bool(args[7])
        prior_scale = float(args[8])
        if deform_enabled:
            expected.append((mode, use_mask, prior_scale))
    return expected


def _resolve_reference_yaml(expected_count: int) -> tuple[Path, list[tuple[str, bool, float]]]:
    forced = os.environ.get("CIRA_IR_REF_YAML", "").strip()
    if forced:
        forced_path = Path(forced)
        expected = _load_expected_ir_config_from_yaml(forced_path)
        return forced_path, expected

    candidates = [_CIRA_LITE_YAML, _CIRA_POSE_YAML]
    matched: list[tuple[Path, list[tuple[str, bool, float]]]] = []
    for cand in candidates:
        expected = _load_expected_ir_config_from_yaml(cand)
        if len(expected) == expected_count:
            matched.append((cand, expected))
    if matched:
        return matched[0]

    details = []
    for cand in candidates:
        expected = _load_expected_ir_config_from_yaml(cand)
        details.append(f"{cand}:{len(expected)}")
    raise AssertionError(
        f"No reference YAML matches deform layer count={expected_count}. "
        f"candidates={', '.join(details)}"
    )


def _extract_ir_index(layer_name: str) -> int:
    match = re.search(r"CIRA_IR_(\d+)_", layer_name)
    if not match:
        raise AssertionError(f"Unexpected deform layer name format: {layer_name}")
    return int(match.group(1))


def _collect_deform_layers(root) -> list[DeformableDepthwiseConv2D]:
    out: list[DeformableDepthwiseConv2D] = []
    seen: set[int] = set()

    def _walk(layer) -> None:
        lid = id(layer)
        if lid in seen:
            return
        seen.add(lid)
        if isinstance(layer, DeformableDepthwiseConv2D):
            out.append(layer)
        for attr in ("layer", "_layer", "wrapped_layer", "inner_layer"):
            inner = getattr(layer, attr, None)
            if inner is not None:
                _walk(inner)
        if hasattr(layer, "layers"):
            for sub in getattr(layer, "layers"):
                _walk(sub)

    _walk(root)
    return out


def main() -> None:
    cfg = AppConfig(
        IMGSZ=128,
        NUM_CLS=7,
        NUM_KPT=15,
        KPT_VALS=3,
        TRAIN_ENGINE="tf-legacy",
        DATA_BACKEND="native",
        TF_LEGACY_BACKBONE="cira-lite",
        TF_CIRA_WIDTH_MULT=0.3,
        TF_CIRA_USE_ATTENTION=True,
        TF_CIRA_USE_DEFORM=True,
        TRAIN_SUPERVISION="label",
        BATCH_SIZE=1,
    )
    cfg.validate()

    model = build_student_qat(cfg)
    out = model(tf.random.uniform((1, cfg.IMGSZ, cfg.IMGSZ, 3), dtype=tf.float32), training=False)
    if len(out.shape) != 3:
        raise AssertionError(f"Unexpected output rank for cira tf-legacy model: {out.shape}")
    if int(out.shape[0]) != 1:
        raise AssertionError(f"Unexpected batch in output shape: {out.shape}")

    deform_layers = _collect_deform_layers(model)
    ref_yaml, expected_cfg = _resolve_reference_yaml(len(deform_layers))
    if not deform_layers:
        raise AssertionError("Expected at least one DeformableDepthwiseConv2D layer in cira tf-legacy model.")
    if len(deform_layers) != len(expected_cfg):
        raise AssertionError(
            f"Unexpected deform layer count: got={len(deform_layers)} "
            f"expected={len(expected_cfg)} from {ref_yaml}"
        )

    deform_layers_sorted = sorted(deform_layers, key=lambda x: _extract_ir_index(str(x.name)))
    for layer, (exp_mode, exp_mask, exp_scale) in zip(deform_layers_sorted, expected_cfg):
        got_mode = str(layer.mode)
        got_mask = bool(layer.use_mask)
        got_scale = float(layer.prior_scale)
        if got_mode != exp_mode or got_mask != exp_mask or abs(got_scale - exp_scale) > 1e-6:
            raise AssertionError(
                f"Layer config mismatch at {layer.name}: "
                f"got(mode={got_mode}, use_mask={got_mask}, prior_scale={got_scale}) "
                f"expected(mode={exp_mode}, use_mask={exp_mask}, prior_scale={exp_scale})"
            )

    exporter = Exporter(cfg)
    exporter._fuse_repvgg(model)
    for layer in deform_layers:
        if not bool(getattr(layer, "force_fallback", False)):
            raise AssertionError("Deform layer fallback flag was not enabled before export.")
    print("verify_tf_legacy_cira_backbone_smoke: OK")


if __name__ == "__main__":
    main()
