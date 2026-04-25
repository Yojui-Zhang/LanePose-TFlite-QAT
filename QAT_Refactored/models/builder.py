import logging
import os
import tensorflow as tf
import tf_keras as K
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

from QAT_Refactored.models.architecture import build_cira_pose_tf_legacy, build_yolov8_pose
from QAT_Refactored.models.heads import TeacherCompatHead, U8PoseCompatHead
from QAT_Refactored.models.layers import DeformableDepthwiseConv2D, RepVGGBlock
from QAT_Refactored.models.qat_utils import DeformableDepthwiseQuantizeConfig, RepVGGQuantizeConfig


class _NoOpLayoutTransform:
    """Skip expensive graph layout transforms for manually annotated custom layers."""

    def apply(self, model, layer_quantize_map):
        return model, layer_quantize_map


class _FastManualAnnotationScheme(quantize_scheme.QuantizeScheme):
    """
    Quantize scheme that preserves default 8-bit registry while bypassing
    layout rewrite transforms. This keeps manual annotation targets unchanged
    and avoids very slow transform-time model reconstruction.
    """

    def __init__(self, *, disable_per_axis: bool = False):
        self._disable_per_axis = bool(disable_per_axis)

    def get_layout_transformer(self):
        return _NoOpLayoutTransform()

    def get_quantize_registry(self):
        return default_8bit_quantize_registry.Default8BitQuantizeRegistry(
            disable_per_axis=self._disable_per_axis
        )


def _use_default_layout_transform() -> bool:
    text = str(os.environ.get("QAT_USE_DEFAULT_LAYOUT_TRANSFORM", "0")).strip().lower()
    return text in {"1", "true", "yes", "on"}

def build_student_qat(cfg):
    logging.info("\n[Builder] Constructing Base Model..")
    policy = K.mixed_precision.global_policy()
    if policy.compute_dtype != "float32" or policy.variable_dtype != "float32":
        logging.warning(
            "[Builder] Mixed precision policy is not float32 "
            f"(compute={policy.compute_dtype}, variable={policy.variable_dtype}). "
            "Forcing float32 for TFMOT QAT compatibility."
        )
        K.mixed_precision.set_global_policy("float32")
    
    legacy_backbone = str(getattr(cfg, "TF_LEGACY_BACKBONE", "yolo-repvgg")).strip().lower()
    if legacy_backbone == "cira-lite":
        logging.info(
            "[Builder] Using tf-legacy backbone: cira-lite (deform=%s, export fallback enabled)",
            bool(getattr(cfg, "TF_CIRA_USE_DEFORM", True)),
        )
        base_model = build_cira_pose_tf_legacy(
            input_shape=(cfg.IMGSZ, cfg.IMGSZ, 3),
            num_classes=cfg.NUM_CLS,
            num_kpt=cfg.NUM_KPT,
            kpt_vals=cfg.KPT_VALS,
            width_mult=float(getattr(cfg, "TF_CIRA_WIDTH_MULT", 0.3)),
            depth_mult=cfg.DEPTH_MULT,
            mode=cfg.TRAIN_SUPERVISION,
            enable_attention=bool(getattr(cfg, "TF_CIRA_USE_ATTENTION", True)),
            use_deform=bool(getattr(cfg, "TF_CIRA_USE_DEFORM", True)),
        )
    else:
        logging.info("[Builder] Using tf-legacy backbone: yolo-repvgg")
        base_model = build_yolov8_pose(
            input_shape=(cfg.IMGSZ, cfg.IMGSZ, 3),
            num_classes=cfg.NUM_CLS,
            num_kpt=cfg.NUM_KPT,
            kpt_vals=cfg.KPT_VALS,
            width_mult=cfg.WIDTH_MULT,
            depth_mult=cfg.DEPTH_MULT,
            mode=cfg.TRAIN_SUPERVISION,
        )

    # Apply QAT by annotating RepVGG blocks with custom quantize config.
    def _annotate(layer):
        if isinstance(layer, RepVGGBlock):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, RepVGGQuantizeConfig()
            )
        if isinstance(layer, DeformableDepthwiseConv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, DeformableDepthwiseQuantizeConfig()
            )
        return layer

    logging.info("[Builder] Stage 1/2: Cloning model with QAT annotations...")
    annotated = K.models.clone_model(base_model, clone_function=_annotate)
    logging.info("[Builder] Stage 1/2 complete.")

    logging.info("[Builder] Stage 2/2: Applying quantize wrappers...")
    with tfmot.quantization.keras.quantize_scope(
        {
            "RepVGGBlock": RepVGGBlock,
            "RepVGGQuantizeConfig": RepVGGQuantizeConfig,
            "DeformableDepthwiseConv2D": DeformableDepthwiseConv2D,
            "DeformableDepthwiseQuantizeConfig": DeformableDepthwiseQuantizeConfig,
        }
    ):
        if _use_default_layout_transform():
            logging.info("[Builder] Using default TFMOT layout transform (compat mode).")
            qat_model = tfmot.quantization.keras.quantize_apply(annotated)
        else:
            logging.info("[Builder] Using fast manual-annotation quantize scheme (no-op layout transform).")
            qat_model = tfmot.quantization.keras.quantize_apply(
                annotated,
                scheme=_FastManualAnnotationScheme(),
            )
    logging.info("[Builder] Stage 2/2 complete.")

    quant_wrappers = sum(
        1 for layer in qat_model.layers if "QuantizeWrapper" in layer.__class__.__name__
    )
    logging.info(f"[Builder] QAT model built. Quantize wrappers: {quant_wrappers}")
    return qat_model
