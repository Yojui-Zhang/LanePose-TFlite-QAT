import logging
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_model_optimization as tfmot

from QAT_Refactored.models.architecture import build_yolov8_pose
from QAT_Refactored.models.heads import TeacherCompatHead, U8PoseCompatHead
from QAT_Refactored.models.layers import RepVGGBlock
from QAT_Refactored.models.qat_utils import RepVGGQuantizeConfig

def build_student_qat(cfg):
    logging.info("\n[Builder] Constructing Base Model..")
    policy = tf.keras.mixed_precision.global_policy()
    if policy.compute_dtype != "float32" or policy.variable_dtype != "float32":
        logging.warning(
            "[Builder] Mixed precision policy is not float32 "
            f"(compute={policy.compute_dtype}, variable={policy.variable_dtype}). "
            "Forcing float32 for TFMOT QAT compatibility."
        )
        tf.keras.mixed_precision.set_global_policy("float32")
    
    base_model = build_yolov8_pose(
        input_shape=(cfg.IMGSZ, cfg.IMGSZ, 3),
        num_classes=cfg.NUM_CLS,
        num_kpt=cfg.NUM_KPT,
        kpt_vals=cfg.KPT_VALS,
        width_mult=cfg.WIDTH_MULT,
        depth_mult=cfg.DEPTH_MULT,
        mode=cfg.TRAIN_SUPERVISION
    )

    # Apply QAT by annotating RepVGG blocks with custom quantize config.
    def _annotate(layer):
        if isinstance(layer, RepVGGBlock):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, RepVGGQuantizeConfig()
            )
        return layer

    annotated = tf.keras.models.clone_model(base_model, clone_function=_annotate)

    with tfmot.quantization.keras.quantize_scope(
        {
            "RepVGGBlock": RepVGGBlock,
            "RepVGGQuantizeConfig": RepVGGQuantizeConfig,
        }
    ):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)

    quant_wrappers = sum(
        1 for layer in qat_model.layers if "QuantizeWrapper" in layer.__class__.__name__
    )
    logging.info(f"[Builder] QAT model built. Quantize wrappers: {quant_wrappers}")
    return qat_model
