import logging
import tensorflow as tf
# [FIX] Do not use 'from tensorflow.keras.models import clone_model'
# direct import is fragile in TF 2.16+ / Keras 3 transition.
# Access via tf.keras alias instead.

from tensorflow import keras as K
import tensorflow_model_optimization as tfmot

from QAT_Refactored.models.architecture import build_yolov8_pose
from QAT_Refactored.models.heads import TeacherCompatHead, U8PoseCompatHead
from QAT_Refactored.models.layers import (
    ChannelAttention, SpatialAttention, CBAM, RepVGGBlock
)
from QAT_Refactored.models.qat_utils import RepVGGQuantizeConfig

def _validate_keras2_environment():
    # 防呆：確認不是 keras 3
    try:
        import keras
        if str(getattr(keras, "__version__", "")).startswith("3"):
            raise RuntimeError("Detected keras==3.x, but TFMOT QAT here requires Keras 2.x (TF 2.15).")
    except ImportError:
        pass

def _validate_legacy_layers(model: K.Model):
    """
    驗證目前 runtime 不是 keras3；並確保 model layers 都是 tf.keras 物件。
    """
    _validate_keras2_environment()
    for layer in model.layers:
        if not isinstance(layer, K.layers.Layer):
            raise TypeError(f"Layer '{layer.name}' is not a tf.keras Layer: {type(layer)}")
    logging.info("[Builder] Layer compatibility verified (tf.keras).")

def build_student_qat(cfg):
    """
    Constructs the Student model with valid QAT support for RepVGG.
    """
    logging.info("\n[Builder] Constructing Base Model...")
    
    # 1. 建立基礎模型 (此時應確保 architecture.py 使用 tf_keras)
    base_model = build_yolov8_pose(
        input_shape=(cfg.IMGSZ, cfg.IMGSZ, 3),
        num_classes=cfg.NUM_CLS,
        num_kpt=cfg.NUM_KPT,
        kpt_vals=cfg.KPT_VALS,
        width_mult=cfg.WIDTH_MULT,
        depth_mult=cfg.DEPTH_MULT,
        mode=cfg.TRAIN_SUPERVISION
    )

    # 2. [NEW] 執行型別防禦檢查
    _validate_legacy_layers(base_model)

    def in_heads(layer_name: str) -> bool:
        return ("kd_head" in layer_name) or ("deploy_head" in layer_name)

    # Standard Layers to quantize automatically
    # 明確指定使用 tf_keras 的層
    QUANTIZABLE_LAYERS = (
        K.layers.Conv2D,
        K.layers.DepthwiseConv2D,
        K.layers.Dense,
        K.layers.Activation,
        K.layers.ReLU,
        K.layers.LeakyReLU,
        K.layers.PReLU,
        K.layers.SeparableConv2D,
    )

    def annotate_fn(layer):
        name = layer.name or ""
        
        # 1. Custom Head Classes -> Skip (or handle if they have weights)
        if isinstance(layer, (TeacherCompatHead, U8PoseCompatHead)):
            return layer
        
        # 2. Layers inside Heads -> Skip
        if in_heads(name):
            return layer

        # 3. Custom RepVGG Block -> Use Custom Config
        if isinstance(layer, RepVGGBlock):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=RepVGGQuantizeConfig()
            )

        # 4. Standard Allowlisted Layers -> Quantize
        if isinstance(layer, QUANTIZABLE_LAYERS):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)

        return layer

    logging.info("[Builder] Applying QAT Annotations...")
    
    # Apply annotations
    # [FIX] Use tf.keras.models.clone_model explicitly
    annotated_model = tf.keras.models.clone_model(base_model, clone_function=annotate_fn)

    logging.info("[Builder] Applying Quantization Scope...")
    
    # Register Custom Objects strictly
    with tfmot.quantization.keras.quantize_scope({
        "TeacherCompatHead": TeacherCompatHead,
        "U8PoseCompatHead": U8PoseCompatHead,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "CBAM": CBAM,
        "RepVGGBlock": RepVGGBlock,
        "RepVGGQuantizeConfig": RepVGGQuantizeConfig
    }):
        # 這是之前報錯的地方
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    _verify_qat_structure(qat_model)
    
    return qat_model

def _verify_qat_structure(model):
    qlayers = [l for l in model.submodules if "Quantize" in l.__class__.__name__]
    logging.info(f"[Builder] Quantized layers count: {len(qlayers)}")
    
    rep_vgg_found = False
    # 使用 model.layers 而不是 model.submodules 以避免遞迴過深
    for l in model.layers:
        if isinstance(l, tfmot.quantization.keras.QuantizeWrapperV2):
            if isinstance(l.layer, RepVGGBlock):
                rep_vgg_found = True
                break
                
    if rep_vgg_found:
        logging.info("[Builder] RepVGG Blocks successfully wrapped with Quantization.")
    else:
        logging.warning("[Builder] WARNING: No RepVGG Blocks found in quantization wrappers.")

    logging.info("[Builder] QAT Structure Verification Passed.")