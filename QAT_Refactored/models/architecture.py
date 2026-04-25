# Source File: QAT_Refactored/models/architecture.py

import tensorflow as tf
import tf_keras as K
from tf_keras import layers as L

from QAT_Refactored.models.layers import (
    CBAM,
    DeformableDepthwiseConv2D,
    RepVGGBlock,
    SpatialAttention,
    conv_bn_act,
    dw_conv_bn_act,
    sppf_block,
)
from QAT_Refactored.models.heads import TeacherCompatHead


def _build_pose_heads(
    inp: tf.Tensor,
    neck_feats: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    *,
    num_classes: int,
    num_kpt: int,
    kpt_vals: int,
    mode: str,
    model_name: str,
) -> K.Model:
    head_dep = TeacherCompatHead(
        num_cls=num_classes,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        name="deploy_head",
    )
    deploy_preds = head_dep(neck_feats)

    if mode == "distill":
        head_kd = TeacherCompatHead(
            num_cls=num_classes,
            num_kpt=num_kpt,
            kpt_vals=kpt_vals,
            name="kd_head",
        )
        kd_preds = head_kd(neck_feats)
        return K.Model(inp, [deploy_preds, kd_preds], name=f"{model_name}_dual")

    return K.Model(inp, deploy_preds, name=model_name)


def _cira_ir_fallback(
    x: tf.Tensor,
    *,
    out_ch: int,
    use_attention: bool,
    use_deform: bool,
    dcn_mode: str,
    use_mask: bool,
    prior_scale: float,
    name: str,
) -> tf.Tensor:
    """
    TensorFlow fallback block that approximates CIRA IR behavior without deform_conv ops.
    """
    shortcut = x
    if use_deform:
        y = conv_bn_act(x, out_ch, k=1, s=1, name=f"{name}_pre_pw", act="relu6")
        y = DeformableDepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            mode=str(dcn_mode),
            use_mask=bool(use_mask),
            prior_scale=float(prior_scale),
            deform_enabled=True,
            force_fallback=False,
            name=f"{name}_deform_dw",
        )(y)
        y = L.BatchNormalization(name=f"{name}_deform_bn")(y)
        y = L.ReLU(max_value=6.0, name=f"{name}_deform_relu6")(y)
        y = conv_bn_act(y, out_ch, k=1, s=1, name=f"{name}_post_pw", act="relu6")
    else:
        y = dw_conv_bn_act(x, out_ch, k=3, s=1, name=f"{name}/dw_pw", act="relu6")
    if int(shortcut.shape[-1]) == out_ch:
        y = L.Add(name=f"{name}/res_add")([shortcut, y])
    y = L.ReLU(max_value=6.0, name=f"{name}/out_relu6")(y)
    if use_attention:
        y = CBAM(ratio=16, kernel_size=7, name=f"{name}/cbam")(y)
    return y


def build_cira_pose_tf_legacy(
    input_shape=(640, 640, 3),
    num_classes=7,
    num_kpt=15,
    kpt_vals=3,
    width_mult=0.3,
    depth_mult=1.0,  # kept for signature compatibility
    mode="label",  # "label" or "distill"
    enable_attention=True,
    use_deform=True,
):
    """
    CIRA-like TensorFlow model for tf-legacy route.
    When ``use_deform=True``, CIRA IR blocks use DeformableDepthwiseConv2D
    (DCN-like TF implementation with prior/residual modes). Export path
    can still switch these layers to depthwise fallback for deployment
    compatibility.
    """
    del depth_mult

    def ch(c: int) -> int:
        return max(8, int(c * width_mult))

    inp = L.Input(shape=input_shape, name="images")

    # Backbone (mirrors CIRA-Pose stage layout, deform-free fallback)
    x0 = conv_bn_act(inp, ch(64), k=3, s=2, name="CIRA_Conv_0", act="relu6")

    x1 = dw_conv_bn_act(x0, ch(128), k=3, s=2, name="CIRA_DWConv_1", act="relu6")
    x2 = _cira_ir_fallback(
        x1,
        out_ch=ch(128),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_only",
        use_mask=False,
        prior_scale=0.20,
        name="CIRA_IR_2",
    )

    x3 = dw_conv_bn_act(x2, ch(256), k=3, s=2, name="CIRA_DWConv_3", act="relu6")
    x4 = _cira_ir_fallback(
        x3,
        out_ch=ch(256),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_only",
        use_mask=False,
        prior_scale=0.25,
        name="CIRA_IR_4",
    )

    x5 = dw_conv_bn_act(x4, ch(512), k=3, s=2, name="CIRA_DWConv_5", act="relu6")
    x6 = _cira_ir_fallback(
        x5,
        out_ch=ch(512),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.30,
        name="CIRA_IR_6",
    )
    x7 = _cira_ir_fallback(
        x6,
        out_ch=ch(512),
        use_attention=bool(enable_attention),
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.30,
        name="CIRA_IR_7",
    )

    x8 = dw_conv_bn_act(x7, ch(1024), k=3, s=2, name="CIRA_DWConv_8", act="relu6")
    x9 = _cira_ir_fallback(
        x8,
        out_ch=ch(1024),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.25,
        name="CIRA_IR_9",
    )
    x10 = sppf_block(x9, ch(1024), name="CIRA_SPPF_10")

    # Neck (CIRA-Pose inspired PAN/FPN)
    x11 = L.UpSampling2D(size=(2, 2), name="CIRA_UP_11")(x10)
    x12 = L.Concatenate(axis=-1, name="CIRA_cat_12")([x11, x7])
    x13 = conv_bn_act(x12, ch(512), k=1, s=1, name="CIRA_Conv_13", act="relu6")
    x14 = _cira_ir_fallback(
        x13,
        out_ch=ch(512),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.30,
        name="CIRA_IR_14",
    )
    x15 = _cira_ir_fallback(
        x14,
        out_ch=ch(512),
        use_attention=bool(enable_attention),
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.30,
        name="CIRA_IR_15",
    )

    x16 = L.UpSampling2D(size=(2, 2), name="CIRA_UP_16")(x15)
    x17 = L.Concatenate(axis=-1, name="CIRA_cat_17")([x16, x4])
    x18 = conv_bn_act(x17, ch(384), k=1, s=1, name="CIRA_Conv_18", act="relu6")
    x19 = _cira_ir_fallback(
        x18,
        out_ch=ch(384),
        use_attention=bool(enable_attention),
        use_deform=bool(use_deform),
        dcn_mode="prior_only",
        use_mask=True,
        prior_scale=0.20,
        name="CIRA_IR_19",
    )
    x20 = _cira_ir_fallback(
        x19,
        out_ch=ch(384),
        use_attention=bool(enable_attention),
        use_deform=bool(use_deform),
        dcn_mode="residual_only",
        use_mask=True,
        prior_scale=0.15,
        name="CIRA_IR_20",
    )

    x21 = dw_conv_bn_act(x2, ch(384), k=3, s=2, name="CIRA_DWConv_21", act="relu6")
    x22 = L.Concatenate(axis=-1, name="CIRA_cat_22")([x21, x20])
    x23 = conv_bn_act(x22, ch(384), k=1, s=1, name="CIRA_Conv_23", act="relu6")
    x24 = RepVGGBlock(ch(384), k=3, s=1, use_se=True, act="relu6", name="CIRA_Rep_24_1")(x23)
    x24 = RepVGGBlock(ch(384), k=3, s=1, use_se=True, act="relu6", name="CIRA_Rep_24_2")(x24)
    x24 = SpatialAttention(kernel_size=7, name="CIRA_SA_24")(x24)

    x25 = dw_conv_bn_act(x24, ch(512), k=3, s=2, name="CIRA_DWConv_25", act="relu6")
    x26 = L.Concatenate(axis=-1, name="CIRA_cat_26")([x25, x15])
    x27 = conv_bn_act(x26, ch(512), k=1, s=1, name="CIRA_Conv_27", act="relu6")
    x28 = _cira_ir_fallback(
        x27,
        out_ch=ch(512),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="prior_residual",
        use_mask=False,
        prior_scale=0.30,
        name="CIRA_IR_28",
    )

    x29 = dw_conv_bn_act(x28, ch(1024), k=3, s=2, name="CIRA_DWConv_29", act="relu6")
    x30 = L.Concatenate(axis=-1, name="CIRA_cat_30")([x29, x10])
    x31 = conv_bn_act(x30, ch(1024), k=1, s=1, name="CIRA_Conv_31", act="relu6")
    x32 = _cira_ir_fallback(
        x31,
        out_ch=ch(1024),
        use_attention=False,
        use_deform=bool(use_deform),
        dcn_mode="residual_only",
        use_mask=False,
        prior_scale=0.25,
        name="CIRA_IR_32",
    )

    neck_feats = (x24, x28, x32)
    return _build_pose_heads(
        inp,
        neck_feats,
        num_classes=num_classes,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        mode=mode,
        model_name="cira_pose_tf_legacy",
    )


def build_yolov8_pose(
    input_shape=(640, 640, 3),
    num_classes=7,
    num_kpt=15,
    kpt_vals=3,
    width_mult=2.0,
    depth_mult=1.0,
    mode="label",  # "label" (single head) or "distill" (dual head)
):
    """
    Constructs YOLOv8-S pose model with RepVGG blocks.
    """
    del depth_mult

    def ch(c):
        return max(8, int(c * width_mult))

    inp = L.Input(shape=input_shape, name="images")

    # Backbone
    x0 = conv_bn_act(inp, ch(8), k=3, s=2, name="Conv_0", act="relu6")
    x1 = dw_conv_bn_act(x0, ch(16), k=3, s=2, name="DWConv_1")
    x2 = RepVGGBlock(ch(16), k=3, s=1, use_se=True, act="relu6", name="RepVGG_2")(x1)

    x3 = dw_conv_bn_act(x2, ch(32), k=3, s=2, name="DWConv_3")
    x4 = RepVGGBlock(ch(32), k=3, s=1, use_se=True, act="relu6", name="RepVGG_4")(x3)

    x5 = dw_conv_bn_act(x4, ch(64), k=3, s=2, name="DWConv_5")
    x6 = RepVGGBlock(ch(64), k=3, s=1, use_se=True, act="relu6", name="RepVGG_6")(x5)
    x6 = SpatialAttention(kernel_size=7, name="SA_6")(x6)
    x7 = RepVGGBlock(ch(64), k=3, s=1, use_se=True, act="relu6", name="RepVGG_7")(x6)
    x7 = SpatialAttention(kernel_size=7, name="SA_7")(x7)

    x8 = dw_conv_bn_act(x7, ch(128), k=3, s=2, name="DWConv_8")
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act="relu6", name="RepVGG_9_1")(x8)
    x9 = SpatialAttention(kernel_size=7, name="SA_9_1")(x9)
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act="relu6", name="RepVGG_9_2")(x9)
    x9 = SpatialAttention(kernel_size=7, name="SA_9_2")(x9)
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act="relu6", name="RepVGG_9_3")(x9)
    x10 = SpatialAttention(kernel_size=7, name="SA_9_3")(x9)

    x11 = sppf_block(x10, ch(128), name="SPPF_11")

    # Neck
    x12 = L.UpSampling2D(size=(2, 2), name="UP_12")(x11)
    x13 = L.Concatenate(axis=-1, name="cat_13")([x12, x6])
    x14 = RepVGGBlock(ch(64), k=3, s=1, act="lrelu", name="RepVGG_14")(x13)
    x15 = SpatialAttention(kernel_size=7, name="SA_15")(x14)

    x16 = L.UpSampling2D(size=(2, 2), name="UP_16")(x15)
    x17 = L.Concatenate(axis=-1, name="cat_17")([x16, x4])
    x18 = RepVGGBlock(ch(32), k=3, s=1, act="lrelu", name="RepVGG_18")(x17)
    x19 = SpatialAttention(kernel_size=7, name="SA_19")(x18)

    neck_feats = (x19, x15, x11)
    return _build_pose_heads(
        inp,
        neck_feats,
        num_classes=num_classes,
        num_kpt=num_kpt,
        kpt_vals=kpt_vals,
        mode=mode,
        model_name="u8s_pose_keras",
    )
