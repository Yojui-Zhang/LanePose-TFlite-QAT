# Source File: QAT_Refactored/models/architecture.py

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from QAT_Refactored.models.layers import (
    conv_bn_act, dw_conv_bn_act, sppf_block, 
    SpatialAttention, ChannelAttention, RepVGGBlock
)
from QAT_Refactored.models.heads import TeacherCompatHead, U8PoseCompatHead

def build_yolov8_pose(
    input_shape=(640, 640, 3),
    num_classes=7,
    num_kpt=15,
    kpt_vals=3,
    width_mult=2.0,
    depth_mult=1.0,
    mode='label' # 'label' (Dual Head) or 'distill'
):
    """
    Constructs YOLOv8-S Pose model with RepVGG blocks.
    Now utilizes the class-based RepVGGBlock for re-parameterization support.
    """
    def ch(c): return max(8, int(c * width_mult))
    
    inp = L.Input(shape=input_shape, name='images')

    # ==========================================
    # Backbone
    # ==========================================
    # P1 (1/2)
    x0 = conv_bn_act(inp, ch(8), k=3, s=2, name='Conv_0', act='relu6') 
    
    # P2 (1/4)
    x1 = dw_conv_bn_act(x0, ch(16), k=3, s=2, name='DWConv_1')
    # Refactored: Use RepVGGBlock Class
    x2 = RepVGGBlock(ch(16), k=3, s=1, use_se=True, act='relu6', name='RepVGG_2')(x1)
    
    # P3 (1/8)
    x3 = dw_conv_bn_act(x2, ch(32), k=3, s=2, name='DWConv_3')
    x4 = RepVGGBlock(ch(32), k=3, s=1, use_se=True, act='relu6', name='RepVGG_4')(x3)
    
    # P4 (1/16)
    x5 = dw_conv_bn_act(x4, ch(64), k=3, s=2, name='DWConv_5')
    x6 = RepVGGBlock(ch(64), k=3, s=1, use_se=True, act='relu6', name='RepVGG_6')(x5)
    x6 = SpatialAttention(kernel_size=7, name="SA_6")(x6)
    x7 = RepVGGBlock(ch(64), k=3, s=1, use_se=True, act='relu6', name='RepVGG_7')(x6)
    x7 = SpatialAttention(kernel_size=7, name="SA_7")(x7)

    # P5 (1/32)
    x8 = dw_conv_bn_act(x7, ch(128), k=3, s=2, name='DWConv_8')
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act='relu6', name='RepVGG_9_1')(x8)
    x9 = SpatialAttention(kernel_size=7, name="SA_9_1")(x9)
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act='relu6', name='RepVGG_9_2')(x9)
    x9 = SpatialAttention(kernel_size=7, name="SA_9_2")(x9)
    x9 = RepVGGBlock(ch(128), k=3, s=1, use_se=True, act='relu6', name='RepVGG_9_3')(x9)
    x10 = SpatialAttention(kernel_size=7, name="SA_9_3")(x9)

    x11 = sppf_block(x10, ch(128), name='SPPF_11')

    # ==========================================
    # Neck (PANet)
    # ==========================================
    # Up 1
    x12 = L.UpSampling2D(size=(2,2), name='UP_12')(x11)
    x13 = L.Concatenate(axis=-1, name='cat_13')([x12, x6]) # Cat P4
    x14 = RepVGGBlock(ch(64), k=3, s=1, act='LeakyRelu', name='RepVGG_14')(x13)
    x15 = SpatialAttention(kernel_size=7, name="SA_15")(x14)

    # Up 2
    x16 = L.UpSampling2D(size=(2,2), name='UP_16')(x15)
    x17 = L.Concatenate(axis=-1, name='cat_17')([x16, x4]) # Cat P3
    x18 = RepVGGBlock(ch(32), k=3, s=1, act='LeakyRelu', name='RepVGG_18')(x17)
    x19 = SpatialAttention(kernel_size=7, name="SA_19")(x18)

    # Feature Pyramid for Heads (P3, P4, P5)
    neck_feats = (x19, x15, x11)

    # ==========================================
    # Heads
    # ==========================================
    
    # Deploy Head
    head_dep = TeacherCompatHead(
        num_cls=num_classes, 
        num_kpt=num_kpt, 
        kpt_vals=kpt_vals, 
        name="deploy_head"
    )
    deploy_preds = head_dep(neck_feats)

    # Keep dual-head only for distillation; label training follows a single-head path.
    if mode == 'distill':
        head_kd = TeacherCompatHead(
            num_cls=num_classes,
            num_kpt=num_kpt,
            kpt_vals=kpt_vals,
            name="kd_head"
        )
        kd_preds = head_kd(neck_feats)
        return K.Model(inp, [deploy_preds, kd_preds], name='u8s_pose_keras_dual')

    return K.Model(inp, deploy_preds, name='u8s_pose_keras')
