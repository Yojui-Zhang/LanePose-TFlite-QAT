"""
Pure-functional Keras builder for a lightweight YOLOv8s-Pose head.
Uses ONLY built-in tf_keras layers so TFMOT can auto-quantize.
Output: [B, N, 4 + num_classes + num_kpt*kpt_vals]
"""
# === TOP-OF-FILE SHIM: put this at the very top of main.py BEFORE any import of tfmot/keras/etc ===
import os, sys

# Prefer tf.keras (legacy) and try to avoid independent keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import tensorflow early
import tensorflow as tf
from tensorflow import keras as K

# Force any "import keras" in other libs to resolve to tf.keras
sys.modules["keras"] = K
sys.modules["keras.models"] = K.models
sys.modules["keras.layers"] = K.layers
sys.modules["keras.activations"] = K.activations
sys.modules["keras.initializers"] = K.initializers
sys.modules["keras.utils"] = K.utils
sys.modules["keras.losses"] = K.losses
sys.modules["keras.backend"] = K.backend
# ===========================================================

from tensorflow.keras import layers as L
import tensorflow_model_optimization as tfmot

from typing import Tuple
import config

# 參數
REG_MAX = 16
NC = config.NUM_CLS
NK = config.NUM_KPT
KPT_DIM = config.KPT_VALS
RAW_C = 4 * REG_MAX + NC + NK * KPT_DIM  # 每個格點的 raw 通道數

def conv_bn_act(x, out_ch: int, k: int = 3, s: int = 1, name: str = None):
    x = L.Conv2D(out_ch, k, strides=s, padding='same', use_bias=False,
                 name=None if not name else f"{name}/conv")(x)
    x = L.BatchNormalization(name=None if not name else f"{name}/bn")(x)
    # use built-in swish to avoid Lambda
    x = L.Activation('swish', name=None if not name else f"{name}/swish")(x)
    # x = L.Activation(tf.nn.silu, name=f'{name}.act{i}')(x)
    return x

def c2f_block(x, out_ch: int, n: int = 2, name: str = None):
    y = conv_bn_act(x, out_ch, k=1, s=1, name=None if not name else f"{name}/cv1")
    parts = [y]
    for i in range(n):
        y = conv_bn_act(y, out_ch, k=3, s=1, name=None if not name else f"{name}/m{i}")
        parts.append(y)
    z = L.Concatenate(axis=-1, name=None if not name else f"{name}/concat")(parts)
    z = conv_bn_act(z, out_ch, k=1, s=1, name=None if not name else f"{name}/cv2")
    return z

def sppf_block(x, out_ch: int, k: int = 5, name: str = None):
    x1 = conv_bn_act(x, out_ch, k=1, s=1, name=None if not name else f"{name}/cv1")
    p1 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p1")(x1)
    p2 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p2")(p1)
    p3 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p3")(p2)
    cat = L.Concatenate(axis=-1, name=None if not name else f"{name}/cat")([x1, p1, p2, p3])
    y = conv_bn_act(cat, out_ch, k=1, s=1, name=None if not name else f"{name}/cv2")
    return y

def make_head(x, out_ch: int, mid_ch: int, name: str):
    x = conv_bn_act(x, mid_ch, k=3, s=1, name=f"{name}/h1")
    x = conv_bn_act(x, mid_ch, k=3, s=1, name=f"{name}/h2")
    x = L.Conv2D(out_ch, 1, padding='same', name=f"{name}/out")(x)
    return x

def dfl_pose_head(p3, p4, p5, ch=128):
    # 三個子塔：回歸(DFL)、分類/obj、關鍵點
    def tower(x, name):
        x = conv_bn_act(x, ch, 3, 1, name=f'{name}.0')
        x = conv_bn_act(x, ch, 3, 1, name=f'{name}.1')
        return x

    r3, r4, r5 = tower(p3,'head.reg.p3'), tower(p4,'head.reg.p4'), tower(p5,'head.reg.p5')
    c3, c4, c5 = tower(p3,'head.cls.p3'), tower(p4,'head.cls.p4'), tower(p5,'head.cls.p5')
    k3, k4, k5 = tower(p3,'head.kpt.p3'), tower(p4,'head.kpt.p4'), tower(p5,'head.kpt.p5')

    reg3 = L.Conv2D(4*REG_MAX, 1, name='head.regout.p3')(r3)
    reg4 = L.Conv2D(4*REG_MAX, 1, name='head.regout.p4')(r4)
    reg5 = L.Conv2D(4*REG_MAX, 1, name='head.regout.p5')(r5)

    co3  = L.Conv2D(NC, 1, name='head.coout.p3')(c3)
    co4  = L.Conv2D(NC, 1, name='head.coout.p4')(c4)
    co5  = L.Conv2D(NC, 1, name='head.coout.p5')(c5)

    kp3  = L.Conv2D(NK*KPT_DIM, 1, name='head.kptout.p3')(k3)
    kp4  = L.Conv2D(NK*KPT_DIM, 1, name='head.kptout.p4')(k4)
    kp5  = L.Conv2D(NK*KPT_DIM, 1, name='head.kptout.p5')(k5)

    # 同尺度拼接 -> [B,H,W, RAW_C]
    def fuse(r, co, kp): return L.Concatenate(axis=-1)([r, co, kp])
    o3, o4, o5 = fuse(reg3,co3,kp3), fuse(reg4,co4,kp4), fuse(reg5,co5,kp5)

    # 展平成 [B, H*W, RAW_C]；按 P3→P4→P5 在 N 維拼接成 [B, 8400, RAW_C]
    o3_bnc = L.Reshape((-1, RAW_C), name='head.flat.p3')(o3)  # 80*80=6400
    o4_bnc = L.Reshape((-1, RAW_C), name='head.flat.p4')(o4)  # 40*40=1600
    o5_bnc = L.Reshape((-1, RAW_C), name='head.flat.p5')(o5)  # 20*20= 400
    # preds  = L.Concatenate(axis=1, name='head.concat.bnc')([o3_bnc, o4_bnc, o5_bnc])  # [B,8400,RAW_C]
    return o3_bnc, o4_bnc, o5_bnc

def build_u8s_pose(
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    num_classes: int = 7,
    num_kpt: int = 15,
    kpt_vals: int = 3,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
):
    C = 4 + num_classes + num_kpt * kpt_vals

    def ch(c): return max(8, int(c * width_mult))
    def n(d):  return max(1, int(d * depth_mult))

    inp = L.Input(shape=input_shape, name='images')

    # Backbone
    x  = conv_bn_act(inp, ch(64),  k=3, s=2, name='stem')
    x  = c2f_block(x, ch(64),  n(2), name='c2f_1')

    x  = conv_bn_act(x, ch(128), k=3, s=2, name='down_2')
    x  = c2f_block(x, ch(128), n(3), name='c2f_2')

    x  = conv_bn_act(x, ch(256), k=3, s=2, name='down_3')
    c3 = c2f_block(x, ch(256), n(3), name='c2f_3')

    x  = conv_bn_act(c3, ch(512), k=3, s=2, name='down_4')
    c4 = c2f_block(x, ch(512), n(3), name='c2f_4')

    x  = conv_bn_act(c4, ch(512), k=3, s=2, name='down_5')
    x  = c2f_block(x, ch(512), n(3), name='c2f_5')
    c5 = sppf_block(x, ch(512), name='sppf')

    # Neck
    concat = L.Concatenate(axis=-1)
    p5_up  = L.UpSampling2D(name='p5_up')(c5)
    p4_td  = c2f_block(concat([p5_up, c4]), ch(256), n(2), name='p4_td')

    p4_up  = L.UpSampling2D(name='p4_up')(p4_td)
    p3_out = c2f_block(concat([p4_up, c3]), ch(128), n(2), name='p3_out')

    p3_dn  = conv_bn_act(p3_out, ch(256), k=3, s=2, name='p3_down')
    p4_out = c2f_block(concat([p3_dn, p4_td]), ch(256), n(2), name='p4_out')

    p4_dn  = conv_bn_act(p4_out, ch(512), k=3, s=2, name='p4_down')
    p5_out = c2f_block(concat([p4_dn, c5]), ch(512), n(2), name='p5_out')

    # Head
    out_p3 = make_head(p3_out, C, ch(128), name='head_p3')
    out_p4 = make_head(p4_out, C, ch(256), name='head_p4')
    out_p5 = make_head(p5_out, C, ch(512), name='head_p5')
    # out_p3, out_p4, out_p5 = dfl_pose_head(p3_out, p4_out, p5_out)

    # Flatten HW -> N，避免 Lambda：用 Reshape
    f3 = L.Reshape(target_shape=(-1, C), name='flat_p3')(out_p3)
    f4 = L.Reshape(target_shape=(-1, C), name='flat_p4')(out_p4)
    f5 = L.Reshape(target_shape=(-1, C), name='flat_p5')(out_p5)
    out = L.Concatenate(axis=1, name='preds')([f3, f4, f5])
    # out = L.Concatenate(axis=1, name='preds')([out_p3, out_p4, out_p5])

    return K.Model(inp, out, name='u8s_pose_keras')

if __name__ == '__main__':
    m = build_u8s_pose((config.IMGSZ,config.IMGSZ,3), num_classes=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS)
    m.summary(line_length=120)
    x = tf.random.uniform([2,640,640,3], 0, 1, dtype=tf.float32)
    y = m(x)
    print('Output shape:', y.shape)
