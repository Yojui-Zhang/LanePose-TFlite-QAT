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

from typing import Tuple
import config

# 參數
REG_MAX = 16
NC = config.NUM_CLS
NK = config.NUM_KPT
KPT_DIM = config.KPT_VALS
if not config.USE_DFL:
    RAW_C = 4 + NC + NK * KPT_DIM  # 每個格點的 raw 通道數
else:
    RAW_C = 4 * REG_MAX + NC + NK * KPT_DIM  # 每個格點的 raw 通道數

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

@tf.keras.utils.register_keras_serializable(package="QAT")
class TeacherCompatHead(tf.keras.layers.Layer):
    def __init__(self, num_cls, num_kpt, kpt_vals, name="head", apply_sigmoid=True, **kw):
        super().__init__(name=name, **kw)
        self.num_cls  = int(num_cls)
        self.num_kpt  = int(num_kpt)
        self.kpt_vals = int(kpt_vals)
        self.C = 4 + self.num_cls + self.num_kpt * self.kpt_vals
        self.apply_sigmoid = bool(apply_sigmoid)
        self.p3_conv = tf.keras.layers.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p3_out")
        self.p4_conv = tf.keras.layers.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p4_out")
        self.p5_conv = tf.keras.layers.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p5_out")

    def get_config(self):
        # Get the default config from the parent class (tf.keras.layers.Layer)
        cfg = super().get_config()
        # Add your custom parameters to the dictionary
        cfg.update({
            "num_cls": self.num_cls,
            "num_kpt": self.num_kpt,
            "kpt_vals": self.kpt_vals,
            "apply_sigmoid": self.apply_sigmoid
        })
        return cfg

    def _apply_prob_activations(self, y):
        # 使用 L.Lambda 提取張量切片，確保圖的連續性
        box = L.Lambda(lambda t: t[..., :4], name=f"{self.name}/slice_box")(y)
        cls = L.Lambda(lambda t: t[..., 4:4+self.num_cls], name=f"{self.name}/slice_cls")(y)
        kpt = L.Lambda(lambda t: t[..., 4+self.num_cls:], name=f"{self.name}/slice_kpt")(y)

        if self.apply_sigmoid:
            cls = L.Activation('sigmoid', name=f"{self.name}/cls_sigmoid")(cls)

        if self.kpt_vals >= 3:
            B, H, W, _ = y.shape
            # 使用 Keras Layer 進行 reshape
            kpt = L.Reshape((H, W, self.num_kpt, self.kpt_vals), name=f"{self.name}/kpt_reshape1")(kpt)
            
            kxy = L.Lambda(lambda t: t[..., :2], name=f"{self.name}/slice_kxy")(kpt)
            kv = L.Lambda(lambda t: t[..., 2:3], name=f"{self.name}/slice_kv")(kpt)
            
            if self.apply_sigmoid:
                kv = L.Activation('sigmoid', name=f"{self.name}/kv_sigmoid")(kv)
            
            # 使用 Keras Layer 進行 concat
            kpt = L.Concatenate(axis=-1, name=f"{self.name}/kpt_concat")([kxy, kv])
            kpt = L.Reshape((H, W, self.num_kpt * self.kpt_vals), name=f"{self.name}/kpt_reshape2")(kpt)
        
        return L.Concatenate(axis=-1, name=f"{self.name}/output_concat")([box, cls, kpt])

    def _to_BCN(self, t):
        # (B,H,W,C) -> (B,C,H*W)
        # 使用 Keras Layer 進行 transpose 和 reshape
        y = L.Permute((3, 1, 2), name=f"{self.name}/transpose_bchw")(t)  # (B,C,H,W)
        y = L.Reshape((t.shape[-1], -1), name=f"{self.name}/flatten_hw")(y)  # (B,C,N)
        return y

    def call(self, feats, training=False):
        p3, p4, p5 = feats  # 1/8, 1/16, 1/32
        y3 = self._apply_prob_activations(self.p3_conv(p3))
        y4 = self._apply_prob_activations(self.p4_conv(p4))
        y5 = self._apply_prob_activations(self.p5_conv(p5))
        bcn3 = self._to_BCN(y3)
        bcn4 = self._to_BCN(y4)
        bcn5 = self._to_BCN(y5)
        return tf.keras.layers.Concatenate(axis=2, name=f"{self.name}/preds")([bcn3, bcn4, bcn5])



# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

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

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
def conv_bn_act(x, out_ch: int, k: int = 3, s: int = 1, name: str = None, act: str = "relu6"):
    x = L.Conv2D(out_ch, k, strides=s, padding='same', use_bias=False,
                 name=None if not name else f"{name}/conv")(x)
    x = L.BatchNormalization(name=None if not name else f"{name}/bn")(x)

    if act == "relu6":
        x = L.ReLU(max_value=6.0, name=None if not name else f"{name}/relu6")(x)
    elif act == "relu":
        x = L.ReLU(name=None if not name else f"{name}/relu")(x)
    else:
        x = L.LeakyReLU(0.1, name=None if not name else f"{name}/lrelu")(x)
    
    return x

def c2f_block(x, out_ch: int, n: int = 2, name: str = None):
    y = conv_bn_act(x, out_ch, k=1, s=1, name=None if not name else f"{name}/cv1", act='relu6')
    parts = [y]
    for i in range(n):
        y = conv_bn_act(y, out_ch, k=3, s=1, name=None if not name else f"{name}/m{i}", act='relu6')
        parts.append(y)
    z = L.Concatenate(axis=-1, name=None if not name else f"{name}/concat")(parts)
    z = conv_bn_act(z, out_ch, k=1, s=1, name=None if not name else f"{name}/cv2", act='relu6')
    return z

def sppf_block(x, out_ch: int, k: int = 5, name: str = None):
    x1 = conv_bn_act(x, out_ch, k=1, s=1, name=None if not name else f"{name}/cv1", act='relu6')
    p1 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p1")(x1)
    p2 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p2")(p1)
    p3 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=None if not name else f"{name}/p3")(p2)
    cat = L.Concatenate(axis=-1, name=None if not name else f"{name}/cat")([x1, p1, p2, p3])
    y = conv_bn_act(cat, out_ch, k=1, s=1, name=None if not name else f"{name}/cv2", act='relu6')
    return y
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
def dw_conv_bn_act(x, out_ch, k=3, s=1, name=None, act='relu6'):
    x = L.DepthwiseConv2D(k, strides=s, padding='same', use_bias=False,
                          name=None if not name else f'{name}/dw')(x)
    x = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                             name=None if not name else f'{name}/dw_bn')(x)
    if act == 'relu6':
        x = L.ReLU(max_value=6.0, name=None if not name else f'{name}/dw_relu6')(x)
    elif act == 'relu':
        x = L.ReLU(name=None if not name else f'{name}/dw_relu')(x)
    else:
        x = L.LeakyReLU(0.1, name=None if not name else f'{name}/dw_lrelu')(x)

    x = L.Conv2D(out_ch, 1, strides=1, padding='same', use_bias=False,
                 name=None if not name else f'{name}/pw')(x)
    x = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                             name=None if not name else f'{name}/pw_bn')(x)
    if act == 'relu6':
        x = L.ReLU(max_value=6.0, name=None if not name else f'{name}/pw_relu6')(x)
    elif act == 'relu':
        x = L.ReLU(name=None if not name else f'{name}/pw_relu')(x)
    else:
        x = L.LeakyReLU(0.1, name=None if not name else f'{name}/pw_lrelu')(x)
    return x

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def se_block(x, ratio=16, name=None):
    ch = x.shape[-1]
    s  = L.GlobalAveragePooling2D(keepdims=True, name=None if not name else f'{name}/gap')(x)
    s  = L.Conv2D(ch // ratio, 1, activation='relu',
                  name=None if not name else f'{name}/down')(s)
    s  = L.Conv2D(ch, 1, activation='sigmoid',
                  name=None if not name else f'{name}/up')(s)
    return L.Multiply(name=None if not name else f'{name}/scale')([x, s])

def repvgg_block(x, out_ch, k=3, s=1, use_se=False, name='rep', act='relu6'):
    # 三條支路：3x3+BN、1x1+BN、Identity BN（條件成立才有）

    y3  = conv_bn_act(x, out_ch, k=k, s=s, name=f'{name}/rbr_dense', act='relu6')
    y1  = conv_bn_act(x, out_ch, k=1, s=s, name=f'{name}/rbr_1x1', act='relu6')
    yid = None
    if x.shape[-1] == out_ch and s == 1:
        yid = L.BatchNormalization(epsilon=1e-3, momentum=0.99, name=f'{name}/rbr_identity')(x)

    # 相加 -> （可選）SE -> 激活
    parts = [t for t in (y3, y1, yid) if t is not None]
    y = L.Add(name=f'{name}/sum')(parts)
    if use_se:
        y = se_block(y, ratio=16, name=f'{name}/se')

    if act == 'relu6':
        y = L.ReLU(max_value=6.0, name=f'{name}/relu6')(y)
    elif act == 'relu':
        y = L.ReLU(name=f'{name}/relu')(y)
    else:
        y = L.Activation('swish', name=f'{name}/silu')(y)
    return y


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

@tf.keras.utils.register_keras_serializable(package="QAT")
class ChannelAttention(L.Layer):
    def __init__(self, ratio=16, **kw):
        super().__init__(**kw)
        self.ratio = int(ratio)
        self.mlp1 = L.Conv2D(None, 1)  # 佔位，build 時根據輸入通道設置
        self.mlp2 = L.Conv2D(None, 1)

    def build(self, input_shape):
        C = int(input_shape[-1])
        self.mlp1 = L.Conv2D(C // self.ratio, 1, activation="relu", use_bias=True, name=f"{self.name}/mlp1")
        self.mlp2 = L.Conv2D(C,              1, activation=None,   use_bias=True, name=f"{self.name}/mlp2")

    def call(self, x):
        avg = L.GlobalAveragePooling2D(keepdims=True, name=f"{self.name}/gap")(x)
        mx  = L.GlobalMaxPooling2D(keepdims=True,     name=f"{self.name}/gmp")(x)
        avg = self.mlp2(self.mlp1(avg))
        mx  = self.mlp2(self.mlp1(mx))
        
        # 修正：使用 Keras Layer
        sum_feat = L.Add(name=f"{self.name}/add")([avg, mx])
        scale = L.Activation('sigmoid', name=f"{self.name}/sigmoid")(sum_feat)
        return L.Multiply(name=f"{self.name}/scale")([x, scale])

    def get_config(self):
        cfg = super().get_config(); cfg.update({"ratio": self.ratio}); return cfg

@tf.keras.utils.register_keras_serializable(package="QAT")
class SpatialAttention(L.Layer):
    def __init__(self, kernel_size=7, **kw):
        super().__init__(**kw)
        self.kernel_size = int(kernel_size)
        self.conv = L.Conv2D(1, self.kernel_size, padding="same", use_bias=False, activation="sigmoid",
                             name=f"{self.name}/conv")

    def call(self, x):
        # 修正：使用 Lambda Layer 包裝 reduce 操作
        avg = L.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{self.name}/avg_pool")(x)
        mx  = L.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{self.name}/max_pool")(x)
        
        # 修正：使用 Keras Layer
        cat = L.Concatenate(axis=-1, name=f"{self.name}/concat")([avg, mx])
        scale = self.conv(cat)
        return L.Multiply(name=f"{self.name}/scale")([x, scale])

    def get_config(self):
        cfg = super().get_config(); cfg.update({"kernel_size": self.kernel_size}); return cfg

@tf.keras.utils.register_keras_serializable(package="QAT")
class CBAM(L.Layer):
    def __init__(self, ratio=16, kernel_size=7, **kw):
        super().__init__(**kw)
        self.ca = ChannelAttention(ratio=ratio, name=f"{self.name}/ca")
        self.sa = SpatialAttention(kernel_size=kernel_size, name=f"{self.name}/sa")

    def call(self, x):
        return self.sa(self.ca(x))
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 新增：Ultralytics 相容的 YOLOv8-Pose Head（兩分支輸出）
class U8PoseCompatHead(tf.keras.layers.Layer):
    def __init__(self, num_cls, num_kpt, kpt_vals, reg_max=16, ch=128, name="u8pose_head", **kw):
        super().__init__(name=name, **kw)
        self.nc = int(num_cls)
        self.nk = int(num_kpt)
        self.kv = int(kpt_vals)
        self.reg_max = int(reg_max)
        self.no = self.nc + (4 if not config.USE_DFL else 4 * self.reg_max)  # <-- Ultralytics: cls + DFL(4*reg_max)

        # 你原本的三塔結構（可直接重用）
        def tower(prefix):
            return [
                L.Conv2D(ch, 3, padding='same', use_bias=False, name=f'{prefix}.0/conv'),
                L.BatchNormalization(name=f'{prefix}.0/bn'),
                L.ReLU(max_value=6.0, name=f'{prefix}.0/relu6'),
                L.Conv2D(ch, 3, padding='same', use_bias=False, name=f'{prefix}.1/conv'),
                L.BatchNormalization(name=f'{prefix}.1/bn'),
                L.ReLU(max_value=6.0, name=f'{prefix}.1/relu6'),
            ]

        self.twr_reg = [tower('head.reg.p3'), tower('head.reg.p4'), tower('head.reg.p5')]
        self.twr_cls = [tower('head.cls.p3'), tower('head.cls.p4'), tower('head.cls.p5')]
        self.twr_kpt = [tower('head.kpt.p3'), tower('head.kpt.p4'), tower('head.kpt.p5')]

        box_ch = (4 if not config.USE_DFL else 4 * self.reg_max)
        # 最終 1x1 輸出層
        self.out_reg = [
            L.Conv2D(box_ch, 1, name='head.regout.p3'),
            L.Conv2D(box_ch, 1, name='head.regout.p4'),
            L.Conv2D(box_ch, 1, name='head.regout.p5'),
        ]
        self.out_cls = [
            L.Conv2D(self.nc, 1, name='head.coout.p3'),
            L.Conv2D(self.nc, 1, name='head.coout.p4'),
            L.Conv2D(self.nc, 1, name='head.coout.p5'),
        ]
        self.out_kpt = [
            L.Conv2D(self.nk * self.kv, 1, name='head.kptout.p3'),
            L.Conv2D(self.nk * self.kv, 1, name='head.kptout.p4'),
            L.Conv2D(self.nk * self.kv, 1, name='head.kptout.p5'),
        ]

    def _run_tower(self, x, blocks):
        y = x
        # 依序套兩個 conv-bn-relu6
        for i in range(0, len(blocks), 3):
            y = blocks[i](y); y = blocks[i+1](y); y = blocks[i+2](y)
        return y

    def _to_bchw(self, t):
        # Keras 是 channels_last，Ultralytics loss 期望 (B, C, H, W)
        return L.Permute((3, 1, 2))(t)

    def call(self, feats, training=False):
        p3, p4, p5 = feats  # P3=1/8, P4=1/16, P5=1/32 —— 必須是這個順序

        feats_out = []
        kpts_out  = []
        for i, p in enumerate((p3, p4, p5)):
            r = self._run_tower(p, self.twr_reg[i])
            c = self._run_tower(p, self.twr_cls[i])
            k = self._run_tower(p, self.twr_kpt[i])

            r = self.out_reg[i](r)  # (B,H,W, 4*reg_max)
            c = self.out_cls[i](c)  # (B,H,W, nc)
            k = self.out_kpt[i](k)  # (B,H,W, nk*3)

            # rc = L.Concatenate(axis=-1)([r, c])  # (B,H,W, nc+4*reg_max)
            rc = L.Concatenate(axis=-1)([c, r])  # 先 cls, 再 reg —— 與 loss 假設一致
            
            feats_out.append(self._to_bchw(rc))  # -> (B, no, H, W)
            kpts_out.append(self._to_bchw(k))    # -> (B, nk*3, H, W)

        # 回傳格式與 Ultralytics 一致： (feats_list, kpts_list)
        return feats_out, kpts_out
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝


def build_u8s_pose_dual(
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
    x0  = conv_bn_act(inp, ch(8),  k=3, s=2, name='Conv_0', act='relu6')                    # 1/2

    x1 = dw_conv_bn_act(x0, ch(16), k=3, s=2, name='DWConv_1')                              # 1/4
    x2 = repvgg_block(x1, ch(16), k=3, s=1, use_se=True, name='RepVGG_2', act='relu6')

    x3 = dw_conv_bn_act(x2, ch(32), k=3, s=2, name='DWConv_3')                              # 1/8
    x4 = repvgg_block(x3, ch(32), k=3, s=1, use_se=True, name='RepVGG_4', act='relu6')

    x5 = dw_conv_bn_act(x4, ch(64), k=3, s=2, name='DWConv_5')                              # 1/16
    x6 = repvgg_block(x5, ch(64), k=3, s=1, use_se=True, name='RepVGG_6', act='relu6')
    # x7 = CBAM(ratio=16, kernel_size=7, name="CBAM_7")(x6)

    x8 = dw_conv_bn_act(x6, ch(128), k=3, s=2, name='DWConv_8')                             # 1/32
    x9 = repvgg_block(x8, ch(128), k=3, s=1, use_se=True, name='RepVGG_9', act='relu6')
    # x10 = CBAM(ratio=16, kernel_size=7, name="CBAM_10")(x9)

    x11 = sppf_block(x9, ch(128), name='SPPF_11')

    # Neck
    x12   = L.UpSampling2D(size=(2,2), name='UP_12')(x11)                                   # 1/16
    x13   = L.Concatenate(axis=-1, name='cat_13')([x12, x6])
    x14   = repvgg_block(x13, ch(64), k=3, s=1, name='RepVGG_14', act='LeakyRelu')
    # x15   = CBAM(ratio=16, kernel_size=7, name="CBAM_15")(x14)

    x16   = L.UpSampling2D(size=(2,2), name='UP_16')(x14)                                   # 1/8
    x17   = L.Concatenate(axis=-1, name='cat_17')([x16, x4])
    x18   = repvgg_block(x17, ch(32), k=3, s=1, name='RepVGG_18', act='LeakyRelu')


    head_kd  = U8PoseCompatHead(num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS, reg_max=REG_MAX, name="kd_head")
    kd_feats, kd_kpts = head_kd((x18, x14, x9))  # ← Ultralytics 訓練態格式

    head_dep = U8PoseCompatHead(num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS, reg_max=REG_MAX, name="deploy_head")
    deploy_feats, deploy_kpts = head_dep((x18, x14, x9))        # ← 維持你原先 (B,N,C) 便於部署

    model = K.Model(inp, [(deploy_feats, deploy_kpts), (kd_feats, kd_kpts)], name="u8s_pose_keras_dual")
    return model


def build_u8s_pose_dual_distill(
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
    x0  = conv_bn_act(inp, ch(8),  k=3, s=2, name='Conv_0', act='relu6')                    # 1/2

    x1 = dw_conv_bn_act(x0, ch(16), k=3, s=2, name='DWConv_1')                              # 1/4
    x2 = repvgg_block(x1, ch(16), k=3, s=1, use_se=True, name='RepVGG_2', act='relu6')

    x3 = dw_conv_bn_act(x2, ch(32), k=3, s=2, name='DWConv_3')                              # 1/8
    x4 = repvgg_block(x3, ch(32), k=3, s=1, use_se=True, name='RepVGG_4', act='relu6')

    x5 = dw_conv_bn_act(x4, ch(64), k=3, s=2, name='DWConv_5')                              # 1/16
    x6 = repvgg_block(x5, ch(64), k=3, s=1, use_se=True, name='RepVGG_6', act='relu6')
    # x7 = CBAM(ratio=16, kernel_size=7, name="CBAM_7")(x6)

    x8 = dw_conv_bn_act(x6, ch(128), k=3, s=2, name='DWConv_8')                             # 1/32
    x9 = repvgg_block(x8, ch(128), k=3, s=1, use_se=True, name='RepVGG_9', act='relu6')
    # x10 = CBAM(ratio=16, kernel_size=7, name="CBAM_10")(x9)

    x11 = sppf_block(x9, ch(128), name='SPPF_11')

    # Neck
    x12   = L.UpSampling2D(size=(2,2), name='UP_12')(x11)                                   # 1/16
    x13   = L.Concatenate(axis=-1, name='cat_13')([x12, x6])
    x14   = repvgg_block(x13, ch(64), k=3, s=1, name='RepVGG_14', act='LeakyRelu')
    # x15   = CBAM(ratio=16, kernel_size=7, name="CBAM_15")(x14)

    x16   = L.UpSampling2D(size=(2,2), name='UP_16')(x14)                                   # 1/8
    x17   = L.Concatenate(axis=-1, name='cat_17')([x16, x4])
    x18   = repvgg_block(x17, ch(32), k=3, s=1, name='RepVGG_18', act='LeakyRelu')

    # x19   = L.UpSampling2D(size=(2,2), name='UP_19')(x18)                                   # 1/4
    # x20   = L.Concatenate(axis=-1, name='cat_20')([x19, x2])
    # x21   = repvgg_block(x20, ch(16), k=3, s=1, name='RepVGG_21', act='LeakyRelu')

    # feats = (p3, p4, p5)  # 一定要這個順序！
    head_kd = TeacherCompatHead(num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS, name="kd_head", apply_sigmoid=False)
    kd_preds = head_kd((x18, x14, x9))

    head_dep = TeacherCompatHead(num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS, name="deploy_head", apply_sigmoid=False)
    deploy_preds = head_dep((x18, x14, x9))   

    return K.Model(inp, [deploy_preds, kd_preds], name='u8s_pose_keras_dual')



if __name__ == '__main__':
    m = build_u8s_pose_dual((config.IMGSZ,config.IMGSZ,3), num_classes=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS)
    m.summary(line_length=120)
    x = tf.random.uniform([2, 640, 640, 3], 0, 1, dtype=tf.float32)
    y = m(x)
    print('Output shape:', y.shape)