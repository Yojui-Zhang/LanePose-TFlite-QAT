"""
Keras reimplementation of a lightweight YOLOv8s-Pose (anchor-free) forward model
that outputs a single tensor shaped [B, N, 5 + num_classes + num_kpt*kpt_vals]
with N = sum(H_i*W_i) over strides {8, 16, 32}. This is designed to be
compatible with TensorFlow Model Optimization (tfmot) QAT and your distillation
loop (teacher graph SavedModel -> student Keras model).

Notes
-----
- Boxes are predicted as raw XYWH (no DFL). Obj, cls, and kpt score are logits.
- Per-scale predictions are concatenated along the N dimension (B,N,C).
- Backbone/neck loosely follow YOLOv8s (C2f + SPPF + PAN-FPN), but simplified.
- Keep activations as SiLU (a.k.a. Swish) to match Ultralytics.
- All layers are standard Keras so tfmot can insert fake-quant nodes.
- **Keras 3 friendly**: avoid `tf.*` graph ops on KerasTensor; use `keras.layers`.

Author: ChatGPT (GPT-5 Thinking)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as L

# ---------------------------- Building Blocks ---------------------------- #

def SiLU(x):
    return tf.nn.silu(x)

class ConvBNAct(L.Layer):
    def __init__(self, out_ch: int, k: int = 3, s: int = 1, name: str = None):
        super().__init__(name=name)
        p = 'same'
        self.conv = L.Conv2D(out_ch, k, strides=s, padding=p, use_bias=False)
        self.bn = L.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return SiLU(x)

class C2f(L.Layer):
    """Simplified C2f (Cross-Stage Partial w/ more fusions).
    Args:
        out_ch: output channels
        n: number of internal convs
    """
    def __init__(self, out_ch: int, n: int = 2, name: str = None):
        super().__init__(name=name)
        self.cv1 = ConvBNAct(out_ch, k=1, s=1)
        self.m = [ConvBNAct(out_ch, k=3, s=1) for _ in range(n)]
        self.cv2 = ConvBNAct(out_ch, k=1, s=1)
        self.concat = L.Concatenate(axis=-1)

    def call(self, x, training=False):
        y = self.cv1(x, training=training)
        parts = [y]
        for block in self.m:
            y = block(y, training=training)
            parts.append(y)
        x = self.concat(parts)
        return self.cv2(x, training=training)

class SPPF(L.Layer):
    """Spatial Pyramid Pooling - Fast (as used by YOLOv5/8).
    Use three maxpools with k=5 and concat.
    """
    def __init__(self, out_ch: int, k: int = 5, name: str = None):
        super().__init__(name=name)
        self.cv1 = ConvBNAct(out_ch, k=1, s=1)
        self.pool = L.MaxPool2D(pool_size=k, strides=1, padding='same')
        self.cv2 = ConvBNAct(out_ch, k=1, s=1)
        self.concat = L.Concatenate(axis=-1)

    def call(self, x, training=False):
        x = self.cv1(x, training=training)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        x = self.concat([x, y1, y2, y3])
        return self.cv2(x, training=training)

# ---------------------------- Model Builder ---------------------------- #

def _make_head(x, out_ch: int, mid_ch: int):
    """Detection head for a single scale. Keep logits for obj/cls/kpt score.
    Returns per-cell predictions with shape [B, H, W, out_ch].
    """
    x = ConvBNAct(mid_ch, k=3, s=1)(x)
    x = ConvBNAct(mid_ch, k=3, s=1)(x)
    return L.Conv2D(out_ch, 1, padding='same')(x)  # logits for cls/obj/kpt score, raw for bbox


def build_u8s_pose(
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    num_classes: int = 7,
    num_kpt: int = 15,
    kpt_vals: int = 3,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
):
    """Build a YOLOv8s-Pose-like Keras model (anchor-free) for QAT distillation.

    Output: Tensor [B, N, C] where C = 5 + num_classes + num_kpt*kpt_vals.
            The 5 = [x, y, w, h, obj_logit].
    """
    C = 5 + num_classes + num_kpt * kpt_vals

    def ch(c):
        return max(8, int(c * width_mult))

    def n(d):
        return max(1, int(d * depth_mult))

    inp = L.Input(shape=input_shape, name='images')

    # ---------------- Backbone ----------------
    x = ConvBNAct(ch(64), k=3, s=2, name='stem') (inp)      # 320x320
    x = C2f(ch(64),  n(2), name='c2f_1')        (x)

    x = ConvBNAct(ch(128), k=3, s=2, name='down_2')(x)      # 160x160
    x = C2f(ch(128), n(3), name='c2f_2')        (x)

    x = ConvBNAct(ch(256), k=3, s=2, name='down_3')(x)      # 80x80
    c3 = C2f(ch(256), n(3), name='c2f_3')       (x)         # save for FPN (P3 base)

    x = ConvBNAct(ch(512), k=3, s=2, name='down_4')(c3)     # 40x40
    c4 = C2f(ch(512), n(3), name='c2f_4')       (x)         # save for FPN (P4 base)

    x = ConvBNAct(ch(512), k=3, s=2, name='down_5')(c4)     # 20x20
    x = C2f(ch(512), n(3), name='c2f_5')        (x)
    c5 = SPPF(ch(512), name='sppf')             (x)         # deepest (P5 base)

    # ---------------- Neck (FPN + PAN) ----------------
    concat = L.Concatenate(axis=-1)

    # Top-down
    p5_up = L.UpSampling2D()(c5)                               # 40x40
    p4_td = C2f(ch(256), n(2), name='p4_td')(concat([p5_up, c4]))

    p4_up = L.UpSampling2D()(p4_td)                            # 80x80
    p3_out = C2f(ch(128), n(2), name='p3_out')(concat([p4_up, c3]))

    # Bottom-up
    p3_dn = ConvBNAct(ch(256), k=3, s=2, name='p3_down')(p3_out)  # 40x40
    p4_out = C2f(ch(256), n(2), name='p4_out')(concat([p3_dn, p4_td]))

    p4_dn = ConvBNAct(ch(512), k=3, s=2, name='p4_down')(p4_out)   # 20x20
    p5_out = C2f(ch(512), n(2), name='p5_out')(concat([p4_dn, c5]))

    # ---------------- Head (per-scale) ----------------
    mid_p3, mid_p4, mid_p5 = ch(128), ch(256), ch(512)
    out_p3 = _make_head(p3_out, C, mid_p3)  # [B,80,80,C]
    out_p4 = _make_head(p4_out, C, mid_p4)  # [B,40,40,C]
    out_p5 = _make_head(p5_out, C, mid_p5)  # [B,20,20,C]

    def flatten_hw(x):
        shp = tf.shape(x)
        b, h, w, c = shp[0], shp[1], shp[2], shp[3]
        x = tf.reshape(x, [b, h * w, c])
        return x

    f3 = L.Lambda(flatten_hw, name='flat_p3')(out_p3)
    f4 = L.Lambda(flatten_hw, name='flat_p4')(out_p4)
    f5 = L.Lambda(flatten_hw, name='flat_p5')(out_p5)
    out = L.Concatenate(axis=1, name='preds')([f3, f4, f5])  # [B, 8400, C]

    model = tf.keras.Model(inp, out, name='u8s_pose_keras')
    return model


# ---------------------------- Quick Self-Test ---------------------------- #
if __name__ == '__main__':
    m = build_u8s_pose((640,640,3), num_classes=7, num_kpt=15, kpt_vals=3)
    m.summary(line_length=120)
    x = tf.random.uniform([2,640,640,3], 0, 1, dtype=tf.float32)
    y = m(x)
    print('Output shape:', y.shape)  # expect (2, 8400, 5 + 7 + 15*3) = (2, 8400, 57)
