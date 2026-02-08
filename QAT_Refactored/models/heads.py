import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L
from typing import Tuple, List, Optional

@K.utils.register_keras_serializable(package="QAT")
class TeacherCompatHead(L.Layer):
    """
    Universal Head for YOLOv8-Pose.
    
    [Architecture Alignment]
    - Training: Outputs (B, C, N) Logits for efficient loss calculation.
    - Export: Outputs (B, C, N) to match C++ 'TFlite.h' memory layout.
      Layout: [Box (4) | Cls (NUM_CLS) | Kpts (NUM_KPT*KPT_VALS)]
    """
    def __init__(self, num_cls: int, num_kpt: int, kpt_vals: int, ch: int = 256, name: str = "head", **kw):
        super().__init__(name=name, **kw)
        self.num_cls = int(num_cls)
        self.num_kpt = int(num_kpt)
        self.kpt_vals = int(kpt_vals)
        self.ch = int(ch)
        
        # Calculate total output channels
        # Matches C++: 4 (Box) + NUM_CLASS + Keypoint_NUM * 3
        self.C = 4 + self.num_cls + self.num_kpt * self.kpt_vals
        
        # Projection Layers (1x1 Conv) for P3, P4, P5
        self.p3_conv = L.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p3_out")
        self.p4_conv = L.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p4_out")
        self.p5_conv = L.Conv2D(self.C, 1, padding="same", use_bias=True, name=f"{name}/p5_out")

    def _to_BCN(self, x: tf.Tensor) -> tf.Tensor:
        """
        Transforms Feature Map (B, H, W, C) -> Flattened Anchors (B, C, N).
        
        Mathematical Proof for C++ Compatibility:
        1. Input: (B, H, W, C)
        2. Permute(3, 1, 2) -> (B, C, H, W)
           Memory: [B, C0_H0_W0...C0_Hn_Wn, C1...]
        3. Reshape(B, C, -1) -> (B, C, N)
           Memory remains: [C0_Anchor0...C0_AnchorN, C1_Anchor0...]
           
        This matches C++ access: data[channel * NUM_BOXES + anchor_idx]
        """
        # 1. BHWC -> BCHW
        x = L.Permute((3, 1, 2))(x)
        
        # 2. Flatten spatial dims (H*W = N_stride)
        # Note: We use dynamic reshape (-1) here to handle variable batch size, 
        # but C dimension is fixed.
        x = L.Reshape((self.C, -1))(x) 
        return x

    def call(self, feats: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Args:
            feats: Tuple of (P3, P4, P5) feature maps.
        Returns:
            Concatenated Logits (B, Total_Channels, Total_Anchors)
        """
        p3, p4, p5 = feats
        
        # 1. Project to Output Channels (1x1 Conv)
        # Output: (B, H/8, W/8, C)
        y3 = self.p3_conv(p3)
        y4 = self.p4_conv(p4)
        y5 = self.p5_conv(p5)
        
        # 2. Flatten and Transpose to (B, C, N_stride)
        bcn3 = self._to_BCN(y3)
        bcn4 = self._to_BCN(y4)
        bcn5 = self._to_BCN(y5)
        
        # 3. Concatenate all strides along Anchor dimension (Last dim)
        # Result: (B, C, N_total)
        return L.Concatenate(axis=2, name=f"{self.name}/concat_anchors")([bcn3, bcn4, bcn5])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_cls": self.num_cls,
            "num_kpt": self.num_kpt,
            "kpt_vals": self.kpt_vals,
            "ch": self.ch
        })
        return cfg


@K.utils.register_keras_serializable(package="QAT")
class U8PoseCompatHead(L.Layer):
    """
    Ultralytics YOLOv8-Pose 相容頭部。
    輸出未經 Flatten 的 Feature Maps (用於 Loss 計算)。
    """
    def __init__(self, num_cls, num_kpt, kpt_vals, reg_max=16, ch=128, name="u8pose_head", **kw):
        super().__init__(name=name, **kw)
        self.nc = int(num_cls)
        self.nk = int(num_kpt)
        self.kv = int(kpt_vals)
        self.reg_max = int(reg_max)
        self.ch = int(ch)
        
        # Helper to build towers
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

        # Final Projections
        box_ch = 4 * self.reg_max # DFL required channels
        
        self.out_reg = [L.Conv2D(box_ch, 1, name=f'head.regout.p{i}') for i in [3,4,5]]
        self.out_cls = [L.Conv2D(self.nc, 1, name=f'head.coout.p{i}') for i in [3,4,5]]
        self.out_kpt = [L.Conv2D(self.nk * self.kv, 1, name=f'head.kptout.p{i}') for i in [3,4,5]]

    def _run_tower(self, x, blocks):
        y = x
        for i in range(0, len(blocks), 3):
            y = blocks[i](y)
            y = blocks[i+1](y)
            y = blocks[i+2](y)
        return y

    def _to_bchw(self, t):
        # Keras (BHWC) -> Loss Expectation (BCHW)
        return L.Permute((3, 1, 2))(t)

    def call(self, feats, training=False):
        p3, p4, p5 = feats
        feats_out = []
        kpts_out = []

        for i, p in enumerate((p3, p4, p5)):
            r = self._run_tower(p, self.twr_reg[i])
            c = self._run_tower(p, self.twr_cls[i])
            k = self._run_tower(p, self.twr_kpt[i])

            r = self.out_reg[i](r)
            c = self.out_cls[i](c)
            k = self.out_kpt[i](k)

            # Concatenate [Cls, Box] for consistency with loss function expectations
            rc = L.Concatenate(axis=-1)([c, r])
            
            feats_out.append(self._to_bchw(rc))
            kpts_out.append(self._to_bchw(k))

        return feats_out, kpts_out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_cls": self.nc,
            "num_kpt": self.nk,
            "kpt_vals": self.kv,
            "reg_max": self.reg_max,
            "ch": self.ch
        })