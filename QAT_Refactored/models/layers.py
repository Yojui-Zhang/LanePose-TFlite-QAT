# Source File: QAT_Refactored/models/layers.py

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

from tensorflow import keras as K
from tensorflow.keras import layers as L

# ==============================================================================
# Helper Functions for RepVGG Fusion
# ==============================================================================

def _fuse_bn_tensor(conv: L.Conv2D, bn: L.BatchNormalization) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core logic to fuse Conv2D and BatchNormalization layers into a single set of weights.
    Formula:
        W_fused = W_conv * (gamma / std)
        b_fused = b_conv + (beta - mean * gamma / std)
        where std = sqrt(var + epsilon)
    """
    # 1. Get Conv weights
    # kernel shape: (k, k, in_ch, out_ch)
    kernel = conv.kernel.numpy()
    if conv.use_bias:
        bias = conv.bias.numpy()
    else:
        bias = np.zeros(kernel.shape[-1], dtype=np.float32)

    # 2. Get BN weights
    # BN weights order: [gamma, beta, moving_mean, moving_variance] (if scale=True, center=True)
    # Check if scale/center are enabled
    gamma = bn.gamma.numpy() if bn.scale else np.ones(bn.moving_variance.shape, dtype=np.float32)
    beta = bn.beta.numpy() if bn.center else np.zeros(bn.moving_variance.shape, dtype=np.float32)
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()
    epsilon = bn.epsilon

    # 3. Compute Fusion Factor
    std = np.sqrt(var + epsilon)
    t = gamma / std  # (out_ch,)
    
    # 4. Fuse Kernel
    # Reshape t for broadcasting: (1, 1, 1, out_ch)
    t_reshape = t.reshape((1, 1, 1, -1))
    kernel_fused = kernel * t_reshape
    
    # 5. Fuse Bias
    # bias_fused = beta - mean * (gamma / std) + conv_bias
    #            = beta - mean * t + bias
    bias_fused = beta - mean * t + bias
    
    return kernel_fused, bias_fused

# ==============================================================================
# Basic Blocks
# ==============================================================================

def conv_bn_act(x: tf.Tensor, out_ch: int, k: int = 3, s: int = 1, 
                name: Optional[str] = None, act: str = "relu6") -> tf.Tensor:
    """Standard Conv-BN-Act block with named scope support."""
    prefix = name if name else None
    
    x = L.Conv2D(out_ch, k, strides=s, padding='same', use_bias=False,
                  name=f"{prefix}/conv" if prefix else None)(x)
    x = L.BatchNormalization(name=f"{prefix}/bn" if prefix else None)(x)

    if act == "relu6":
        x = L.ReLU(max_value=6.0, name=f"{prefix}/relu6" if prefix else None)(x)
    elif act == "relu":
        x = L.ReLU(name=f"{prefix}/relu" if prefix else None)(x)
    elif act == "lrelu":
        x = L.LeakyReLU(0.1, name=f"{prefix}/lrelu" if prefix else None)(x)
    elif act == "swish":
        x = L.Activation("swish", name=f"{prefix}/swish" if prefix else None)(x)
    return x

def dw_conv_bn_act(x: tf.Tensor, out_ch: int, k: int = 3, s: int = 1, 
                   name: Optional[str] = None, act: str = 'relu6') -> tf.Tensor:
    """Depthwise Separable Convolution Block."""
    prefix = name if name else None
    
    # Depthwise
    x = L.DepthwiseConv2D(k, strides=s, padding='same', use_bias=False,
                          name=f'{prefix}/dw' if prefix else None)(x)
    x = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                             name=f'{prefix}/dw_bn' if prefix else None)(x)
    if act == 'relu6':
        x = L.ReLU(max_value=6.0, name=f'{prefix}/dw_relu6' if prefix else None)(x)
        
    # Pointwise
    x = L.Conv2D(out_ch, 1, strides=1, padding='same', use_bias=False,
                 name=f'{prefix}/pw' if prefix else None)(x)
    x = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                             name=f'{prefix}/pw_bn' if prefix else None)(x)
    if act == 'relu6':
        x = L.ReLU(max_value=6.0, name=f'{prefix}/pw_relu6' if prefix else None)(x)
        
    return x

def sppf_block(x: tf.Tensor, out_ch: int, k: int = 5, name: Optional[str] = None) -> tf.Tensor:
    """Spatial Pyramid Pooling - Fast (SPPF)."""
    prefix = name if name else None
    
    x1 = conv_bn_act(x, out_ch, k=1, s=1, name=f"{prefix}/cv1" if prefix else None, act='relu6')
    p1 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=f"{prefix}/p1" if prefix else None)(x1)
    p2 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=f"{prefix}/p2" if prefix else None)(p1)
    p3 = L.MaxPool2D(pool_size=k, strides=1, padding='same',
                     name=f"{prefix}/p3" if prefix else None)(p2)
    
    cat = L.Concatenate(axis=-1, name=f"{prefix}/cat" if prefix else None)([x1, p1, p2, p3])
    y = conv_bn_act(cat, out_ch, k=1, s=1, name=f"{prefix}/cv2" if prefix else None, act='relu6')
    return y

# [CRITICAL REFACTOR] Convert functional se_block to Class-based Layer
# to prevent variable creation inside tf.function/loops.
@K.utils.register_keras_serializable(package="QAT")
class SEBlock(L.Layer):
    """Squeeze-and-Excitation Block as a Layer."""
    def __init__(self, channels: int, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        
        # Ensure minimum 1 channel
        squeeze_ch = max(1, self.channels // self.ratio)
        
        self.gap = L.GlobalAveragePooling2D(keepdims=True, name="gap")
        self.conv_down = L.Conv2D(squeeze_ch, 1, activation='relu', name="down")
        self.conv_up = L.Conv2D(self.channels, 1, activation='sigmoid', name="up")
        self.multiply = L.Multiply(name="scale")

    def call(self, x):
        s = self.gap(x)
        s = self.conv_down(s)
        s = self.conv_up(s)
        return self.multiply([x, s])

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio
        })
        return config

# ==============================================================================
# Advanced Blocks (RepVGG & Attention)
# ==============================================================================

@K.utils.register_keras_serializable(package="QAT")
class RepVGGBlock(L.Layer):
    """
    RepVGG Block supporting structural re-parameterization and QAT.
    It contains 3 branches during training:
        1. 3x3 Conv + BN
        2. 1x1 Conv + BN
        3. Identity + BN (if dimensions match)
    During inference (deploy=True), these are fused into a single 3x3 Conv.
    """
    def __init__(self, out_ch: int, k: int = 3, s: int = 1, 
                 use_se: bool = False, act: str = 'relu6', deploy: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.use_se = use_se
        self.act_name = act
        self.deploy = deploy
        
        # Explicitly init attributes for reflection/QAT safety
        self.rbr_dense_conv: Optional[L.Conv2D] = None
        self.rbr_dense_bn: Optional[L.BatchNormalization] = None
        self.rbr_1x1_conv: Optional[L.Conv2D] = None
        self.rbr_1x1_bn: Optional[L.BatchNormalization] = None
        self.rbr_identity_bn: Optional[L.BatchNormalization] = None
        self.rbr_reparam: Optional[L.Conv2D] = None
        self.activation: Optional[L.Layer] = None
        self.se_layer: Optional[SEBlock] = None # Placeholder

        # [FIX] Initialize SEBlock here if needed, to ensure variables are tracked.
        if self.use_se:
             self.se_layer = SEBlock(channels=self.out_ch, ratio=16, name="se")

    def build(self, input_shape):
        in_ch = input_shape[-1]
        
        if self.deploy:
            # Inference Mode: Single Conv Layer
            self.rbr_reparam = L.Conv2D(self.out_ch, self.k, strides=self.s, padding='same',
                                        use_bias=True, name=f"{self.name}/rbr_reparam")
        else:
            # Training Mode: Multi-branch
            
            # 1. 3x3 Branch
            self.rbr_dense_conv = L.Conv2D(self.out_ch, self.k, strides=self.s, padding='same',
                                           use_bias=False, name=f"{self.name}/rbr_dense/conv")
            self.rbr_dense_bn = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                                                     name=f"{self.name}/rbr_dense/bn")
            
            # 2. 1x1 Branch
            self.rbr_1x1_conv = L.Conv2D(self.out_ch, 1, strides=self.s, padding='same',
                                         use_bias=False, name=f"{self.name}/rbr_1x1/conv")
            self.rbr_1x1_bn = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                                                   name=f"{self.name}/rbr_1x1/bn")
            
            # 3. Identity Branch (Only if dimensions match)
            if in_ch == self.out_ch and self.s == 1:
                self.rbr_identity_bn = L.BatchNormalization(epsilon=1e-3, momentum=0.99,
                                                            name=f"{self.name}/rbr_identity/bn")
        
        # Activation
        if self.act_name == 'relu6':
            self.activation = L.ReLU(max_value=6.0, name=f"{self.name}/relu6")
        elif self.act_name == 'relu':
            self.activation = L.ReLU(name=f"{self.name}/relu")
        elif self.act_name == 'swish':
            self.activation = L.Activation('swish', name=f"{self.name}/swish")
        elif self.act_name == "lrelu":
            self.activation = L.LeakyReLU(0.1, name=f"{self.name}/lrelu")

        super().build(input_shape)

    def call(self, x, training=None):
        if self.deploy:
            y = self.rbr_reparam(x)
        else:
            # Training Mode
            # Sum of branches
            x_dense = self.rbr_dense_bn(self.rbr_dense_conv(x), training=training)
            x_1x1 = self.rbr_1x1_bn(self.rbr_1x1_conv(x), training=training)
            
            y = x_dense + x_1x1
            
            if self.rbr_identity_bn is not None:
                y += self.rbr_identity_bn(x, training=training)

        # [FIX] Use class-based SEBlock
        if self.use_se and self.se_layer is not None:
             y = self.se_layer(y)

        if self.activation:
            y = self.activation(y)
            
        return y

    def switch_to_deploy(self):
        """
        Structural Re-parameterization:
        Fuses the 3x3, 1x1, and Identity branches into a single 3x3 Convolution.
        This method MUST be called before exporting the model.
        """
        if self.deploy:
            return

        print(f"[RepVGGBlock] Switching to deploy mode: {self.name}")
        
        # 1. Get fused weights for 3x3 Branch
        kernel_3x3, bias_3x3 = _fuse_bn_tensor(self.rbr_dense_conv, self.rbr_dense_bn)
        
        # 2. Get fused weights for 1x1 Branch
        kernel_1x1, bias_1x1 = _fuse_bn_tensor(self.rbr_1x1_conv, self.rbr_1x1_bn)
        
        # Pad 1x1 kernel to 3x3 (Assuming k=3 for the main branch)
        # 1x1 is at the center of 3x3
        pad_h = (self.k - 1) // 2
        pad_w = (self.k - 1) // 2
        kernel_1x1_padded = np.pad(
            kernel_1x1, 
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0)), 
            mode='constant'
        )

        # 3. Get fused weights for Identity Branch
        kernel_id = np.zeros_like(kernel_3x3)
        bias_id = np.zeros_like(bias_3x3)
        
        if self.rbr_identity_bn is not None:
            # Create a pseudo Identity Conv Kernel (1x1 conv acting as identity)
            # Shape: (k, k, in_ch, out_ch), but effectively 1x1 at center
            # Identity means output channel i gets input channel i.
            in_ch = kernel_3x3.shape[2]
            # Construct identity kernel: 1.0 at center spatial pos for matching channels
            t = np.zeros((1, 1, in_ch, self.out_ch), dtype=np.float32)
            for i in range(in_ch):
                t[0, 0, i, i % self.out_ch] = 1.0 # Ensure broadcast if out > in (rare) or match

            # Fuse with the Identity BN
            # We pretend we have a conv layer with 't' as kernel and 0 bias
            # But since we don't have the layer object, we manually do the math:
            # W_fused = W_id * (gamma/std)
            # b_fused = beta - mean * (gamma/std)
            
            gamma = self.rbr_identity_bn.gamma.numpy() if self.rbr_identity_bn.scale else np.ones(self.out_ch)
            beta = self.rbr_identity_bn.beta.numpy() if self.rbr_identity_bn.center else np.zeros(self.out_ch)
            mean = self.rbr_identity_bn.moving_mean.numpy()
            var = self.rbr_identity_bn.moving_variance.numpy()
            eps = self.rbr_identity_bn.epsilon
            
            std = np.sqrt(var + eps)
            factor = gamma / std
            
            # W_id is 1.0 at diagonals. So W_fused is just 'factor' at diagonals.
            # We pad 't' to 3x3 first
            t_padded = np.pad(
                t, 
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0)), 
                mode='constant'
            )
            
            kernel_id = t_padded * factor.reshape(1, 1, 1, -1)
            bias_id = beta - mean * factor

        # 4. Sum Everything Up
        final_kernel = kernel_3x3 + kernel_1x1_padded + kernel_id
        final_bias = bias_3x3 + bias_1x1 + bias_id

        # 5. Create the new Deploy Layer
        self.rbr_reparam = L.Conv2D(
            filters=self.out_ch,
            kernel_size=self.k,
            strides=self.s,
            padding='same',
            use_bias=True,
            name=f"{self.name}/rbr_reparam"
        )
        
        # Build it to initialize weights tensor
        # Input shape: (None, H, W, in_ch) -> Take in_ch from kernel shape
        in_ch = final_kernel.shape[2]
        self.rbr_reparam.build((None, None, None, in_ch))
        
        # Set weights
        self.rbr_reparam.set_weights([final_kernel, final_bias])

        # 6. Cleanup (Release memory of training branches)
        self.__delattr__('rbr_dense_conv')
        self.__delattr__('rbr_dense_bn')
        self.__delattr__('rbr_1x1_conv')
        self.__delattr__('rbr_1x1_bn')
        if hasattr(self, 'rbr_identity_bn'):
            self.__delattr__('rbr_identity_bn')
            
        self.deploy = True

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "out_ch": self.out_ch,
            "k": self.k,
            "s": self.s,
            "use_se": self.use_se,
            "act": self.act_name,
            "deploy": self.deploy
        })
        return cfg

@K.utils.register_keras_serializable(package="QAT")
class ChannelAttention(L.Layer):
    def __init__(self, ratio=16, **kw):
        super().__init__(**kw)
        self.ratio = int(ratio)
        self.mlp1 = None
        self.mlp2 = None

    def build(self, input_shape):
        C = int(input_shape[-1])
        self.mlp1 = L.Conv2D(C // self.ratio, 1, activation="relu", use_bias=True, name=f"{self.name}/mlp1")
        self.mlp2 = L.Conv2D(C, 1, activation=None, use_bias=True, name=f"{self.name}/mlp2")
        super().build(input_shape)

    def call(self, x):
        avg = L.GlobalAveragePooling2D(keepdims=True, name=f"{self.name}/gap")(x)
        mx = L.GlobalMaxPooling2D(keepdims=True, name=f"{self.name}/gmp")(x)
        
        # Shared MLP
        a1 = self.mlp2(self.mlp1(avg))
        a2 = self.mlp2(self.mlp1(mx))
        
        scale = L.Activation('sigmoid', name=f"{self.name}/sigmoid")(L.Add()([a1, a2]))
        return L.Multiply(name=f"{self.name}/scale")([x, scale])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg

@K.utils.register_keras_serializable(package="QAT")
class SpatialAttention(L.Layer):
    def __init__(self, kernel_size=7, **kw):
        super().__init__(**kw)
        self.kernel_size = int(kernel_size)
        self.conv = L.Conv2D(1, self.kernel_size, padding="same", use_bias=False, activation="sigmoid",
                             name=f"{self.name}/conv")

    def call(self, x):
        # Use simple reduction ops
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        cat = L.Concatenate(axis=-1)([avg, mx])
        scale = self.conv(cat)
        return L.Multiply(name=f"{self.name}/scale")([x, scale])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg

@K.utils.register_keras_serializable(package="QAT")
class CBAM(L.Layer):
    def __init__(self, ratio=16, kernel_size=7, **kw):
        super().__init__(**kw)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(ratio=ratio, name=f"{self.name}/ca")
        self.sa = SpatialAttention(kernel_size=kernel_size, name=f"{self.name}/sa")

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return cfg