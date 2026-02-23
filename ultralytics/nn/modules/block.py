# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Sequence, Dict, Any, Callable, Final
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

try:
    # torchvision >= 0.13 通常提供
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
except Exception:
    _tv_deform_conv2d = None

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "RepVGGBlock",
    "ConformableBlock",
    "ConformableInvertedResidual",
    "RepEdgeACBlock",
    "MobileNetV3_Bneck",
    "GhostBottleneckV2",
    "ShuffleNetV2Block",
)


# ============================================================
# MobileNetV3 Units
# ============================================================
class SEModule_v3(nn.Module):
    """Squeeze-and-Excitation block specifically for MobileNetV3 using Hardsigmoid."""
    def __init__(self, c, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, max(8, c // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, c // reduction), c, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNetV3_Bneck(nn.Module):
    """MobileNetV3 Inverted Residual Bottleneck."""
    def __init__(self, c1, c2, c3, k=3, s=1, use_se=False, use_hs=False):
        super().__init__()
        self.use_res_connect = (s == 1 and c1 == c2)
        act = nn.Hardswish if use_hs else nn.ReLU

        layers = []
        if c3 != c1:
            layers.append(Conv(c1, c3, 1, 1, act=act()))  # pw
        layers.append(DWConv(c3, c3, k, s, act=act()))    # dw
        if use_se:
            layers.append(SEModule_v3(c3))
        layers.append(Conv(c3, c2, 1, 1, act=False))      # pw-linear
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


# ============================================================
# GhostNetV2 Units
# ============================================================
class GhostModuleV2(nn.Module):
    """GhostModuleV2 with DFC Attention."""
    def __init__(self, c1, c2, k=1, ratio=2, dw_size=3, s=1, relu=True):
        super().__init__()
        self.oup = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        # DFC Attention Branch (Simplified for YOLO backbone integration)
        self.short_conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 1, 1, 0, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        res = self.short_conv(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out * torch.sigmoid(res)

class GhostBottleneckV2(nn.Module):
    """GhostNetV2 Bottleneck."""
    def __init__(self, c1, c2, c3, k=3, s=1):
        super().__init__()
        self.use_res_connect = (s == 1 and c1 == c2)
        self.conv = nn.Sequential(
            GhostModuleV2(c1, c3, relu=True),
            DWConv(c3, c3, k, s, act=False) if s == 2 else nn.Identity(),
            GhostModuleV2(c3, c2, relu=False)
        )
        if not self.use_res_connect:
            self.shortcut = nn.Sequential(
                DWConv(c1, c1, k, s, act=False),
                Conv(c1, c2, 1, 1, act=False)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ============================================================
# ShuffleNetV2 Units
# ============================================================
class ShuffleNetV2Block(nn.Module):
    """ShuffleNetV2 Inverted Residual Block."""
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.stride = stride
        assert c2 % 2 == 0, "Output channels must be even"
        c_ = c2 // 2
        self.use_split = (stride == 1 and c1 == c2)

        if not self.use_split:
            self.branch1 = nn.Sequential(
                DWConv(c1, c1, 3, stride, act=False),
                Conv(c1, c_, 1, 1, act=True)
            )
        else:
            self.branch1 = nn.Identity()

        in_channels = c_ if self.use_split else c1
        self.branch2 = nn.Sequential(
            Conv(in_channels, c_, 1, 1, act=True),
            DWConv(c_, c_, 3, stride, act=False),
            Conv(c_, c_, 1, 1, act=True)
        )

    def forward(self, x):
        if self.use_split:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # channel_shuffle 取自您代碼中現有的函數
        return channel_shuffle(out, 2)

# ===================================================================================================================
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):

        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.SiLU()
        # self.nonlinearity = nn.ReLU()
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
 
        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_1x1'):
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                    kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                    padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.rbr_dense = self.rbr_reparam
            # self.__delattr__('rbr_dense')
            self.__delattr__('rbr_1x1')
            if hasattr(self, 'rbr_identity'):
                self.__delattr__('rbr_identity')
            if hasattr(self, 'id_tensor'):
                self.__delattr__('id_tensor')
            self.deploy = True
 
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
 
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
 
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
 
    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_dense(inputs))
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
 
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


# ===================================================================================================================
# ============================================================
# RepEdgeACBlock
# ============================================================

@dataclass(frozen=True)
class _FuseResult:
    kernel: torch.Tensor
    bias: torch.Tensor


class RepEdgeACBlock(nn.Module):
    """
    RepVGG-like block with additional anisotropic depthwise branches (1x3, 3x1).
    Train-time: multi-branch -> sum -> activation.
    Deploy-time: single 3x3 conv (bias=True) after fusing all branches.

    Design constraints:
    - Pure conv/BN/activation.
    - DW branches enabled only when representable under a single dense 3x3 conv (groups=1 and c1==c2).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: bool = True,
        deploy: bool = False,
        use_dw_branches: bool = True,
    ) -> None:
        super().__init__()

        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"Invalid channels: c1={c1}, c2={c2}")
        if k != 3:
            raise ValueError("RepEdgeACBlock only supports k=3 for deterministic fusion.")
        if s not in (1, 2):
            raise ValueError(f"Unsupported stride: s={s} (expected 1 or 2)")
        if g <= 0:
            raise ValueError(f"Invalid groups: g={g}")

        self.in_channels: int = c1
        self.out_channels: int = c2
        self.kernel_size: int = k
        self.stride: int = s
        self.groups: int = g
        self.padding: int = (k // 2) if p is None else p
        self.deploy: bool = deploy

        self.act: nn.Module = nn.SiLU(inplace=True) if act else nn.Identity()

        # Why: DW anisotropic branches are only safe to collapse into a single dense conv when:
        # - groups==1 (target deploy conv is dense) and
        # - c1==c2 (DW branch maps channel i->i; summation requires aligned channel semantics).
        self._dw_enabled: bool = bool(use_dw_branches and (g == 1) and (c1 == c2))

        if deploy:
            self.rbr_reparam: nn.Conv2d = nn.Conv2d(
                c1, c2, k, s, self.padding, groups=g, bias=True
            )
            self.rbr_dense = None
            self.rbr_1x1 = None
            self.rbr_identity = None
            self.rbr_dw_1x3 = None
            self.rbr_dw_3x1 = None
            return

        self.rbr_reparam = None

        self.rbr_dense: nn.Sequential = conv_bn(
            c1, c2, kernel_size=3, stride=s, padding=self.padding, groups=g
        )
        self.rbr_1x1: nn.Sequential = conv_bn(
            c1, c2, kernel_size=1, stride=s, padding=0, groups=g
        )

        self.rbr_identity: Optional[nn.BatchNorm2d] = None
        if c1 == c2 and s == 1:
            # Why: identity branch stabilizes early optimization without adding conv parameters.
            self.rbr_identity = nn.BatchNorm2d(c1)

        self.rbr_dw_1x3: Optional[nn.Sequential] = None
        self.rbr_dw_3x1: Optional[nn.Sequential] = None
        if self._dw_enabled:
            # Why: anisotropic DW adds directional bias at negligible parameter cost.
            self.rbr_dw_1x3 = nn.Sequential(
                nn.Conv2d(c1, c1, (1, 3), s, (0, 1), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )
            self.rbr_dw_3x1 = nn.Sequential(
                nn.Conv2d(c1, c1, (3, 1), s, (1, 0), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)}")

        if self.deploy:
            assert self.rbr_reparam is not None
            return self.act(self.rbr_reparam(x))

        out = self.rbr_dense(x) + self.rbr_1x1(x)

        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)

        if self._dw_enabled:
            assert self.rbr_dw_1x3 is not None and self.rbr_dw_3x1 is not None
            out = out + self.rbr_dw_1x3(x) + self.rbr_dw_3x1(x)

        return self.act(out)

    @staticmethod
    def _fuse_conv_bn_branch(branch: nn.Sequential) -> _FuseResult:
        if not isinstance(branch, nn.Sequential) or len(branch) != 2:
            raise TypeError("Branch must be nn.Sequential(conv, bn).")
        conv, bn = branch[0], branch[1]
        # if not _is_conv(conv) or not _is_bn(bn):
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            raise TypeError("Branch must be nn.Sequential(conv, bn).")

        w = conv.weight
        if conv.bias is None:
            b = torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
        else:
            b = conv.bias

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)

        fused_w = w * scale
        fused_b = beta + (b - mean) * (gamma / std)

        return _FuseResult(kernel=fused_w, bias=fused_b)

    @staticmethod
    def _fuse_identity_bn(
        bn: nn.BatchNorm2d,
        channels: int,
        groups: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _FuseResult:
        if groups != 1:
            raise ValueError("Identity fusion is only defined for groups=1 in this implementation.")
        if channels <= 0:
            raise ValueError(f"Invalid channels: {channels}")

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)

        # identity 3x3 kernel: diag at center
        kernel = torch.zeros((channels, channels, 3, 3), device=device, dtype=dtype)
        idx = torch.arange(channels, device=device)
        kernel[idx, idx, 1, 1] = 1.0

        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_w = kernel * scale
        fused_b = beta - mean * (gamma / std)

        return _FuseResult(kernel=fused_w, bias=fused_b)

    @staticmethod
    def _pad_1x1_to_3x3(k1: torch.Tensor) -> torch.Tensor:
        if k1.dim() != 4 or k1.size(-1) != 1 or k1.size(-2) != 1:
            raise ValueError(f"Expected 1x1 kernel, got shape={tuple(k1.shape)}")
        return F.pad(k1, [1, 1, 1, 1])

    @staticmethod
    def _embed_dw_to_dense_3x3(
        dw_kernel: torch.Tensor,
        out_channels: int,
        device: torch.device,
        dtype: torch.dtype,
        kind: str,
    ) -> torch.Tensor:
        """
        Convert depthwise (C,1,1,3) or (C,1,3,1) into dense (C,C,3,3) diagonal kernel.
        """
        if out_channels <= 0:
            raise ValueError(f"Invalid out_channels: {out_channels}")
        if dw_kernel.dim() != 4 or dw_kernel.size(0) != out_channels:
            raise ValueError(f"Invalid dw_kernel shape={tuple(dw_kernel.shape)} for out_channels={out_channels}")

        dense = torch.zeros((out_channels, out_channels, 3, 3), device=device, dtype=dtype)
        idx = torch.arange(out_channels, device=device)

        if kind == "1x3":
            if dw_kernel.size(2) != 1 or dw_kernel.size(3) != 3:
                raise ValueError(f"Expected (C,1,1,3), got {tuple(dw_kernel.shape)}")
            dense[idx, idx, 1, 0:3] = dw_kernel[:, 0, 0, :]
        elif kind == "3x1":
            if dw_kernel.size(2) != 3 or dw_kernel.size(3) != 1:
                raise ValueError(f"Expected (C,1,3,1), got {tuple(dw_kernel.shape)}")
            dense[idx, idx, 0:3, 1] = dw_kernel[:, 0, :, 0]
        else:
            raise ValueError(f"Unknown kind: {kind}")

        return dense

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            assert self.rbr_reparam is not None
            return self.rbr_reparam.weight, self.rbr_reparam.bias

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        k_sum = torch.zeros(
            (self.out_channels, self.in_channels, 3, 3),
            device=device,
            dtype=dtype,
        )
        b_sum = torch.zeros((self.out_channels,), device=device, dtype=dtype)

        # 3x3 branch
        r = self._fuse_conv_bn_branch(self.rbr_dense)
        k_sum += r.kernel
        b_sum += r.bias

        # 1x1 branch (pad to 3x3)
        r = self._fuse_conv_bn_branch(self.rbr_1x1)
        k_sum += self._pad_1x1_to_3x3(r.kernel)
        b_sum += r.bias

        # identity BN branch
        if self.rbr_identity is not None:
            r = self._fuse_identity_bn(
                self.rbr_identity,
                channels=self.in_channels,
                groups=self.groups,
                device=device,
                dtype=dtype,
            )
            k_sum += r.kernel
            b_sum += r.bias

        # anisotropic DW branches
        if self._dw_enabled:
            assert self.rbr_dw_1x3 is not None and self.rbr_dw_3x1 is not None

            r13 = self._fuse_conv_bn_branch(self.rbr_dw_1x3)
            r31 = self._fuse_conv_bn_branch(self.rbr_dw_3x1)

            # represent DW as diagonal dense kernel
            k_sum += self._embed_dw_to_dense_3x3(r13.kernel, self.out_channels, device, dtype, kind="1x3")
            b_sum += r13.bias

            k_sum += self._embed_dw_to_dense_3x3(r31.kernel, self.out_channels, device, dtype, kind="3x1")
            b_sum += r31.bias

        return k_sum, b_sum

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        k, b = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
        )

        self.rbr_reparam.weight.data.copy_(k)
        self.rbr_reparam.bias.data.copy_(b)

        # Why: remove training branches to avoid accidental parameter updates / state drift.
        if hasattr(self, "rbr_dense"):
            del self.rbr_dense
        if hasattr(self, "rbr_1x1"):
            del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "rbr_dw_1x3"):
            del self.rbr_dw_1x3
        if hasattr(self, "rbr_dw_3x1"):
            del self.rbr_dw_3x1

        self.deploy = True

# ===================================================================================================================

# ============================================================
# DeformConv2d Safe Wrapper (AMP/bf16 stable + optional fallback)
# ============================================================

IntOrPair = Union[int, Tuple[int, int]]

def _to_2tuple(v: IntOrPair) -> Tuple[int, int]:
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"expected 2-tuple, got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


@dataclass
class DeformStatsLogger:
    log_every: int = 5000
    max_topk: int = 3
    _calls: int = 0
    _deform: int = 0
    _fallback: int = 0
    _reason: Counter = field(default_factory=Counter)
    _exc: Counter = field(default_factory=Counter)
    _printed_reasons: set[str] = field(default_factory=set)
    _last_backend: str = "unknown"

    def on_deform(self, backend: str) -> None:
        self._calls += 1
        self._deform += 1
        self._last_backend = backend
        self._maybe_flush()

    def on_fallback(self, reason: str, backend: str, exc: Optional[BaseException] = None) -> None:
        self._calls += 1
        self._fallback += 1
        self._reason[reason] += 1
        self._last_backend = backend
        if exc is not None:
            self._exc[type(exc).__name__] += 1

        if reason not in self._printed_reasons:
            self._printed_reasons.add(reason)
            # Why: 讓訓練 log 第一時間可見「為何在 fallback」，但不打斷訓練。
            print(f"[DEFORM_FALLBACK] reason={reason} backend={backend}", flush=True)

        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if self.log_every <= 0:
            return
        if (self._calls % self.log_every) != 0:
            return

        top_reason = self._reason.most_common(self.max_topk)
        top_exc = self._exc.most_common(self.max_topk)
        print(
            f"[DEFORM_STATS] calls={self._calls} deform={self._deform} fallback={self._fallback} "
            f"top_reason={top_reason} top_exc={top_exc} last_backend={self._last_backend}",
            flush=True,
        )


_DEFAULT_DEFORM_LOGGER = DeformStatsLogger(log_every=5000, max_topk=3)


@dataclass(frozen=True)
class _TorchvisionDeformCaps:
    available: bool
    supports_groups: bool
    supports_mask: bool
    supports_deformable_groups: bool


_TV_DEFORM_CAPS: Optional[_TorchvisionDeformCaps] = None
_TV_DEFORM_FN: Optional[Callable[..., torch.Tensor]] = None


def _get_torchvision_deform_caps() -> _TorchvisionDeformCaps:
    global _TV_DEFORM_CAPS, _TV_DEFORM_FN
    if _TV_DEFORM_CAPS is not None:
        return _TV_DEFORM_CAPS

    try:
        from torchvision.ops import deform_conv2d as tv_deform_conv2d  # type: ignore
        _TV_DEFORM_FN = tv_deform_conv2d
    except Exception:
        _TV_DEFORM_FN = None
        _TV_DEFORM_CAPS = _TorchvisionDeformCaps(
            available=False,
            supports_groups=False,
            supports_mask=False,
            supports_deformable_groups=False,
        )
        return _TV_DEFORM_CAPS

    sig = inspect.signature(_TV_DEFORM_FN)
    params = sig.parameters
    supports_groups = "groups" in params
    supports_mask = "mask" in params
    supports_dg = ("deformable_groups" in params) or ("offset_groups" in params)
    _TV_DEFORM_CAPS = _TorchvisionDeformCaps(
        available=True,
        supports_groups=supports_groups,
        supports_mask=supports_mask,
        supports_deformable_groups=supports_dg,
    )
    return _TV_DEFORM_CAPS


def _grid_sample_normalize(x: torch.Tensor, denom: int) -> torch.Tensor:
    # Why: 避免 W/H=1 時除以 0，並穩定 grid_sample 的 normalizing。
    if denom <= 0:
        return torch.zeros_like(x)
    return (2.0 * x / float(denom)) - 1.0


def _depthwise_deform_conv2d_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    offset: torch.Tensor,
    mask: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    deformable_groups: int,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be NCHW, got {tuple(x.shape)}")
    if weight.dim() != 4:
        raise ValueError(f"weight must be OIHW, got {tuple(weight.shape)}")

    n, c, hin, win = x.shape
    oc, ic_per_group, kh, kw = weight.shape

    if oc != c:
        raise ValueError(f"depthwise requires out_channels==in_channels, got oc={oc}, c={c}")
    if ic_per_group != 1:
        raise ValueError(f"depthwise requires weight.shape[1]==1, got {ic_per_group}")
    if deformable_groups <= 0:
        raise ValueError(f"deformable_groups must be >0, got {deformable_groups}")
    if c % deformable_groups != 0:
        raise ValueError(f"channels must be divisible by deformable_groups, got c={c}, dg={deformable_groups}")

    h_out, w_out = int(offset.shape[2]), int(offset.shape[3])
    p = kh * kw

    expected_off_ch = 2 * p * deformable_groups
    if offset.shape[1] != expected_off_ch:
        raise ValueError(f"offset channels mismatch: got {offset.shape[1]}, expected {expected_off_ch}")

    if mask is not None:
        expected_mask_ch = p * deformable_groups
        if mask.shape[1] != expected_mask_ch:
            raise ValueError(f"mask channels mismatch: got {mask.shape[1]}, expected {expected_mask_ch}")

    sx, sy = stride[1], stride[0]
    px, py = padding[1], padding[0]
    dx, dy = dilation[1], dilation[0]

    device = x.device
    dtype_out = x.dtype

    x_f = x.float()
    offset_f = offset.float()
    mask_f = mask.float() if mask is not None else None

    oy = torch.arange(h_out, device=device, dtype=torch.float32)
    ox = torch.arange(w_out, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(oy, ox, indexing="ij")  # (Hout, Wout)
    base_y0 = yy * float(sy) - float(py)
    base_x0 = xx * float(sx) - float(px)

    denom_x = max(win - 1, 1)
    denom_y = max(hin - 1, 1)

    c_per_dg = c // deformable_groups
    out = torch.empty((n, c, h_out, w_out), device=device, dtype=torch.float32)

    w_dw = weight[:, 0, :, :].reshape(c, p).float()  # (C, P)

    output_parts = []

    for dg in range(deformable_groups):
        c0 = dg * c_per_dg
        c1 = c0 + c_per_dg

        x_g = x_f[:, c0:c1, :, :]  # (N, Cg, Hin, Win)
        w_g = w_dw[c0:c1, :]       # (Cg, P)

        off_g = offset_f[:, dg * (2 * p):(dg + 1) * (2 * p), :, :]  # (N, 2P, Hout, Wout)
        if mask_f is None:
            m_g = None
        else:
            m_g = mask_f[:, dg * p:(dg + 1) * p, :, :]  # (N, P, Hout, Wout)

        samples: list[torch.Tensor] = []
        idx = 0
        for ky in range(kh):
            for kx in range(kw):
                off_y = off_g[:, idx * 2 + 0, :, :]  # (N, Hout, Wout)
                off_x = off_g[:, idx * 2 + 1, :, :]  # (N, Hout, Wout)

                yy_s = base_y0 + float(ky * dy) + off_y
                xx_s = base_x0 + float(kx * dx) + off_x

                grid_x = _grid_sample_normalize(xx_s, denom_x)
                grid_y = _grid_sample_normalize(yy_s, denom_y)
                grid = torch.stack((grid_x, grid_y), dim=-1)  # (N, Hout, Wout, 2)

                v = F.grid_sample(
                    x_g,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )  # (N, Cg, Hout, Wout)

                if m_g is not None:
                    v = v * m_g[:, idx, :, :].unsqueeze(1)

                samples.append(v)
                idx += 1

        col = torch.stack(samples, dim=2)  # (N, Cg, P, Hout, Wout)
        y_g = torch.einsum("n c p h w, c p -> n c h w", col, w_g)
        # out[:, c0:c1, :, :] = y_g
        output_parts.append(y_g)

    out = torch.cat(output_parts, dim=1)

    if bias is not None:
        if bias.numel() != c:
            raise ValueError(f"bias numel mismatch: got {bias.numel()}, expected {c}")
        out = out + bias.reshape(1, c, 1, 1).float()

    return out.to(dtype_out)

def _is_rank0() -> bool:
    # Why: DDP/多卡時避免重複刷屏；單卡時永遠 True。
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v) == 0
            except Exception:
                return True
    return True


@dataclass
class DeformHubLogger:
    log_every: int = 5000
    max_topk: int = 3
    enable_rank0_only: bool = True

    _by_tag: Dict[str, "DeformStatsLogger"] = field(default_factory=dict)

    def _get(self, tag: str) -> "DeformStatsLogger":
        t = tag if tag else "unknown"
        lg = self._by_tag.get(t)
        if lg is None:
            lg = DeformStatsLogger(log_every=self.log_every, max_topk=self.max_topk)
            self._by_tag[t] = lg
        return lg

    def on_deform(self, backend: str, tag: str = "unknown") -> None:
        if self.enable_rank0_only and not _is_rank0():
            return
        self._get(tag).on_deform(backend)

    def on_fallback(self, reason: str, backend: str, tag: str = "unknown", exc: Optional[BaseException] = None) -> None:
        if self.enable_rank0_only and not _is_rank0():
            return
        # Why: 讓每個 tag 的 reason 更可讀（tag 前綴由 safe_deform_conv2d 統一）。
        self._get(tag).on_fallback(reason=reason, backend=backend, exc=exc)

    def flush_all(self) -> None:
        if self.enable_rank0_only and not _is_rank0():
            return
        for tag, lg in sorted(self._by_tag.items(), key=lambda kv: kv[0]):
            # Why: 每個 epoch 結束可手動 dump 一次「各 stage 使用率」。
            print(
                f"[DEFORM_TAG] tag={tag} calls={lg._calls} deform={lg._deform} fallback={lg._fallback} "
                f"top_reason={lg._reason.most_common(self.max_topk)} top_exc={lg._exc.most_common(self.max_topk)}",
                flush=True,
            )

def _log_deform(logger: Any, *, backend: str, tag: str) -> None:
    # Why: 相容舊 logger（只吃 backend）與新 hub logger（吃 backend+tag）。
    try:
        logger.on_deform(backend=backend, tag=tag)
    except TypeError:
        logger.on_deform(backend)


def _log_fallback(logger: Any, *, reason: str, backend: str, tag: str, exc: Optional[BaseException]) -> None:
    # Why: 相容舊 logger（reason/backend/exc）與新 hub logger（多 tag）。
    try:
        logger.on_fallback(reason=reason, backend=backend, tag=tag, exc=exc)
    except TypeError:
        logger.on_fallback(reason=reason, backend=backend, exc=exc)

def _build_tv_deform_kwargs(
    caps: _TorchvisionDeformCaps,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    mask: Optional[torch.Tensor],
    groups: int,
    deformable_groups: int,
) -> Tuple[dict[str, Any], Optional[str]]:
    """
    回傳: (kwargs, not_supported_reason)
    not_supported_reason != None 表示組合無法由當前 torchvision deform_conv2d 表達。
    """
    kwargs: dict[str, Any] = {"stride": stride, "padding": padding, "dilation": dilation}

    if mask is not None:
        if not caps.supports_mask:
            return kwargs, "mask_unsupported"
        kwargs["mask"] = mask

    if groups != 1:
        if not caps.supports_groups:
            return kwargs, "groups_unsupported"
        kwargs["groups"] = int(groups)

    if deformable_groups != 1:
        # torchvision 版本差異：有些叫 deformable_groups，有些叫 offset_groups
        if not caps.supports_deformable_groups:
            return kwargs, "deformable_groups_unsupported"
        if "deformable_groups" in inspect.signature(_TV_DEFORM_FN).parameters:  # type: ignore[arg-type]
            kwargs["deformable_groups"] = int(deformable_groups)
        else:
            kwargs["offset_groups"] = int(deformable_groups)

    return kwargs, None



def safe_deform_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    offset: torch.Tensor,
    mask: Optional[torch.Tensor],
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deformable_groups: int = 1,
    *,
    logger=None,
    tag: str = "deform",
) -> torch.Tensor:
    lg = logger if logger is not None else _DEFAULT_DEFORM_LOGGER
    t = str(tag) if tag else "unknown"

    try:
        s = _to_2tuple(stride)
        p = _to_2tuple(padding)
        d = _to_2tuple(dilation)
    except Exception as e:
        # 如果連參數都解不出來，ONNX 導出也沒救了，直接走 fallback
        if not torch.onnx.is_in_onnx_export():
             _log_fallback(lg, reason=f"{t}:bad_hyperparam", backend="fallback_conv2d", tag=t, exc=e)
        return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # ==============================================================================
    # [ONNX Export Block] 新增區塊：導出時強制走標準路徑
    # ==============================================================================
    if torch.onnx.is_in_onnx_export():
        # Why: export 時不可用「try/except 探測」避免 tracer 生成不一致圖；改用 caps 決策。
        caps = _get_torchvision_deform_caps()
        is_depthwise = (groups == x.shape[1] and weight.shape[0] == x.shape[1] and weight.shape[1] == 1)

        if caps.available and _TV_DEFORM_FN is not None:
            kwargs, why_not = _build_tv_deform_kwargs(
                caps,
                stride=s, padding=p, dilation=d,
                mask=mask,
                groups=groups,
                deformable_groups=deformable_groups,
            )
            if why_not is None:
                return _TV_DEFORM_FN(x, offset, weight, bias=bias, **kwargs)  # type: ignore[misc]

        # torchvision 不可表達：只允許 depthwise 走自家實作；其餘直接退回 conv2d（導出穩定優先）
        if is_depthwise:
            return _depthwise_deform_conv2d_pytorch(
                x=x,
                weight=weight,
                bias=bias,
                offset=offset,
                mask=mask,
                stride=s,
                padding=p,
                dilation=d,
                deformable_groups=deformable_groups,
            )
        return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)
    # ==============================================================================

    if x is None or weight is None or offset is None:
        _log_fallback(lg, reason=f"{t}:param_none", backend="fallback_conv2d", tag=t, exc=None)
        return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


    if x.dim() != 4 or weight.dim() != 4 or offset.dim() != 4:
        _log_fallback(lg, reason=f"{t}:bad_dim", backend="fallback_conv2d", tag=t, exc=None)
        return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)

    n, c, _, _ = x.shape
    oc, _, kh, kw = weight.shape
    if c <= 0 or oc <= 0 or kh <= 0 or kw <= 0:
        _log_fallback(lg, reason=f"{t}:bad_shape", backend="fallback_conv2d", tag=t, exc=None)
        return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)

    caps = _get_torchvision_deform_caps()

    # 優先走 torchvision：只要它能表達 (mask/groups/dg) 組合
    if caps.available and _TV_DEFORM_FN is not None:
        kwargs, why_not = _build_tv_deform_kwargs(
            caps,
            stride=s, padding=p, dilation=d,
            mask=mask,
            groups=groups,
            deformable_groups=deformable_groups,
        )
        if why_not is None:
            try:
                y = _TV_DEFORM_FN(x, offset, weight, bias=bias, **kwargs)
                _log_deform(lg, backend="torchvision", tag=t)
                return y
            except Exception as e:
                _log_fallback(lg, reason=f"{t}:torchvision_exc", backend="fallback_conv2d", tag=t, exc=e)
                return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)
        else:
            _log_fallback(lg, reason=f"{t}:tv_{why_not}", backend="fallback_conv2d", tag=t, exc=None)
 

    is_depthwise = (groups == c and oc == c and weight.shape[1] == 1)
    if is_depthwise:
        try:
            y = _depthwise_deform_conv2d_pytorch(
                x=x,
                weight=weight,
                bias=bias,
                offset=offset,
                mask=mask,
                stride=s,
                padding=p,
                dilation=d,
                deformable_groups=deformable_groups,
            )
            _log_deform(lg, backend="pytorch_depthwise", tag=t)
            return y
        except Exception as e:
            _log_fallback(lg, reason=f"{t}:depthwise_deform_exc", backend="fallback_conv2d", tag=t, exc=e)
            return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)

    # 非 depthwise 且 torchvision 無法表達的 groups 情況：明確 fallback（避免 silent baseline）
    if groups != 1:
        return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)
 

    _log_fallback(lg, reason=f"{t}:torchvision_unavailable", backend="fallback_conv2d", tag=t, exc=None)
    return F.conv2d(x, weight, bias=bias, stride=s, padding=p, dilation=d, groups=groups)



# ============================================================
# Max-Min CBAM
# ============================================================

class MaxMinChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden_planes = max(in_planes // ratio, 4)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_planes, hidden_planes),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_planes, in_planes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        max_pool = x.reshape(b, c, -1).max(dim=2)[0].reshape(b, c, 1, 1)
        min_pool = x.reshape(b, c, -1).min(dim=2)[0].reshape(b, c, 1, 1)
        out = self.mlp(max_pool) + self.mlp(min_pool)
        return self.sigmoid(out).reshape(b, c, 1, 1)


class MaxMinSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([max_out, min_out], dim=1)))


class MaxMinCBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = MaxMinChannelAttention(in_planes, ratio)
        self.sa = MaxMinSpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


# ============================================================
# TPG (Sobel) + Gating
# ============================================================

class OptimizedTPG(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, is_depthwise: bool = False):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        reduced_channels = max(1, in_channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )

        self.fusion: Optional[nn.Module]
        if is_depthwise:
            self.fusion = None
        else:
            self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        kx = self.sobel_x.expand(c, 1, 3, 3)
        ky = self.sobel_y.expand(c, 1, 3, 3)
        edge_x = F.conv2d(x, kx, padding=1, groups=c)
        edge_y = F.conv2d(x, ky, padding=1, groups=c)
        mag = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)

        refined = mag * self.attention(mag)
        if self.fusion is None:
            return x + refined

        out = self.fusion(torch.cat([x, refined], dim=1))
        return self.act(self.bn(out))


class StabilizedTPG(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        reduced_channels = max(1, in_channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    @staticmethod
    def _per_channel_minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        kx = self.sobel_x.expand(c, 1, 3, 3)
        ky = self.sobel_y.expand(c, 1, 3, 3)
        edge_x = F.conv2d(x, kx, padding=1, groups=c)
        edge_y = F.conv2d(x, ky, padding=1, groups=c)
        mag = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
        mag = self._per_channel_minmax_norm(mag)
        return x + self.alpha * (mag * self.attention(mag))


# ============================================================
# Topological Offset Prior (Explainable)
# ============================================================

class TopologicalOffsetPrior(nn.Module):
    """
    offset channel order: interleaved (offset_y, offset_x)
    output shape: [B, 2 * deformable_groups * k*k, Hout, Wout]
    """
    def __init__(self, k: int, deformable_groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.k = int(k)
        self.dg = int(deformable_groups)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _norm01_by_max(x: torch.Tensor, eps: float) -> torch.Tensor:
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return x / (x_max + eps)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError("TopologicalOffsetPrior: feat 必須為 NCHW")

        b, c, _, _ = feat.shape

        # [FIX] Why: Ultralytics/AMP 可能把 buffer 轉成 fp16/bf16，這裡強制 conv2d 的 input/weight 同 dtype
        f32 = feat.to(dtype=torch.float32)

        sobel_x = self.sobel_x.to(dtype=f32.dtype, device=f32.device)
        sobel_y = self.sobel_y.to(dtype=f32.dtype, device=f32.device)
        kx = sobel_x.expand(c, 1, 3, 3).contiguous()
        ky = sobel_y.expand(c, 1, 3, 3).contiguous()

        gx = F.conv2d(f32, kx, padding=1, groups=c)
        gy = F.conv2d(f32, ky, padding=1, groups=c)

        gx_m = gx.mean(dim=1, keepdim=True)
        gy_m = gy.mean(dim=1, keepdim=True)

        mag_raw = torch.sqrt(gx_m.square() + gy_m.square() + self.eps)
        mag = self._norm01_by_max(mag_raw, self.eps)

        dy = (gy_m / (mag_raw + self.eps)) * mag
        dx = (gx_m / (mag_raw + self.eps)) * mag

        if (dy.shape[-2], dy.shape[-1]) != out_hw:
            dy = F.interpolate(dy, size=out_hw, mode="bilinear", align_corners=False)
            dx = F.interpolate(dx, size=out_hw, mode="bilinear", align_corners=False)

        rep = self.dg * self.k * self.k
        dy_rep = dy.repeat(1, rep, 1, 1)
        dx_rep = dx.repeat(1, rep, 1, 1)

        prior = torch.stack([dy_rep, dx_rep], dim=2).reshape(b, 2 * rep, out_hw[0], out_hw[1])
        return prior.to(dtype=feat.dtype)


# ============================================================
# ConformableConv2d with Ablation Modes + Industrial Fallback
# ============================================================

class ConformableConv2d(nn.Module):
    """
    mode:
      - "baseline"       : standard conv2d (no deform)
      - "prior_only"     : offset = clamp(beta * prior)
      - "residual_only"  : offset = clamp(gamma * delta)
      - "prior_residual" : offset = clamp(beta * prior + gamma * delta)
    """
    _VALID_MODES = ("baseline", "prior_only", "residual_only", "prior_residual")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        g: int = 1,
        d: int = 1,
        deformable_groups: int = 1,
        mode: str = "prior_residual",
        use_mask: bool = True,
        offset_clamp: Optional[float] = None,
        mask_init_bias: float = 2.0,
        prior_scale: float = 0.35,
        deform_enabled: bool = True,
        force_fallback: bool = False,
        allow_fallback: bool = True,
         tag: str = "cira", 
         deform_logger=None,
    ):
        super().__init__()

        self._tag = tag
        self._deform_logger = deform_logger  # 可傳 DeformHubLogger


        if mode not in self._VALID_MODES:
            raise ValueError(f"ConformableConv2d: mode 必須為 {self._VALID_MODES} 之一")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels 必須為正整數")
        if g <= 0 or (in_channels % g != 0) or (out_channels % g != 0):
            raise ValueError("groups 不合法：需整除 in_channels 且 out_channels")
        if k <= 0 or k % 2 == 0:
            raise ValueError("k 建議為正奇數（3/5/7）")

        self.k = int(k)
        self.s = int(s)
        self.p = int(p)
        self.g = int(g)
        self.d = int(d)
        self.dg = int(deformable_groups)

        # --- compat aliases (Ultralytics / wrappers 可能讀取這些名稱) ---
        self.stride = self.s
        self.padding = self.p
        self.dilation = self.d
        self.groups = self.g
        self.deformable_groups = self.dg

        # --- optional logger/tag (不存在也不該讓 forward 爆掉) ---
        self._tag = getattr(self, "_tag", "cira")
        self._deform_logger = getattr(self, "_deform_logger", None)

        self.mode = mode
        self.use_mask = bool(use_mask)
        self.deform_enabled = bool(deform_enabled)
        self.force_fallback = bool(force_fallback)
        self.allow_fallback = bool(allow_fallback)

        # Why: depthwise 分支更容易被先驗擾動，使用門控 TPG 提升穩定
        is_depthwise = (self.g == in_channels) and (in_channels > 1) and (out_channels == in_channels)
        self.tpg = StabilizedTPG(in_channels) if is_depthwise else OptimizedTPG(in_channels, is_depthwise=False)

        self.prior = TopologicalOffsetPrior(k=self.k, deformable_groups=self.dg)
        self.prior_scale = float(prior_scale)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual gate

        self.num_offsets = 2 * self.dg * self.k * self.k
        self.num_masks = self.dg * self.k * self.k
        out_ch = self.num_offsets + (self.num_masks if self.use_mask else 0)

        self.adaptor = nn.Conv2d(
            in_channels,
            out_ch,
            kernel_size=3,
            padding=1,
            stride=self.s,
            bias=True,
        )

        nn.init.constant_(self.adaptor.weight, 0.0)
        nn.init.constant_(self.adaptor.bias, 0.0)

        # Why: 初期 mask 接近 1，讓 backbone 分佈更接近 baseline
        if self.use_mask:
            with torch.no_grad():
                self.adaptor.bias[self.num_offsets : self.num_offsets + self.num_masks].fill_(float(mask_init_bias))

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // self.g, self.k, self.k))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.offset_clamp = float(offset_clamp) if offset_clamp is not None else float(max(self.k, 2))

    @staticmethod
    def _clamp(x: torch.Tensor, v: float) -> torch.Tensor:
        return torch.clamp(x, min=-v, max=v)

    def set_mode(self, mode: str) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"set_mode: mode 必須為 {self._VALID_MODES} 之一")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "baseline" or (not self.deform_enabled):
            return F.conv2d(x, self.weight, self.bias, stride=self.s, padding=self.p, dilation=self.d, groups=self.g)

        feat = self.tpg(x)

        params = self.adaptor(feat)
        if self.use_mask:
            delta_offset, mask_logits = torch.split(params, [self.num_offsets, self.num_masks], dim=1)
            mask = torch.sigmoid(mask_logits)
        else:
            delta_offset = params
            mask = None

        out_hw = (delta_offset.shape[-2], delta_offset.shape[-1])
        offset_prior = self.prior(feat, out_hw=out_hw) * self.prior_scale

        delta = torch.tanh(delta_offset) * self.offset_clamp
        gamma = torch.sigmoid(self.gamma)

        if self.mode == "prior_only":
            offset = self._clamp(offset_prior, self.offset_clamp)
        elif self.mode == "residual_only":
            offset = self._clamp(gamma * delta, self.offset_clamp)
        else:  # "prior_residual"
            offset = self._clamp(offset_prior + gamma * delta, self.offset_clamp)

        return safe_deform_conv2d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            offset=offset,
            mask=mask,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deformable_groups=self.deformable_groups,
            logger=getattr(self, "_deform_logger", None),
            tag=getattr(self, "_tag", "cira"),
        )



class RobustConformableConv2d(ConformableConv2d):
    """
    向後相容：強制使用 StabilizedTPG + prior_residual 預設
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        g: int = 1,
        d: int = 1,
        deformable_groups: int = 1,
        mode: str = "prior_residual",
        use_mask: bool = True,
        offset_clamp: Optional[float] = None,
        mask_init_bias: float = 2.0,
        prior_scale: float = 0.35,
        deform_enabled: bool = True,
        force_fallback: bool = False,
        allow_fallback: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            k=k,
            s=s,
            p=p,
            g=g,
            d=d,
            deformable_groups=deformable_groups,
            mode=mode,
            use_mask=use_mask,
            offset_clamp=offset_clamp,
            mask_init_bias=mask_init_bias,
            prior_scale=prior_scale,
            deform_enabled=deform_enabled,
            force_fallback=force_fallback,
            allow_fallback=allow_fallback,
        )
        self.tpg = StabilizedTPG(in_channels)

# ============================================================
# ConformableBlock Units
# ============================================================

class ConformableBlock(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1, e: float = 0.5, shortcut: bool = True):
        super().__init__()
        c_hidden = int(c2 * e)

        self.cv1 = nn.Conv2d(c1, c_hidden, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        self.act1 = nn.SiLU(inplace=True)

        self.cv2 = ConformableConv2d(c_hidden, c_hidden, k=k, s=s, p=k // 2, g=g, d=1)
        self.bn2 = nn.BatchNorm2d(c_hidden)
        self.act2 = nn.SiLU(inplace=True)

        self.cv3 = nn.Conv2d(c_hidden, c2, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act3 = nn.SiLU(inplace=True)

        self.add = bool(shortcut and c1 == c2 and s == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.cv1(x)))
        y = self.act2(self.bn2(self.cv2(y)))
        y = self.act3(self.bn3(self.cv3(y)))
        return x + y if self.add else y


# ============================================================
# ShuffleNetV2 Units
# ============================================================

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("channel_shuffle: x 必須為 NCHW")
    if groups <= 0:
        raise ValueError("channel_shuffle: groups 必須為正整數")

    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError(f"channel_shuffle: channels({c}) 必須可被 groups({groups}) 整除")

    ch_per_g = c // groups
    x = x.view(b, groups, ch_per_g, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ConformableInvertedResidual(nn.Module):
    """
    YAML args（向後相容）：
      [oup, k, stride, use_attention]  # 舊
      [oup, k, stride, use_attention, dcn_mode, deform_enabled, force_fallback, use_mask, prior_scale]  # 新
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        k: int = 3,
        stride: int = 1,
        use_attention: bool = True,
        dcn_mode: str = "prior_residual",
        deform_enabled: bool = True,
        force_fallback: bool = False,
        use_mask: bool = True,
        prior_scale: float = 0.35,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ShuffleNetV2 stride 建議限制在 1 或 2")
        if (inp % 2 != 0) or (oup % 2 != 0):
            raise ValueError("ShuffleNetV2 通道需為偶數（chunk + shuffle）")

        self.stride = int(stride)
        branch_features = oup // 2
        if branch_features <= 0:
            raise ValueError("branch_features 不可為 0")

        self.use_split = bool(self.stride == 1 and inp == oup)

        # branch1
        if not self.use_split:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, k, self.stride, k // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.SiLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        # branch2
        input_channels = branch_features if self.use_split else inp

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),

            # 只在 stride=1/2 的 depthwise 位置用（保持 ShuffleNetV2 輕量特性）
            ConformableConv2d(
                in_channels=branch_features,
                out_channels=branch_features,
                k=k,
                s=self.stride,
                p=k // 2,
                g=branch_features,                 # depthwise
                d=1,
                deformable_groups=1,
                mode=str(dcn_mode),
                use_mask=bool(use_mask),
                offset_clamp=float(max(k, 2)),
                mask_init_bias=2.0,
                prior_scale=float(prior_scale),
                deform_enabled=bool(deform_enabled),
                force_fallback=bool(force_fallback),
                allow_fallback=True,
            ),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),
        )

        self.use_attention = bool(use_attention)
        if self.use_attention:
            self.attention = MaxMinCBAM(oup, ratio=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_split:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        if self.use_attention:
            out = self.attention(out)

        return channel_shuffle(out, 2)


# ===================================================================================================================

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: Tuple[int, ...] = (5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: Tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: List[int], k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: List[int]):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """
        Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """
        Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: List[int], c3: int, embed: int):
        """
        Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (List[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: List[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
