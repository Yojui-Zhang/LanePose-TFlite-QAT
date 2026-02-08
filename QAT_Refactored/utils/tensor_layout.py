# Source File: QAT_Refactored/utils/tensor_layout.py

from __future__ import annotations

from typing import Final, Optional

import numpy as np
import tensorflow as tf

_ERR_RANK: Final[str] = "Expected rank-3 tensor (B,*,*)."
_ERR_AMBIG: Final[str] = "Ambiguous layout: both dim1 and dim2 match total_channels."
_ERR_MISMATCH: Final[str] = "Invalid layout: neither dim1 nor dim2 matches total_channels."


def ensure_bnc_tf(y: tf.Tensor, total_channels: int) -> tf.Tensor:
    """
    Why: 讓所有 loss/assigner 一律以 (B, N, C) 解碼，避免散落的 transpose 猜測邏輯。
    Accepts: (B, C, N) or (B, N, C)
    Returns: (B, N, C)
    """
    if total_channels <= 0:
        raise ValueError(f"total_channels must be > 0, got {total_channels}")

    y = tf.convert_to_tensor(y)
    tf.debugging.assert_rank(y, 3, message=_ERR_RANK)

    s = tf.shape(y)
    c = tf.cast(total_channels, tf.int32)

    dim1_is_c = tf.equal(s[1], c)  # (B, C, N)
    dim2_is_c = tf.equal(s[2], c)  # (B, N, C)

    tf.debugging.assert_equal(
        tf.logical_or(dim1_is_c, dim2_is_c),
        True,
        message=_ERR_MISMATCH,
    )
    tf.debugging.assert_equal(
        tf.logical_and(dim1_is_c, dim2_is_c),
        False,
        message=_ERR_AMBIG,
    )

    return tf.cond(
        dim1_is_c,
        lambda: tf.transpose(y, [0, 2, 1]),
        lambda: y,
    )


def ensure_bcn_tf(
    y: tf.Tensor,
    total_channels: int | None = None,
    *,
    total_c: int | None = None,
    name: str = "pred",
) -> tf.Tensor:
    """
    Why: 匯出/部署路徑（含 C++）固定使用 (B, C, N)，需要一致的 fail-fast 驗證與轉回。
    Accepts: (B, C, N) or (B, N, C)
    Returns: (B, C, N)
    """
    tc = total_channels if total_channels is not None else total_c
    if tc is None:
        raise ValueError(f"{name}: total_channels/total_c must be provided")
    if tc <= 0:
        raise ValueError(f"{name}: total_channels must be > 0, got {tc}")

    y = tf.convert_to_tensor(y)
    tf.debugging.assert_rank(y, 3, message=_ERR_RANK)

    s = tf.shape(y)
    c = tf.cast(tc, tf.int32)

    dim1_is_c = tf.equal(s[1], c)  # (B, C, N)
    dim2_is_c = tf.equal(s[2], c)  # (B, N, C)

    tf.debugging.assert_equal(
        tf.logical_or(dim1_is_c, dim2_is_c),
        True,
        message=_ERR_MISMATCH,
    )
    tf.debugging.assert_equal(
        tf.logical_and(dim1_is_c, dim2_is_c),
        False,
        message=_ERR_AMBIG,
    )

    return tf.cond(
        dim2_is_c,
        lambda: tf.transpose(y, [0, 2, 1]),
        lambda: y,
    )


def assert_layout_tf(y: tf.Tensor, total_channels: int, name: str = "tensor") -> tf.Tensor:
    """
    Why: 在 graph 邊界 fail-fast，避免下游以錯誤 layout 進行切片卻不自知。
    Accepts: (B,C,N) or (B,N,C) where one of dim1/dim2 equals total_channels
    Returns: 原 tensor（不 transpose）
    """
    if total_channels <= 0:
        raise ValueError(f"total_channels must be > 0, got {total_channels}")

    y = tf.convert_to_tensor(y)
    tf.debugging.assert_rank(y, 3, message=f"{name}: {_ERR_RANK}")

    s = tf.shape(y)
    c = tf.cast(total_channels, tf.int32)

    dim1_is_c = tf.equal(s[1], c)  # (B, C, N)
    dim2_is_c = tf.equal(s[2], c)  # (B, N, C)

    tf.debugging.assert_equal(
        tf.logical_or(dim1_is_c, dim2_is_c),
        tf.constant(True),
        message=f"{name}: {_ERR_MISMATCH}",
    )
    tf.debugging.assert_equal(
        tf.logical_and(dim1_is_c, dim2_is_c),
        tf.constant(False),
        message=f"{name}: {_ERR_AMBIG}",
    )

    return y


class TensorLayoutError(ValueError):
    pass

def ensure_bnc_np(x: np.ndarray, *, total_c: Optional[int] = None, num_cls: Optional[int] = None) -> np.ndarray:
    """
    Ensure prediction layout is (B, N, C).
    Accepts (B, C, N) or (B, N, C).

    Why: visualization/decoder slices channels on the last axis (C).
    """
    if x.ndim != 3:
        raise TensorLayoutError(f"preds must be rank-3, got shape={x.shape}")

    _, d1, d2 = x.shape

    if total_c is not None:
        if d2 == total_c:  # already (B, N, C)
            return x
        if d1 == total_c:  # (B, C, N) -> (B, N, C)
            return np.transpose(x, (0, 2, 1))
        raise TensorLayoutError(f"preds shape mismatch: shape={x.shape}, total_c={total_c}")

    # Fallback (best-effort) when total_c is unavailable.
    min_c = 4 + int(num_cls or 0)
    if d1 >= min_c and d2 > d1:  # likely (B, C, N) because N > C
        return np.transpose(x, (0, 2, 1))
    return x


