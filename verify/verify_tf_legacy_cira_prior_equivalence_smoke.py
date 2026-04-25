from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf

from QAT_Refactored.models.layers import DeformableDepthwiseConv2D


def _legacy_prior(
    x: tf.Tensor,
    out_h: tf.Tensor,
    out_w: tf.Tensor,
    *,
    kernel_size: int,
    prior_scale: float,
    eps: float,
) -> tf.Tensor:
    sobel = tf.image.sobel_edges(x)  # [B, H, W, C, 2]
    dy = tf.reduce_mean(sobel[..., 0], axis=-1, keepdims=True)
    dx = tf.reduce_mean(sobel[..., 1], axis=-1, keepdims=True)

    mag_raw = tf.sqrt(tf.square(dy) + tf.square(dx) + eps)
    mag_max = tf.reduce_max(mag_raw, axis=[1, 2, 3], keepdims=True)
    mag = mag_raw / (mag_max + eps)

    dy = (dy / (mag_raw + eps)) * mag
    dx = (dx / (mag_raw + eps)) * mag

    target_hw = tf.stack([out_h, out_w])
    dy = tf.image.resize(dy, target_hw, method="bilinear")
    dx = tf.image.resize(dx, target_hw, method="bilinear")

    prior = tf.stack([dy, dx], axis=-1)
    prior = tf.tile(prior, [1, 1, 1, kernel_size * kernel_size, 1])
    return prior * float(prior_scale)


def main() -> None:
    tf.random.set_seed(7)

    layer = DeformableDepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        mode="prior_residual",
        use_mask=True,
        prior_scale=0.30,
        deform_enabled=True,
        force_fallback=False,
        name="verify_prior_equivalence",
    )

    x = tf.random.uniform((2, 64, 64, 96), dtype=tf.float32)
    _ = layer(x, training=False)  # build layer internals

    out_h = tf.shape(x)[1]
    out_w = tf.shape(x)[2]

    fast_prior = layer._build_prior_offsets(x, out_h, out_w)
    legacy_prior = _legacy_prior(
        x,
        out_h,
        out_w,
        kernel_size=layer.kernel_size,
        prior_scale=layer.prior_scale,
        eps=layer.eps,
    )
    fast_tiled = tf.tile(fast_prior, [1, 1, 1, layer.kernel_size * layer.kernel_size, 1])

    max_abs = float(tf.reduce_max(tf.abs(fast_tiled - legacy_prior)).numpy())
    mean_abs = float(tf.reduce_mean(tf.abs(fast_tiled - legacy_prior)).numpy())

    if max_abs > 1e-5:
        raise AssertionError(
            f"Fast prior deviates too much from legacy path: max_abs={max_abs}, mean_abs={mean_abs}"
        )

    print(
        "verify_tf_legacy_cira_prior_equivalence_smoke: OK "
        f"(max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e})"
    )


if __name__ == "__main__":
    main()
