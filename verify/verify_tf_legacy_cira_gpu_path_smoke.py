from __future__ import annotations

try:
    from ._bootstrap import setup_runtime_env
except ImportError:
    from _bootstrap import setup_runtime_env

setup_runtime_env()

import tensorflow as tf

from QAT_Refactored.models.layers import DeformableDepthwiseConv2D


def _run_mode(mode: str, use_mask: bool) -> None:
    layer = DeformableDepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        mode=mode,
        use_mask=use_mask,
        prior_scale=0.30,
        deform_enabled=True,
        force_fallback=False,
        name=f"verify_{mode}_{'mask' if use_mask else 'nomask'}",
    )
    x = tf.random.uniform((1, 80, 80, 128), dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = layer(x, training=True)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, layer.trainable_variables)

    if not layer.trainable_variables:
        raise AssertionError(f"{mode}: expected trainable variables in deform layer.")
    if any(g is None for g in grads):
        raise AssertionError(f"{mode}: got None gradient under forced GPU execution.")
    if y.shape.rank != 4:
        raise AssertionError(f"{mode}: unexpected output rank {y.shape}.")


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("verify_tf_legacy_cira_gpu_path_smoke: SKIP (no GPU visible)")
        return

    tf.config.set_soft_device_placement(False)

    try:
        with tf.device("/GPU:0"):
            _run_mode("prior_only", use_mask=True)
            _run_mode("residual_only", use_mask=True)
            _run_mode("prior_residual", use_mask=True)
    except Exception as exc:
        raise AssertionError(
            "Deform path cannot stay on forced GPU execution; "
            "this indicates a potential CPU fallback or missing GPU kernel."
        ) from exc

    print("verify_tf_legacy_cira_gpu_path_smoke: OK")


if __name__ == "__main__":
    main()
