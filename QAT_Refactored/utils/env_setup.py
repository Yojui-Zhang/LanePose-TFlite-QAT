import os
import sys
import logging
import warnings

def _patch_protobuf_message_factory() -> None:
    """
    TensorFlow 2.15 still calls MessageFactory.GetPrototype(), but protobuf 6.x
    removed that method. Add a small compatibility shim before importing TF.
    """
    try:
        from google.protobuf import message_factory
    except Exception:
        return

    if hasattr(message_factory.MessageFactory, "GetPrototype"):
        return

    def _get_prototype(self, descriptor):  # type: ignore[unused-argument]
        return message_factory.GetMessageClass(descriptor)

    message_factory.MessageFactory.GetPrototype = _get_prototype  # type: ignore[attr-defined]


def setup_environment() -> None:
    """
    配置全域環境變數。
    CRITICAL: 必須在導入 tensorflow 或 tf_keras 之前呼叫。
    """
    # 1. 強制設定 Legacy Keras 標誌 (最優先執行)
    # 這必須在任何 TF import 發生前設定才有效
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["KERAS_FORCE_REBATCH"] = "1"
    # TFMOT + tf_keras may emit false-positive initializer compatibility warnings
    # when objects come from keras.src. This warning is non-fatal and noisy.
    warnings.filterwarnings(
        "ignore",
        message=r"The `keras\.initializers\.serialize\(\)` API should only be used.*",
        module=r"tf_keras\.src\.initializers(\.__init__)?",
    )
    _patch_protobuf_message_factory()
    
    # 2. 設定基礎 Logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def check_tf_version() -> None:
    """
    導入並檢查 TensorFlow 與 Keras 版本一致性 (TF 2.15 / Keras 2.x)。
    """
    import tensorflow as tf
    from tensorflow import keras as K
    import sys
    import logging

    logging.info(f"[Env] TensorFlow Version: {tf.__version__}")
    logging.info(f"[Env] tf.keras Version: {getattr(K, '__version__', 'unknown')}")

    # Keras 3.x compatibility with TF 2.16
    # When TF_USE_LEGACY_KERAS=1, TF uses tf.keras which is Keras 2.x compatible
    # standalone keras 3.x is OK as long as we're using tf.keras internally
    logging.info("[Env] Keras 3.x compatibility mode enabled via TF_USE_LEGACY_KERAS=1")
