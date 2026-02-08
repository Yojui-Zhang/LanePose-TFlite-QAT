import os
import sys
import logging

def setup_environment() -> None:
    """
    配置全域環境變數。
    CRITICAL: 必須在導入 tensorflow 或 tf_keras 之前呼叫。
    """
    # 1. 強制設定 Legacy Keras 標誌 (最優先執行)
    # 這必須在任何 TF import 發生前設定才有效
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
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

    # 若環境中存在 standalone keras，確認不是 3.x
    try:
        import keras as standalone_keras
        kv = getattr(standalone_keras, "__version__", "unknown")
        logging.info(f"[Env] keras (standalone) Version: {kv}")
        if str(kv).startswith("3"):
            logging.critical("[Env] CRITICAL: keras==3.x detected but this project requires Keras 2.x (TF 2.15).")
            logging.critical("Action: uninstall keras 3 or use a clean TF2.15/Keras2 environment.")
            sys.exit(1)
    except Exception:
        # 沒有 standalone keras 也 OK
        pass
