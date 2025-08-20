'''
===================================================
Tensor 版本強制設定
===================================================
'''
import os, sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras as K

# force any "import keras" to resolve to tf.keras
sys.modules["keras"] = K
sys.modules["keras.models"] = K.models
sys.modules["keras.layers"] = K.layers
sys.modules["keras.activations"] = K.activations
sys.modules["keras.initializers"] = K.initializers
sys.modules["keras.utils"] = K.utils
sys.modules["keras.losses"] = K.losses
sys.modules["keras.backend"] = K.backend

import config

'''
====================================================
偵測系統記憶體
====================================================
'''

def enable_gpu_mem_growth():
    """設定 GPU 記憶體為動態增長模式。"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ No GPU detected. Running on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        lg = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPUs, {len(lg)} Logical GPUs. Memory growth enabled.")
    except RuntimeError as e:
        print(f"❌ TF GPU config error: {e}")


'''
====================================================
偵測中斷訊號以儲存模型
====================================================
'''

def setup_mixed_precision():
    """如果 config.USE_AMP 為 True，則啟用混合精度訓練。"""
    if config.USE_AMP:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision (AMP) enabled.")
        except ImportError:
            print("⚠️ Could not import mixed_precision. Skipping AMP setup.")
    else:
        print("ℹ️ Mixed precision (AMP) is disabled.")