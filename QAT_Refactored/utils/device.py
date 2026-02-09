import tensorflow as tf
import os

def enable_gpu_mem_growth():
    """開啟 GPU 記憶體按需增長 (避免一次佔用全部顯存)"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Device] GPU Memory Growth Enabled: {len(gpus)} GPUs found.")
        except RuntimeError as e:
            print(f"[Device] GPU Error: {e}")

def setup_mixed_precision(use_amp=False):
    """設定混合精度 (Mixed Precision)"""
    if use_amp:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"[Device] Mixed Precision Enabled: {policy.compute_dtype}")
    else:
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"[Device] Using Float32 Precision: {policy.compute_dtype}")
