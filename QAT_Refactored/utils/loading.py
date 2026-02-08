# File: QAT_Refactored/utils/loading.py

import tensorflow as tf
import os

class SMWrapper(tf.keras.Model):
    """
    SavedModel 的 Keras 包裝器。
    用於將 TF SavedModel 的 signatures['serving_default'] 轉換為可訓練/可推論的 Keras Layer 行為。
    """
    def __init__(self, saved_model, input_name, output_key, single_out_spec):
        super().__init__()
        # 保存整個 saved_model 物件以維持資源追蹤
        self.saved_model = saved_model 
        self.fn = self.saved_model.signatures["serving_default"]
        self.input_name = input_name
        self.output_key = output_key
        self.single_out_spec = single_out_spec

    @tf.function
    def single_fn(self, img):
        # 擴展維度以符合 signature 預期 (通常是 batch dimension)
        img = tf.expand_dims(img, 0)
        out = self.fn(**{self.input_name: img})
        return out[self.output_key][0]

    @tf.function
    def call(self, x):
        # 使用 map_fn 處理 batch 輸入，因為原始 signature 可能只接受固定 shape
        mapped = tf.map_fn(self.single_fn, x, fn_output_signature=self.single_out_spec)
        return mapped

def try_load_keras_model(model_path):
    """
    嘗試載入模型，自動識別 .keras, .h5 或 SavedModel 目錄。
    Returns:
        model: Keras Model or SMWrapper
        is_keras_native: bool
    """
    model_path = str(model_path)
    
    # 1. 優先嘗試標準 Keras 載入
    try:
        m = tf.keras.models.load_model(model_path)
        print(f"[Loader] Loaded with tf.keras.models.load_model: {model_path}")
        return m, True
    except Exception as e:
        print(f"[Loader] Standard load failed, falling back to SavedModel wrapper. Error: {e}")

    # 2. 降級處理：載入 SavedModel 並包裝
    try:
        saved = tf.saved_model.load(model_path)
    except Exception as e:
        raise RuntimeError(f"[Loader] Failed to load SavedModel from {model_path}: {e}")

    if "serving_default" not in saved.signatures:
        raise RuntimeError("[Loader] SavedModel has no 'serving_default' signature.")
    
    fn = saved.signatures["serving_default"]

    # 解析 Input
    if hasattr(fn, 'structured_input_signature'):
        input_keys = list(fn.structured_input_signature[1].keys())
    else:
        # Fallback if structured_input_signature is missing (older TF)
        input_keys = [t.name.split(':')[0] for t in fn.inputs]
        
    if len(input_keys) != 1:
        # 若有多個輸入，通常需要更複雜的處理，這裡假設單一輸入 (image)
        print(f"[Loader] Warning: Multiple inputs detected: {input_keys}. Using the first one.")
    
    input_name = input_keys[0]

    # 解析 Output
    out_keys = list(fn.structured_outputs.keys())
    if len(out_keys) == 0:
        raise RuntimeError("[Loader] SavedModel has no outputs.")
    output_key = out_keys[0]

    # 解析 Output Spec
    out_spec_proto = fn.structured_outputs[output_key]
    try:
        out_shape_list = out_spec_proto.shape.as_list()
    except Exception:
        out_shape_list = list(out_spec_proto.shape)

    out_dtype = out_spec_proto.dtype
    # 移除 batch dim (通常是 None 或 1)
    single_out_shape = tuple(out_shape_list[1:])
    single_out_spec = tf.TensorSpec(shape=single_out_shape, dtype=out_dtype)

    wrapped = SMWrapper(saved, input_name, output_key, single_out_spec)
    print(f"[Loader] Wrapped SavedModel signature. Input: {input_name}, Output: {output_key}")
    return wrapped, False