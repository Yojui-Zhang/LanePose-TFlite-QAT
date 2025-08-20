
import tensorflow as tf

'''
載入 Teacher Keras 模型（優先 .keras / Keras SavedModel）
'''

def try_load_keras_model(export_dir):
    # 1) 優先用 tf.keras loader（若 SavedModel 是 Keras 格式就會成功）
    try:
        m = tf.keras.models.load_model(export_dir)
        print("[INFO] Loaded with tf.keras.models.load_model")
        return m, True
    except Exception as e:
        print("[INFO] tf.keras.models.load_model failed, falling back to saved_model signature:", e)

    # 2) 用 saved_model.signatures["serving_default"] 包成 Keras-like wrapper
    saved = tf.saved_model.load(export_dir)
    if "serving_default" not in saved.signatures:
        raise RuntimeError("SavedModel has no 'serving_default' signature, can't wrap automatically.")
    fn = saved.signatures["serving_default"]

    # 取得 signature 的輸入/輸出名稱
    input_keys = list(fn.structured_input_signature[1].keys())
    if len(input_keys) != 1:
        raise RuntimeError("SavedModel serving_default expects multiple inputs; wrapper only supports single-image input signatures.")
    input_name = input_keys[0]

    out_keys = list(fn.structured_outputs.keys())
    if len(out_keys) == 0:
        raise RuntimeError("SavedModel serving_default has no outputs.")
    output_key = out_keys[0]

    # 由 signature 的輸出推斷輸出 single-sample shape/dtype
    out_spec_proto = fn.structured_outputs[output_key]
    try:
        out_shape_list = out_spec_proto.shape.as_list()  # e.g. [1, 56, 8400]
    except Exception:
        out_shape_list = list(out_spec_proto.shape)

    out_dtype = out_spec_proto.dtype

    # single-sample output spec: 去掉 batch 維 (index 0)
    single_out_shape = tuple(out_shape_list[1:])  # e.g. (56, 8400)
    single_out_spec = tf.TensorSpec(shape=single_out_shape, dtype=out_dtype)

    class SMWrapper(tf.keras.Model):
        def __init__(self, concrete_fn, input_name, output_key, single_out_spec):
            super().__init__()
            self.fn = concrete_fn
            self.input_name = input_name
            self.output_key = output_key
            self.single_out_spec = single_out_spec

        @tf.function
        def call(self, x):
            # x: [B,H,W,3] float32
            def single_fn(img):
                img = tf.expand_dims(img, 0)  # -> [1,H,W,3]
                out = self.fn(**{self.input_name: img})
                # out[self.output_key] shape [1, ...]
                return out[self.output_key][0]  # squeeze batch dim -> shape single_out_shape

            # 使用 tf.map_fn 在 batch 維度上逐張呼叫 single_fn
            mapped = tf.map_fn(single_fn, x, fn_output_signature=self.single_out_spec)
            # mapped shape: [B, ...] ，直接回傳
            return mapped

    wrapped = SMWrapper(fn, input_name, output_key, single_out_spec)
    print(f"[INFO] Wrapped SavedModel signature into Keras-like model. input_name={input_name}, output_key={output_key}, single_out_shape={single_out_shape}")
    return wrapped, False
