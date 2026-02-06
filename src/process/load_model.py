import tensorflow as tf

# 步驟 1: 將 SMWrapper 類別的定義移到檔案的頂層
class SMWrapper(tf.keras.Model):
    def __init__(self, saved_model, input_name, output_key, single_out_spec):
        super().__init__()
        # 步驟 2: 保存整個 saved_model 物件，而不只是 function signature
        self.saved_model = saved_model 
        self.fn = self.saved_model.signatures["serving_default"]
        self.input_name = input_name
        self.output_key = output_key
        self.single_out_spec = single_out_spec

    @tf.function
    def single_fn(self, img):
        img = tf.expand_dims(img, 0)
        # 使用 self.fn 進行呼叫
        out = self.fn(**{self.input_name: img})
        return out[self.output_key][0]

    @tf.function
    def call(self, x):
        mapped = tf.map_fn(self.single_fn, x, fn_output_signature=self.single_out_spec)
        return mapped

'''
載入 Teacher Keras 模型（優先 .keras / Keras SavedModel）
'''
def try_load_keras_model(export_dir):
    # 1) 優先用 tf.keras loader
    try:
        m = tf.keras.models.load_model(export_dir)
        print("\n[INFO] Loaded with tf.keras.models.load_model")
        return m, True
    except Exception as e:
        print("\n[INFO] tf.keras.models.load_model failed, falling back to saved_model signature:", e)

    # 2) 用 saved_model.signatures["serving_default"] 包成 Keras-like wrapper
    saved = tf.saved_model.load(export_dir) # 'saved' 物件持有變數
    if "serving_default" not in saved.signatures:
        raise RuntimeError("SavedModel has no 'serving_default' signature, can't wrap automatically.")
    
    fn = saved.signatures["serving_default"]

    # ... (這裡所有解析 input/output 名稱和 shape 的程式碼都維持不變) ...
    input_keys = list(fn.structured_input_signature[1].keys())
    if len(input_keys) != 1:
        raise RuntimeError("SavedModel serving_default expects multiple inputs; wrapper only supports single-image input signatures.")
    input_name = input_keys[0]

    out_keys = list(fn.structured_outputs.keys())
    if len(out_keys) == 0:
        raise RuntimeError("SavedModel serving_default has no outputs.")
    output_key = out_keys[0]

    out_spec_proto = fn.structured_outputs[output_key]
    try:
        out_shape_list = out_spec_proto.shape.as_list()
    except Exception:
        out_shape_list = list(out_spec_proto.shape)

    out_dtype = out_spec_proto.dtype
    single_out_shape = tuple(out_shape_list[1:])
    single_out_spec = tf.TensorSpec(shape=single_out_shape, dtype=out_dtype)

    # 現在使用頂層定義的 SMWrapper 類別，並傳入整個 saved 物件
    wrapped = SMWrapper(saved, input_name, output_key, single_out_spec)
    print(f"\n[INFO] Wrapped SavedModel signature into Keras-like model. input_name={input_name}, output_key={output_key}, single_out_shape={single_out_shape}")
    return wrapped, False

