# qat_tf/qat_distill.py
import glob
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

import config


# ---------- 前處理（與 Ultralytics 部署一致） ----------
def letterbox(img, new_size=config.IMGSZ):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top  = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114,114,114))
    return img

def _decode_path(p):
    if isinstance(p, bytes):
        return p.decode('utf-8')
    return str(p)

def parse_img(path):

    p = _decode_path(path)
    bgr = cv2.imread(p)
    rgb = bgr[:, :, ::-1]
    img = letterbox(rgb, config.IMGSZ).astype(np.float32) / 255.0
    return img

def tf_parse(path):
    img = tf.numpy_function(parse_img, [path], Tout=tf.float32)
    img.set_shape([config.IMGSZ, config.IMGSZ, 3])
    return img

def build_dataset(img_glob, batch=config.BATCH, shuffle=True, repeat=True):
    files = sorted(glob.glob(img_glob))

    if len(files) == 0:
        raise FileNotFoundError(f"No images found for pattern: {img_glob}")

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle: ds = ds.shuffle(len(files))
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat: ds = ds.repeat()
    return ds, len(files)


def normalize_teacher_pred(y, expected_C):
    """
    將 teacher 輸出標準化為 [B, N, C]。
    - 若輸入已是 [B, N, C]（最後維度 == expected_C），直接回傳。
    - 若輸入是 [B, C, N]，嘗試 transpose -> [B, N, C] 並檢查最後維度是否為 expected_C。
    - 否則在 graph 中會觸發 assert（會中止），並給出有意義錯誤訊息。
    注意：此實作完全使用 TF ops，可在 @tf.function 中呼叫。
    """
    y = tf.convert_to_tensor(y)
    # 確保 rank==3（會在 graph 中觸發檢查）
    tf.debugging.assert_rank(y, 3, message="normalize_teacher_pred: input rank must be 3 (B,?,?)")

    # 動態 shape
    sh = tf.shape(y)
    dim1 = sh[1]
    dim2 = sh[2]
    expected = tf.cast(expected_C, dtype=dim2.dtype)

    # 判斷最後維度是否等於 expected_C
    cond_last_is_expected = tf.equal(dim2, expected)

    def _return_as_is():
        return y

    def _try_transpose_and_check():
        y_t = tf.transpose(y, perm=[0, 2, 1])  # [B, N, C] <- [B, C, N]
        # 檢查 transpose 後最後維度是 expected_C，若不是會觸發 assert
        tf.debugging.assert_equal(tf.shape(y_t)[2], expected,
                                  message=("normalize_teacher_pred: after transpose, "
                                           "last dim != expected_C"))
        return y_t

    result = tf.cond(cond_last_is_expected, _return_as_is, _try_transpose_and_check)
    return result


# ---------- 載入 Keras 模型（優先 .keras / Keras SavedModel） ----------

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


# ---------- 代表集 generator（給 TFLite） ----------
def rep_data_gen():
    paths = sorted(glob.glob(config.REP_DIR))     # ← 這行改了

    num_picture = 0
    for p in paths:
        img = parse_img(p)                        # float32 [H,W,3] /255
        img = np.expand_dims(img, 0).astype(np.float32)
        yield [img]
        num_picture += 1

    print(f"\n\nRead the data = {num_picture}\n")