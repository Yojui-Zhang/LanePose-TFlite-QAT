# qat_tf/qat_distill.py
import os, glob
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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

def parse_img(path):
    bgr = cv2.imread(path.decode())
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

# ---------- 蒸餾損失（簡化版） ----------
def distill_loss(y_t, y_s, num_cls=config.NUM_CLS):
    # 預期輸出 shape: [B, N, 5+num_cls] = [xywh, obj, cls...]
    box_t, obj_t, cls_t = y_t[..., :4], y_t[..., 4:5], y_t[..., 5:5+num_cls]
    box_s, obj_s, cls_s = y_s[..., :4], y_s[..., 4:5], y_s[..., 5:5+num_cls]

    box_l = tf.reduce_mean(tf.abs(box_t - box_s))

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    obj_l = bce(obj_t, obj_s)

    # 溫度可調，例如 2.0
    p_t = tf.nn.softmax(cls_t)
    p_s = tf.nn.softmax(cls_s)
    kl = tf.keras.losses.KLDivergence()
    cls_l = kl(p_t, p_s)

    return box_l + obj_l + cls_l

# =============================================

def _split_outputs(y, num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS):
    """把 [B,N,5+num_cls+num_kpt*kpt_vals] 切成 box/obj/cls/kpt(xy,s)"""
    box = y[..., :4]                  # [B,N,4]
    obj = y[..., 4:5]                 # [B,N,1]
    cls = y[..., 5:5+num_cls]         # [B,N,num_cls]
    kpt = y[..., 5+num_cls : 5+num_cls + num_kpt*kpt_vals]  # [B,N,num_kpt*kpt_vals]
    kpt = tf.reshape(kpt, [-1, tf.shape(y)[1], num_kpt, kpt_vals])  # [B,N,K,3]
    kxy = kpt[..., :2]                # [B,N,K,2]
    ks  = kpt[..., 2:3]               # [B,N,K,1]  (關鍵點 score/logit)
    return box, obj, cls, kxy, ks

def kl_div_weighted(p_t, p_s, weight):
    # 手寫 KL，支援 sample-wise 權重
    eps = 1e-8
    p_t = tf.clip_by_value(p_t, eps, 1.0)
    p_s = tf.clip_by_value(p_s, eps, 1.0)
    kl = tf.reduce_sum(p_t * (tf.math.log(p_t) - tf.math.log(p_s)), axis=-1)  # [B,N]
    # 將 [B,N,1] 的 weight 壓到 [B,N]
    w = tf.squeeze(weight, axis=-1)
    return tf.reduce_mean(w * kl)

def distill_loss_pose(y_teacher, y_student,
                      num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS):
    # 拆 teacher / student 的輸出
    box_t, obj_t, cls_t, kxy_t, ks_t = _split_outputs(y_teacher, num_cls, num_kpt, kpt_vals)
    box_s, obj_s, cls_s, kxy_s, ks_s = _split_outputs(y_student, num_cls, num_kpt, kpt_vals)

    # 物件度權重（抑制背景框）
    w_obj = tf.stop_gradient(tf.nn.sigmoid(obj_t))  # [B,N,1]

    # Box: L1，加權
    box_l = tf.reduce_mean(w_obj * tf.abs(box_t - box_s))

    # Obj: BCE（logits）
    obj_l = config.BCE(obj_t, obj_s)

    # Class: KL（logits->softmax），加權
    p_t = tf.nn.softmax(cls_t)
    p_s = tf.nn.softmax(cls_s)
    cls_l = kl_div_weighted(p_t, p_s, weight=w_obj)

    # KPT (x,y): L1，加權；(score): BCE（logits），加權
    kxy_l = tf.reduce_mean(w_obj[..., None] * tf.abs(kxy_t - kxy_s))  # [B,N,1,K,2] 對齊 broadcast
    ks_l  = tf.reduce_mean(w_obj * config.BCE(ks_t, ks_s))

    # 總損失
    loss = (config.W_BOX * box_l +
            config.W_OBJ * obj_l +
            config.W_CLS * cls_l +
            config.W_KPT_XY * kxy_l +
            config.W_KPT_S  * ks_l)
    return loss

# ---------- 代表集 generator（給 TFLite） ----------
def rep_data_gen():
    paths = sorted(glob.glob(str(Path(config.REP_DIR) / "*")))
    for p in paths:
        img = parse_img(p)  # float32 [H,W,3] /255
        img = np.expand_dims(img, 0).astype(np.float32)
        yield [img]
