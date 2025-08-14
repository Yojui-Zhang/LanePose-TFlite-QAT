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
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle: ds = ds.shuffle(len(files))
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat: ds = ds.repeat()
    return ds, len(files)

# ---------- 載入 Keras 模型（優先 .keras / Keras SavedModel） ----------
def try_load_keras_model(export_dir):
    # 1) 直接用 keras loader
    try:
        m = tf.keras.models.load_model(export_dir)
        return m, True
    except Exception:
        pass
    # 2) 退回 SavedModel signature -> 包成 Keras 可呼叫模型（可能無法被 tfmot 量化）
    saved = tf.saved_model.load(export_dir)
    fn = saved.signatures["serving_default"]
    # 建立一個薄包裝層：輸入 [B,H,W,3] -> dict -> 取第一個輸出張量
    class SMWrapper(tf.keras.Model):
        def __init__(self, concrete_fn):
            super().__init__()
            self.fn = concrete_fn
        @tf.function
        def call(self, x):
            outs = self.fn(x)
            # 取第一個 key（依實際 keys 調整）
            key = list(outs.keys())[0]
            return outs[key]
    return SMWrapper(fn), False

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
