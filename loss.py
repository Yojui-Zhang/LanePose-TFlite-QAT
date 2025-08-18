
import tensorflow as tf
import config

def _split_outputs(y, num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS):
    C = tf.shape(y)[-1]
    expect_no_obj   = 4 + num_cls + num_kpt * kpt_vals
    expect_with_obj = 5 + num_cls + num_kpt * kpt_vals
    has_obj = tf.equal(C, expect_with_obj)

    box = y[..., :4]
    cls_base  = tf.where(has_obj, 5, 4)
    kpt_base  = cls_base + num_cls
    obj = tf.cond(has_obj, lambda: y[..., 4:5], lambda: tf.zeros_like(y[..., :1]))
    cls = y[..., cls_base:cls_base+num_cls]
    kpt = y[..., kpt_base:kpt_base+num_kpt*kpt_vals]
    kpt = tf.reshape(kpt, [-1, tf.shape(y)[1], num_kpt, kpt_vals])
    kxy = kpt[..., :2]
    ks  = kpt[..., 2:3]
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
    # w_obj = tf.stop_gradient(tf.nn.sigmoid(obj_t))  # [B,N,1]
    # 老師沒有 obj 通道時，用 class 機率當權重
    p_t = tf.nn.softmax(cls_t)
    w_obj = tf.reduce_max(p_t, axis=-1, keepdims=True)  # [B,N,1]


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

