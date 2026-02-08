# Source File: QAT_Refactored/losses/distill_loss.py

from __future__ import annotations

import tensorflow as tf
from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.utils.tensor_layout import ensure_bnc_tf


def split_outputs(y_bnc: tf.Tensor, num_cls: int, num_kpt: int, kpt_vals: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Why: distill loss 的切片必須以最後一維為 channel；統一先確保輸入為 (B,N,C) 後再切片，避免 (B,C,N) 切歪。
    """
    expected_c = 4 + num_cls + (num_kpt * kpt_vals)
    tf.debugging.assert_rank(y_bnc, 3, message="split_outputs expects rank-3 (B,N,C).")
    tf.debugging.assert_equal(
        tf.shape(y_bnc)[-1],
        tf.cast(expected_c, tf.int32),
        message="split_outputs channel mismatch (last dim must be total channels).",
    )

    box = y_bnc[..., :4]
    cls = y_bnc[..., 4 : 4 + num_cls]
    kpt = y_bnc[..., 4 + num_cls :]
    return box, cls, kpt


class DistillLossPose(tf.keras.layers.Layer):
    """
    Distillation Loss.
    """
    def __init__(self, cfg: AppConfig, **kwargs):
        super().__init__(name="DistillLossPose", **kwargs)
        self.cfg = cfg

    def call(self, y_teacher: tf.Tensor, y_student: tf.Tensor) -> tf.Tensor:
        """
        Accepts: (B,C,N) or (B,N,C) logits
        Returns: scalar KD loss
        """
        total_c = self.cfg.total_output_channels

        y_teacher = ensure_bnc_tf(y_teacher, total_c)
        y_student = ensure_bnc_tf(y_student, total_c)

        tf.debugging.assert_equal(
            tf.shape(y_teacher),
            tf.shape(y_student),
            message="Teacher/Student logits shape mismatch after layout normalization.",
        )

        # Why: AMP/float16 下 log/kl 更穩定
        y_teacher = tf.cast(y_teacher, tf.float32)
        y_student = tf.cast(y_student, tf.float32)

        box_t, cls_t, kpt_t = split_outputs(y_teacher, self.cfg.NUM_CLS, self.cfg.NUM_KPT, self.cfg.KPT_VALS)
        box_s, cls_s, kpt_s = split_outputs(y_student, self.cfg.NUM_CLS, self.cfg.NUM_KPT, self.cfg.KPT_VALS)

        # Soft targets on cls
        p_t_cls = tf.nn.softmax(cls_t, axis=-1)
        p_s_cls = tf.nn.softmax(cls_s, axis=-1)

        # Objectness weighting (teacher confidence)
        w_obj = tf.reduce_max(p_t_cls, axis=-1, keepdims=True)  # (B,N,1)

        # 1) Box loss (L1 on sigmoid)
        box_t_act = tf.sigmoid(box_t)
        box_s_act = tf.sigmoid(box_s)
        l_box = tf.reduce_mean(w_obj * tf.abs(box_t_act - box_s_act))

        # 2) Cls loss (KL)
        kl = tf.reduce_sum(
            p_t_cls * (tf.math.log(p_t_cls + 1e-9) - tf.math.log(p_s_cls + 1e-9)),
            axis=-1,
        )  # (B,N)
        l_cls = tf.reduce_mean(tf.squeeze(w_obj, axis=-1) * kl)

        # 3) Kpt loss (L1 on sigmoid xy)
        kxy_t = tf.sigmoid(kpt_t[..., : self.cfg.NUM_KPT * 2])
        kxy_s = tf.sigmoid(kpt_s[..., : self.cfg.NUM_KPT * 2])
        l_kxy = tf.reduce_mean(
            w_obj * tf.reduce_mean(tf.abs(kxy_t - kxy_s), axis=-1, keepdims=True)
        )

        total = (self.cfg.W_BOX * l_box) + (self.cfg.W_CLS * l_cls) + (self.cfg.W_KPT_XY * l_kxy)
        return total
