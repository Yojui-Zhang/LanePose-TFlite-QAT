# verify_pose_loss_smoke.py
import tensorflow as tf
import numpy as np

from QAT_Refactored.config.config import cfg
from QAT_Refactored.losses.pose_loss import PoseLabelLoss, get_anchors, build_batch_dict_from_padded_labels

@tf.function
def f(loss_fn, y_pred, batch_labels, anchors, class_weights):
    batch_dict = build_batch_dict_from_padded_labels(
        batch_labels,
        num_cls=cfg.NUM_CLS,
        num_kpt=cfg.NUM_KPT,
        kpt_vals=cfg.KPT_VALS,
    )
    return loss_fn(y_pred, batch_dict, anchors, class_weights)

def main():
    loss_fn = PoseLabelLoss(cfg)

    anchors = get_anchors(imgsz=cfg.IMGSZ, strides=[8, 16, 32])                 # 預期 (8400,4)
    N = anchors.shape[0]
    C = cfg.total_output_channels
    B = 2

    # 假資料：y_pred 直接模擬 (B,C,N)（符合你目前 head 輸出）:contentReference[oaicite:3]{index=3}
    y_pred = tf.random.uniform([B, C, N], dtype=tf.float32)

    # batch_labels: 依你專案格式（這裡只給 padding 的空標註，測 tracing）
    # 若你的 build_batch_dict_from_padded_labels 需要特定 shape，請對齊它的輸入規格。
    max_gt = 64
    # 假設 label tensor 內部格式由 build_batch_dict... 解碼；這裡用全 0 讓 valid_mask 全 False
    batch_labels = tf.zeros([B, max_gt, 1 + 4 + cfg.NUM_KPT * cfg.KPT_VALS], dtype=tf.float32)

    class_weights = tf.ones([cfg.NUM_CLS], dtype=tf.float32)

    out = f(loss_fn, y_pred, batch_labels, anchors, class_weights)
    tf.nest.map_structure(lambda t: tf.debugging.assert_all_finite(t, "non-finite"), out)
    print("verify_pose_loss_smoke: OK")

if __name__ == "__main__":
    main()
