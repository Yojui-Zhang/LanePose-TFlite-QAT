# qat_tf/qat_distill.py
import os, glob
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import cv2
import keras as K
import u_8_s_pose_keras_qat as U

import config

from importlib import reload
from pathlib import Path
from process import build_dataset, try_load_keras_model, distill_loss, _split_outputs, distill_loss_pose, distill_loss, rep_data_gen





def main():

# =====================================
    # # 1) 載入模型（優先 Keras）
    base_model, is_keras = try_load_keras_model(config.EXPORTED_DIR)
    # if not is_keras:
    #     print("[WARN] 匯出物看起來不是純 Keras 模型，tfmot 可能無法插入 fake-quant。"
    #           "建議重試 export(format='saved_model', keras=True, nms=False)。")
# =====================================
    # Teacher（凍結）
    teacher = base_model
    teacher.trainable = False

# =====================================
    # Student：Keras 可量化
    reload(U)  # 確保載到最新
    student = U.build_u8s_pose(
        input_shape=(config.IMGSZ, config.IMGSZ, 3),
        num_classes=config.NUM_CLS,
        num_kpt=config.NUM_KPT,
        kpt_vals=config.KPT_VALS
    )

    print(type(student), isinstance(student, K.Model), getattr(student, "_is_graph_network", None))

    student = tfmot.quantization.keras.quantize_model(student)

# =====================================

    # # Student（QAT）
    # if is_keras:
    #     student = tfmot.quantization.keras.quantize_model(
    #         tf.keras.models.clone_model(teacher)
    #     )
    # else:
    #     # 退路：對 wrapper 嘗試量化（可能失敗）；若失敗改用 PTQ。
    #     try:
    #         student = tfmot.quantization.keras.quantize_model(
    #             tf.keras.models.clone_model(teacher)
    #         )
    #     except Exception as e:
    #         print("[ERROR] 量化包裝失敗：", e)
    #         print("請改用 Keras 匯出，或先走 PTQ（int8）流程。")
    #         return
# =====================================
    # 權重對齊（clone_model 通常會複製權重；這裡保險起見再 copy 一次）
    student.set_weights(teacher.get_weights())

    opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(batch_imgs):
        with tf.GradientTape() as tape:
            y_t = teacher(batch_imgs, training=False)
            y_s = student(batch_imgs, training=True)
            # loss = distill_loss(y_t, y_s, NUM_CLS)
            loss = distill_loss_pose(y_t, y_s, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
        grads = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # 2) 建立無標註影像資料流（用你的訓練/驗證影像路徑萬用字元）
    ds, n_files = build_dataset(img_glob="your_train_images/**/*.jpg", batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    # =================== 一次性自檢開始 =======================

    # 取一個 batch，檢查前處理是否正確
    it = iter(ds)
    sample = next(it)  # 期望 [B,H,W,3]、float32、值域約在[0,1]
    print("[CHECK] sample shape/dtype/range:",
          sample.shape, sample.dtype,
          float(tf.reduce_min(sample)), float(tf.reduce_max(sample)))

    # 有些匯出會回傳 dict 或 list，統一取第一個主要輸出張量
    def _pick_tensor(y):
        if isinstance(y, dict):
            k = list(y.keys())[0]
            return y[k]
        if isinstance(y, (list, tuple)):
            return y[0]
        return y
    
    # 前向一次，檢查 Teacher/Student 輸出 shape
    y_t_dbg = _pick_tensor(teacher(sample, training=False))
    y_s_dbg = _pick_tensor(student(sample, training=False))
    print("[CHECK] teacher out:", y_t_dbg.shape, y_t_dbg.dtype)
    print("[CHECK] student out:", y_s_dbg.shape, y_s_dbg.dtype)

    # 針對 YOLO-Pose：確認最後維度是否符合 [5 + NUM_CLS + NUM_KPT*KPT_VALS]
    expected = 5 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    feat_dim = int(y_t_dbg.shape[-1])
    if feat_dim != expected:
        print(f"[WARN] teacher 最後維度={feat_dim} != 預期={expected}，"
              f"請調整 _split_outputs() 的切片索引或確認實際輸出佈局。")

    # 乾跑一次 train_step，確認梯度可回傳、loss 無 NaN/Inf
    # （若想看到更詳細錯誤，可臨時啟用 eager：tf.config.run_functions_eagerly(True)）
    loss_test = train_step(sample)
    tf.debugging.assert_all_finite(loss_test, "loss 出現 NaN/Inf，請檢查前處理或切片索引。")
    print(f"[CHECK] dry-run distill_loss={float(loss_test):.6f}")

    # 確認 QAT 真的插入了量化包裝層（可選）
    qlayers = [l for l in student.submodules if "Quantize" in l.__class__.__name__]
    print(f"[CHECK] quantization layers count: {len(qlayers)}")
    # =================== 一次性自檢到此結束 =======================


    # 3) 訓練（QAT 微調）
    for e in range(config.EPOCHS):
        it = iter(ds)
        for _ in range(steps_per_epoch):
            imgs = next(it)
            l = train_step(imgs)
        print(f"Epoch {e+1}/{config.EPOCHS}  distill_loss={float(l):.4f}")

    # 4) 另存 QAT 模型（SavedModel）
    SAVE_DIR = "qat_saved_model"
    tf.saved_model.save(student, SAVE_DIR)
    print("Saved:", SAVE_DIR)

    # 5) 轉 TFLite 全 INT8
    conv = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflm = conv.convert()
    Path(config.TFLITE_OUT).write_bytes(tflm)
    print("Wrote", config.TFLITE_OUT)

if __name__ == "__main__":
    main()

