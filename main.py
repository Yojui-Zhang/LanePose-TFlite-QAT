# === TOP-OF-FILE SHIM: put this at the very top of main.py BEFORE any import of tfmot/keras/etc ===
import os, sys

# Prefer tf.keras (legacy) and try to avoid independent keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import tensorflow early
import tensorflow as tf
from tensorflow import keras as K

# Force any "import keras" in other libs to resolve to tf.keras
sys.modules["keras"] = K
sys.modules["keras.models"] = K.models
sys.modules["keras.layers"] = K.layers
sys.modules["keras.activations"] = K.activations
sys.modules["keras.initializers"] = K.initializers
sys.modules["keras.utils"] = K.utils
sys.modules["keras.losses"] = K.losses
sys.modules["keras.backend"] = K.backend
# ===========================================================

from tensorflow.keras import layers as L
import tensorflow_model_optimization as tfmot
from pathlib import Path

import numpy as np
import cv2

import u_8_s_pose_keras_qat as cfg
import config

from importlib import reload
from pathlib import Path
from process import build_dataset, try_load_keras_model, distill_loss, _split_outputs, distill_loss_pose, distill_loss, rep_data_gen, normalize_teacher_pred


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 開啟 memory growth，逐步分配顯存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print("TF GPU config error:", e)


def main():

    # 1) 載入 Teacher（優先 Keras）
    base_model, is_keras = try_load_keras_model(config.EXPORTED_DIR)
    teacher = base_model
    teacher.trainable = False

    reload(cfg)  # 確保載到最新
    
    # 2) 建 Student 並插 QAT wrapper
    student = cfg.build_u8s_pose(
        input_shape=(config.IMGSZ, config.IMGSZ, 3),
        num_classes=config.NUM_CLS,
        num_kpt=config.NUM_KPT,
        kpt_vals=config.KPT_VALS
    )
    student = tfmot.quantization.keras.quantize_model(student)

    # 檢查 QAT 層是否插入
    qlayers = [l for l in student.submodules if "Quantize" in l.__class__.__name__]
    print("[CHECK] quantization layers count:", len(qlayers))

    opt = tf.keras.optimizers.Adam(1e-4)
    expected = 5 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS

    @tf.function
    def train_step(batch_imgs):
        with tf.GradientTape() as tape:
            y_t = teacher(batch_imgs, training=False)
            y_t = normalize_teacher_pred(y_t, expected_C=expected)   # -> [B,N,C]
            y_s = student(batch_imgs, training=True)
            loss = distill_loss_pose(y_t, y_s, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
        grads = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # 3) 建立資料流
    ds, n_files = build_dataset(img_glob=config.REP_DIR, batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    # ============== 一次性自檢 ==============
    it = iter(ds)
    sample = next(it)  # 期望 [B,H,W,3]、float32、[0,1]
    print("[CHECK] sample shape/dtype/range:",
          sample.shape, sample.dtype,
          float(tf.reduce_min(sample)), float(tf.reduce_max(sample)))

    def _pick_tensor(y):
        if isinstance(y, dict):
            k = list(y.keys())[0]
            return y[k]
        if isinstance(y, (list, tuple)):
            return y[0]
        return y
    
    y_t_raw = teacher(sample, training=False)
    y_t_norm = normalize_teacher_pred(y_t_raw, expected_C=expected)
    y_s_norm = _pick_tensor(student(sample, training=False))
    print("[CHECK] teacher out (normalized):", y_t_norm.shape, y_t_norm.dtype)
    print("[CHECK] student out:", y_s_norm.shape, y_s_norm.dtype)

    feat_dim = int(y_t_norm.shape[-1])
    if feat_dim != expected:
        raise ValueError(f"[FATAL] teacher normalized 最後維度={feat_dim} != 預期={expected}。")

    loss_test = train_step(sample)
    tf.debugging.assert_all_finite(loss_test, "loss 出現 NaN/Inf，請檢查前處理或切片索引。")
    print(f"[CHECK] dry-run distill_loss={float(loss_test):.6f}")
    # ============== 自檢結束 ==============

    # 4) QAT 微調
    for e in range(config.EPOCHS):
        it = iter(ds)
        for _ in range(steps_per_epoch):
            imgs = next(it)
            l = train_step(imgs)
        print(f"Epoch {e+1}/{config.EPOCHS}  distill_loss={float(l):.4f}")

    # 5) 計算 N, C（u8s-pose）
    N = (config.IMGSZ // 8) * (config.IMGSZ // 8) \
        + (config.IMGSZ // 16) * (config.IMGSZ // 16) \
        + (config.IMGSZ // 32) * (config.IMGSZ // 32)
    C = 5 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    print("Expected N, C:", N, C)  # e.g. 8400, 56/57

    # 6) Strip QAT wrappers if available (fallback to no-strip)
    if hasattr(tfmot.quantization.keras, "strip_quantization"):
        print("[INFO] Using tfmot.strip_quantization to remove QAT wrappers.")
        student_infer = tfmot.quantization.keras.strip_quantization(student)
    else:
        print("[WARN] tfmot.strip_quantization 不存在，跳過移除 QAT wrapper（格式仍會與 good 一致）。")
        student_infer = student  # 直接用 QAT 包裝的模型進行匯出

    # 7) 觀察一次實際輸出 shape
    try:
        sample_one = sample[:1]
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], dtype=tf.float32)

    y_test = student_infer(sample_one, training=False)
    y_shape = tuple(y_test.shape.as_list())
    print("[EXPORT CHECK] student_infer runtime output shape:", y_shape)

    # 決策：先把動態 layout 統一成 [1, N, C]（nc），最後再轉為 [1, C, N]
    action = "identity"
    if len(y_shape) == 3:
        _, d1, d2 = y_shape
        if d1 == N and d2 == C:
            action = "identity"              # (1,N,C) -> (1,N,C)
        elif d1 == C and d2 == N:
            action = "transpose"             # (1,C,N) -> (1,N,C)
        elif d1 == N and d2 == C - 1:
            action = "pad_last"              # (1,N,C-1) -> pad -> (1,N,C)
        elif d1 == C - 1 and d2 == N:
            action = "transpose_then_pad"    # (1,C-1,N) -> T -> pad -> (1,N,C)
        else:
            print("[EXPORT CHECK] WARNING: unusual shape; will force reshape to [1,N,C]")
            action = "force_reshape"
    else:
        total = int(tf.size(y_test).numpy())
        if total == 1 * N * C:
            action = "force_reshape"
        else:
            print("[EXPORT CHECK] ERROR: cannot map to [1,N,C] safely; still try force reshape")
            action = "force_reshape"

    print("[EXPORT CHECK] chosen action to make [1,N,C]:", action)

    # 8) 封裝 ExportModule：輸出 **最終固定為 [1, C, N]**（= good）
    class ExportModule(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, config.IMGSZ, config.IMGSZ, 3], dtype=tf.float32, name="images")])
        def serving_fn(self, x):
            y = self.model(x, training=False)  # float graph

            # 先變成 [1, N, C]（nc）
            if action == "identity":
                out_nc = y
            elif action == "transpose":
                out_nc = tf.transpose(y, perm=[0, 2, 1])              # (B,C,N)->(B,N,C)
            elif action == "pad_last":
                out_nc = tf.pad(y, paddings=[[0, 0], [0, 0], [0, 1]]) # (B,N,C-1)->(B,N,C)
            elif action == "transpose_then_pad":
                tmp = tf.transpose(y, perm=[0, 2, 1])                 # (B,C-1,N)->(B,N,C-1)
                out_nc = tf.pad(tmp, paddings=[[0, 0], [0, 0], [0, 1]])
            elif action == "force_reshape":
                out_nc = tf.reshape(y, [1, N, C])
            else:
                out_nc = y

            # 最終改成 [1, C, N]（= good）
            out_cn = tf.transpose(out_nc, perm=[0, 2, 1])             # (B,N,C)->(B,C,N)
            out_cn = tf.reshape(out_cn, [1, C, N])                    # 靜態 shape 保證

            # ★ key 用 "PartitionedCall:0" 以對齊 good
            return {"PartitionedCall:0": out_cn}

    export_mod = ExportModule(student_infer)

    # 9) 存 SavedModel（固定 batch=1, float I/O）
    SAVE_DIR = "qat_saved_model_fixed_fpIO"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    concrete_fn = export_mod.serving_fn.get_concrete_function()
    tf.saved_model.save(export_mod, SAVE_DIR, signatures=concrete_fn)
    print("Saved fixed-batch SavedModel to", SAVE_DIR)

    # 10) 轉 TFLite（★ float32 I/O + 內部 INT8 = 與 good 一致）
    conv = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # ★ 關鍵：外部 I/O = float32（內部仍 INT8）
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32

    conv.experimental_new_converter = True
    try:
        conv.experimental_new_quantizer = True
    except Exception:
        pass

    # 輸出檔
    Path(config.TFLITE_OUT).parent.mkdir(parents=True, exist_ok=True)
    tflm = conv.convert()
    Path(config.TFLITE_OUT).write_bytes(tflm)
    print("Wrote", config.TFLITE_OUT)

    # === 10b) 產生純浮點 TFLite（不量化）作為對照 ===
    conv_fp = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    # 不開啟任何優化、不給代表性資料
    # conv_fp.optimizations = []
    # 不設定 supported_ops，讓它保持 float kernels
    tflm_fp = conv_fp.convert()
    Path(config.TFLITE_OUT.replace(".tflite", "_float.tflite")).write_bytes(tflm_fp)
    print("Wrote", config.TFLITE_OUT.replace(".tflite", "_float.tflite"))


if __name__ == "__main__":
    main()
