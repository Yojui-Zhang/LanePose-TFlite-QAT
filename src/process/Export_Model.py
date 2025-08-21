'''
===================================================
Tensor 版本強制設定
===================================================
'''
import os, sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras as K

# force any "import keras" to resolve to tf.keras
sys.modules["keras"] = K
sys.modules["keras.models"] = K.models
sys.modules["keras.layers"] = K.layers
sys.modules["keras.activations"] = K.activations
sys.modules["keras.initializers"] = K.initializers
sys.modules["keras.utils"] = K.utils
sys.modules["keras.losses"] = K.losses
sys.modules["keras.backend"] = K.backend

'''
===================================================
import Depance file
===================================================
'''
import numpy as np 
from pathlib import Path

'''
===================================================
Local imports from your project
===================================================
'''
import config
from src.Loss_function.loss import _split_outputs
from src.process.data import rep_data_gen
from src.process.pred_model import normalize_teacher_pred
from src.process.Train_Model import _ensure_bhwc4

'''
==================================================================================
Export Setting 
==================================================================================
'''
def create_and_configure_tflite_converter(saved_model_path):
    """建立並回傳一個已設定好的 TFLite INT8 converter"""
    print(f"  > Creating TFLite converter from: {saved_model_path}")
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.float32
    conv.inference_output_type     = tf.float32
    conv.experimental_new_converter = True
    try:
        conv.experimental_new_quantizer = True
    except AttributeError:
        pass # 如果 TensorFlow 版本不支援此屬性，則忽略
    return conv


'''
==================================================================================
Export Module 
==================================================================================
'''
class ExportModule(tf.Module):
    """
    ==============================================================================
       Export wrapper to:
       - unify to (B,N,C)
       - reorder P3/P4/P5 grid flatten modes (N)
       - apply channel mapping (C)
       - (optional) convert xywh -> ltrb distances in stride units (match Ultralytics decode)
       - final (1,C,N) with name 'output0'
    ==============================================================================
    """
    def __init__(self, model, C, lens_perm, reorder_idx,
                 grid_modes=config.GRID_MODES, porder=config.PORDER, ch_map=config.CHANNEL_MAPPING,
                 xywh_to_ltrb=config.XYWH_TO_LTRB, xywh_is_norm01=config.XYWH_IS_NORMALIZED_01):
        super().__init__()
        self.model = model
        self.C = int(C)
        self.lens_perm = tuple(int(x) for x in lens_perm)
        self.reorder_idx = tuple(int(x) for x in reorder_idx)
        self.grid_modes = tuple(grid_modes)
        self.porder = tuple(porder)
        self.ch_map = tf.constant(list(map(int, ch_map)), tf.int32)
        self.xywh_to_ltrb = bool(xywh_to_ltrb)
        self.xywh_is_norm01 = bool(xywh_is_norm01)

    @tf.function(input_signature=[tf.TensorSpec([1, config.IMGSZ, config.IMGSZ, 3], tf.float32, name="images")])
    def serving_fn(self, x):
        y = self.model(x, training=False)
        if y.shape.rank != 3: raise ValueError(f"export expects rank=3, got {y.shape}")
        out_nc = tf.transpose(y, [0, 2, 1]) if (y.shape[1] == self.C) else y

        segs = tf.split(out_nc, num_or_size_splits=list(self.lens_perm), axis=1)
        segs = [segs[i] for i in self.reorder_idx]
        out_nc = tf.concat(segs, axis=1)

        H3, W3 = config.IMGSZ // 8, config.IMGSZ // 8
        H4, W4 = config.IMGSZ // 16, config.IMGSZ // 16
        H5, W5 = config.IMGSZ // 32, config.IMGSZ // 32
        N3, N4, N5 = H3*W3, H4*W4, H5*W5
        
        def reorder_block_tf(block_BNC, H, W, scan='row', flip_y=False, flip_x=False):
            B, C = tf.shape(block_BNC)[0], tf.shape(block_BNC)[2]
            x_ = tf.reshape(block_BNC, [B, H, W, C])
            if scan == 'col': x_ = tf.transpose(x_, [0, 2, 1, 3])
            if flip_y: x_ = tf.reverse(x_, axis=[1])
            if flip_x: x_ = tf.reverse(x_, axis=[2])
            return tf.reshape(x_, [B, -1, C])

        p3, p4, p5 = tf.split(out_nc, [N3, N4, N5], axis=1)
        m3, m4, m5 = self.grid_modes
        p3r = reorder_block_tf(p3, H3, W3, scan=m3[0], flip_y=bool(m3[1]), flip_x=bool(m3[2]))
        p4r = reorder_block_tf(p4, H4, W4, scan=m4[0], flip_y=bool(m4[1]), flip_x=bool(m4[2]))
        p5r = reorder_block_tf(p5, H5, W5, scan=m5[0], flip_y=bool(m5[1]), flip_x=bool(m5[2]))

        preds_list = [p3r, p4r, p5r]
        out_nc = tf.concat([preds_list[i] for i in self.porder], axis=1)
        out_nc = tf.gather(out_nc, self.ch_map, axis=-1)

        # 輸出是 logits，但 TFLite 需要的是機率，因此在這裡加上 sigmoid
        raw_box, raw_cls, raw_kpt = tf.split(out_nc, [4, config.NUM_CLS, -1], axis=-1)

        box = tf.sigmoid(raw_box)
        cls = tf.sigmoid(raw_cls)
        
        kpt_reshaped = tf.reshape(raw_kpt, [-1, N3+N4+N5, config.NUM_KPT, config.KPT_VALS])
        # kpt_xy = kpt_reshaped[..., :2] # xy 是 logits，但在 C++ 端處理
        kpt_xy = tf.sigmoid(kpt_reshaped[..., :2]) # 新的程式碼，輸出歸一化座標
        kpt_v = tf.sigmoid(kpt_reshaped[..., 2:3]) # v 是機率
        kpt = tf.reshape(tf.concat([kpt_xy, kpt_v], axis=-1), [-1, N3+N4+N5, config.NUM_KPT * config.KPT_VALS])
        
        # 產生和 C++ code 預期一致的輸出：[box_prob, cls_prob, kpt_logits_v_prob]
        pred_ultra = tf.concat([box, cls, kpt], axis=-1)
        out_cn = tf.transpose(pred_ultra, [0, 2, 1])
        return {"output0": tf.reshape(out_cn, [1, self.C, -1])}

def _detect_resume_kind(p: str):
    """
    ==============================================================================
    辨識 RESUME_WEIGHTS 格式, 回傳 'ckpt' | 'h5' | 'savedmodel' | 'tflite' | None
    ==============================================================================
    """
    if not p: return None
    P = Path(p)
    if P.is_dir():
        # SavedModel: 目錄內含 saved_model.pb 或 variables/
        if (P / "saved_model.pb").exists() or (P / "variables").exists():
            return "savedmodel"
        return None
    suf = P.suffix.lower()
    if suf == ".tflite":
        return "tflite"
    if suf in (".h5", ".hdf5"):
        return "h5"
    if suf == ".ckpt" or (P.with_suffix(".ckpt.index").exists()) or (P.with_suffix(".index").exists()):
        return "ckpt"
    # 一些 ckpt 不帶副檔名：xxx.index/xxx.data-00000-of-00001
    if (P.with_suffix(".index").exists()) or any(str(P).endswith(s) for s in [".data-00000-of-00001", ".index"]):
        return "ckpt"
    return None


def mae_student_keras_vs_teacher(student, teacher, ds, lens_perm, reorder_idx, C):
    """
    ==============================================================================
    用『記憶體中的 student（ckpt/h5 已載入）』做 prob-domain MAE（與 teacher 比）。
    ==============================================================================
    """
    try:
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    # 對齊到 (B,N,C) 並做分段重排
    def align_student_BNC(y_s_raw):
        y_s_nc = tf.transpose(y_s_raw, [0, 2, 1]) if (y_s_raw.shape.rank == 3 and int(y_s_raw.shape[1]) == C) else y_s_raw
        segs = tf.split(y_s_nc, num_or_size_splits=list(lens_perm), axis=1)
        segs = [segs[i] for i in reorder_idx]
        return tf.concat(segs, axis=1)  # (B,N,C)

    @tf.function
    def _eval(x_eval):
        y_t = normalize_teacher_pred(
                    teacher(x_eval, training=False),
                    expected_C=C,
                    num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS,
                    batch_imgs=x_eval,
                    target_domain='unit',          # ★ 改 'pixel' 可走像素域
                    return_detected=False
                    )
        y_s = align_student_BNC(student(x_eval, training=False))                     # (B,N,C)，logits/未激活

        box_t, obj_t, cls_t, kxy_t, ks_t = _split_outputs(y_t, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
        box_s, obj_s, cls_s, kxy_s, ks_s = _split_outputs(y_s, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)

        # 學生轉到機率域
        box_s = tf.nn.sigmoid(box_s)
        kxy_s = tf.nn.sigmoid(kxy_s)
        p_t   = tf.nn.softmax(cls_t)
        p_s   = tf.nn.softmax(cls_s)

        mae_box = tf.reduce_mean(tf.abs(box_t - box_s))
        mae_kpt = tf.reduce_mean(tf.abs(kxy_t - kxy_s))
        mae_cls = tf.reduce_mean(tf.abs(p_t   - p_s))  # 分佈 L1
        return mae_box, mae_cls, mae_kpt

    mb, mc, mk = _eval(sample_one)
    print(f"\n[MAE Keras(ckpt/h5) vs Teacher] box={float(mb.numpy()):.4f}  cls={float(mc.numpy()):.4f}  kpt={float(mk.numpy()):.4f}")


def mae_student_savedmodel_vs_teacher(student_saved_dir, teacher, ds):
    """
    ==============================================================================
    用『已 export 的學生 SavedModel』直接評估 Prob-domain MAE（Student vs Teacher）。
    不需 ckpt；兩邊都是機率域（box/cls/kpt 都已經過 sigmoid/softmax）。
    ==============================================================================
    """
    # 取 1 張樣本
    try:
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS

    # 學生 SavedModel 載入 → (1, C, N) 機率
    sm = tf.saved_model.load(student_saved_dir)
    sig = sm.signatures.get("serving_default")
    if sig is None:
        raise RuntimeError(f"No serving_default in {student_saved_dir}")
    out_cn = sig(images=sample_one)["output0"]  # 通常是 (1, C, N)
    if out_cn.shape.rank != 3:
        raise RuntimeError(f"Unexpected output shape from student SavedModel: {out_cn.shape}")
    if out_cn.shape[1] == C:
        st_nc = tf.transpose(out_cn, [0, 2, 1])  # -> (1, N, C)
    elif out_cn.shape[2] == C:
        st_nc = out_cn  # 已是 (1, N, C)
    else:
        raise RuntimeError(f"Channel mismatch: got {out_cn.shape}, expected C={C}")

    # 老師 → normalize_teacher_pred: (1, N, C) 機率
    te_nc = normalize_teacher_pred(
                    teacher(sample_one, training=False),
                    expected_C=C,
                    num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS,
                    batch_imgs=sample_one,
                    target_domain='unit',          # ★ 改 'pixel' 可走像素域
                    return_detected=False
                    )

    # 切片計 MAE（機率域）
    c0_box, c1_box = 0, 4
    c0_cls, c1_cls = 4, 4 + config.NUM_CLS
    c0_kpt, c1_kpt = c1_cls, C

    mae_box = tf.reduce_mean(tf.abs(st_nc[..., c0_box:c1_box] - te_nc[..., c0_box:c1_box]))
    mae_cls = tf.reduce_mean(tf.abs(st_nc[..., c0_cls:c1_cls] - te_nc[..., c0_cls:c1_cls]))
    mae_kpt = tf.reduce_mean(tf.abs(st_nc[..., c0_kpt:c1_kpt] - te_nc[..., c0_kpt:c1_kpt]))

    print(f"\n[MAE SavedModel vs Teacher] box={float(mae_box.numpy()):.4f}  "
          f"cls={float(mae_cls.numpy()):.4f}  kpt={float(mae_kpt.numpy()):.4f}")


def mae_student_tflite_vs_teacher(tflite_path, teacher, ds):
    """
    ==============================================================================
    只有 TFLite 也可以：用 TFLite 輸出 (1, C, N) 機率，與 Teacher 機率做 MAE。
    ==============================================================================
    """
    try:
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS

    # TFLite 推論 → (1, C, N)
    out_cn = _run_tflite_infer(tflite_path, sample_one)  # 你已有這個 helper
    if out_cn.ndim != 3 or out_cn.shape[1] != C:
        raise RuntimeError(f"TFLite output shape unexpected: {out_cn.shape}")
    st_nc = np.transpose(out_cn, (0, 2, 1))  # -> (1, N, C), numpy

    # Teacher 機率
    te_nc = normalize_teacher_pred(
                    teacher(sample_one, training=False),
                    expected_C=C,
                    num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS,
                    batch_imgs=sample_one,
                    target_domain='unit',          # ★ 改 'pixel' 可走像素域
                    return_detected=False
                    )

    # MAE
    c0_box, c1_box = 0, 4
    c0_cls, c1_cls = 4, 4 + config.NUM_CLS
    c0_kpt, c1_kpt = c1_cls, C

    mae_box = np.mean(np.abs(st_nc[..., c0_box:c1_box] - te_nc[..., c0_box:c1_box]))
    mae_cls = np.mean(np.abs(st_nc[..., c0_cls:c1_cls] - te_nc[..., c0_cls:c1_cls]))
    mae_kpt = np.mean(np.abs(st_nc[..., c0_kpt:c1_kpt] - te_nc[..., c0_kpt:c1_kpt]))

    print(f"\n[MAE TFLite vs Teacher]    box={mae_box:.4f}  cls={mae_cls:.4f}  kpt={mae_kpt:.4f}")


def export_only(student, teacher, ds, output_paths, tag="export_only_diagnostics"):
    """
    ==============================================================================
    [新] Export-only 模式 (僅診斷)：
      - 根據 RESUME_WEIGHTS 載入一個已有的 SavedModel 或 TFLite 檔案。
      - 如果提供的是 SavedModel，會額外將其轉換為 TFLite。
      - 對提供的模型檔案執行 MAE 診斷，將其與 Teacher 模型進行比較。
      - 注意：此模式不再從 student 物件匯出新模型，而是專注於分析現有檔案。
    ==============================================================================
    """
    # ---- 1. 基本設定與樣本 ----
    rw = getattr(config, "RESUME_WEIGHTS", None)
    if not rw:
        print("[ERROR] `RESUME_WEIGHTS` must be set in config.py for diagnostics mode. Aborting.")
        return

    kind = _detect_resume_kind(rw)
    if not kind or kind in ("ckpt", "h5"):
        print(f"[ERROR] `RESUME_WEIGHTS` must be a 'savedmodel' or 'tflite' directory/file. Found: {kind}. Aborting.")
        return

    print(f"\n--- Running Diagnostics on Existing Model ---")
    print(f"  - Model Path: {rw}")
    print(f"  - Detected Kind: {kind}")

    try:
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)
    print(f"  - Sanity Check: sample_one range = [{float(tf.reduce_min(sample_one)):.4f}, {float(tf.reduce_max(sample_one)):.4f}]")

    # ---- 2. 根據檔案類型執行診斷 ----
    try:
        if kind == "savedmodel":
            print("\n--- Analyzing SavedModel ---")

            # 2a. 從此 SavedModel 轉換出 TFLite
            print("\n--- Converting SavedModel to TFLite for further analysis ---")
            conv = create_and_configure_tflite_converter(rw)
            tfl_bytes = conv.convert()
            
            # 決定 TFLite 輸出路徑
            tflite_filename = Path(rw).name + "_converted.tflite"
            tflite_path = output_paths['models'] / tflite_filename
            tflite_path.write_bytes(tfl_bytes)
            print(f"✅ TFLite model converted and saved to → {tflite_path}")

            # 2b. 對Save_model / TFLite 算 MAE
            mae_student_savedmodel_vs_teacher(rw, teacher, ds)
            mae_student_tflite_vs_teacher(str(tflite_path), teacher, ds)

        elif kind == "tflite":
            print("\n--- Analyzing TFLite ---")
            # 直接對該 TFLite 算 MAE
            mae_student_tflite_vs_teacher(rw, teacher, ds)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] A failure occurred during diagnostics for {rw}: {e}")

    print("\n--- Diagnostics Complete ---")


'''
==================================================================================
One-shot diagnostics helpers 
==================================================================================
'''

def _run_tflite_infer(tflite_path: str, x_bhwc: tf.Tensor) -> np.ndarray:
    """
    ==============================================================================
    Run TFLite and return (1, C, N) float32.
    ==============================================================================
    """

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    idt = interp.get_input_details()[0]
    odt = interp.get_output_details()[0]
    # set input
    xin = x_bhwc.numpy().astype(np.float32)
    interp.set_tensor(idt["index"], xin)
    interp.invoke()
    y = interp.get_tensor(odt["index"]).astype(np.float32)  # expect (1, C, N) or (1, N, C)
    # robust to layout
    if y.ndim != 3:
        raise RuntimeError(f"Unexpected TFLite output rank: {y.shape}")
    Ccand = [y.shape[1], y.shape[2]]
    # Let caller decide C, so we just return [1,C,N] if possible
    if y.shape[1] < y.shape[2]:
        return y  # (1, C, N)
    else:
        return np.transpose(y, (0, 2, 1))  # -> (1, C, N)


def run_diagnostics_once(
    export_mod: "ExportModule",
    teacher: tf.keras.Model,
    tflite_path: str,
    sample_one: tf.Tensor,
    C: int,
    NUM_CLS: int,
    NUM_KPT: int,
    KPT_VALS: int,
):
    """
    ==============================================================================
    Do three checks on the same image:
      1) Keras(student serving_fn) vs TFLite(student)  -> MAE box/cls/kpt
      2) Teacher(normalized) vs Student(serving_fn)    -> MAE box/cls/kpt
      3) Variance across N for several channels        -> detect plateau
    ==============================================================================
    """
    # ---- 1) Keras student (serving_fn -> [1,C,N]) ----
    out_keras = export_mod.serving_fn(sample_one)["output0"].numpy()  # (1, C, N)

    # ---- 2) TFLite student (-> [1,C,N]) ----
    out_tfl = _run_tflite_infer(tflite_path, sample_one)              # (1, C, N)

    # ---- 3) Teacher normalized (B,N,C) -> transpose to (1,C,N) ----
    expected_C = C
    te_nc = normalize_teacher_pred(
                        teacher(sample_one, training=False),
                        expected_C=C,
                        num_cls=config.NUM_CLS, num_kpt=config.NUM_KPT, kpt_vals=config.KPT_VALS,
                        batch_imgs=sample_one,
                        target_domain='unit',          # ★ 改 'pixel' 可走像素域
                        return_detected=False
                        )

    te_cn = np.transpose(te_nc, (0, 2, 1))  # (1, C, N)

    # sanity shape
    assert out_keras.shape[1] == C and out_tfl.shape[1] == C and te_cn.shape[1] == C, \
        f"Channel mismatch: keras={out_keras.shape}, tflite={out_tfl.shape}, teacher={te_cn.shape}"

    # slices
    c0_box = 0
    c1_box = 4
    c0_cls = 4
    c1_cls = 4 + NUM_CLS
    c0_kpt = c1_cls
    c1_kpt = C

    def mae_slice(a, b, s0, s1):
        return float(np.mean(np.abs(a[:, s0:s1, :] - b[:, s0:s1, :])))

    # ---- MAE: TFLite vs Keras (量化/匯出一致性) ----
    mae_box_tk = mae_slice(out_tfl, out_keras, c0_box, c1_box)
    mae_cls_tk = mae_slice(out_tfl, out_keras, c0_cls, c1_cls)
    mae_kpt_tk = mae_slice(out_tfl, out_keras, c0_kpt, c1_kpt)

    # ---- MAE: Student(Keras) vs Teacher (對齊/訓練程度) ----
    mae_box_st = mae_slice(out_keras, te_cn, c0_box, c1_box)
    mae_cls_st = mae_slice(out_keras, te_cn, c0_cls, c1_cls)
    mae_kpt_st = mae_slice(out_keras, te_cn, c0_kpt, c1_kpt)

    # ---- Variance across N（檢查是否又變成常數） ----
    def var_across_N(y_cn, ch, limit=500):
        N = y_cn.shape[2]
        n = min(N, limit)
        seg = y_cn[0, ch, :n]
        return float(np.var(seg))

    probe_channels = {
        "box_x": 0,
        "box_y": 1,
        "cls_0": c0_cls,
        "cls_last": c1_cls - 1,
        "kpt_last": C - 1,
    }

    print("\n========== One-shot Diagnostics ==========")
    print(f"[Shapes] keras={out_keras.shape}, tflite={out_tfl.shape}, teacher={te_cn.shape}")
    print("\n---- MAE: Student TFLite  vs Student Keras (serving_fn) ----")
    print(f"  box: {mae_box_tk:.6f} | cls: {mae_cls_tk:.6f} | kpt: {mae_kpt_tk:.6f}")
    print("\n---- MAE: Student Keras  vs Teacher (normalized) ----")
    print(f"  box: {mae_box_st:.6f} | cls: {mae_cls_st:.6f} | kpt: {mae_kpt_st:.6f}")

    print("\n---- Variance across N (first 500 anchors) ----")
    for name, ch in probe_channels.items():
        v_k = var_across_N(out_keras, ch)
        v_t = var_across_N(out_tfl, ch)
        print(f"  ch {name:>8s} | var Keras={v_k:.3e} | var TFLite={v_t:.3e}")


