# ==================== top-of-file shim (keep this at the very top) ====================
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
# =====================================================================================

from tensorflow.keras import layers as L
import tensorflow_model_optimization as tfmot
from pathlib import Path
import numpy as np

import u_8_s_pose_keras_qat as cfg
import config

from importlib import reload
from process import (
    build_dataset, try_load_keras_model,
    rep_data_gen, normalize_teacher_pred
)
from loss import distill_loss_pose

# ---------------------------- alignment knobs (from your search) ----------------------------
# BEST P-order: (0, 1, 2)  -> P3, P4, P5
PORDER = (0, 1, 2)

# BEST grid modes: (('col',1,0), ('row',0,0), ('col',1,0)) for P3, P4, P5
GRID_MODES = (('col', 1, 0), ('row', 0, 0), ('col', 1, 0))

# BEST channel mapping (official i -> your qat j)
# CHANNEL_MAPPING = [
#     0, 32, 50, 44, 33, 11, 1, 3, 25, 26, 4, 35, 14, 22, 53, 52,
#     13, 23, 30, 5, 29, 27, 9, 34, 45, 7, 24, 21, 40, 17, 38, 48,
#     31, 39, 37, 15, 54, 8, 28, 12, 16, 49, 2, 47, 51, 41, 10, 19,
#     18, 6, 55, 20, 46, 36, 42, 43
# ]
CHANNEL_MAPPING = [
    0,1,2,3,4,5,6,7,8,9,
    10,11,12,13,14,15,16,17,18,19,
    20,21,22,23,24,25,26,27,28,29,
    30,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,45,46,47,48,49,
    50,51,52,53,54,55
]


# 如果你的 head 輸出是 xywh（多數情況），開啟這個把它轉成 [l,t,r,b]（以 grid center、stride 為基準）
XYWH_TO_LTRB = False
# 若你的 xywh 是 0~1 範圍（常見：/255 後的輸入），開啟這個把 xywh 乘上 IMGSZ 轉像素
XYWH_IS_NORMALIZED_01 = False
# ------------------------------------------------------------------------------------------------


def enable_gpu_mem_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        lg = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(lg)} Logical GPUs")
    except RuntimeError as e:
        print("TF GPU config error:", e)


def build_student_qat():
    reload(cfg)
    student = cfg.build_u8s_pose(
        input_shape=(config.IMGSZ, config.IMGSZ, 3),
        num_classes=config.NUM_CLS,
        num_kpt=config.NUM_KPT,
        kpt_vals=config.KPT_VALS
    )
    student = tfmot.quantization.keras.quantize_model(student)
    qlayers = [l for l in student.submodules if "Quantize" in l.__class__.__name__]
    print("[CHECK] quantization layers count:", len(qlayers))
    return student


def run_qat(student, teacher, ds, steps_per_epoch):
    opt = tf.keras.optimizers.Adam(1e-4)
    expected_C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS

    @tf.function
    def train_step(batch_imgs):
        with tf.GradientTape() as tape:
            y_t = teacher(batch_imgs, training=False)
            y_t = normalize_teacher_pred(y_t, expected_C=expected_C)   # -> [B,N,C]
            y_s = student(batch_imgs, training=True)
            loss = distill_loss_pose(y_t, y_s, config.NUM_CLS, config.NUM_KPT, config.KPT_VALS)
        grads = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    # dry-run
    it = iter(ds)
    sample = next(it)
    y_tn = normalize_teacher_pred(teacher(sample, training=False), expected_C=expected_C)
    y_sn = (student(sample, training=False))
    print("[CHECK] teacher out (normalized):", y_tn.shape, y_tn.dtype)
    print("[CHECK] student out (probe):     ", y_sn.shape, y_sn.dtype)
    l = float(train_step(sample))
    print(f"[CHECK] dry-run distill_loss={l:.6f}")

    # epochs
    for e in range(config.EPOCHS):
        it = iter(ds)
        for _ in range(steps_per_epoch):
            imgs = next(it)
            loss = train_step(imgs)
        print(f"Epoch {e+1}/{config.EPOCHS}  distill_loss={float(loss):.4f}")


def choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, expected_C):
    # Teacher normalized
    y_te_nc = normalize_teacher_pred(teacher(sample_one, training=False), expected_C=expected_C)
    # Student -> [1,N,C] or transpose
    y_st = student_infer(sample_one, training=False)
    if len(y_st.shape) != 3:
        raise ValueError(f"[EXPORT] Student single output must be rank-3, got {y_st.shape}")
    if int(y_st.shape[1]) == int(y_te_nc.shape[1]):
        y_st_nc = y_st
        single_action = "identity"
    elif int(y_st.shape[2]) == int(y_te_nc.shape[1]):
        y_st_nc = tf.transpose(y_st, [0, 2, 1])
        single_action = "transpose"
    else:
        raise ValueError(f"[EXPORT] Student shape must be (1,N,C) or (1,C,N), got {y_st.shape}")
    print("[ALIGN] student single_action =", single_action)

    # try 6 permutations x (forward/reverse teacher) → pick lowest MAE
    from itertools import permutations
    lens = [N3, N4, N5]
    teacher_orders = {"forward": [N3, N4, N5], "reverse": [N5, N4, N3]}
    y_te_orders = {
        "forward": y_te_nc,
        "reverse": tf.concat([y_te_nc[:, N3+N4:, :], y_te_nc[:, N3:N3+N4, :], y_te_nc[:, :N3, :]], axis=1),
    }

    best_perm, best_order, best_mae = None, None, float("inf")
    for perm in permutations(lens, 3):
        s0, s1, s2 = perm
        split0 = y_st_nc[:, :s0, :]
        split1 = y_st_nc[:, s0:s0+s1, :]
        split2 = y_st_nc[:, s0+s1:s0+s1+s2, :]
        for name, to_order in teacher_orders.items():
            target = []
            for need in to_order:
                target.append(split0 if need == s0 else split1 if need == s1 else split2 if need == s2 else None)
            if None in target:  # pragma: no cover
                continue
            y_st_nc_aligned = tf.concat(target, axis=1)
            mae = tf.reduce_mean(tf.abs(y_st_nc_aligned - y_te_orders[name])).numpy()
            if mae < best_mae:
                best_mae, best_perm, best_order = mae, perm, name

    if best_perm is None:
        raise RuntimeError("[ALIGN] failed to decide student N-order.")
    # convert teacher order (lengths) -> indices for tf.split result
    split_index_by_len = {best_perm[0]: 0, best_perm[1]: 1, best_perm[2]: 2}
    reorder_idx = [split_index_by_len[l] for l in teacher_orders[best_order]]
    print(f"[ALIGN] lens_perm={best_perm}, teacher_order={best_order}, MAE={best_mae:.6e}")
    return best_perm, reorder_idx


# ================================== Export Module ==================================
class ExportModule(tf.Module):
    """Export wrapper to:
       - unify to (B,N,C)
       - reorder P3/P4/P5 grid flatten modes (N)
       - apply channel mapping (C)
       - (optional) convert xywh -> ltrb distances in stride units (match Ultralytics decode)
       - final (1,C,N) with name 'output0'
    """
    def __init__(self, model, C, lens_perm, reorder_idx,
                 grid_modes=GRID_MODES, porder=PORDER, ch_map=CHANNEL_MAPPING,
                 xywh_to_ltrb=XYWH_TO_LTRB, xywh_is_norm01=XYWH_IS_NORMALIZED_01):
        super().__init__()
        self.model = model
        self.C = int(C)
        self.lens_perm = tuple(int(x) for x in lens_perm)       # e.g. (6400,1600,400)
        self.reorder_idx = tuple(int(x) for x in reorder_idx)   # e.g. (0,1,2)
        self.grid_modes = tuple(grid_modes)                     # (('col',1,0), ('row',0,0), ('col',1,0))
        self.porder = tuple(porder)                             # (0,1,2)
        self.ch_map = tf.constant(list(map(int, ch_map)), tf.int32)
        self.xywh_to_ltrb = bool(xywh_to_ltrb)
        self.xywh_is_norm01 = bool(xywh_is_norm01)

    @tf.function(input_signature=[tf.TensorSpec([1, config.IMGSZ, config.IMGSZ, 3], tf.float32, name="images")])
    def serving_fn(self, x):
        y = self.model(x, training=False)   # (B,N,C) or (B,C,N)
        # --- to (B,N,C)
        if y.shape.rank != 3:
            raise ValueError(f"export expects rank=3, got {y.shape}")
        out_nc = tf.transpose(y, [0, 2, 1]) if (y.shape[1] == self.C) else y  # (B,N,C)

        # --- coarse N reorder via lens_perm + reorder_idx (align P3,P4,P5 block order)
        segs = tf.split(out_nc, num_or_size_splits=list(self.lens_perm), axis=1)   # [seg0, seg1, seg2]
        segs = [segs[i] for i in self.reorder_idx]
        out_nc = tf.concat(segs, axis=1)  # (B,N,C) with P3,P4,P5 ordered

        # --- fine N reorder: grid flatten modes for each P level
        # P dims derived from IMGSZ
        H3, W3 = config.IMGSZ // 8,  config.IMGSZ // 8   # 80,80
        H4, W4 = config.IMGSZ // 16, config.IMGSZ // 16  # 40,40
        H5, W5 = config.IMGSZ // 32, config.IMGSZ // 32  # 20,20
        N3, N4, N5 = H3*W3, H4*W4, H5*W5

        def reorder_block_tf(block_BNC, H, W, scan='row', flip_y=False, flip_x=False):
            B = tf.shape(block_BNC)[0]
            C = tf.shape(block_BNC)[2]
            x = tf.reshape(block_BNC, [B, H, W, C])      # [B,H,W,C]
            if scan == 'col':
                x = tf.transpose(x, [0, 2, 1, 3])        # [B,W,H,C]
                H, W = W, H
            if flip_y:
                x = tf.reverse(x, axis=[1])
            if flip_x:
                x = tf.reverse(x, axis=[2])
            return tf.reshape(x, [B, H * W, C])

        # split to p3/p4/p5 in current P-order
        p3, p4, p5 = tf.split(out_nc, [N3, N4, N5], axis=1)

        # apply your best grid modes
        (m3, m4, m5) = self.grid_modes
        p3r = reorder_block_tf(p3, H3, W3, scan=m3[0], flip_y=bool(m3[1]), flip_x=bool(m3[2]))
        p4r = reorder_block_tf(p4, H4, W4, scan=m4[0], flip_y=bool(m4[1]), flip_x=bool(m4[2]))
        p5r = reorder_block_tf(p5, H5, W5, scan=m5[0], flip_y=bool(m5[1]), flip_x=bool(m5[2]))

        # apply PORDER (here it's (0,1,2), i.e., p3,p4,p5)
        preds_list = [p3r, p4r, p5r]
        out_nc = tf.concat([preds_list[i] for i in self.porder], axis=1)  # (B,N,C) final N-order

        # --- C reorder via channel mapping to match official semantic ordering
        out_nc = tf.gather(out_nc, self.ch_map, axis=-1)  # (B,N,C)

        # --- optional: xywh -> ltrb distances in stride units (to match Ultralytics decode)
        if self.xywh_to_ltrb:
            N = tf.shape(out_nc)[1]
            raw_xywh = out_nc[:, :, 0:4]  # (B,N,4)
            x, y_c, w, h = tf.split(raw_xywh, 4, axis=-1)

            # grid centers in pixels using the SAME grid modes/PORDER as above
            def make_grid(hh, ww):
                yy, xx = tf.meshgrid(tf.range(hh), tf.range(ww), indexing='ij')
                return tf.stack([tf.cast(xx, tf.float32), tf.cast(yy, tf.float32)], axis=-1)  # [H,W,2]

            g3 = make_grid(H3, W3); g4 = make_grid(H4, W4); g5 = make_grid(H5, W5)

            def reorder_grid_tf(g, scan='row', flip_y=False, flip_x=False):
                x = g
                if scan == 'col':
                    x = tf.transpose(x, [1, 0, 2])
                if flip_y: x = tf.reverse(x, axis=[0])
                if flip_x: x = tf.reverse(x, axis=[1])
                return x

            g3r = reorder_grid_tf(g3, scan=m3[0], flip_y=bool(m3[1]), flip_x=bool(m3[2]))
            g4r = reorder_grid_tf(g4, scan=m4[0], flip_y=bool(m4[1]), flip_x=bool(m4[2]))
            g5r = reorder_grid_tf(g5, scan=m5[0], flip_y=bool(m5[1]), flip_x=bool(m5[2]))

            g3r = tf.reshape(g3r, [N3, 2])
            g4r = tf.reshape(g4r, [N4, 2])
            g5r = tf.reshape(g5r, [N5, 2])

            # concatenate by PORDER
            grid_feat = tf.concat([ [g3r, g4r, g5r][i] for i in self.porder ], axis=0)  # [N,2]
            # stride per P-level
            stride = tf.concat([
                tf.fill([N3, 1], 8.0), tf.fill([N4, 1], 16.0), tf.fill([N5, 1], 32.0)
            ], axis=0)
            # reorder stride to match PORDER
            stride = tf.concat([ [tf.fill([N3,1],8.0), tf.fill([N4,1],16.0), tf.fill([N5,1],32.0)][i] for i in self.porder ], axis=0)

            grid_center_xy_px = (grid_feat + 0.5) * stride  # [N,2] in pixels
            grid_center_xy_px = tf.reshape(grid_center_xy_px, [1, -1, 2])  # [1,N,2]
            gcx, gcy = grid_center_xy_px[:, :, 0:1], grid_center_xy_px[:, :, 1:2]

            # scale xywh to pixels if normalized
            if self.xywh_is_norm01:
                s = tf.cast(config.IMGSZ, raw_xywh.dtype)
                x   = x   * s
                y_c = y_c * s
                w   = w   * s
                h   = h   * s

            # xywh -> [x_min, y_min, x_max, y_max]
            x_min = x - 0.5 * w
            x_max = x + 0.5 * w
            y_min = y_c - 0.5 * h
            y_max = y_c + 0.5 * h

            # convert to distances (ltrb) normalized by stride (>=0)
            l = tf.maximum((gcx - x_min) / stride, 0.0)
            t = tf.maximum((gcy - y_min) / stride, 0.0)
            r = tf.maximum((x_max - gcx) / stride, 0.0)
            b = tf.maximum((y_max - gcy) / stride, 0.0)
            box_ltrb = tf.concat([l, t, r, b], axis=-1)  # (B,N,4)

            # replace first 4 channels with ltrb distances
            out_nc = tf.concat([box_ltrb, out_nc[:, :, 4:]], axis=-1)

        raw_xywh = out_nc[:, :, 0:4]
        xy = tf.math.sigmoid(raw_xywh[:, :, 0:2])
        wh = tf.math.sigmoid(raw_xywh[:, :, 2:4])
        out_nc = tf.concat([tf.concat([xy, wh], axis=-1), out_nc[:, :, 4:]], axis=-1)

        # --- activations for objectness / cls / keypoint v
        C = self.C
        nc = int(config.NUM_CLS)
        kdim = int(config.KPT_VALS)
        nk = (C - 4 - nc) // kdim

        # 先取出 logits
        cls_logit = out_nc[:, :, 4:4+nc]        # <-- class 緊接在 xywh 後
        kpt      = out_nc[:, :, 4+nc:4+nc+nk*kdim]

        # 從舊 teacher 的語義推斷：class 應該是已 gated 的 conf
        # 若你仍有 obj logit，可在 head 內部先算 conf = sigmoid(obj)*sigmoid(raw_cls)
        # 但匯出這層只保留 class 機率，不保留獨立 obj 通道
        cls = tf.math.sigmoid(cls_logit)

        # keypoint 只對 v 走 sigmoid
        N_dyn = tf.shape(out_nc)[1]
        kpt4 = tf.reshape(kpt, [-1, N_dyn, nk, kdim])
        xy = kpt4[..., :2]
        v  = tf.math.sigmoid(kpt4[..., 2:3])
        kpt = tf.reshape(tf.concat([xy, v], axis=-1), [-1, N_dyn, nk*kdim])

        # 產生和舊模型一致的輸出：[xywh, cls, kpt]
        pred_ultra = tf.concat([out_nc[:, :, 0:4], cls, kpt], axis=-1)  # (B,N, 4+nc+nk*kdim)
        out_cn = tf.transpose(pred_ultra, [0, 2, 1])                    # (B,C,N)
        return {"output0": tf.reshape(out_cn, [1, C, -1])}


def main():
    enable_gpu_mem_growth()

    # 1) Teacher
    base_model, _ = try_load_keras_model(config.EXPORTED_DIR)
    teacher = base_model
    teacher.trainable = False

    # 2) Student + QAT
    student = build_student_qat()

    # 3) Data
    ds, n_files = build_dataset(img_glob=config.REP_DIR, batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    # 4) QAT fine-tune
    run_qat(student, teacher, ds, steps_per_epoch)

    # 5) Shapes
    N3 = (config.IMGSZ // 8)  * (config.IMGSZ // 8)   # 80*80
    N4 = (config.IMGSZ // 16) * (config.IMGSZ // 16)  # 40*40
    N5 = (config.IMGSZ // 32) * (config.IMGSZ // 32)  # 20*20
    N = N3 + N4 + N5
    C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
    print("Expected N, C:", N, C)

    # 6) Strip QAT wrappers
    if hasattr(tfmot.quantization.keras, "strip_quantization"):
        print("[INFO] strip_quantization()")
        student_infer = tfmot.quantization.keras.strip_quantization(student)
    else:
        print("[WARN] strip_quantization not found; exporting wrapped model.")
        student_infer = student

    # 7) Auto-derive coarse P-order split (still keep your robust logic)
    try:
        sample_one = next(iter(ds))[:1]
    except Exception:
        sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

    expected_C = C
    lens_perm, reorder_idx = choose_student_split_order(
        student_infer, teacher, sample_one, N3, N4, N5, expected_C
    )

    # 8) Export SavedModel (float I/O, fixed batch=1)
    export_mod = ExportModule(
        student_infer, C=C, lens_perm=lens_perm, reorder_idx=reorder_idx,
        grid_modes=GRID_MODES, porder=PORDER, ch_map=CHANNEL_MAPPING,
        xywh_to_ltrb=XYWH_TO_LTRB, xywh_is_norm01=XYWH_IS_NORMALIZED_01
    )

    SAVE_DIR = str(Path(config.TFLITE_OUT) / "qat_saved_model_fixed_fpIO")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    concrete_fn = export_mod.serving_fn.get_concrete_function()
    tf.saved_model.save(export_mod, SAVE_DIR, signatures=concrete_fn)
    print("Saved SavedModel →", SAVE_DIR)

    # quick signature check
    loaded = tf.saved_model.load(SAVE_DIR)
    sig = loaded.signatures["serving_default"]
    print("Signature inputs :", sig.structured_input_signature)
    print("Signature outputs:", sig.structured_outputs)

    # 9) Convert TFLite: float I/O + INT8 internals
    conv = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    conv.experimental_new_converter = True
    try:
        conv.experimental_new_quantizer = True   # 也可以試 False，選 QUANTIZE 最少的版本
    except Exception:
        pass

    tfl_bytes = conv.convert()
    out_path = str(Path(config.TFLITE_OUT) / "best_qat_int8.tflite")
    Path(out_path).write_bytes(tfl_bytes)
    print("Wrote", out_path)

    # 10) Inspect TFLite I/O
    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    print("TFLite inputs :", interp.get_input_details())
    print("TFLite outputs:", interp.get_output_details())


if __name__ == "__main__":
    main()
