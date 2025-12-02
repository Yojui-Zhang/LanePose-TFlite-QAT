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
import time
import cv2
from datetime import datetime
from pathlib import Path
import numpy as np 

import tensorflow_model_optimization as tfmot

'''
===================================================
Local imports from your project
===================================================
'''
import config

from src.process.data import (build_dataset)
from src.process.load_model import try_load_keras_model
from src.process.interrupt_signal import install_interrupt_handlers
from src.process.device import (enable_gpu_mem_growth, setup_mixed_precision)
from src.process.Train_Model import (build_student_qat, run_qat, choose_student_split_order, 
                                     assert_kd_path_not_quantized, probe_kd_output_distribution,
                                     _ensure_bhwc4)

from src.process.Export_Model import (ExportModule, run_diagnostics_once,export_only, 
                                      create_and_configure_tflite_converter)

from src.process.pred_model import (ensure_BNC_static, make_ultra_infer_model)

if config.PLOT_Switch == True:
    from src.process.Plot_Data import plot_and_save_loss_curve
    
NUM_CLS  = config.NUM_CLS
NUM_KPT  = config.NUM_KPT
KPT_VALS = config.KPT_VALS

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
def _to_uint8_img(bhwc, num):
    # 取 batch 第 1 張，前 3 通道（RGB）
    img = bhwc[num, ..., :3].numpy() if hasattr(bhwc, 'numpy') else bhwc[num, ..., :3]
    img = np.asarray(img)
    # 轉 uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.5:  # 0~1 規模
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _apply_sigmoid_for_pose(y_row):
    """對一行 (C,) 套用 sigmoid：box(0:4), cls(4:4+NUM_CLS), kpt(xy,v)"""
    z = y_row.copy()
    z[0:4] = 1 / (1 + np.exp(-z[0:4]))  # box xywh
    if NUM_CLS > 0:
        z[4:4+NUM_CLS] = 1 / (1 + np.exp(-z[4:4+NUM_CLS]))
    kpt = z[4+NUM_CLS:].reshape(NUM_KPT, KPT_VALS)
    kpt[:, :2] = 1 / (1 + np.exp(-kpt[:, :2]))  # x,y
    if KPT_VALS >= 3:
        kpt[:, 2:3] = 1 / (1 + np.exp(-kpt[:, 2:3]))  # v
    z[4+NUM_CLS:] = kpt.reshape(-1)
    return z

def _pick_best_and_draw(img_bgr, y_BNC, is_logits=False, out_path=None, title_text=None):
    """
    img_bgr: (H,W,3) uint8
    y_BNC:   (N,C) numpy
    is_logits: 若為 True 先做 sigmoid 映到 0~1
    """
    H, W = img_bgr.shape[:2]
    arr = y_BNC.numpy() if hasattr(y_BNC, 'numpy') else y_BNC
    arr = np.asarray(arr)

    # 先把所有候選轉換到 unit 域便於打分
    if is_logits:
        arr_unit = np.stack([_apply_sigmoid_for_pose(r) for r in arr], axis=0)
    else:
        arr_unit = arr.copy()
    # 計分：用 keypoint v 的平均值（若無 v 就用框面積當作備援）
    kpt = arr_unit[:, 4+NUM_CLS:].reshape(-1, NUM_KPT, KPT_VALS)
    if KPT_VALS >= 3:
        v = kpt[:, :, 2]
        scores = v.mean(axis=1)
    else:
        xywh = arr_unit[:, 0:4]
        scores = (xywh[:, 2] * xywh[:, 3])  # 寬×高

    best = int(scores.argmax())
    det  = arr_unit[best]  # 已在 unit 域

    # 取 box
    x, y, w, h = det[0:4]
    x1 = int((x - w/2) * W); y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W); y2 = int((y + h/2) * H)

    # 畫圖
    canvas = img_bgr.copy()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 20), 2)
    # 取 kpts
    kpt_flat = det[4+NUM_CLS:].reshape(NUM_KPT, KPT_VALS)
    for i in range(NUM_KPT):
        kx = int(kpt_flat[i, 0] * W)
        ky = int(kpt_flat[i, 1] * H)
        kv = float(kpt_flat[i, 2] if KPT_VALS >= 3 else 1.0)
        color = (40, 220, 40) if kv >= 0.5 else (40, 40, 220)
        cv2.circle(canvas, (kx, ky), 3, color, -1)

    if title_text:
        cv2.putText(canvas, title_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    if out_path is not None:
        cv2.imwrite(str(out_path), canvas)
    return canvas

def _iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _apply_sigmoid_for_pose(vec):
    # 你原本有的：把 logits 映到 0~1
    return 1.0 / (1.0 + np.exp(-vec))

def _pick_and_draw_multi(
    img_bgr, y_BNC, is_logits=False,
    det_score_thr=0.5, top_k=None,
    do_nms=True, iou_thr=0.5,
    kpt_vis_thr=0.5,
    out_path=None, title_text=None
):
    """
    依門檻畫出多個物件與關鍵點。
    img_bgr: (H,W,3) uint8
    y_BNC:   (N,C) numpy / tensor；排版為 [x,y,w,h, NUM_CLS, kpts...]
    is_logits: 若為 True 先做 sigmoid
    det_score_thr: 以關鍵點 v 平均(或面積)為分數，≥此值才保留
    top_k: 最多保留幾個（None 表示不限制）
    do_nms: 是否進行簡單 NMS
    iou_thr: NMS IoU 門檻
    kpt_vis_thr: 關鍵點著色分界（≥明亮綠；< 深藍）
    out_path/title_text: 與你原本相同
    回傳: canvas, kept_indices
    """
    H, W = img_bgr.shape[:2]
    arr = y_BNC.numpy() if hasattr(y_BNC, 'numpy') else y_BNC
    arr = np.asarray(arr)

    if arr.ndim != 2 or arr.shape[0] == 0:
        # 沒偵測到任何候選
        canvas = img_bgr.copy()
        if title_text:
            cv2.putText(canvas, title_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        if out_path is not None:
            cv2.imwrite(str(out_path), canvas)
        return canvas, []

    # logits → unit
    if is_logits:
        arr_unit = np.stack([_apply_sigmoid_for_pose(r) for r in arr], axis=0)
    else:
        arr_unit = arr.copy()

    # 你原本的結構：前4為 xywh，其後 +NUM_CLS，再往後是 kpts
    # 需已定義以下全域常數：NUM_CLS, NUM_KPT, KPT_VALS
    kpt = arr_unit[:, 4+NUM_CLS:].reshape(-1, NUM_KPT, KPT_VALS)

    # 分數：有 v 用 v 平均；否則用框面積
    if KPT_VALS >= 3:
        v = kpt[:, :, 2]
        scores = v.mean(axis=1)  # (N,)
    else:
        xywh = arr_unit[:, 0:4]
        scores = (xywh[:, 2] * xywh[:, 3])  # 寬×高，(N,)

    # 先以門檻過濾
    keep = np.where(scores >= float(det_score_thr))[0]
    if keep.size == 0:
        # 若無通過門檻，則退而求其次：挑 top-1
        keep = np.array([int(scores.argmax())], dtype=int)

    # 依分數高到低排序
    keep = keep[np.argsort(scores[keep])[::-1]]

    # top_k 限制
    if top_k is not None and top_k > 0:
        keep = keep[:top_k]

    # 先把對應框轉為像素座標，後面 NMS / 繪製共用
    boxes_xyxy = []
    for idx in keep:
        det = arr_unit[idx]
        x, y, w, h = det[0:4]
        x1 = int(np.clip((x - w/2) * W, 0, W-1))
        y1 = int(np.clip((y - h/2) * H, 0, H-1))
        x2 = int(np.clip((x + w/2) * W, 0, W-1))
        y2 = int(np.clip((y + h/2) * H, 0, H-1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        boxes_xyxy.append([x1, y1, x2, y2])

    # 簡單 NMS（以門檻後、分數排序過的 keep 為順序）
    if do_nms and len(keep) > 1:
        final_keep = []
        for i, idx in enumerate(keep):
            bi = boxes_xyxy[i]
            suppr = False
            for j in final_keep:
                # final_keep 存的是 keep 的索引位置，所以取其對應 box
                bj = boxes_xyxy[keep.tolist().index(j)]
                if _iou_xyxy(bi, bj) > iou_thr:
                    suppr = True
                    break
            if not suppr:
                final_keep.append(idx)
        keep = np.array(final_keep, dtype=int)

    # 繪製
    canvas = img_bgr.copy()
    palette = [
        (60,220,20), (20,180,255), (255,160,20),
        (220,80,200), (60,140,240), (140,220,140),
        (200,200,60), (240,120,120), (160,100,255), (120,220,220)
    ]

    kept_indices = []
    for n, idx in enumerate(keep):
        det = arr_unit[idx]
        # box
        x, y, w, h = det[0:4]
        x1 = int(np.clip((x - w/2) * W, 0, W-1))
        y1 = int(np.clip((y - h/2) * H, 0, H-1))
        x2 = int(np.clip((x + w/2) * W, 0, W-1))
        y2 = int(np.clip((y + h/2) * H, 0, H-1))

        color_box = palette[n % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_box, 2)

        # 類別（若需要）
        if NUM_CLS > 0:
            cls_probs = det[4:4+NUM_CLS]
            cls_id = int(np.argmax(cls_probs))
            cls_sc = float(cls_probs[cls_id])
            cv2.putText(canvas, f"id{idx} c{cls_id}:{cls_sc:.2f}",
                        (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"id{idx} s:{scores[idx]:.2f}",
                        (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2, cv2.LINE_AA)

        # kpts
        kpt_flat = det[4+NUM_CLS:].reshape(NUM_KPT, KPT_VALS)
        for i in range(NUM_KPT):
            kx = int(np.clip(kpt_flat[i, 0] * W, 0, W-1))
            ky = int(np.clip(kpt_flat[i, 1] * H, 0, H-1))
            kv = float(kpt_flat[i, 2] if KPT_VALS >= 3 else 1.0)
            color_k = (40, 220, 40) if kv >= kpt_vis_thr else (40, 40, 220)
            cv2.circle(canvas, (kx, ky), 3, color_k, -1)

        kept_indices.append(int(idx))

    if title_text:
        cv2.putText(canvas, title_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    if out_path is not None:
        cv2.imwrite(str(out_path), canvas)

    return canvas, kept_indices
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def save_model_outputs_to_txt(teacher, student, ds, C, output_paths):
    """
    執行一次推論，並將 teacher 和 student 的輸出結果儲存為 .txt 檔案。
    """
    print("\n--- Saving model outputs to .txt files ---")
    
    # 1) 準備樣本
    sample_batch = next(iter(ds))
    sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
    sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)

    # 2) 執行推論
    y_t_raw = teacher(sample_one, training=False)
    y_s_out = student(sample_one, training=False)

    # 3) 將輸出統一到 (B,N,C) 格式
    y_t_BNC = ensure_BNC_static(y_t_raw, C)
    kd_raw = y_s_out[1] if isinstance(y_s_out, (list, tuple)) else y_s_out
    kd_BNC = ensure_BNC_static(kd_raw, C)
    
    dep_raw = y_s_out[0] if isinstance(y_s_out, (list, tuple)) else None
    dep_BNC = None
    if dep_raw is not None:
        dep_BNC = ensure_BNC_static(dep_raw, C)

    # 4) 攤平成 (B*N, C) 並存檔
    teacher_flat = tf.reshape(y_t_BNC, [-1, C]).numpy()
    student_kd_flat = tf.reshape(kd_BNC, [-1, C]).numpy()
    np.savetxt(output_paths['logs'] / 'teacher_output.txt', teacher_flat, fmt='%.4f', delimiter=',')
    np.savetxt(output_paths['logs'] / 'student_kd_output.txt', student_kd_flat, fmt='%.4f', delimiter=',')
    print(f"[txt] Saved: teacher_output.txt, student_kd_output.txt")

    if dep_BNC is not None:
        student_dep_flat = tf.reshape(dep_BNC, [-1, C]).numpy()
        np.savetxt(output_paths['logs'] / 'student_deploy_output.txt', student_dep_flat, fmt='%.4f', delimiter=',')
        print(f"[txt] Saved: student_deploy_output.txt")
        
    # 返回推論結果，給下一個函式使用
    return sample_one, y_t_BNC, kd_BNC, dep_BNC

def save_visualization_results_to_png(sample_one, y_t_BNC, kd_BNC, dep_BNC, output_paths, images_num, conf_thr):
    """
    將模型推論結果繪製到樣本圖片上，並儲存為 .png 檔案。
    此函式能安全地處理不同大小的 batch。
    """
    print("\n--- Saving visualization results to .png files ---")
    plots_dir = output_paths['plots']
    os.makedirs(plots_dir, exist_ok=True)
    
    # 取得實際的 batch size，並決定要處理幾張圖 (最多3張)
    num_images_to_process = sample_one.shape[0]
    
    if num_images_to_process == 0:
        print("[viz] No images in the batch to visualize.")
        return

    for i in range(num_images_to_process):
        # 1) 安全地取出影像
        img = _to_uint8_img(sample_one, i)

        # 2) 取出對應的推論結果 (N, C)
        t_NC = y_t_BNC[i].numpy() if hasattr(y_t_BNC, 'numpy') else y_t_BNC[i]
        kd_NC = kd_BNC[i].numpy() if hasattr(kd_BNC, 'numpy') else kd_BNC[i]
        
        dep_NC = None
        if dep_BNC is not None:
            dep_NC = dep_BNC[i].numpy() if hasattr(dep_BNC, 'numpy') else dep_BNC[i]
            
        # 3) 設定儲存路徑
        path_teacher = plots_dir / f"result{images_num}_teacher.png"
        path_student_kd = plots_dir / f"result{images_num}_student_kd.png"
        path_student_deploy = plots_dir / f"result{images_num}_student_deploy.png"

        # 4) 繪圖並儲存
        # _ = _pick_best_and_draw(img, t_NC, is_logits=False, out_path=path_teacher, title_text="Teacher")
        # _ = _pick_best_and_draw(img, kd_NC, is_logits=True, out_path=path_student_kd, title_text="Student KD")

        _, _ = _pick_and_draw_multi(
            img,
            t_NC,                 # (N,C)
            is_logits=False,       # 若 y_BNC 是後處理前的 logits 設 True
            det_score_thr=conf_thr,     # 分數門檻
            top_k=10,              # 最多保留 10 個；不設即不限
            do_nms=True,           # 開 NMS
            iou_thr=0.5,           # NMS IoU 門檻
            kpt_vis_thr=0.5,       # 關鍵點顏色門檻
            out_path=path_teacher,
            title_text="Multi detections"
        )

        _, _ = _pick_and_draw_multi(
            img,
            kd_NC,                 # (N,C)
            is_logits=False,       # 若 y_BNC 是後處理前的 logits 設 True
            det_score_thr=conf_thr,     # 分數門檻
            top_k=10,              # 最多保留 10 個；不設即不限
            do_nms=True,           # 開 NMS
            iou_thr=0.5,           # NMS IoU 門檻
            kpt_vis_thr=0.5,       # 關鍵點顏色門檻
            out_path=path_student_kd,
            title_text="Multi detections"
        )
        
        if dep_NC is not None:
            _, _ = _pick_and_draw_multi(
                img,
                dep_NC,                 # (N,C)
                is_logits=False,       # 若 y_BNC 是後處理前的 logits 設 True
                det_score_thr=conf_thr,     # 分數門檻
                top_k=10,              # 最多保留 10 個；不設即不限
                do_nms=True,           # 開 NMS
                iou_thr=0.5,           # NMS IoU 門檻
                kpt_vis_thr=0.5,       # 關鍵點顏色門檻
                out_path=path_student_deploy,
                title_text="Multi detections"
            )

        saved_paths = f"{path_teacher.name}, {path_student_kd.name}, {path_student_deploy.name}"
        
        if dep_NC is not None:
            _ = _pick_best_and_draw(img, dep_NC, is_logits=False, out_path=path_student_deploy, title_text="Student Deploy")
            saved_paths += f", {path_student_deploy.name}"
            
        print(f"[viz] Saved for image {images_num}: {saved_paths}")

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def main():
    # 0) 初始化設定
    start_time = time.time()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.TFLITE_OUT) / run_timestamp
    output_paths = {
        'logs': output_dir / 'logs',
        'plots': output_dir / 'plots',
        'models': output_dir / 'models',
    }
    for p in output_paths.values():
        p.mkdir(parents=True, exist_ok=True)
    
    output_paths['log_csv'] = output_paths['logs'] / 'training_log.csv'
    output_paths['lr_plot'] = output_paths['plots'] / 'learning_rate_schedule.png'
    output_paths['loss_plot'] = output_paths['plots'] / 'loss_curve.png'
    
    print(f"\n--- QAT Script Started at {run_timestamp} ---")
    print(f"\n--- All outputs will be saved in: {output_dir} ---")
    
    install_interrupt_handlers()

    enable_gpu_mem_growth()
    setup_mixed_precision()

    # 1) 載入教師模型
    print("\n--- Loading Teacher Model ---")
    teacher, _ = try_load_keras_model(config.EXPORTED_DIR)
    teacher.trainable = False
    print("✅ Teacher model loaded and frozen.")

    # 2) 建立學生模型
    print("\n--- Building Student Model ---")
    student = build_student_qat()
    assert_kd_path_not_quantized(student)
    student.summary(line_length=120)

    # 3) 準備資料集
    print("\n--- Preparing Dataset ---")
    ds, n_files = build_dataset(img_glob=config.REP_DIR_train, batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    try:
        if getattr(config, "EXPORT_ONLY", False):
            print("\n=== EXPORT_ONLY: skip training, use current/loaded weights ===")
            export_only(student, teacher, ds, output_paths, tag="export_only")
            end_time = time.time()
            print(f"\n--- 🎉 Done (EXPORT_ONLY) in {((end_time - start_time) / 60):.2f} minutes. ---")
            return
        else:
            loss_history = run_qat(student, teacher, ds, steps_per_epoch, output_paths)

            if getattr(config, "PLOT_Switch", False):
                plot_and_save_loss_curve(loss_history, output_paths['loss_plot'])

    except KeyboardInterrupt:
        print("\n[⚠️ Interrupt] KeyboardInterrupt caught. Will export current weights...\n")
    finally:
        
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ ERROR ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # # if config.TRAIN_SUPERVISION == 'label':
        
        # student = make_ultra_infer_model(student, branch='kd')   # 或 'deploy' 視你需要

        # # 計算 C 值
        # C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
        # conf_thr = 0.4

        # for i in range(3):
        #     # 執行推論並儲存 .txt 檔案
        #     sample_one, y_t_BNC, kd_BNC, dep_BNC = save_model_outputs_to_txt(
        #         teacher, student, ds, C, output_paths
        #     )

        #     # 繪製視覺化結果並儲存 .png 檔案
        #     save_visualization_results_to_png(
        #         sample_one, y_t_BNC, kd_BNC, dep_BNC, output_paths, i, conf_thr
        #     )
        
        
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        
        if not getattr(config, "EXPORT_ONLY", False):
            # 5) 導出前準備（你原本的第 5～9 步）
            print("\n--- Preparing for Export ---")
            N3 = (config.IMGSZ // 8)  ** 2
            N4 = (config.IMGSZ // 16) ** 2
            N5 = (config.IMGSZ // 32) ** 2
            C  = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
            print(f"\nExpected N={N3+N4+N5}, C={C}")

# =================================================================================================================
            # if hasattr(tfmot.quantization.keras, "strip_quantization"):
            #     print("\n[INFO] Stripping quantization wrappers from the KD-only submodel.")
            #     student_kd = tf.keras.Model(student.input, student.outputs[1], name="student_kd_only")  # 取 KD
            #     student_infer = tfmot.quantization.keras.strip_quantization(student_kd)
            # else:
            #     print("\n[WARN] `strip_quantization` not found; exporting wrapped KD-only submodel.")
            #     student_infer = tf.keras.Model(student.input, student.outputs[1], name="student_kd_only")
# =================================================================================================================

            # if hasattr(tfmot.quantization.keras, "strip_quantization"):
            #     print("\n[INFO] Stripping quantization wrappers from the model.")
            #     # 只取 deploy 分支輸出（index 0）
            #     student_deploy = tf.keras.Model(student.input, student.outputs[0], name="student_deploy_only")
            #     student_infer = tfmot.quantization.keras.strip_quantization(student_deploy)
            # else:
            #     print("\n[WARN] `strip_quantization` not found; exporting wrapped model.")
            #     student_infer = tf.keras.Model(student.input, student.outputs[0], name="student_deploy_only")
            
            student_infer = tf.keras.Model(student.input, student.outputs[0], name="student_deploy_only")

# =================================================================================================================

            # 6) 自動對齊輸出順序
#             try:
#                 sample_batch = next(iter(ds))
#                 sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
#                 sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)
#             except Exception:
#                 sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

#             lens_perm, reorder_idx = choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, C, 
#                                                                 config.NUM_CLS, config.NUM_KPT, config.KPT_VALS,)
            
            # 7) 導出 SavedModel
            print("\n--- Exporting SavedModel ---")
            export_mod = ExportModule( student_infer, C=C, apply_chmap=False, ch_map=None, apply_sigmoid_cls=False, apply_sigmoid_kptv=False )
            
            saved_model_path = str(output_paths['models'] / ("qat_saved_model_interrupted" if config.STOP_REQUESTED else "qat_saved_model"))
            concrete_fn = export_mod.serving_fn.get_concrete_function()
            tf.saved_model.save(export_mod, saved_model_path, signatures=concrete_fn)
            print(f"\n✅ SavedModel exported to → {saved_model_path}")

            print("\n--- Converting to TFLite INT8 ---")
            conv = create_and_configure_tflite_converter(saved_model_path)

            tfl_bytes   = conv.convert()

            if config.TFLITE_QUANT_MODE == 'fp32':
                tflite_path = str(output_paths['models'] / ("best_qat_FP32_interrupted.tflite" if config.STOP_REQUESTED else "best_qat_FP32.tflite"))
            elif config.TFLITE_QUANT_MODE == 'fp16':
                tflite_path = str(output_paths['models'] / ("best_qat_FP16_interrupted.tflite" if config.STOP_REQUESTED else "best_qat_FP16.tflite"))
            elif config.TFLITE_QUANT_MODE == 'int8':
                tflite_path = str(output_paths['models'] / ("best_qat_int8_interrupted.tflite" if config.STOP_REQUESTED else "best_qat_int8.tflite"))
            else:
                tflite_path = str(output_paths['models'] / ("best_qat_unknow_interrupted.tflite" if config.STOP_REQUESTED else "best_qat_unknow.tflite"))

            Path(tflite_path).write_bytes(tfl_bytes)
            print(f"\n✅ TFLite model written to → {tflite_path}")

            # 9) 檢查 TFLite I/O
            interp = tf.lite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            print("\n TFLite inputs:", interp.get_input_details())
            print("\n TFLite outputs:", interp.get_output_details())
                
            # === One-shot diagnostics ===
            print("\n--- Running one-shot diagnostics ---")
            run_diagnostics_once(
                export_mod=export_mod,
                teacher=teacher,
                tflite_path=tflite_path,
                sample_one=sample_one,     # 與部署端同一張預處理影像
                C=C,
                NUM_CLS=config.NUM_CLS,
                NUM_KPT=config.NUM_KPT,
                KPT_VALS=config.KPT_VALS,
            )
            
    # 10) 完成
    end_time = time.time()
    print(f"\n--- 🎉 All tasks completed in {((end_time - start_time) / 60):.2f} minutes. ---")

if __name__ == "__main__":
    main()