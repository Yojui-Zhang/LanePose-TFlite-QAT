# qat_tf/qat_distill.py
import glob
import tensorflow as tf
import numpy as np
import cv2
import os
import re

from src.process.labels_yolo_pose_tf import parse_label_lines
import config

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
前處理（與 Ultralytics 部署一致）
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

AUTOTUNE = tf.data.AUTOTUNE

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

def _decode_path(p):
    if isinstance(p, bytes):
        return p.decode('utf-8')
    return str(p)

def parse_img(path):

    p = _decode_path(path)
    bgr = cv2.imread(p)
    
    #bgr to rgb
    rgb = bgr[:, :, ::-1]
    img = letterbox(rgb, config.IMGSZ).astype(np.float32) / 255.0
    return img

def tf_parse(path):
    img = tf.numpy_function(parse_img, [path], Tout=tf.float32)
    img.set_shape([config.IMGSZ, config.IMGSZ, 3])
    return img


def tf_parse_load(img_path):
    # 保險：確保是 scalar tf.string
    img_path = tf.ensure_shape(img_path, [])
    tf.debugging.assert_type(img_path, tf.string,
                                message="img_path must be tf.string. Check build_dataset inputs.")
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (config.IMGSZ, config.IMGSZ))

    # ../images/xxx.jpg -> ../labels/xxx.txt
    lbl_path = tf.strings.regex_replace(img_path, r"/images/", "/labels/")
    lbl_path = tf.strings.regex_replace(lbl_path, r"\.(jpg|jpeg|png|bmp)$", ".txt")

    return img, lbl_path  # (H,W,C), scalar string


def build_dataset(img_glob, batch=config.BATCH, shuffle=True, repeat=True):
    
    # 初始化一個空列表來存放所有檔案路徑
    all_files = []
    
    # 判斷輸入是單一路徑字串還是多路徑列表
    if isinstance(img_glob, str):
        # 如果是單一字串，像以前一樣處理
        patterns = [img_glob]
    else:
        # 如果是列表或元組，直接使用
        patterns = img_glob
        
    # 遍歷所有路徑模式，並將找到的檔案加入總列表
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
        
    # 使用 set 去除重複的路徑，然後排序以保持一致性
    files = sorted(list(set(all_files)))

    if len(files) == 0:
        # 更新錯誤訊息以反映可能的多路徑輸入
        raise FileNotFoundError(f"No images found for patterns: {img_glob}")
    else:
        # 顯示讀取到的總圖片數量
        print(f"\nRead image: {len(files)} from {len(patterns)} directories")

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True) # 加上 reshuffle_each_iteration
        
    ds = ds.map(tf_parse_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
        
    return ds, len(files)


def compute_class_weights(img_glob, num_classes, num_kpt, kpt_vals):
    """
    掃描所有 label 檔，依 cls 出現次數計算 class weights。

    回傳：np.ndarray, shape = (num_classes,)
    """
    # 1. 展開所有圖片路徑（跟 build_dataset 一樣的寫法）
    if isinstance(img_glob, str):
        patterns = [img_glob]
    else:
        patterns = list(img_glob)

    all_files = []
    for g in patterns:
        all_files.extend(glob.glob(g))
    files = sorted(list(set(all_files)))

    cls_counts = np.zeros(num_classes, dtype=np.int64)

    for img_path in files:
        # 2. 由 image path 推 label path
        #    /images/xxx.jpg -> /labels/xxx.txt（跟 tf_parse_load 一致）
        label_path = re.sub(r"/images/", "/labels/", img_path)
        label_path = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".txt",
                            label_path, flags=re.IGNORECASE)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            continue

        # 3. 用你原本的 parse_label_lines 解析
        arr = parse_label_lines(lines, num_kpt=num_kpt, kpt_vals=kpt_vals)
        # arr shape: (M, 5 + K*V)；第 0 欄是 cls
        if arr.size == 0:
            continue

        cls_ids = arr[:, 0].astype(np.int64)
        for c in cls_ids:
            if 0 <= c < num_classes:
                cls_counts[c] += 1

    # 4. 避免有類別完全沒出現，防止除以 0
    cls_counts_safe = cls_counts.copy()
    cls_counts_safe[cls_counts_safe == 0] = 1

    # 5. 計算權重：出現少的 → 權重大
    #    weight_c ∝ max_count / count_c
    max_count = float(cls_counts_safe.max()) if cls_counts_safe.max() > 0 else 1.0
    raw_weights = max_count / cls_counts_safe.astype(np.float32)

    # 6. 正規化一下，讓平均 weight ≈ 1，比較穩定
    mean_w = float(raw_weights.mean()) if raw_weights.mean() > 0 else 1.0
    class_weights = raw_weights / mean_w

    print("[class balance] counts =", cls_counts)
    print("[class balance] weights =", class_weights)

    return class_weights.astype(np.float32)


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
代表集 generator（ 轉 TFLite）
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

def rep_data_gen():
    paths = sorted(glob.glob(config.REP_DIR_export))     # ← 這行改了

    num_picture = 0
    for p in paths:
        img = parse_img(p)                        # float32 [H,W,3] /255
        img = np.expand_dims(img, 0).astype(np.float32)
        yield [img]
        num_picture += 1

    print(f"\n\nRead the data = {num_picture}\n")