# qat_tf/qat_distill.py
import glob
import tensorflow as tf
import numpy as np
import os
import re

from src.process.labels_yolo_pose_tf import parse_label_lines
from src.process.preprocess_tf import decode_and_letterbox
from src.process.labels_yolo_pose_tf import parse_label_file_tf 
import config

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
前處理（與 Ultralytics 部署一致）
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

AUTOTUNE = tf.data.AUTOTUNE

@tf.function
def img_to_label_path_tf(img_path: tf.Tensor) -> tf.Tensor:
    # 1) folder: /images/ -> /labels/
    lbl_path = tf.strings.regex_replace(img_path, r"/images/", "/labels/")
    # 2) ext: .anything -> .txt  (支援 .jpg/.JPG/.png...，不需要列舉)
    lbl_path = tf.strings.regex_replace(lbl_path, r"\.[^/.]+$", ".txt")

    # 防呆：確認結尾是 .txt（如果不是，直接讓你早期爆掉，避免默默讀錯）
    n = tf.strings.length(lbl_path)
    suffix = tf.strings.substr(lbl_path, tf.maximum(n - 4, 0), 4)
    tf.debugging.assert_equal(suffix, ".txt", message="[path] lbl_path is not .txt, check regex!")
    return lbl_path

def _decode_path(p):
    if isinstance(p, bytes):
        return p.decode('utf-8')
    return str(p)



def tf_parse_load(img_path):
    img_path = tf.ensure_shape(img_path, [])
    tf.debugging.assert_type(img_path, tf.string,
                             message="img_path must be tf.string. Check build_dataset inputs.")

    img, meta = decode_and_letterbox(
        img_path,
        new_size=config.IMGSZ,
        pad_value=getattr(config, "LETTERBOX_PAD_VALUE", 114.0/255.0),
        scaleup=True
    )

    # ../images/xxx.jpg -> ../labels/xxx.txt
    lbl_path = img_to_label_path_tf(img_path)

    # 回傳 meta：給訓練時把 label 做 letterbox 座標映射
    return img, lbl_path, meta

def tf_parse_load_with_labels(img_path):
    img_path = tf.ensure_shape(img_path, [])
    tf.debugging.assert_type(img_path, tf.string,
                             message="img_path must be tf.string. Check build_dataset inputs.")

    img, meta = decode_and_letterbox(
        img_path,
        new_size=config.IMGSZ,
        pad_value=getattr(config, "LETTERBOX_PAD_VALUE", 114.0/255.0),
        scaleup=True
    )

    lbl_path = img_to_label_path_tf(img_path)

    labels = parse_label_file_tf(
        lbl_path, meta,
        new_size=int(config.IMGSZ),
        num_kpt=int(config.NUM_KPT),
        kpt_vals=int(config.KPT_VALS)
    )

    return img, labels, meta


def build_dataset(img_glob, batch=config.BATCH, shuffle=True, repeat=True, with_labels=False):
    
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
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)

    if with_labels:
        # （可選）避免少數資料 label txt 缺檔造成 tf.io.read_file crash：先在 Python 端補空檔（只做一次，不在每 step）
        if getattr(config, "AUTO_TOUCH_MISSING_LABELS", True):
            for f in files:
                lp = re.sub(r"/images/", "/labels/", f)
                lp = re.sub(r"\.[^/.]+$", ".txt", lp)
                if not os.path.exists(lp):
                    os.makedirs(os.path.dirname(lp), exist_ok=True)
                    open(lp, "a", encoding="utf-8").close()

        ds = ds.map(tf_parse_load_with_labels, num_parallel_calls=AUTOTUNE)

        D = 5 + int(config.NUM_KPT) * int(config.KPT_VALS)
        ds = ds.padded_batch(
            batch,
            padded_shapes=(
                [int(config.IMGSZ), int(config.IMGSZ), 3],  # img
                [None, D],                                  # labels
                [5],                                        # meta
            ),
            padding_values=(
                tf.constant(0.0, tf.float32),
                tf.constant(0.0, tf.float32),
                tf.constant(0.0, tf.float32),
            ),
            drop_remainder=False
        ).prefetch(AUTOTUNE)

    else:
        ds = ds.map(tf_parse_load, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch).prefetch(AUTOTUNE)

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
    paths = sorted(glob.glob(config.REP_DIR_export))
    for p in paths:
        img_lb, _meta = decode_and_letterbox(
            tf.constant(p),
            new_size=config.IMGSZ,
            pad_value=getattr(config, "LETTERBOX_PAD_VALUE", 114.0/255.0),
            scaleup=True
        )
        yield [tf.expand_dims(img_lb, 0).numpy().astype(np.float32)]
