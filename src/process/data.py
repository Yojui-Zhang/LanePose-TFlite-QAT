# qat_tf/qat_distill.py
import glob
import tensorflow as tf
import numpy as np
import cv2

import config

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
前處理（與 Ultralytics 部署一致）
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

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
    rgb = bgr[:, :, ::-1]
    img = letterbox(rgb, config.IMGSZ).astype(np.float32) / 255.0
    return img

def tf_parse(path):
    img = tf.numpy_function(parse_img, [path], Tout=tf.float32)
    img.set_shape([config.IMGSZ, config.IMGSZ, 3])
    return img

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
        print(f"Read image: {len(files)} from {len(patterns)} directories")

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True) # 加上 reshuffle_each_iteration
        
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
        
    return ds, len(files)

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