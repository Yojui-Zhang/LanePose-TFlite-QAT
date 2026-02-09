import tensorflow as tf
import glob
import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Generator

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.data.transforms import decode_and_letterbox
from QAT_Refactored.data.parser import (
    img_path_to_label_path_tf, 
    parse_label_file_tf, 
    parse_label_lines_numpy
)

class DataPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.autotune = tf.data.AUTOTUNE

    def list_files(self, patterns: List[str]) -> List[str]:
        """
        根據 Glob Pattern 列表取得所有檔案路徑。
        具備 Fail-Fast 檢查。
        """
        all_files = []
        if isinstance(patterns, str):
            patterns = [patterns]
            
        for p in patterns:
            # 支援 pathlib 物件自動轉字串
            p_str = str(p)
            if os.path.isdir(p_str):
                found = []
                for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
                    found.extend(glob.glob(os.path.join(p_str, "**", f"*.{ext}"), recursive=True))
                    found.extend(glob.glob(os.path.join(p_str, "**", f"*.{ext.upper()}"), recursive=True))
            elif os.path.exists(p_str) and os.path.isfile(p_str):
                found = [p_str]
            else:
                found = glob.glob(p_str)
            if not found:
                logging.warning(f"[Data] Pattern matched 0 files: {p_str}")
            all_files.extend(found)
            
        files = sorted(list(set(all_files)))
        
        if not files:
            msg = f"[Data] CRITICAL: No files found! Checked patterns: {patterns}"
            logging.error(msg)
            raise FileNotFoundError(msg)
            
        return files

    def split_train_val(self, files: List[str]) -> Tuple[List[str], List[str]]:
        """依設定比例分割訓練與驗證集 (Deterministic)"""
        n = len(files)
        if self.cfg.VAL_SPLIT <= 0.0:
            return files, []
            
        n_val = max(1, int(round(n * self.cfg.VAL_SPLIT)))
        train_files = files[:-n_val]
        val_files = files[-n_val:]
        return train_files, val_files

    def _parse_function_no_label(self, img_path):
        """僅讀取圖片 (Distill Mode / Inference)"""
        img, meta = decode_and_letterbox(
            img_path, 
            new_size=self.cfg.IMGSZ, 
            pad_value=self.cfg.LETTERBOX_PAD_VALUE
        )
        # Distill 模式下，labels 返回空，但需返回 path 供 debug
        lbl_path = img_path_to_label_path_tf(img_path) 
        return img, lbl_path, meta

    def _parse_function_with_label(self, img_path):
        """讀取圖片與標籤 (Label Supervision Mode)"""
        img, meta = decode_and_letterbox(
            img_path, 
            new_size=self.cfg.IMGSZ, 
            pad_value=self.cfg.LETTERBOX_PAD_VALUE
        )
        
        lbl_path = img_path_to_label_path_tf(img_path)
        
        labels = parse_label_file_tf(
            lbl_path, 
            meta, 
            new_size=self.cfg.IMGSZ, 
            num_kpt=self.cfg.NUM_KPT, 
            kpt_vals=self.cfg.KPT_VALS
        )
        return img, labels, meta

    def build_dataset(self, files: List[str], training: bool = True, with_labels: bool = False) -> tf.data.Dataset:
        """建立 tf.data.Dataset"""
        ds = tf.data.Dataset.from_tensor_slices(files)
        
        if training:
            ds = ds.shuffle(len(files), reshuffle_each_iteration=True, seed=self.cfg.SEED)

        if with_labels:
            ds = ds.map(self._parse_function_with_label, num_parallel_calls=self.autotune)
            
            # Padded Batch (因每張圖的標籤數量不同)
            # Shapes: Img, Labels, Meta
            D = 5 + self.cfg.NUM_KPT * self.cfg.KPT_VALS
            MAX_M = self.cfg.MAX_OBJS
            
            ds = ds.padded_batch(
                self.cfg.BATCH_SIZE,
                padded_shapes=(
                    [self.cfg.IMGSZ, self.cfg.IMGSZ, 3], # Img
                    [MAX_M, D],                          # Labels (Fixed to MAX_M)
                    [5]                                  # Meta
                ),
                padding_values=(0.0, 0.0, 0.0),
                drop_remainder=(training and self.cfg.TRAIN_DROP_REMAINDER)
            )
        else:
            ds = ds.map(self._parse_function_no_label, num_parallel_calls=self.autotune)
            ds = ds.batch(self.cfg.BATCH_SIZE)

        ds = ds.prefetch(self.autotune)
        
        if training:
            ds = ds.repeat()
            
        return ds

    def get_train_val_datasets(self):
        """
        對外主要接口：取得 Train 與 Val Dataset
        """
        logging.info("[Data] Scanning for training files...")
        train_files = self.list_files(self.cfg.TRAIN_PATTERNS)
        logging.info(f"[Data] Found {len(train_files)} training images.")

        if self.cfg.VAL_PATTERNS:
            val_files = self.list_files(self.cfg.VAL_PATTERNS)
            logging.info(f"[Data] Using explicit VAL_PATTERNS with {len(val_files)} images.")
        elif self.cfg.VAL_PATTERN:
            val_files = self.list_files([self.cfg.VAL_PATTERN])
            logging.info(f"[Data] Using explicit VAL_PATTERN with {len(val_files)} images.")
        else:
            train_files, val_files = self.split_train_val(train_files)
            logging.info(f"[Data] Split: {len(train_files)} Train, {len(val_files)} Val")
        
        # 只要不是純推論 (Inference)，訓練時通常都需要 Label (包含 Distill 模式)
        with_labels = (self.cfg.TRAIN_SUPERVISION in ['label', 'distill'])
        
        if self.cfg.TRAIN_SUPERVISION == 'distill':
            logging.info("[Data] Distillation Mode: Enabling Label loading for Hybrid Supervision.")

        ds_train = self.build_dataset(train_files, training=True, with_labels=with_labels)
        
        ds_val = None
        if val_files:
            # 驗證集通常也需要 Label 來計算 Loss/mAP
            ds_val = self.build_dataset(val_files, training=False, with_labels=with_labels)
            
        return ds_train, ds_val, len(train_files), len(val_files)

    def compute_class_weights(self) -> np.ndarray:
        """計算類別權重"""
        logging.info("[Data] Computing class weights (scanning all labels)...")
        files = self.list_files(self.cfg.TRAIN_PATTERNS)
        
        counts = np.zeros(self.cfg.NUM_CLS, dtype=np.int64)
        
        # 為了效率，這裡使用簡單的 Numpy loop
        for img_path in files:
            lbl_path = img_path.replace("/images/", "/labels/").rsplit('.', 1)[0] + ".txt"
            
            if not os.path.exists(lbl_path): continue
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            arr = parse_label_lines_numpy(lines, self.cfg.NUM_KPT, self.cfg.KPT_VALS)
            if arr.shape[0] == 0: continue
            
            cls_ids = arr[:, 0].astype(int)
            for c in cls_ids:
                if 0 <= c < self.cfg.NUM_CLS:
                    counts[c] += 1
                    
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        max_count = np.max(counts)
        weights = max_count / counts
        
        # Normalize to mean=1
        weights /= np.mean(weights)
        
        logging.info(f"[Data] Class Counts: {counts}")
        logging.info(f"[Data] Class Weights: {weights}")
        return weights.astype(np.float32)

    def get_rep_dataset_gen(self) -> Generator:
        """
        產生 Representative Dataset Generator (用於 TFLite Int8 量化)。
        """
        # 優先使用 Val，若無則用 Train
        pattern = self.cfg.VAL_PATTERN if self.cfg.VAL_PATTERN else self.cfg.TRAIN_PATTERNS[0]
        try:
            # 只取前 100 張，避免 FileNotFoundError 阻斷流程 (如果 Val pattern 不存在)
            files = self.list_files([pattern])[:100]
        except FileNotFoundError:
             # Fallback
             logging.warning("[Data] Val pattern not found for rep dataset, using first train pattern.")
             files = self.list_files([self.cfg.TRAIN_PATTERNS[0]])[:100]

        def gen():
            for p in files:
                img, meta = decode_and_letterbox(
                    p, 
                    new_size=self.cfg.IMGSZ, 
                    pad_value=self.cfg.LETTERBOX_PAD_VALUE
                )
                # TFLite Converter 預期 [1, H, W, C]
                yield [tf.expand_dims(img, 0)]
                
        return gen
