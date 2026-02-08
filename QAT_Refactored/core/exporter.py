import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Generator, Tuple, List

from QAT_Refactored.config.config import AppConfig
from QAT_Refactored.models.layers import RepVGGBlock
from QAT_Refactored.utils.tensor_layout import ensure_bcn_tf

class ExportModule(tf.Module):
    """
    Wraps the trained model for TFLite export.
    [C++ Compatibility Enforcements] [cite: 65]
    1. Static Input Shape: Prevents dynamic allocation[cite: 65].
    2. Sigmoid Activation: C++ expects normalized coords (0-1)[cite: 66, 70].
    """
    def __init__(self, model: tf.keras.Model, cfg: AppConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
    @tf.function
    def serving_fn(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """TFLite 進入點，強制應用 Sigmoid 以匹配 C++ 邏輯 [cite: 68-70]"""
        y = self.model(x, training=False)
        if isinstance(y, (list, tuple)):
            y = y[0]

        # Why: hard-enforce exporter contract (1, C, N) even if internal model flips to (1, N, C).
        y = ensure_bcn_tf(y, total_c=self.cfg.total_output_channels, name="export_output")

        y = tf.sigmoid(y) 
        return {"output0": y}

class Exporter:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    @staticmethod
    def _unwrap_repvgg(layer: tf.keras.layers.Layer) -> Optional[tf.keras.layers.Layer]:
        # Why: QAT wrapper 可能把 RepVGGBlock 藏在 .layer / ._layer 等欄位，直接 isinstance 會漏掉。
        candidates: List[tf.keras.layers.Layer] = [layer]
        for attr in ("layer", "_layer", "wrapped_layer", "inner_layer"):
            inner = getattr(layer, attr, None)
            if isinstance(inner, tf.keras.layers.Layer):
                candidates.append(inner)

        for cand in candidates:
            if isinstance(cand, RepVGGBlock):
                return cand
        return None

    @staticmethod
    def _iter_children(layer: tf.keras.layers.Layer) -> List[tf.keras.layers.Layer]:
        # Why: Keras container 用 .layers；wrapper 常用 .layer；統一遞迴入口避免漏掃。
        children: List[tf.keras.layers.Layer] = []

        if hasattr(layer, "layers"):
            for sub in getattr(layer, "layers"):
                if isinstance(sub, tf.keras.layers.Layer):
                    children.append(sub)

        for attr in ("layer", "_layer", "wrapped_layer", "inner_layer"):
            inner = getattr(layer, attr, None)
            if isinstance(inner, tf.keras.layers.Layer):
                children.append(inner)

        uniq: List[tf.keras.layers.Layer] = []
        seen: Set[int] = set()
        for ch in children:
            cid = id(ch)
            if cid in seen:
                continue
            seen.add(cid)
            uniq.append(ch)
        return uniq

    def _fuse_repvgg(self, model: tf.keras.Model) -> None:
        if model is None:
            raise ValueError("model is None")

        logging.info("[Exporter] Fusing RepVGG Blocks...")
        fused_count = 0
        visited: Set[int] = set()

        def _fuse_layer(layer: tf.keras.layers.Layer) -> None:
            nonlocal fused_count

            lid = id(layer)
            if lid in visited:
                return
            visited.add(lid)

            rep = self._unwrap_repvgg(layer)
            if rep is not None and (not getattr(rep, "deploy", False)):
                if hasattr(rep, "switch_to_deploy"):
                    rep.switch_to_deploy()
                    fused_count += 1

            for sub in self._iter_children(layer):
                _fuse_layer(sub)

        for layer in model.layers:
            _fuse_layer(layer)

        if fused_count == 0:
            logging.warning("[Exporter] ⚠️ No RepVGGBlocks found to fuse.")
        else:
            logging.info(f"[Exporter] Successfully fused {fused_count} blocks.")


    def export_saved_model(self, model: tf.keras.Model, output_path: Path) -> None:
        """
        對應 main.py 的第 1 步：包含 Pre-fusion -> Fusion -> Post-fusion -> Save [cite: 78-85]
        """
        logging.info("[Exporter] Phase 1: Fusion & SavedModel Export")
        
        # 1. Pre-Fusion Check [cite: 78]
        dummy_input = tf.random.uniform((1, self.cfg.IMGSZ, self.cfg.IMGSZ, 3), dtype=tf.float32)
        y_pre = model(dummy_input, training=False)
        y_pre = y_pre[0].numpy() if isinstance(y_pre, (list, tuple)) else y_pre.numpy()

        # 2. Fuse Model [cite: 79]
        self._fuse_repvgg(model)

        # 3. Post-Fusion Check [cite: 80]
        y_post = model(dummy_input, training=False)
        y_post = y_post[0].numpy() if isinstance(y_post, (list, tuple)) else y_post.numpy()
        
        diff = np.max(np.abs(y_pre - y_post))
        logging.info(f"[Exporter] Fusion Max Error: {diff:.8f}")
        if diff > 1e-3:
            logging.warning("[Exporter] ⚠️ HIGH FUSION ERROR (> 1e-3) [cite: 81-82]!")

        # 4. Save Static SavedModel [cite: 83-85]
        export_mod = ExportModule(model, self.cfg)
        input_spec = tf.TensorSpec(shape=self.cfg.EXPORT_INPUT_SHAPE, dtype=tf.float32, name="images")
        concrete_fn = export_mod.serving_fn.get_concrete_function(input_spec)
        
        tf.saved_model.save(export_mod, str(output_path), signatures=concrete_fn)
        logging.info(f"[Exporter] SavedModel (Static) written to {output_path}")

    def convert_to_tflite(self, saved_model_path: Path, output_path: Path, 
                          rep_dataset_gen: Optional[Generator] = None) -> None:
        """
        對應 main.py 的第 2 步：執行量化轉換 [cite: 86-91]
        """
        logging.info(f"[Exporter] Phase 2: Converting to TFLite ({self.cfg.TFLITE_QUANT_MODE})")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        if self.cfg.TFLITE_QUANT_MODE == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if rep_dataset_gen:
                converter.representative_dataset = rep_dataset_gen
                # 強制使用 INT8 算子以符合 C++ 相容性 [cite: 87]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32 
            else:
                logging.warning("[Exporter] No representative dataset; falling back to dynamic range[cite: 89].")

        elif self.cfg.TFLITE_QUANT_MODE == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        with open(output_path, "wb") as f: f.write(tflite_model)
        
        logging.info(f"[Exporter] TFLite saved: {output_path} ({len(tflite_model)/1024:.1f} KB)")
        self._print_cpp_config()

    def export(self, model: tf.keras.Model, rep_dataset: Optional[Generator] = None) -> None:
        """舊版整合進入點，用於相容 verify_export.py [cite: 77]"""
        sm_path = self.cfg.OUTPUT_DIR / "export/saved_model"
        tf_path = self.cfg.OUTPUT_DIR / f"export/model_{self.cfg.TFLITE_QUANT_MODE}.tflite"
        sm_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.export_saved_model(model, sm_path)
        self.convert_to_tflite(sm_path, tf_path, rep_dataset)

    def _print_cpp_config(self):
        """輸出 C++ config.h 程式碼段 [cite: 91]"""
        print("\n" + "="*60 + "\n [C++ INTEGRATION] Copy this to your 'config.h'\n" + "="*60)
        print(self.cfg.generate_cpp_header_snippet())
        print("="*60 + "\n")