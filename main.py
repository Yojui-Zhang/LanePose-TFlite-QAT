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
from datetime import datetime
from pathlib import Path

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
from src.process.Train_Model import (build_student_qat, run_qat, choose_student_split_order)

from src.process.Export_Model import (ExportModule, run_diagnostics_once,export_only, 
                                      create_and_configure_tflite_converter)

if config.PLOT_Switch == True:
    from src.process.Plot_Data import plot_and_save_loss_curve



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
    
    print(f"--- QAT Script Started at {run_timestamp} ---")
    print(f"--- All outputs will be saved in: {output_dir} ---")
    
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
    student.summary(line_length=120)

    # 3) 準備資料集
    print("\n--- Preparing Dataset ---")
    ds, n_files = build_dataset(img_glob=config.REP_DIR_train, batch=config.BATCH)
    steps_per_epoch = max(1, n_files // config.BATCH)

    try:
        # 4) QAT 微調 or 直接 Export
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
        # 極端情況：某些環境仍會拋出 KeyboardInterrupt；這裡兜底處理
        print("\n[⚠️ Interrupt] KeyboardInterrupt caught. Will export current weights...\n")
    finally:

        if not getattr(config, "EXPORT_ONLY", False):
            # 5) 導出前準備（你原本的第 5～9 步）
            print("\n--- Preparing for Export ---")
            N3 = (config.IMGSZ // 8)  ** 2
            N4 = (config.IMGSZ // 16) ** 2
            N5 = (config.IMGSZ // 32) ** 2
            C  = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
            print(f"Expected N={N3+N4+N5}, C={C}")

            if hasattr(tfmot.quantization.keras, "strip_quantization"):
                print("[INFO] Stripping quantization wrappers from the model.")
                student_infer = tfmot.quantization.keras.strip_quantization(student)
            else:
                print("[WARN] `strip_quantization` not found; exporting wrapped model.")
                student_infer = student

            # 6) 自動對齊輸出順序
            try:
                sample_one = next(iter(ds))[:1]
            except Exception:
                sample_one = tf.zeros([1, config.IMGSZ, config.IMGSZ, 3], tf.float32)

            lens_perm, reorder_idx = choose_student_split_order(student_infer, teacher, sample_one, N3, N4, N5, C)

            # 7) 導出 SavedModel
            print("\n--- Exporting SavedModel ---")
            export_mod = ExportModule(
                student_infer, C=C, lens_perm=lens_perm, reorder_idx=reorder_idx,
                grid_modes=config.GRID_MODES, porder=config.PORDER, ch_map=config.CHANNEL_MAPPING,
                xywh_to_ltrb=config.XYWH_TO_LTRB, xywh_is_norm01=config.YXWH_IS_NORMALIZED_01 if hasattr(config,'YXWH_IS_NORMALIZED_01') else config.XYWH_IS_NORMALIZED_01
            )
            saved_model_path = str(output_paths['models'] / ("qat_saved_model_interrupted" if config.STOP_REQUESTED else "qat_saved_model"))
            concrete_fn = export_mod.serving_fn.get_concrete_function()
            tf.saved_model.save(export_mod, saved_model_path, signatures=concrete_fn)
            print(f"✅ SavedModel exported to → {saved_model_path}")

            print("\n--- Converting to TFLite INT8 ---")
            conv = create_and_configure_tflite_converter(saved_model_path)

            tfl_bytes   = conv.convert()
            tflite_path = str(output_paths['models'] / ("best_qat_int8_interrupted.tflite" if config.STOP_REQUESTED else "best_qat_int8.tflite"))
            Path(tflite_path).write_bytes(tfl_bytes)
            print(f"✅ TFLite model written to → {tflite_path}")

            # 9) 檢查 TFLite I/O
            interp = tf.lite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            print(" TFLite inputs:", interp.get_input_details())
            print(" TFLite outputs:", interp.get_output_details())
                
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