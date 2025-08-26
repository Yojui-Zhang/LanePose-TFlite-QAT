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

from src.process.pred_model import (ensure_BNC_static)

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
        
    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝

        # # 1) 準備樣本
        sample_batch = next(iter(ds))
        sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_one = _ensure_bhwc4(sample_imgs, imgsz=config.IMGSZ)

        # # 2) 重新計算 teacher / student 的對齊資訊（和訓練邏輯一致）
        C = 4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS
        # H3 = W3 = config.IMGSZ // 8
        # H4 = W4 = config.IMGSZ // 16
        # H5 = W5 = config.IMGSZ // 32
        # N3, N4, N5 = H3*W3, H4*W4, H5*W5

        # lens_perm, reorder_idx = choose_student_split_order(
        #     student, teacher, sample_one, N3, N4, N5, C, NUM_CLS, NUM_KPT, KPT_VALS
        # )
        # lens_perm  = tuple(int(x) for x in lens_perm)
        # reorder_idx = [int(x) for x in reorder_idx]

        # def _reorder_N_blocks(y_BNC):
        #     s0, s1, s2 = lens_perm
        #     parts = tf.split(y_BNC, [s0, s1, s2], axis=1)
        #     return tf.concat([parts[reorder_idx[0]], parts[reorder_idx[1]], parts[reorder_idx[2]]], axis=1)

        # 3) 正確取分支 -> 統一到 (B,N,C) -> 學生做重排
        y_t_raw = teacher(sample_one, training=False)
        y_s_out = student(sample_one, training=False)

        # teacher to (B,N,C)
        y_t_BNC = ensure_BNC_static(y_t_raw, C)  # 不要自己 transpose
        # student kd 分支到 (B,N,C) + 重排
        kd_raw = y_s_out[1] if isinstance(y_s_out, (list,tuple)) else y_s_out
        kd_BNC = ensure_BNC_static(kd_raw, C)
        # kd_BNC = _reorder_N_blocks(kd_BNC)

        # （如有需要，也可把 deploy 分支同樣轉為 BNC + 重排後另存，僅供參考）
        dep_raw = y_s_out[0] if isinstance(y_s_out, (list,tuple)) else None
        if dep_raw is not None:
            dep_BNC = ensure_BNC_static(dep_raw, C)
            # dep_BNC = _reorder_N_blocks(dep_BNC)

        # 4) 攤平成 (B*N, C) 存檔
        teacher_flat = tf.reshape(y_t_BNC, [-1, C]).numpy()
        student_kd_flat = tf.reshape(kd_BNC, [-1, C]).numpy()
        np.savetxt(output_paths['logs'] / 'teacher_output.txt', teacher_flat, fmt='%.4f', delimiter=',')
        np.savetxt(output_paths['logs'] / 'student_kd_output.txt', student_kd_flat, fmt='%.4f', delimiter=',')

        if dep_raw is not None:
            student_dep_flat = tf.reshape(dep_BNC, [-1, C]).numpy()
            np.savetxt( output_paths['logs'] / 'student_deploy_output.txt', student_dep_flat, fmt='%.4f', delimiter=',')
        
    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        
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

            if hasattr(tfmot.quantization.keras, "strip_quantization"):
                print("\n[INFO] Stripping quantization wrappers from the model.")
                # 只取 deploy 分支輸出（index 0）
                student_deploy = tf.keras.Model(student.input, student.outputs[0], name="student_deploy_only")
                student_infer = tfmot.quantization.keras.strip_quantization(student_deploy)
            else:
                print("\n[WARN] `strip_quantization` not found; exporting wrapped model.")
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