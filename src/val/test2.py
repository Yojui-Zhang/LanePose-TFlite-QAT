import tensorflow as tf
import numpy as np

WRAPPED_SAVEDMODEL = "./student_models/qat_saved_model_rescaled"
OUT_TFLITE = "./student_models/best_qat_int8_like_teacher.tflite"

# =================================================================
# 1. 創建一個代表性資料集函式
# =================================================================
# 這個函式必須是一個生成器（generator），它會迭代地提供代表性資料。
# 函式中的資料形狀和類型必須與你的模型輸入的形狀和類型完全匹配。
# 這裡假設你的模型輸入形狀是 [1, 640, 640, 3]，類型是 float32。
def representative_dataset():
    # 這裡你需要替換成你自己的資料載入邏輯。
    # 應該從你的訓練資料集中取得一小部分樣本。
    # 假設你有一些圖片檔案路徑...
    # images_paths = [...]

    # 這裡我們用隨機資料作為範例
    for _ in range(100):
        # 產生一個與模型輸入形狀匹配的資料張量
        data = np.random.rand(1, 640, 640, 3)  # 假設圖片尺寸為 640x640x3
        yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(WRAPPED_SAVEDMODEL)

# =================================================================
# 2. 設定優化器與代表性資料集
# =================================================================
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# =================================================================
# 3. 指定全整數量化與輸入輸出類型
# =================================================================
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tfl = converter.convert()
open(OUT_TFLITE, "wb").write(tfl)
print("Saved:", OUT_TFLITE)