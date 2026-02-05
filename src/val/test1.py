import tensorflow as tf, numpy as np

def inspect(path):
    itp = tf.lite.Interpreter(model_path=path); itp.allocate_tensors()
    i = itp.get_input_details()[0]
    o = itp.get_output_details()[0]
    print("\n===", path, "===")
    print("IN :", i["dtype"], "quant=", i["quantization"])
    print("OUT:", o["dtype"], "quant=", o["quantization"])
    # 粗看是否整張圖是整數圖
    ints = sum(t["dtype"] in (np.int8, np.uint8) for t in itp.get_tensor_details())
    floats = sum(t["dtype"] == np.float32 for t in itp.get_tensor_details())
    print(f"tensors int={ints} float={floats}  (float>0 代表可能有回退)")

inspect("./Teacher_models/lanepose20250807_s_model_640_640_6c_v1_integer_quant.tflite")
inspect("./student_models/best_qat_int8_interrupted.tflite")
