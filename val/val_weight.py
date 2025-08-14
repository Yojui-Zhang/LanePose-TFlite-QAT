# inspect_tflite.py
import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = "../output/best_qat_int8.tflite"  # 改成你的路徑
assert Path(TFLITE_PATH).exists(), "tflite not found"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

print(f"Total tensors: {len(details)}")
for i, d in enumerate(details):
    name = d['name']
    shape = d['shape']
    dtype = d['dtype']
    qp = d.get('quantization_parameters', None)  # dict with 'scales','zero_points','quantized_dimension'
    q_info = ""
    if qp:
        scales = qp.get('scales', None)
        zps = qp.get('zero_points', None)
        qdim = qp.get('quantized_dimension', None)
        # format q info
        if scales is None or len(scales) == 0:
            q_info = f"per-tensor? scale/zp: {d.get('quantization', None)}"
        else:
            q_info = f"per-channel scales len={len(scales)}, quant_dim={qdim}"
    print(f"[{i}] name={name}, shape={shape}, dtype={dtype}, quant={q_info}")
