# find_zero_scales.py
import tensorflow as tf
from pathlib import Path
TFLITE = "../output/best_qat_int8.tflite"   # 改成你的路徑
interpreter = tf.lite.Interpreter(model_path=TFLITE)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

zero_scale = []
for d in details:
    q = d.get('quantization_parameters', None)
    if q:
        scales = q.get('scales', None)
        if scales is None or len(scales) == 0:
            # sometimes 'quantization' tuple exists:
            qt = d.get('quantization', (None, None))
            if qt[0] == 0.0:
                zero_scale.append((d['index'], d['name'], d['shape'], d['dtype'], qt))
    else:
        # fall back: check d['quantization'] tuple
        qt = d.get('quantization', (None, None))
        if qt[0] == 0.0:
            zero_scale.append((d['index'], d['name'], d['shape'], d['dtype'], qt))

print("Found", len(zero_scale), "tensors with scale==0. Sample:")
for s in zero_scale[:40]:
    print(s)
