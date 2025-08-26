import tensorflow as tf
import numpy as np
from collections import Counter

def dump_model_report(model_path):
    print('\n===', model_path, '===')
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    # 建 index -> tensor detail 的查表
    tmap = {t['index']: t for t in interp.get_tensor_details()}
    ops = interp._get_ops_details()

    def dtype_of(x):
        # 某些版本 inputs/outputs 是 dict，某些是 int（tensor index）
        if isinstance(x, dict):
            return np.dtype(x['dtype']).name
        return np.dtype(tmap[int(x)]['dtype']).name

    def shape_of(x):
        if isinstance(x, dict):
            return x['shape']
        return tmap[int(x)]['shape']

    # 列出所有 op 的 I/O 型別（前 50 行）
    for i, op in enumerate(ops[:50]):
        in_types = [dtype_of(t) for t in op['inputs']]
        out_types = [dtype_of(t) for t in op['outputs']]
        print(f'{i:03d} {op["op_name"]}: {in_types} -> {out_types}')

    # 統計關鍵算子的 INT8 覆蓋率
    int8_conv = float_conv = 0
    per_op = Counter()
    for op in ops:
        name = op['op_name']
        in_types = [dtype_of(t) for t in op['inputs']]
        out_types = [dtype_of(t) for t in op['outputs']]
        types = set(in_types + out_types)
        per_op[name] += 1

        if name in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'):
            ok_types = set(in_types + out_types) - {'int32'}
            if ok_types <= {'int8'}:
                int8_conv += 1
            elif 'float32' in ok_types:
                float_conv += 1


    print('Total ops:', len(ops), dict(per_op))
    print('INT8 conv-like ops:', int8_conv, 'FLOAT conv-like ops:', float_conv)

    # 輸出張量與其量化（若輸出前有 DEQUANTIZE 可看到對應 scale）
    out_det = interp.get_output_details()[0]
    print('Output detail:', {'index': out_det['index'], 'dtype': np.dtype(out_det['dtype']).name,
                             'shape': out_det['shape'],
                             'quant': out_det.get('quantization_parameters', {})})


# 跑官方與你自己的檔
dump_model_report('lanepose20250807_s_model_640_640_6c_v1_integer_quant.tflite')
dump_model_report('best_qat_int8.tflite')
