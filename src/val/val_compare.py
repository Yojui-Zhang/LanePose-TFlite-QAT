# compare_tflite.py
import numpy as np
import tensorflow as tf
from pathlib import Path
import json, sys

GOOD = "lanepose20250807_s_model_640_640_6c_v1_integer_quant.tflite"
BAD  = "best_qat_int8.tflite"

def summarize(path):
    p = Path(path)
    out = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    out["size"] = p.stat().st_size
    interp = tf.lite.Interpreter(model_path=str(p))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    outp = interp.get_output_details()
    td = interp.get_tensor_details()
    out["num_tensors"] = len(td)
    out["inputs"] = inp
    out["outputs"] = outp

    # find zero-scale tensors
    zero_scale = []
    for d in td:
        qp = d.get("quantization_parameters", {})
        scales = qp.get("scales", None)
        q = d.get("quantization", None)
        # detect zero or missing scale
        s_zero = False
        if scales is not None:
            try:
                arr = np.array(scales)
                if arr.size == 0 or np.all(arr == 0):
                    s_zero = True
            except:
                s_zero = True
        elif q is not None:
            if q[0] == 0:
                s_zero = True
        else:
            # no quantization info => treat as "missing"
            s_zero = True
        if s_zero:
            zero_scale.append({"index": d["index"], "name": d.get("name",""), "shape": d.get("shape")})
    out["zero_scale_count"] = len(zero_scale)
    out["zero_scale_sample"] = zero_scale[:40]
    return out

def run_dummy(path, n=1):
    p = Path(path)
    if not p.exists(): return {"ran": False, "reason":"file missing"}
    interp = tf.lite.Interpreter(model_path=str(p))
    try:
        interp.allocate_tensors()
    except Exception as e:
        return {"ran": False, "reason": f"allocate_tensors failed: {e}"}
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    shape = inp["shape"].copy()
    dtype = inp["dtype"]
    # make dummy input with correct dtype; if quantized, use zero_point or center
    if np.issubdtype(dtype, np.integer):
        zp = 0
        qparams = inp.get("quantization_parameters", {})
        zps = qparams.get("zero_points", None)
        if zps is not None and len(zps) > 0:
            zp = int(zps[0])
        data = np.zeros(shape, dtype=dtype) + zp
    else:
        data = np.random.random(shape).astype(np.float32)
    try:
        interp.set_tensor(inp["index"], data)
        interp.invoke()
        out_tensor = interp.get_tensor(out["index"])
        return {"ran": True, "out_shape": out_tensor.shape, "out_dtype": str(out_tensor.dtype)}
    except Exception as e:
        return {"ran": False, "reason": str(e)}

if __name__ == "__main__":
    for tag, fname in [("good", GOOD), ("bad", BAD)]:
        print("="*60)
        print(tag, fname)
        s = summarize(fname)
        print("exists:", s.get("exists"))
        if not s.get("exists"):
            continue
        print("size:", s["size"])
        print("num_tensors:", s["num_tensors"])
        print("zero_scale_count:", s["zero_scale_count"])
        print("zero_scale_sample (up to 40):")
        for z in s["zero_scale_sample"]:
            print("  ", z)
        print("inputs:")
        for i in s["inputs"]:
            print("   ", i["name"], "shape:", i["shape"], "dtype:", i["dtype"],
                  "quant_params:", i.get("quantization_parameters") or i.get("quantization"))
        print("outputs:")
        for o in s["outputs"]:
            print("   ", o["name"], "shape:", o["shape"], "dtype:", o["dtype"],
                  "quant_params:", o.get("quantization_parameters") or o.get("quantization"))
        print("dummy inference:", run_dummy(fname))
