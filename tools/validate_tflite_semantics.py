import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

import config
from src.process.preprocess_tf import decode_and_letterbox

def _load_tflite(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=getattr(config, "TFLITE_NUM_THREADS", 4))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return interpreter, inp, out

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode_like_tflite_h(output_bcn: np.ndarray, meta: np.ndarray,
                         conf_thres: float = 0.25,
                         topk: int = 300):
    """
    output_bcn: (1, C, N) expected.
    meta: [orig_h, orig_w, scale, pad_x, pad_y] from letterbox (pad in pixels of IMGSZ)
    Return list of dict: {class_id, score, box_xywh_orig, kpts_orig}
    """
    assert output_bcn.ndim == 3 and output_bcn.shape[0] == 1, f"Expected (1,C,N), got {output_bcn.shape}"
    C, N = output_bcn.shape[1], output_bcn.shape[2]
    num_cls = int(config.NUM_CLS)
    num_kpt = int(config.NUM_KPT)
    kval = int(config.KPT_VALS)

    # split
    box = output_bcn[0, 0:4, :].T  # (N,4) [cx,cy,w,h] in 0..1 (IMGSZ space)
    cls = output_bcn[0, 4:4+num_cls, :].T if num_cls > 0 else np.zeros((N, 0), np.float32)
    kpt = output_bcn[0, 4+num_cls:, :].T if num_kpt > 0 else np.zeros((N, 0), np.float32)

    if num_kpt > 0:
        kpt = kpt.reshape(N, num_kpt, kval)  # (N,K,V)

    orig_h, orig_w, scale, pad_x, pad_y = meta.astype(np.float32)
    # Note: in your TF letterbox, scale = IMGSZ / max(orig_h, orig_w). Same idea as C++ code.
    # In TFlite.h they compute scale_factor = max(scale_x, scale_y). With square letterbox, that's 1/scale
    # Here we use the same mapping you use in TF:
    # padded coords -> orig coords: (x*IMGSZ - pad_x)/scale

    results = []
    # choose max class per anchor
    if num_cls > 0:
        cls_ids = np.argmax(cls, axis=1)
        scores = cls[np.arange(N), cls_ids]
    else:
        cls_ids = np.zeros((N,), np.int32)
        scores = np.ones((N,), np.float32)

    keep = scores > conf_thres
    idxs = np.where(keep)[0]
    if idxs.size == 0:
        return results

    # topk by score
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    idxs = idxs[:topk]

    for i in idxs:
        cx, cy, w, h = box[i]
        # to padded pixel coords
        x_p = cx * config.IMGSZ
        y_p = cy * config.IMGSZ
        w_p = w * config.IMGSZ
        h_p = h * config.IMGSZ

        # to original
        x = (x_p - pad_x) / scale
        y = (y_p - pad_y) / scale
        w_o = w_p / scale
        h_o = h_p / scale

        kpts = None
        if num_kpt > 0:
            kpts = []
            for k in range(num_kpt):
                kx, ky = kpt[i, k, 0], kpt[i, k, 1]
                kv = kpt[i, k, 2] if kval >= 3 else 1.0
                kx_p = kx * config.IMGSZ
                ky_p = ky * config.IMGSZ
                kx_o = (kx_p - pad_x) / scale
                ky_o = (ky_p - pad_y) / scale
                kpts.append((float(kx_o), float(ky_o), float(kv)))
        results.append({
            "class_id": int(cls_ids[i]),
            "score": float(scores[i]),
            "box_xywh": (float(x), float(y), float(w_o), float(h_o)),
            "kpts": kpts
        })
    return results

def draw_results(img_bgr, results, kpt_th=0.3):
    for r in results:
        x,y,w,h = r["box_xywh"]
        x1=int(round(x - w/2)); y1=int(round(y - h/2))
        x2=int(round(x + w/2)); y2=int(round(y + h/2))
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"{r['class_id']}:{r['score']:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if r["kpts"] is not None:
            for (kx,ky,kv) in r["kpts"]:
                if kv >= kpt_th:
                    cv2.circle(img_bgr, (int(round(kx)), int(round(ky))), 2, (0,0,255), -1)
    return img_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="Path to .tflite")
    ap.add_argument("--img", required=True, help="Path to an input image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="tflite_semantics_debug.jpg")
    args = ap.parse_args()

    interpreter, inp, out = _load_tflite(args.tflite)

    img_lb, meta = decode_and_letterbox(tf.constant(args.img), new_size=config.IMGSZ,
                                        pad_value=getattr(config,"LETTERBOX_PAD_VALUE",114.0/255.0),
                                        scaleup=True)
    x = tf.expand_dims(img_lb, 0).numpy().astype(np.float32)

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])

    # Expect (1,C,N). If not, try to reshape common cases.
    if y.ndim == 3 and y.shape[0] == 1:
        out_bcn = y
    elif y.ndim == 3:
        # maybe (1,N,C)
        if y.shape[2] == (4 + config.NUM_CLS + config.NUM_KPT * config.KPT_VALS):
            out_bcn = np.transpose(y, (0,2,1))
        else:
            raise ValueError(f"Unexpected output shape: {y.shape}")
    else:
        raise ValueError(f"Unexpected output shape: {y.shape}")

    # range sanity
    mn = float(out_bcn.min()); mx = float(out_bcn.max())
    print(f"[TFLite output] shape={out_bcn.shape} min={mn:.6f} max={mx:.6f}")

    # draw on original image (not letterboxed)
    img0 = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img0 is None:
        raise FileNotFoundError(args.img)

    results = decode_like_tflite_h(out_bcn, meta.numpy(), conf_thres=args.conf)
    print(f"Decoded {len(results)} proposals (conf>{args.conf}).")

    vis = draw_results(img0.copy(), results)
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
