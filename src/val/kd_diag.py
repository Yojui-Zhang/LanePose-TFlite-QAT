#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KD/QAT 診斷腳本：
- 自動讀取 TF SavedModel / TFLite，跑一次推論
- 猜測 head 佈局（XYWH vs DFL；是否含 obj；通道順序）
- 若同時提供 Teacher 與 Student，計算各分支 MAE/相關係數
- 偵測 Student 是否被 1/256 假量化夾死

需要：tensorflow>=2.x, numpy
"""
import argparse, os, sys, math, json
from collections import Counter
from typing import Dict, Tuple, List, Optional

import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    tf = None

# ------------ 基本工具 ------------
def pretty_shape(a):
    return "x".join(map(str, list(a.shape)))

def as_CN(arr: np.ndarray) -> np.ndarray:
    """
    將任意 (B,*,*) 輸出統一成 (C,N)
    常見： (1, C, N) or (1, N, C) or (C, N)
    """
    a = arr
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        # 嘗試找到像 (C,N) 的兩維
        flat = a.reshape(-1, a.shape[-1])
        return flat.T if flat.shape[0] < flat.shape[1] else flat
    # 現在是 (X,Y)
    X, Y = a.shape
    # 8400 這個數常出現在 N；若遇到就以它為 N
    if 8400 in (X, Y):
        return a if Y == 8400 else a.T
    # 否則假設「寬的一維是 N」
    return a if X < Y else a.T

def select_largest_float_tensor(d: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    best_key, best_val, best_numel = None, None, -1
    for k, v in d.items():
        if v.dtype.kind in "fc" and v.size > best_numel:
            best_key, best_val, best_numel = k, v, v.size
    if best_val is None:
        raise RuntimeError("找不到浮點輸出張量")
    return best_key, best_val

def run_savedmodel(sm_path: str) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    assert tf is not None, "需要 tensorflow 來讀 SavedModel"
    model = tf.saved_model.load(sm_path)
    # 優先 serving_default
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]
    else:
        # 隨便拿一個 signature
        infer = list(model.signatures.values())[0]
    # 取第一個輸入張量規格
    in_spec_dict = infer.structured_input_signature[1]
    in_name = list(in_spec_dict.keys())[0]
    input_spec = list(in_spec_dict.values())[0]
    ishape = [d if isinstance(d, int) else (d if d is not None else 1) for d in input_spec.shape]
    if len(ishape) != 4:
        # 預設 1x640x640x3
        ishape = [1, 640, 640, 3]
    dummy = tf.zeros(ishape, dtype=input_spec.dtype if input_spec.dtype else tf.float32)
    out = infer(**{in_name: dummy})
    # 把最大的 float 輸出抓出來
    out_key, out_val = select_largest_float_tensor({k: v.numpy() for k, v in out.items()})
    return out_val, tuple(ishape)

def run_tflite(tfl_path: str) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    assert tf is not None, "需要 tensorflow 來跑 TFLite"
    interpreter = tf.lite.Interpreter(model_path=tfl_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    ishape = tuple(in_det["shape"].tolist())
    dummy = np.zeros(ishape, dtype=np.float32)
    interpreter.set_tensor(in_det["index"], dummy)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det["index"])
    return out, ishape

# ------------ 佈局推斷與解碼 ------------
def guess_layout(C: int, num_classes: int, num_kpts: int,
                 reg_max_candidates=(16, 32), try_obj=(0,1)) -> List[Dict]:
    """
    根據通道數 C 與已知的 nc、nkpts，列出可行的頭部配置
    回傳每個候選 {kind: 'xywh' or 'dfl', reg_max, has_obj, ok:bool}
    """
    ks = []
    for has_obj in try_obj:
        # XYWH：4 + obj + nc + 3*nk
        exp_xywh = 4 + has_obj + num_classes + 3*num_kpts
        if exp_xywh == C:
            ks.append(dict(kind="xywh", reg_max=None, has_obj=has_obj, ok=True))
        # DFL：4*reg_max + obj + nc + 3*nk
        for rm in reg_max_candidates:
            exp_dfl = 4*rm + has_obj + num_classes + 3*num_kpts
            if exp_dfl == C:
                ks.append(dict(kind="dfl", reg_max=rm, has_obj=has_obj, ok=True))
    return ks

def split_segments(CN: np.ndarray, cfg: Dict, num_classes: int, num_kpts: int):
    """
    依照 cfg 切片輸出：(C,N) -> dict(box, obj?, cls, kpt)
    - xywh: 前 4 為 box
    - dfl : 前 4*reg_max 為四邊各 reg_max 個 bin
    - obj: 若 has_obj=1 則接在後面
    - cls: 接著 nc
    - kpt: 最後 3*nk
    注意：你的 C++ 代碼假設「沒有 obj，cls 緊接在 4 之後，再來 kpt」。
    """
    C, N = CN.shape
    i = 0
    seg = {}
    if cfg["kind"] == "xywh":
        seg["box_raw"] = CN[i:i+4]; i += 4
    else:
        rm = cfg["reg_max"]
        seg["box_dfl"] = CN[i:i+4*rm].reshape(4, rm, N)  # (4, reg_max, N)
        i += 4*rm
    if cfg["has_obj"]:
        seg["obj"] = CN[i:i+1]; i += 1
    seg["cls"] = CN[i:i+num_classes]; i += num_classes
    seg["kpt"] = CN[i:i+3*num_kpts].reshape(num_kpts, 3, N)  # (K,3,N)
    return seg

def decode_dfl(box_dfl: np.ndarray) -> np.ndarray:
    """
    box_dfl: (4, reg_max, N), 做 softmax + 期望值，回傳 (4, N)
    """
    four, rm, N = box_dfl.shape
    e = np.exp(box_dfl - box_dfl.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)  # softmax
    bins = np.arange(rm, dtype=np.float32).reshape(1, rm, 1)
    expv = (p * bins).sum(axis=1)  # (4, N)
    return expv

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# ------------ 指標與偵測 ------------
def mae(a,b):
    return float(np.mean(np.abs(a-b)))

def corr(a,b):
    a = a.reshape(-1); b = b.reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def quant_grid_ratio(values: np.ndarray, denom=256, atol=1e-6) -> Tuple[float, List[Tuple[float,int]]]:
    """
    偵測是否落在 1/256 階梯上：回傳比例與最常見的若干值
    """
    v = values.reshape(-1)
    q = np.round(v*denom)
    mask = np.abs(v*denom - q) < atol
    ratio = float(mask.mean())
    from collections import Counter
    c = Counter((q[mask]/denom).tolist()).most_common(10)
    return ratio, c

# ------------ 主流程 ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_savedmodel", type=str, default=None)
    ap.add_argument("--student_savedmodel", type=str, default=None)
    ap.add_argument("--student_tflite", type=str, default=None)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--num_kpts", type=int, required=True)
    ap.add_argument("--reg_max_candidates", type=int, nargs="+", default=[16,32])
    ap.add_argument("--try_obj", type=int, nargs="+", default=[0,1], help="嘗試有無 obj 分支")
    args = ap.parse_args()

    outs = {}

    if args.teacher_savedmodel:
        out, ishape = run_savedmodel(args.teacher_savedmodel)
        outs["teacher"] = {"raw": out, "ishape": ishape}
        print(f"[Teacher] input shape: {ishape}, output: {pretty_shape(out)}")

    if args.student_savedmodel:
        out, ishape = run_savedmodel(args.student_savedmodel)
        outs["student_sm"] = {"raw": out, "ishape": ishape}
        print(f"[Student(SM)] input shape: {ishape}, output: {pretty_shape(out)}")

    if args.student_tflite:
        out, ishape = run_tflite(args.student_tflite)
        outs["student_tfl"] = {"raw": out, "ishape": ishape}
        print(f"[Student(TFLite)] input shape: {ishape}, output: {pretty_shape(out)}")

    if not outs:
        print("請至少提供一個模型路徑")
        sys.exit(1)

    # 逐個輸出做佈局猜測與量化偵測
    parsed = {}
    for name, d in outs.items():
        CN = as_CN(d["raw"]).astype(np.float32)  # (C,N)
        C,N = CN.shape
        print(f"\n=== {name} -> (C,N)=({C},{N}) ===")
        # 猜佈局
        cands = guess_layout(C, args.num_classes, args.num_kpts,
                             reg_max_candidates=tuple(args.reg_max_candidates),
                             try_obj=tuple(args.try_obj))
        if not cands:
            print(f"  [!] 無法用 XYWH/DFL + obj∈{args.try_obj} 拆解 C={C}，請檢查通道定義。")
        else:
            print("  可能的頭部配置：")
            for kk in cands:
                print(f"   - kind={kk['kind']}, reg_max={kk['reg_max']}, has_obj={kk['has_obj']}")

        # 量化偵測（全輸出）
        ratio, topv = quant_grid_ratio(CN, denom=256, atol=1e-6)
        print(f"  量化格點(1/256)命中率：{ratio*100:.2f}%")
        if topv:
            s = ", ".join([f"{v:.6f}×{cnt}" for v,cnt in topv[:5]])
            print(f"  最常見值（前5）：{s}")

        parsed[name] = {"CN": CN, "cands": cands}

    # 如果同時有 Teacher 與 Student，嘗試對齊並計算指標
    if "teacher" in parsed and ("student_tfl" in parsed or "student_sm" in parsed):
        student_key = "student_tfl" if "student_tfl" in parsed else "student_sm"
        T_CN, S_CN = parsed["teacher"]["CN"], parsed[student_key]["CN"]

        # 為 Teacher 與 Student 分別挑一個「最合理」的配置（偏好 DFL>XYWH、has_obj=0>1）
        def pick(cfgs):
            if not cfgs: return None
            cfgs = sorted(cfgs, key=lambda z: (z["kind"]!="dfl", z["has_obj"]))  # DFL優先，無obj優先
            return cfgs[0]
        T_cfg = pick(parsed["teacher"]["cands"])
        S_cfg = pick(parsed[student_key]["cands"])

        if not T_cfg or not S_cfg:
            print("\n[!] 無法同時為 Teacher 與 Student 找到可用頭部配置，略過對齊評估。")
            return

        print(f"\n對齊採用配置：Teacher={T_cfg} | Student={S_cfg}")

        T_seg = split_segments(T_CN, T_cfg, args.num_classes, args.num_kpts)
        S_seg = split_segments(S_CN, S_cfg, args.num_classes, args.num_kpts)

        # Teacher box：若 DFL 則解碼成 xywh-like；若 XYWH 直接取
        if "box_dfl" in T_seg:
            T_box = decode_dfl(T_seg["box_dfl"])
        else:
            T_box = T_seg["box_raw"]
        if "box_dfl" in S_seg:
            S_box = decode_dfl(S_seg["box_dfl"])
        else:
            S_box = S_seg["box_raw"]

        # 對 cls/kpt 做基本激活處理（視情況你可改 logits/KL）
        T_cls = T_seg["cls"]; S_cls = S_seg["cls"]
        # 假設輸出已是 0~1 機率；若你輸出是 logits，改成 sigmoid(T_cls) / sigmoid(S_cls)
        # T_cls, S_cls = sigmoid(T_cls), sigmoid(S_cls)

        T_kpt = T_seg["kpt"]  # (K,3,N)
        S_kpt = S_seg["kpt"]

        print("\n--- 對齊評估 (Teacher vs Student) ---")
        print(f"box: MAE={mae(T_box,S_box):.6f} | corr={corr(T_box,S_box):.4f}")
        # obj（如有）
        if ("obj" in T_seg) and ("obj" in S_seg):
            print(f"obj: MAE={mae(T_seg['obj'], S_seg['obj']):.6f} | corr={corr(T_seg['obj'], S_seg['obj']):.4f}")
        else:
            print("obj: [兩邊至少一邊無 obj 分支]")

        print(f"cls: MAE={mae(T_cls,S_cls):.6f} | corr={corr(T_cls,S_cls):.4f}")
        print(f"kpt(x,y,v): MAE={mae(T_kpt,S_kpt):.6f} | corr={corr(T_kpt,S_kpt):.4f}")

        # 範圍檢查（0~1）
        def range_report(tag, a):
            mn, mx = float(a.min()), float(a.max())
            out_of = float(np.mean((a<0)|(a>1))*100.0)
            print(f"{tag}: range=[{mn:.3f},{mx:.3f}] | 超出[0,1]比例={out_of:.2f}%")
        range_report("Teacher box(decoded)", T_box)
        range_report("Student box(decoded)", S_box)
        range_report("Teacher kpt", T_kpt)
        range_report("Student kpt", S_kpt)

        # 若 Student 有明顯 1/256 階梯，特別提示
        S_ratio, S_top = quant_grid_ratio(S_CN, denom=256, atol=1e-6)
        if S_ratio > 0.3:
            print("\n[提示] Student 的輸出看起來被 8-bit 階梯量化嚴重夾住，建議蒸餾時在『未量化的 logits/浮點節點』取特徵對齊，或關閉輸出層假量化。")
            if S_top:
                print("       最常見階梯值（前5）:", ", ".join([f"{v:.6f}×{cnt}" for v,cnt in S_top[:5]]))

    print("\n完成。若有任何一側的『可能頭部配置』列表為空，通常代表：通道順序/是否 DFL/是否含 obj 與你假設不一致（請對照你的 C++ 解碼邏輯）。")

if __name__ == "__main__":
    main()
