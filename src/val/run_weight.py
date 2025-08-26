# search_n_and_c_alignment.py
import numpy as np
import cv2
import tensorflow as tf
import itertools
from scipy.optimize import linear_sum_assignment

IMG_PATH = 'test.jpg'  # 換成你的測試圖
OFF_PATH = 'lanepose20250807_s_model_640_640_6c_v1_integer_quant.tflite'
QAT_PATH = 'best_qat_int8.tflite'

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw //= 2; dh //= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return im

def run_tflite(model_path, img):
    itp = tf.lite.Interpreter(model_path=model_path)
    itp.allocate_tensors()
    i = itp.get_input_details()[0]['index']; o = itp.get_output_details()[0]['index']
    itp.set_tensor(i, img[None].astype('float32'))
    itp.invoke()
    return itp.get_tensor(o)[0]  # [C,N]

def mean_hungarian_corr(A, B, eps=1e-8):
    # A,B: [C,N]，先做每通道標準化，再做 Hungarian 取最佳配對的平均相關
    C = A.shape[0]
    A0 = A - A.mean(axis=1, keepdims=True); B0 = B - B.mean(axis=1, keepdims=True)
    A0 /= (A0.std(axis=1, keepdims=True) + eps)
    B0 /= (B0.std(axis=1, keepdims=True) + eps)
    sim = A0 @ B0.T / (A.shape[1] - 1)           # [C,C]
    cost = -sim
    r, c = linear_sum_assignment(cost)           # r = 0..C-1
    return float(sim[r, c].mean()), c.tolist()   # 平均相關、mapping: A_i -> B_{c[i]}

def reorder_grid_block(block, H, W, scan, fy, fx):
    # block: [C, H*W]（預設 row-major）
    x = block.reshape(block.shape[0], H, W)
    if scan == 'col':
        x = np.transpose(x, (0,2,1))  # [C,W,H]
        H, W = W, H
    if fy: x = x[:, ::-1, :]
    if fx: x = x[:, :, ::-1]
    return x.reshape(block.shape[0], H*W)

# --- 讀圖 + 前處理 ---
img0 = cv2.imread(IMG_PATH)[:, :, ::-1]
img0 = letterbox(img0, 640) / 255.0

# --- 跑兩個模型 ---
y_off = run_tflite(OFF_PATH, img0)  # [C,8400]
y_qat = run_tflite(QAT_PATH, img0)
C, N = y_off.shape
assert y_qat.shape == y_off.shape == (C, 8400)

# --- 切成 P3/P4/P5 ---
cuts = [6400, 8000, 8400]
def split3(y):
    return y[:, :cuts[0]], y[:, cuts[0]:cuts[1]], y[:, cuts[1]:cuts[2]]
off3 = split3(y_off)
qat3 = split3(y_qat)
Hs, Ws = [80,40,20], [80,40,20]

orders = list(itertools.permutations([0,1,2]))
grid_modes = [(scan, fy, fx) for scan in ['row','col'] for fy in [0,1] for fx in [0,1]]

best = (-1, None, None, None)  # (score, porder, (m0,m1,m2), mapping)

for pord in orders:
    for m0 in grid_modes:
        b0 = reorder_grid_block(qat3[pord[0]], Hs[pord[0]], Ws[pord[0]], *m0)
        for m1 in grid_modes:
            b1 = reorder_grid_block(qat3[pord[1]], Hs[pord[1]], Ws[pord[1]], *m1)
            for m2 in grid_modes:
                b2 = reorder_grid_block(qat3[pord[2]], Hs[pord[2]], Ws[pord[2]], *m2)
                yq = np.concatenate([b0,b1,b2], axis=1)  # [C,8400]
                score, mapping = mean_hungarian_corr(y_off, yq)
                if score > best[0]:
                    best = (score, pord, (m0,m1,m2), mapping)

print('BEST score:', best[0])
print('BEST P-order:', best[1])
print('BEST grid modes:', best[2])  # 三層各自 (scan, fy, fx)
print('BEST channel mapping (official i -> qat j):', best[3])
