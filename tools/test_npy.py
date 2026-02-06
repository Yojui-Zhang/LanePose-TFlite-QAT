import numpy as np

# 載入剛剛存的檔案
data = np.load("debug_yolo_output.csv.npy")

print("="*30)
print(f"資料形狀: {data.shape}")
print("="*30)

# YOLOv8-pose 的輸出通常是 (1, 56, 8400)
# 1: Batch size
# 56: 資訊維度 (4個Box座標 + 1個信心度 + 17*3個關鍵點資訊)
# 8400: 候選框 (Anchor) 數量

# 我們先把它轉置成 (1, 8400, 56) 比較好讀
data = np.transpose(data, (0, 2, 1))
squeezed_data = data[0] # 取出 batch 0 -> (8400, 56)

# --- 檢查 1: 信心度 (Confidence) ---
# 在 YOLOv8 中，index 4 通常是 Objectness/Class score
conf_scores = squeezed_data[:, 4]

max_conf = np.max(conf_scores)
avg_conf = np.mean(conf_scores)

print(f"最大信心度 (Max Confidence): {max_conf:.6f}")
print(f"平均信心度 (Avg Confidence): {avg_conf:.6f}")

# --- 診斷 ---
if max_conf < 0.01:
    print("\n[嚴重警告] 模型輸出的最高信心度極低 (< 0.01)！")
    print("可能原因：")
    print("1. 輸入圖片沒有做歸一化 (除以 255.0)。")
    print("2. 輸入圖片是 BGR，但模型訓練時是用 RGB (顏色通道錯了)。")
    print("3. 模型訓練失敗，權重爛掉了。")
elif max_conf > 1.0:
    print("\n[警告] 信心度大於 1.0，這不正常 (應該在 0~1 之間)。")
    print("可能原因：輸出層沒有包含 Sigmoid 激活函數 (Raw Logits)。")
else:
    print("\n[資訊] 信心度數值範圍正常。")
    # 列出前 5 個信心度最高的框看看數值
    top_indices = np.argsort(conf_scores)[::-1][:5]
    print("\n前 5 個高信心度候選框的數值 (前 5 碼通常是 xywh + conf):")
    for idx in top_indices:
        print(f"Index {idx}: {squeezed_data[idx, :5]}")


