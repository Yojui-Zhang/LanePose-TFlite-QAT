import cv2
import tensorflow as tf
import numpy as np

# ---------------------------------------------------
NUM_CLASSES = 7

Image_Size_X = 640
Image_Size_Y = 640

CONF_thres = 0.4
IOU_thres = 0.4

# ---------------------------------------------------
# 定義骨架連接關係
SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]
# 關鍵點顏色 (B, G, R)
KP_COLOR = (0, 255, 0)
# 連結線顏色
LIMB_COLOR = (0, 255, 255)
# 類別顏色 (多類別時用不同顏色區分框)
CLASS_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)] 

input_shape = (Image_Size_X, Image_Size_Y) 
# ---------------------------------------------------

def run_tf_inference_with_viz(video_path, model_path, conf_thres=0.1, iou_thres=0.01):
    # 1. 載入模型
    print(f"正在載入模型: {model_path}...")
    try:
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    # 取得原始影片尺寸
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"開始推論 (類別數設定: {NUM_CLASSES})... (按 'q' 離開)")

    has_saved_debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 2. 影像前處理 ---
        img_resized = cv2.resize(frame, input_shape)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_norm = img_rgb / 255.0
        img_tensor = tf.convert_to_tensor(img_norm[np.newaxis, ...], dtype=tf.float32)

        # --- 3. 推論 ---
        output = infer(img_tensor)
        raw_tensor = list(output.values())[0].numpy()
        
        # --- [新增] 儲存除錯檔案 (只存第一次) ---
        if not has_saved_debug:
            save_path_npy = "debug_yolo_output.csv"
            np.save(save_path_npy, raw_tensor)
            print(f"\n[DEBUG] 成功儲存推論結果至: {save_path_npy}")
            print(f"[DEBUG] 輸出形狀 (Shape): {raw_tensor.shape}")
            # 順便直接在 Terminal 印出一些統計數據幫你快速判斷
            print(f"[DEBUG] 最大值 (Max): {np.max(raw_tensor)}")
            print(f"[DEBUG] 最小值 (Min): {np.min(raw_tensor)}")
            print(f"[DEBUG] 平均值 (Mean): {np.mean(raw_tensor)}")

            save_path_csv = "debug_yolo_output.csv"
            
            # --- 處理維度問題 ---
            # CSV 只能存 1D 或 2D 資料。如果 tensor 是 3D 以上 (例如 (1, 17, 3))，
            # 需要先 reshape 或 flatten (攤平)。這裡選擇 flatten 確保一定能存。
            if raw_tensor.ndim > 2:
                print(f"[DEBUG] 偵測到高維度資料 {raw_tensor.shape}，已自動攤平為 1D 以存入 CSV")
                data_to_save = raw_tensor.flatten()
            else:
                data_to_save = raw_tensor

            # --- 核心修改：使用 savetxt 存成 CSV ---
            # delimiter=",": 用逗號分隔
            # fmt='%.6f': 格式化為小數點後6位 (避免科學記號看起來很亂)
            np.savetxt(save_path_csv, data_to_save, delimiter=",", fmt='%.6f')
            
            print(f"\n[DEBUG] 成功儲存推論結果至: {save_path_csv}")
            print(f"[DEBUG] 輸出形狀 (Shape): {raw_tensor.shape}")
            
            # 統計數據 (保持不變)
            print(f"[DEBUG] 最大值 (Max): {np.max(raw_tensor)}")
            print(f"[DEBUG] 最小值 (Min): {np.min(raw_tensor)}")
            print(f"[DEBUG] 平均值 (Mean): {np.mean(raw_tensor)}")
            
            has_saved_debug = True # 鎖住，避免下一幀重複存

        # 取得輸出並轉成 numpy
        pred = list(output.values())[0].numpy()
        
        # 確保形狀是 (Batch, Anchors, Channels) -> (1, 8400, 56)
        # 如果模型輸出是 (1, 56, 8400)，需要轉置
        if pred.shape[1] < pred.shape[2]: 
            pred = np.transpose(pred, (0, 2, 1))
        
        pred = pred[0] # 取出 batch 0 -> Shape: (8400, 4 + NUM_CLASSES + Keypoints)

        # --- 4. 後處理 (Post-Processing) - 多類別修改版 ---
        
        # 4.1 解析 Tensor
        # 座標: 前 4 個
        bboxes = pred[:, :4] 
        # 類別分數: 從 index 4 到 index 4 + 類別數
        class_scores = pred[:, 4 : 4 + NUM_CLASSES] 
        # Keypoints: 剩下的部分
        kpts_data = pred[:, 4 + NUM_CLASSES :] 

        # 4.2 找出每個 Anchor 的「最大信心類別」與「分數」
        class_ids = np.argmax(class_scores, axis=1)
        # max 找出該類別的分數
        confidences = np.max(class_scores, axis=1)

        # 4.3 過濾低信心度
        mask = confidences > conf_thres
        
        bboxes_filtered = bboxes[mask]
        class_ids_filtered = class_ids[mask]
        confidences_filtered = confidences[mask]
        kpts_filtered = kpts_data[mask]
        
        if len(bboxes_filtered) == 0:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) == ord('q'): break
            continue

        # 4.4 準備 NMS 數據 (YOLO [cx, cy, w, h] -> [x, y, w, h])
        boxes_xywh = bboxes_filtered.copy()
        boxes_xywh[:, 0] = bboxes_filtered[:, 0] - bboxes_filtered[:, 2] / 2  # x_top_left
        boxes_xywh[:, 1] = bboxes_filtered[:, 1] - bboxes_filtered[:, 3] / 2  # y_top_left
        
        # 4.5 執行 NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(), 
            scores=confidences_filtered.tolist(), 
            score_threshold=conf_thres, 
            nms_threshold=iou_thres
        )

        # --- 5. 繪製結果 ---
        scale_x = orig_w / input_shape[0]
        scale_y = orig_h / input_shape[1]

        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                
                # 取得該物件資訊
                box = boxes_xywh[idx]
                cls_id = class_ids_filtered[idx]
                conf = confidences_filtered[idx]

                # 選擇顏色
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                # 還原座標
                x, y, w, h = box

                x = (x * Image_Size_X) * scale_x
                y = (y * Image_Size_Y) * scale_y
                w = (w * Image_Size_X) * scale_x
                h = (h * Image_Size_Y) * scale_y
                
                # 畫 Bounding Box
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                
                # 畫類別名稱與分數
                label = f"Class {cls_id}: {conf:.2f}"
                cv2.putText(frame, label, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 處理 Keypoints
                kpts = kpts_filtered[idx]
                
                parsed_kpts = []
                for k in range(0, len(kpts), 3):
                    kx, ky, kconf = kpts[k], kpts[k+1], kpts[k+2]
                    if kconf > 0.5: 
                        # Keypoints 座標還原
                        cx = int((kx * Image_Size_X) * scale_x)
                        cy = int((ky * Image_Size_Y) * scale_y)
                        parsed_kpts.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 4, KP_COLOR, -1)
                    else:
                        parsed_kpts.append(None)

                # 畫骨架
                for p1_idx, p2_idx in SKELETON:
                    if p1_idx-1 < len(parsed_kpts) and p2_idx-1 < len(parsed_kpts):
                        pt1 = parsed_kpts[p1_idx-1]
                        pt2 = parsed_kpts[p2_idx-1]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(frame, pt1, pt2, LIMB_COLOR, 2)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    video = "./vecow-demo.mp4"
    # model_dir = "./model/carkeypoint-20251122-Rep-s2_saved_model" 
    model_dir = "./model/20251208_111850/models/qat_saved_model" 
    
    run_tf_inference_with_viz(video, model_dir, CONF_thres, IOU_thres)
