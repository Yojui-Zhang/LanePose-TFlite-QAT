import cv2
import tensorflow as tf
import numpy as np

# ---------------------------------------------------
NUM_CLASSES = 7          # 跟你現在的一樣
Image_Size_X = 640
Image_Size_Y = 640

CONF_thres = 0.4
IOU_thres  = 0.4

# ---------------------------------------------------
# 定義骨架連接關係（沿用你原本的）
SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]
KP_COLOR   = (0, 255, 0)       # 關鍵點顏色 (BGR)
LIMB_COLOR = (0, 255, 255)     # 骨架線顏色
CLASS_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

input_shape = (Image_Size_X, Image_Size_Y)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_tf_inference_with_viz(video_path, model_path,
                              conf_thres=CONF_thres, iou_thres=IOU_thres):
    # 1. 載入模型
    print(f"正在載入模型: {model_path} ...")
    try:
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"開始推論 (NUM_CLASSES = {NUM_CLASSES}) ... (按 'q' 離開)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. 影像前處理
        img_resized = cv2.resize(frame, input_shape)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_tensor = tf.convert_to_tensor(img_norm[np.newaxis, ...], dtype=tf.float32)

        # 3. 推論
        output = infer(img_tensor)
        pred = list(output.values())[0].numpy()  # 例如 (1, 56, 8400) or (1, 8400, 56)

        # 統一成 (1, N, C)
        if pred.shape[1] < pred.shape[2]:  # (1, C, N) -> (1, N, C)
            pred = np.transpose(pred, (0, 2, 1))

        pred = pred[0]   # (N, C)

        # ====== 4. 依照「loss_tf 的假設」重新 decode ======
        # [0:4]   : box logits
        # [4:4+C] : class logits
        # [其餘] : keypoints logits
        box_logits = pred[:, :4]
        cls_logits = pred[:, 4:4 + NUM_CLASSES]
        kpt_logits = pred[:, 4 + NUM_CLASSES:]

        # 把 logits 壓成 0~1
        boxes = _sigmoid(box_logits)   # (N, 4), cx,cy,w,h in (0,1)
        cls_prob = _sigmoid(cls_logits)  # (N, num_classes)
        kpts = _sigmoid(kpt_logits)    # (N, num_kpt * 3)

        # 每個 anchor 的最佳類別與信心
        class_ids = np.argmax(cls_prob, axis=1)
        confidences = np.max(cls_prob, axis=1)

        # 4.1 過濾低信心
        mask = confidences > conf_thres
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        kpts = kpts[mask]

        # 沒預測到就直接顯示原圖
        if len(boxes) == 0:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # 4.2 準備 NMS：YOLO [cx,cy,w,h] -> [x,y,w,h] (左上角座標 + 寬高，仍為 0~1)
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x_min
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y_min

        # cv2.dnn.NMSBoxes 期待的是像素單位，但 IoU 與是否 scale 無關，
        # 這裡直接轉成 640 基準像素再丟進去，避免 threshold 太敏感
        pixel_boxes_xywh = np.zeros_like(boxes_xywh)
        pixel_boxes_xywh[:, 0] = boxes_xywh[:, 0] * Image_Size_X
        pixel_boxes_xywh[:, 1] = boxes_xywh[:, 1] * Image_Size_Y
        pixel_boxes_xywh[:, 2] = boxes[:, 2] * Image_Size_X
        pixel_boxes_xywh[:, 3] = boxes[:, 3] * Image_Size_Y

        indices = cv2.dnn.NMSBoxes(
            bboxes=pixel_boxes_xywh.tolist(),
            scores=confidences.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )

        # 4.3 還原到原始影像尺寸
        scale_x = orig_w / Image_Size_X
        scale_y = orig_h / Image_Size_Y

        if len(indices) > 0:
            for idx in indices:
                # 可能是 [[i]] 或 [i]
                if isinstance(idx, (list, np.ndarray)):
                    idx = idx[0]

                box_xywh = pixel_boxes_xywh[idx]
                cls_id = int(class_ids[idx])
                conf = float(confidences[idx])
                kpts_obj = kpts[idx]

                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                x, y, w, h = box_xywh
                # 轉回原始 frame 尺度
                x = x * scale_x
                y = y * scale_y
                w = w * scale_x
                h = h * scale_y

                # 畫 bbox
                cv2.rectangle(
                    frame,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    color,
                    2
                )

                # 畫 label
                label = f"Class {cls_id}: {conf:.2f}"
                cv2.putText(
                    frame, label,
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2
                )

                # 解析並畫 keypoints
                parsed_kpts = []
                for k in range(0, len(kpts_obj), 3):
                    kx, ky, kconf = kpts_obj[k], kpts_obj[k+1], kpts_obj[k+2]

                    if kconf < 0.5:
                        parsed_kpts.append(None)
                        continue

                    px = int(kx * Image_Size_X * scale_x)
                    py = int(ky * Image_Size_Y * scale_y)
                    parsed_kpts.append((px, py))
                    cv2.circle(frame, (px, py), 4, KP_COLOR, -1)

                # 畫骨架
                for p1_idx, p2_idx in SKELETON:
                    if p1_idx-1 < len(parsed_kpts) and p2_idx-1 < len(parsed_kpts):
                        pt1 = parsed_kpts[p1_idx-1]
                        pt2 = parsed_kpts[p2_idx-1]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(frame, pt1, pt2, LIMB_COLOR, 2)

        # 顯示畫面
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = "./vecow-demo.mp4"
    model_dir = "./model/20251208_200732/models/qat_saved_model_interrupted"
    run_tf_inference_with_viz(video, model_dir, CONF_thres, IOU_thres)
