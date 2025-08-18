import tensorflow as tf

IMGSZ = 640
BATCH = 2
EPOCHS = 100  # 可先跑 5~10 看收斂

base_lr = 3e-3
end_lr = 1e-5
momentum = 0.9


REP_DIR = "../dataset/lanepose/mix_QAT/images/*.jpg"  # 代表集資料夾（放 500~1000 張）

EXPORTED_DIR = "./lanepose20250807_s_model_640_640_6c_v1_saved_model/"  # Ultralytics 匯出路徑
TFLITE_OUT = "./output"

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
NUM_CLS = 7          # 你的資料集類別數(0 ~ 6 = 7, -> ans = 6)
NUM_KPT = 15         # 你的關鍵點數
KPT_VALS = 3         # YOLOv8-Pose 預設每點 3 個值: (x, y, score/logit)
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

# 權重可依驗證結果微調
W_BOX = 1.0
W_OBJ = 0.0
W_CLS = 0.5
W_KPT_XY = 2.0
W_KPT_S  = 0.25  # 關鍵點 score/logit 權重

BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

PORDER = (0, 1, 2)
GRID_MODES = (('col',1,0), ('row',0,0), ('col',1,0))
CHANNEL_MAPPING = list(range(C))
XYWH_TO_LTRB = False
XYWH_IS_NORMALIZED_01 = False