import tensorflow as tf

'''
===================================================
Training Settings
===================================================
'''
IMGSZ = 640
BATCH = 2
EPOCHS = 20              # 可先跑 5~10 看收斂

base_lr = 0.01
end_lr = 0.001
momentum = 0.9

BNSTOP__ = True         # 凍結 BN , Ture不凍結/ False凍結
USE_AMP = False         # 設定為 True 以啟用混合精度訓練 (Tensor 版本不支援)

PLOT_Switch = False     # 是否繪製數據圖, 若 matplotlib 版本不符可關閉
EXPORT_ONLY = False      # True 是否只進行輸出測試（.ckpt）, False 進行蒸餾QAT輸出


'''
===================================================
Location (Input/Output)
===================================================
'''
"""Train Dataset"""
REP_DIR_train = [
    # "../_Dataset/KeyPoint/15point_6class_box0/20220830/images/*.jpg",
    # "../_Dataset/KeyPoint/15point_6class_box0/20240321_night/images/*.jpg",
    # "../_Dataset/KeyPoint/15point_6class_box0/acc_datasets/images/*.jpg",
    # "../_Dataset/KeyPoint/15point_6class_box0/s3_20230803/images/*.jpg",
    # "../_Dataset/KeyPoint/15point_6class_box0/Traffic_dataset_20240720_345_k/images/*.jpg",
    # "../_Dataset/KeyPoint/15point_6class_box0/yolov8data2_20250804/images/*.jpg"
    # "../_Dataset/KeyPoint/temp/mix_QAT/images/*.jpg"

    "../dataset/lanepose/mix_QAT/images/*.jpg"    
]

"""TFlite Validation Dataset"""
# REP_DIR_export = "../_Dataset/KeyPoint/temp/mix_QAT/images/*.jpg"
REP_DIR_export = "../dataset/lanepose/test1/images/*.jpg"

"""Teacher Model"""
EXPORTED_DIR = "./lanepose20250807_s_model_640_640_6c_v1_saved_model/"

"""Output File"""
TFLITE_OUT = "./output"

"""Export_Only Load Model""" 
RESUME_WEIGHTS = "./output/20250820_161954/models/qat_saved_model_interrupted"

'''
===================================================
Model Settings
===================================================
'''
# Seting
NUM_CLS = 7          # 你的資料集類別數(0 ~ 6 = 7, -> ans = 6)
NUM_KPT = 15         # 你的關鍵點數
KPT_VALS = 3         # YOLOv8-Pose 預設每點 3 個值: (x, y, score/logit)

# Weigth
W_BOX = 7.0
W_OBJ = 1.0          # 沒有
W_CLS = 1.0
W_KPT_XY = 12.0
W_KPT_V  = 1.0       # 關鍵點 score/logit 權重


'''
===================================================
System Settings
===================================================
'''
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
C = 4 + NUM_CLS + NUM_KPT * KPT_VALS

PORDER = (0, 1, 2)
GRID_MODES = (('col',1,0), ('row',0,0), ('col',1,0))

CHANNEL_MAPPING = list(range(C))
XYWH_TO_LTRB = False            # 模型輸出為ltrb:True, 輸出為xy:False
XYWH_IS_NORMALIZED_01 = False   # 模型輸出是否經過歸一化

STOP_REQUESTED = False          # 全域旗標：一旦收到中斷訊號就設 True