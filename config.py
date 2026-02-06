import tensorflow as tf

'''
===================================================
Training Settings
===================================================
'''

IMGSZ = 640
SEED = 42
VAL_SPLIT = 0.05  # 建議 2%~10% 做驗證 (label 模式)
BATCH = 2
EPOCHS = 20              # 可先跑 5~10 看收斂

base_lr = 0.001  # QAT 建議從小 LR fine-tune 起跑
end_lr = 0.0001
momentum = 0.9

LETTERBOX_PAD_VALUE = 114.0 / 255.0 

MAX_OBJS = 64   # 單張圖最大標註數

BNSTOP__ = True         # 凍結 BN , Ture不凍結/ False凍結
USE_AMP = False         # 設定為 True 以啟用混合精度訓練 (Tensor 版本不支援)

PLOT_Switch = True     # 是否繪製數據圖, 若 matplotlib 版本不符可關閉
EXPORT_ONLY = False      # True 是否只進行輸出測試（.ckpt）, False 進行蒸餾QAT輸出

TFLITE_QUANT_MODE = "int8"  # 可選: "int8" | "fp16" | "fp32"

# 監督來源：'distill' 或 'label#
TRAIN_SUPERVISION = 'label'  # 'label' 或 'distill'
KD_LOSS_WEIGHT = 1.0
DEPLOY_LOSS_WEIGHT = 1.0

USE_DFL  = False       # ← 關掉 DFL

# 你的資料：v=0 仍有 xy，所以要開 True
KPT_SUPERVISE_XY_WHEN_V0 = False

AUTO_TOUCH_MISSING_LABELS = False

KPT_CLASS_MASK = [
    # cls=0 lane: 15 points
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],

    # cls=1 vehicle skeleton: 12 points
    # 依你這份標註：第9、14、15點是 v=0（1-based）
    # => 0-based index = 8,13,14 關掉
    [1,1,1,1,1,1,1,1,0,1,1,1,1,0,0],

    # cls=2~6: placeholder center point with v=0 => disable all kpt loss
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # cls=2
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # cls=3
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # cls=4
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # cls=5
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # cls=6
]

TAL_TOPK = 10
TAL_ALPHA = 0.5
TAL_BETA  = 6.0
TAL_CENTER_RADIUS = 2.5

VFL_ALPHA = 0.75
VFL_GAMMA = 2.0

EMA_DECAY = 0.9998
KPT_VIS_W = 0.5


'''
===================================================
Location (Input/Output)
===================================================
'''
"""Train Dataset"""
REP_DIR_train = [
    
    # "../Dataset/20220830/images/*.jpg",
    # "../Dataset/20220830_enhance/images/*.jpg",
    # "../Dataset/20220830-Stable_Diffusion-enhance/images/*.png",

    # "../Dataset/20240116-nocheck/images/*.jpg",
    # "../Dataset/20240321_night/images/*.jpg",
    # "../Dataset/20240321_night-Stable_Diffusion-enhance/images/*.png",
    # "../Dataset/20240603-nocheck/images/*.jpg",
    # "../Dataset/20240803-nocheck/images/*.jpg",
    # "../Dataset/20240923-BigSun-nocheck/images/*.jpg",
    # "../Dataset/20241010-nocheck/images/*.jpg",
    # "../Dataset/20241126-Bridge-nocheck/images/*.jpg",
    # "../Dataset/20250213-nocheck/images/*.jpg",

    # "../Dataset/acc_dataset/images/*.jpg",
    # "../Dataset/acc_dataset_enhance/images/*.jpg",
    # "../Dataset/acc_dataset-Stable_Diffusion-enhance/images/*.png",
    # "../Dataset/s3_20230803/images/*.jpg",
    # "../Dataset/s3_20230803_enhance/images/*.jpg",
    # "../Dataset/s3_20230803-Stable_Diffusion-enhance/images/*.png",
    # "../Dataset/Traffic_dataset_20240720_345_k/images/*.jpg",
    # "../Dataset/Traffic_dataset_20240720_345_k_enhance/images/*.jpg",

    # "../Dataset/vecow-demo-nocheck/images/*.jpg",

    # "../Dataset/yolov8data2_20250804/images/*.jpg",
    # "../Dataset/yolov8data2_20250804_enhance/images/*.jpg"
    #"../dataset/lanepose/acc_datasets/images/*.jpg"

    "../dataset/lanepose/acc_datasets/images/*.jpg"
    
]

"""TFlite Validation Dataset"""
# REP_DIR_export = "../_Dataset/KeyPoint/temp/mix_QAT/images/*.jpg"
REP_DIR_export = "../dataset/lanepose/test1/images/*.jpg"

"""Teacher Model"""
EXPORTED_DIR = "./lanepose20250807_s_model_640_640_6c_v1_saved_model/"

"""Output File"""
TFLITE_OUT = "./output"

"""Export_Only Load Model""" 
RESUME_WEIGHTS = "./output/20250910_135119/models/qat_saved_model"

'''
===================================================
Model Settings
===================================================
'''
# Seting
NUM_CLS = 7          # 你的資料集類別數
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





# ===================================================
# Export / TFLite output semantics (must match TFlite.h)
# ===================================================
EXPORT_APPLY_SIGMOID_BOX = True
EXPORT_APPLY_SIGMOID_CLS = True
EXPORT_APPLY_SIGMOID_KPTXY = True
EXPORT_APPLY_SIGMOID_KPTV = True
