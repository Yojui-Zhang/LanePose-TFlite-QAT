# Dyn KD weight

python run_paper_experiments.py \
    --datasets kitti --studies B --seeds 0,1,2,3,4 \
    --epochs 200 --batch 64 --close-mosaic 10 \
    --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --device-kitti 1 --export-fraction 0.25 \
    --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
    --data-root ./Paper-Data/Data_kitti_B_pureKD_tflite_int8_0309 \
    --qat-kd-temperature 1.0 \
    --qat-kd-cls-distill bce \
    --qat-kd-dfl-distill kldiv \
    --qat-kd-fg-threshold 0.25 \
    --qat-kd-fg-topk 800 \
    --qat-kd-fg-min-pos 200 \
    --qat-kd-fg-apply-to both \
    --include-b-pure-kd \
    --skip-existing \
    --eval-tflite-map \
    --tflite-map-split val

  kd_deploy 則用這個：

  python run_paper_experiments.py \
    --datasets kitti --studies B --seeds 0,1,2,3,4 \
    --epochs 200 --batch 64 --close-mosaic 10 \
    --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --device-kitti 1 --export-fraction 0.25 \
    --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
    --data-root ./Paper-Data/Data_kitti_B_kdDeploy_tflite_int8_0309 \
    --qat-kd-temperature 1.0 \
    --qat-kd-cls-distill bce \
    --qat-kd-dfl-distill kldiv \
    --qat-kd-fg-threshold 0.25 \
    --qat-kd-fg-topk 800 \
    --qat-kd-fg-min-pos 200 \
    --qat-kd-fg-apply-to both \
    --qat-balance-strategy grad_norm \
    --qat-balance-shared-group head \
    --qat-balance-min 0.10 \
    --qat-balance-max 2.0 \
    --qat-balance-warmup-steps 2000 \
    --qat-balance-deploy-ramp-steps 800 \
    --qat-balance-update-interval 10 \
    --qat-balance-max-step-change 1.20 \
    --qat-balance-adapt-power 0.50 \
    --qat-balance-log-interval 20 \
    --include-b-kd-deploy \
    --skip-existing \
    --eval-tflite-map \
    --tflite-map-split val


# kd-weight

python run_paper_experiments.py \
  --datasets kitti --studies B --seeds 3,4 \
  --epochs 200 --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 0 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --data-root ../Paper-Data/Data_kitti_onlyB_kdweight025 \
  --qat-kd-weight 0.25 \
  --qat-balance-log-interval 20 \
  --skip-existing


# ------------------------------------------------------------------------------
# mulit model

python run_paper_experiments.py \
  --datasets kitti \
  --studies A \
  --seeds 0 \
  --epochs 10 \
  --kitti-mobilenetv3-model ./ultralytics/cfg/models/Yojui/yolov8_MobileNetV3-Lite.yaml \
  --kitti-ghostnetv2-model  ./ultralytics/cfg/models/Yojui/yolov8_GhostNetV2-Lite.yaml \
  --kitti-shufflenetv2-model ./ultralytics/cfg/models/Yojui/yolov8_ShuffleNetV2-Lite.yaml \
  --kitti-cira-lite-model ./ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml \
  --skip-existing


python run_paper_experiments.py \
  --datasets kitti \
  --studies B \
  --seeds 0,1,2,3,4 \
  --epochs 200 \
  --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 0 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --data-root ../Paper-Data/Data_kitti_OnlyB_and_onlyKD \
  --qat-kd-weight 1.0 \
  --qat-balance-min 0.0 \
  --qat-balance-max 1.0 \
  --qat-balance-log-interval 20 \
  --skip-existing


# ------------------------------------------------------------------------------
# Smoke Test
python run_paper_experiments.py --smoke --max-runs 1 --datasets kitti --studies A --seeds 0

# ------------------------------------------------------------------------------
# Val
yolo val model=../Paper-Data/Data_kitti_OnlyB_onlyDeploy_dynKD_onlyKD_tflite_int8_0225/paper_reports/artifacts/B/kitti/kd_deploy/kd-deploy/seed0/B_kitti_kd_deploy_seed0_20260226_0134580800/model_fp32.tflite data=./dataset/KITTI.yaml imgsz=640

# ------------------------------------------------------------------------------

python train_QAT.py \
    --TRAIN_ENGINE tf-legacy \
    --QAT_LOSS_MODE kd-deploy \
    --TF_LEGACY_BACKBONE cira-lite \
    --TF_CIRA_USE_DEFORM True \
    --DATA_BACKEND ultralytics \
    --DATA_YAML ./dataset/lanepose-carkeypoint.yaml \
    --ULTRA_TASK pose \
    --IMGSZ 640 \
    --BATCH_SIZE 64 \
    --EPOCHS 200 \
    --OUTPUT_DIR ./runs/tf_legacy_kd_deploy_cira_Lanecarkeypoint-20260227 \
    --TFLITE_QUANT_MODE int8 \
    --TRAIN_SUPERVISION label \
    --AUX_KD_HEAD_LABEL_LOSS True \
    --KD_LOSS_WEIGHT 1.0 \
    --DEPLOY_LOSS_WEIGHT 1.0 \
    --KD_TEMPERATURE 1.0 \
    --KD_CLS_DISTILL bce \
    --KD_DFL_DISTILL kldiv \
    --KD_FG_THRESHOLD 0.25 \
    --KD_FG_TOPK 800 \
    --KD_FG_MIN_POS 200 \
    --KD_FG_APPLY_TO both \
    --KD_BALANCE_STRATEGY grad_norm \
    --KD_BALANCE_SHARED_PARAM_GROUP head \
    --KD_BALANCE_MIN_WEIGHT 0.10 \
    --KD_BALANCE_MAX_WEIGHT 2.0 \
    --KD_BALANCE_WARMUP_STEPS 2000 \
    --KD_BALANCE_DEPLOY_RAMP_STEPS 800 \
    --KD_BALANCE_UPDATE_INTERVAL 10 \
    --KD_BALANCE_MAX_STEP_CHANGE 1.20 \
    --KD_BALANCE_ADAPT_POWER 0.50 \
    --KD_BALANCE_LOG_INTERVAL 20 \
    --ULTRA_EXPORT_FRACTION 0.25
  
  
  如果你要 teacher distill，把上面改成：

  - "TRAIN_SUPERVISION": "distill"
  - "EXPORTED_TEACHER_DIR": "<teacher_saved_model_or_h5_path>"
  
  
CUDA_VISIBLE_DEVICES= 0 python train_QAT.py \
    --TRAIN_ENGINE tf-legacy \
    --QAT_LOSS_MODE kd-deploy \
    --TF_LEGACY_BACKBONE cira-lite \
    --TF_CIRA_USE_DEFORM True \
    --DATA_BACKEND ultralytics \
    --DATA_YAML ./dataset/lanepose-carkeypoint.yaml \
    --ULTRA_TASK pose \
    --IMGSZ 640 \
    --BATCH_SIZE 64 \
    --EPOCHS 200 \
    --SEED 0 \
    --OUTPUT_DIR ./runs/alldataset/ciralite/0227/ \
    --TFLITE_QUANT_MODE int8 \
    --TRAIN_SUPERVISION distill \
    --KD_LOSS_WEIGHT 1.0 \
    --DEPLOY_LOSS_WEIGHT 1.0 \
    --KD_TEMPERATURE 1.0 \
    --KD_CLS_DISTILL bce \
    --KD_DFL_DISTILL kldiv \
    --KD_FG_THRESHOLD 0.25 \
    --KD_FG_TOPK 800 \
    --KD_FG_MIN_POS 200 \
    --KD_FG_APPLY_TO both \
    --KD_BALANCE_STRATEGY grad_norm \
    --KD_BALANCE_SHARED_PARAM_GROUP head \
    --KD_BALANCE_MIN_WEIGHT 0.10 \
    --KD_BALANCE_MAX_WEIGHT 2.0 \
    --KD_BALANCE_WARMUP_STEPS 2000 \
    --KD_BALANCE_DEPLOY_RAMP_STEPS 800 \
    --KD_BALANCE_UPDATE_INTERVAL 10 \
    --KD_BALANCE_MAX_STEP_CHANGE 1.20 \
    --KD_BALANCE_ADAPT_POWER 0.50 \
    --KD_BALANCE_LOG_INTERVAL 20 \
    --BASE_LR 0.01 \
    --END_LR 0.0001 \
    --MOMENTUM 0.937 \
    --WEIGHT_DECAY 0.0005 \
    --ULTRA_EXPORT_FRACTION 0.25
    
    --EXPORTED_TEACHER_DIR ./Teacher-model/acc-dataset/carkeypoint-20251207-Rep-AutoLabel-192GFlops \
    
    
