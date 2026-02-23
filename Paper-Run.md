# Dyn KD weight

python run_paper_experiments.py \
  --datasets kitti --studies A,B --seeds 0,1,2,3,4 \
  --epochs 200 --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 0 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --data-root ../Paper-Data/Data_kitti_AandB_dynKD_near025 \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-min 0.22 \
  --qat-balance-max 0.32 \
  --qat-balance-warmup-steps 0 \
  --qat-balance-deploy-ramp-steps 4000 \
  --qat-balance-update-interval 10 \
  --qat-balance-max-step-change 1.05 \
  --qat-balance-adapt-power 0.25 \
  --qat-balance-log-interval 20 \
  --kitti-mobilenetv3-model ./ultralytics/cfg/models/Yojui/yolov8_MobileNetV3-Lite.yaml \
  --kitti-ghostnetv2-model  ./ultralytics/cfg/models/Yojui/yolov8_GhostNetV2-Lite.yaml \
  --kitti-shufflenetv2-model ./ultralytics/cfg/models/Yojui/yolov8_ShuffleNetV2-Lite.yaml \
  --kitti-cira-lite-model ./ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml \
  --skip-existing


python run_paper_experiments.py \
  --datasets kitti --studies B --seeds 0,1,2,3,4 \
  --epochs 200 --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 1 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --data-root ../Paper-Data/Data_kitti_onlyB_dynKD_TEST \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-min 0.15 \
  --qat-balance-max 0.60 \
  --qat-balance-warmup-steps 0 \
  --qat-balance-deploy-ramp-steps 2000 \
  --qat-balance-update-interval 10 \
  --qat-balance-max-step-change 1.10 \
  --qat-balance-adapt-power 0.50 \
  --qat-balance-log-interval 20 \
  --kitti-mobilenetv3-model ./ultralytics/cfg/models/Yojui/yolov8_MobileNetV3-Lite.yaml \
  --kitti-ghostnetv2-model  ./ultralytics/cfg/models/Yojui/yolov8_GhostNetV2-Lite.yaml \
  --kitti-shufflenetv2-model ./ultralytics/cfg/models/Yojui/yolov8_ShuffleNetV2-Lite.yaml \
  --kitti-cira-lite-model ./ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml \
  --skip-existing

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
  --studies A,B \
  --seeds 0,1,2,3,4 \
  --epochs 200 \
  --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 0 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --kitti-cira-lite-model ./ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml \
  --data-root ../Paper-Data/Data_kitti_AandB_Cira_lite_v2_and_onlyDepoly \
  --qat-kd-weight 0.0 \
  --qat-balance-min 0.0 \
  --qat-balance-log-interval 20 \
  --skip-existing


# ------------------------------------------------------------------------------
# Smoke Test
python run_paper_experiments.py --smoke --max-runs 1 --datasets kitti --studies A --seeds 0