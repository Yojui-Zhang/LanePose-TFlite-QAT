# QAT-TEST Usage

## 1. Select Runtime Mode

- `ultralytics`: PyTorch/Ultralytics route (`train_pose.py`), faster training/export path.
- `tf-legacy-qat`: TensorFlow QAT route (`train_QAT.py` with `TRAIN_ENGINE=tf-legacy`).

Use separate environments for the two modes to avoid CUDA/cuDNN dependency conflicts.

## 2. Install

### 2.1 Ultralytics environment

```bash
conda create -n qat_ultra python=3.10 -y
conda activate qat_ultra
python -m pip install --upgrade pip
python -m pip install -r requirement.txt -r requirements.ultralytics.txt
```

GPU check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY
```

### 2.2 TensorFlow QAT environment

```bash
conda create -n qat_tf python=3.10 -y
conda activate qat_tf
python -m pip install --upgrade pip
python -m pip install -r requirement.txt -r requirements.qat-tflegacy.txt
```

GPU check:

```bash
python - <<'PY'
import tensorflow as tf
print("tf:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
PY
```

## 3. Train

### 3.1 Ultralytics official loss (`original`)

```bash
python train_pose.py \
  --model yolo11n.pt \
  --data ./dataset/KITTI.yaml \
  --task detect \
  --epochs 200 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode original \
  --export-tflite --export-int8 --export-fraction 0.25
```

### 3.2 Ultralytics KD+Deploy (`kd-deploy`)

```bash
python train_pose.py \
  --model yolo11n.pt \
  --data ./dataset/KITTI.yaml \
  --task detect \
  --epochs 200 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-min 0.2 \
  --qat-balance-max 5.0 \
  --qat-balance-log-interval 50 \
  --export-tflite --export-int8 --export-fraction 0.25
```

### 3.3 TensorFlow legacy QAT (`train_QAT.py`)

```bash
python train_QAT.py \
  --TRAIN_ENGINE tf-legacy \
  --QAT_LOSS_MODE kd-deploy \
  --TF_LEGACY_BACKBONE cira-lite \
  --TF_CIRA_USE_DEFORM True \
  --DATA_BACKEND ultralytics \
  --DATA_YAML ./dataset/KITTI.yaml \
  --ULTRA_TASK detect \
  --IMGSZ 640 \
  --BATCH_SIZE 64 \
  --EPOCHS 200 \
  --OUTPUT_DIR ./runs/tf_legacy_kd_deploy_cira_kitti \
  --TFLITE_QUANT_MODE int8 \
  --TRAIN_SUPERVISION label \
  --AUX_KD_HEAD_LABEL_LOSS True \
  --KD_LOSS_WEIGHT 1.0 \
  --DEPLOY_LOSS_WEIGHT 1.0
```

## 4. Export Only

```bash
python train_pose.py \
  --model ./runs/detect/exp/weights/best.pt \
  --data ./dataset/KITTI.yaml \
  --task detect \
  --skip-train \
  --export-tflite --export-int8 --export-fraction 0.25
```

## 5. Useful Verifications

```bash
python verify/verify_tf_legacy_cira_backbone_smoke.py
python verify/verify_tf_legacy_cira_export_fallback_smoke.py
python verify/verify_tf_legacy_cira_gpu_path_smoke.py
```

## 6. `run_paper_experiments.py` Unified CLI Usage

### 6.1 Study-B variants (all default OFF)

- `--include-b-deploy-only`: run `deploy_only`
- `--include-b-kd-only`: run `KdDepoly_half` (fixed-alpha KD+deploy; old alias kept for compatibility)
- `--include-b-pure-kd`: run `pure_kd`
- `--include-b-kd-deploy`: run `kd_deploy`
- `--b-delta-baseline {deploy_only,KdDepoly_half,pure_kd,kd_deploy}`: choose delta baseline in report

If `--studies B` is set but none of the four `--include-b-*` flags are provided, the runner will raise an error.

### 6.2 Study-A extra variants (all default OFF)

- `--include-a-cira`: include CIRA (acc + kitti)
- `--include-a-kitti-cira-lite`: include KITTI cira-lite
- `--include-a-kitti-mobilenetv3`: include KITTI mobilenetv3 (requires `--kitti-mobilenetv3-model`)
- `--include-a-kitti-ghostnetv2`: include KITTI ghostnetv2 (requires `--kitti-ghostnetv2-model`)
- `--include-a-kitti-shufflenetv2`: include KITTI shufflenetv2 (requires `--kitti-shufflenetv2-model`)

Without these `--include-a-*` flags, Study-A runs only YOLO baseline.

### 6.3 Run Study-B on KITTI

```bash
python run_paper_experiments.py \
  --datasets kitti --studies B --seeds 0,1,2,3,4 \
  --epochs 200 --batch 64 --close-mosaic 10 \
  --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
  --device-kitti 1 --export-fraction 0.25 \
  --kitti-teacher ./Teacher-model/KITTI/KILLI-CIRA-AutoLabel-20260211 \
  --data-root ./Paper-Data/Data_kitti_OnlyB_onlyDeploy_dynKD_onlyKD_tflite_int8_0306 \
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
  --include-b-kd-only \
  --include-b-pure-kd \
  --include-b-kd-deploy \
  --b-kd-only-weight 1.0 \
  --skip-existing \
  --eval-tflite-map \
  --tflite-map-split val
```

Notes:

- `--include-b-kd-only` now generates `KdDepoly_half`, which keeps the historical fixed-alpha KD+deploy behavior.
- `--include-b-pure-kd` is the true pure-KD branch (`loss = kd_loss`) and requires a teacher path.
- `kd_deploy` now uses a more conservative Study-B preset than the fixed-alpha branch to reduce alpha saturation on KITTI.
