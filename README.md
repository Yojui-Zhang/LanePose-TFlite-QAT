# QAT 專案 README

本專案目標是讓 `train_QAT.py` 與 Ultralytics 官方訓練/匯出流程高度對齊，並保留 `KD + deploy` 雙輸出頭與組合 loss：

- `total_loss = lambda_kd * L_kd + lambda_dep * L_deploy`（`lambda` 由動態平衡器更新）
- `QAT_LOSS_MODE=original`：走官方 Ultralytics API（對齊模式）
- `QAT_LOSS_MODE=kd-deploy`：走自訂 KD+deploy 訓練器（保留蒸餾能力）

---

## 1. 專案現況與重點

- 已完成 `train_QAT.py` 對齊 Ultralytics 訓練、驗證、TFLite 匯出流程。
- 已保留 `kd-deploy` 路線（不破壞 distill 訓練能力）。
- 已補上 TFLite `int8` 匯出後相容別名：
  - `*_integer_quant.tflite`
  - `*_full_integer_quant.tflite`
- 已新增 parity 驗證腳本：`verify/verify_ultralytics_bit_parity_smoke.py`。

### 1.1 功能對齊定義

本專案使用兩層對齊標準：

1. `功能對齊（Functional parity）`：
   - `.pt` 權重逐 tensor 完全一致
   - `.tflite` 同輸入下輸出逐元素完全一致
2. `位元對齊（Byte parity）`：
   - `.tflite` 檔案 SHA256 完全相同

目前可穩定達成的是功能對齊；位元對齊在 TFLite 序列化流程下不保證穩定成立。

---

## 2. 目錄結構

```text
QAT/
├── train_pose.py
├── train_QAT.py
├── QAT_Refactored/
│   ├── config/config.py
│   ├── core/ultralytics_kd.py
│   ├── core/ultralytics_route2.py
│   └── data/ultralytics_bridge.py
├── ultralytics/
├── verify/
│   ├── verify_install.py
│   ├── verify_route2_config.py
│   ├── verify_ultralytics_bridge_smoke.py
│   ├── verify_ultralytics_kd_loss_smoke.py
│   ├── verify_ultralytics_bit_parity_smoke.py
│   └── verify_tflite_contract.py
├── dataset/
├── output/
├── ERROR_LOG.md
└── README.md
```

---

## 3. 核心模式與語意

## 3.1 `train_pose.py`（Route2）

- `--qat-loss-mode original`：
  - 直接使用 Ultralytics 官方 `YOLO(...).train()` + export。
- `--qat-loss-mode kd-deploy`：
  - 轉交給 `train_QAT.run_train_qat(...)`。
  - 保留 KD+deploy 雙頭與動態平衡 loss（`lambda_kd * L_kd + lambda_dep * L_deploy`）。

## 3.2 `train_QAT.py`

`train_QAT.py` 目前支援兩種 engine/模式：

- `TRAIN_ENGINE=ultralytics`（預設推薦）
  - `QAT_LOSS_MODE=original`：官方 API 對齊模式。
  - `QAT_LOSS_MODE=kd-deploy`：KDPoseTrainer 模式。
- `TRAIN_ENGINE=tf-legacy`
  - 走舊有 TensorFlow QAT pipeline（保留相容）。

---

## 4. 安裝

```bash
python -m pip install -r requirement.txt
```

---

## 5. 常用指令

## 5.1 用 Ultralytics 官方路線訓練（建議）

```bash
python train_pose.py \
  --model ./runs/pose/acc-dataset-YOLOn-20260209/weights/best.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode original
```

## 5.2 用 KD+deploy 路線訓練

```bash
python train_pose.py \
  --model ./runs/pose/acc-dataset-YOLOn-20260209/weights/best.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-ema-decay 0.95 \
  --qat-balance-update-interval 10 \
  --qat-balance-deploy-ramp-steps 1000 \
  --qat-balance-min 0.2 \
  --qat-balance-max 5.0 \
  --export-tflite \
  --export-int8 \
  --export-fraction 0.25
  
python train_pose.py \
  --model ./runs/detect/KILLI-Yolov8n-20260211/weights/best.pt \
  --data ./dataset/KITTI.yaml \
  --task detect \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-ema-decay 0.95 \
  --qat-balance-update-interval 10 \
  --qat-balance-deploy-ramp-steps 1000 \
  --qat-balance-min 0.2 \
  --qat-balance-max 5.0 \
  --export-tflite \
  --export-int8 \
  --export-fraction 0.25
  
```

可選 teacher：

```bash
--qat-teacher-exported-dir <teacher_saved_model_dir>
```

## 5.3 只做匯出

```bash
python train_pose.py \
  --model ./runs/pose/.../weights/best.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --skip-train \
  --export-tflite \
  --export-int8
```

## 5.4 直接呼叫 `train_QAT.py`

```bash
python train_QAT.py
```

`train_QAT.py` 參數由 `QAT_Refactored/config/config.py` 管理；需要切模式時請調整：

- `TRAIN_ENGINE`
- `QAT_LOSS_MODE`
- `TRAIN_SUPERVISION`
- `KD_BALANCE_STRATEGY`
- `KD_BALANCE_SHARED_PARAM_GROUP`
- `KD_BALANCE_EMA_DECAY`
- `KD_BALANCE_UPDATE_INTERVAL`
- `KD_BALANCE_DEPLOY_RAMP_STEPS`
- `TFLITE_QUANT_MODE`

---

## 5.5 `train_pose.py` 參數說明

### 基本訓練參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--model` | `cfg/models/v8/yolov8-pose.yaml` | 訓練起始模型，可用 `.pt`、`.yaml`、內建模型名稱。 |
| `--data` | `./dataset/lanepose-carkeypoint.yaml` | 資料集 YAML 路徑。 |
| `--task` | `None` | 任務類型（`pose`/`detect`/...），一般 pose 專案可不填。 |
| `--epochs` | `200` | 訓練 epoch 數。 |
| `--batch` | `64` | batch size。 |
| `--imgsz` | `640 640` | 訓練輸入尺寸；`kd-deploy` 目前要求正方形。 |
| `--device` | `0` | 裝置，例如 `0`、`cpu`。 |
| `--workers` | `4` | DataLoader workers。 |
| `--fliplr` | `0.0` | 水平翻轉機率。 |
| `--project` | `./runs/pose` | 輸出專案目錄。 |
| `--name` | `route2` | 本次 run 名稱。 |
| `--resume` | `False` | 從中斷訓練續跑。 |
| `--cache` | `False` | 啟用資料快取。 |
| `--skip-train` | `False` | 跳過訓練只做匯出（僅 `original` 路線可用）。 |
| `--no-amp` | `False` | 關閉 AMP。 |
| `--no-cos-lr` | `False` | 關閉 cosine LR。 |
| `--strict-run` | `False` | 設 `exist_ok=False`，避免覆寫舊 run。 |

### 匯出參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--export-tflite` | `False` | 訓練後匯出 TFLite。 |
| `--export-int8` | `False` | 匯出 INT8 量化模型。 |
| `--export-half` | `False` | 匯出 FP16 模型。 |
| `--export-nms` | `False` | 匯出含 NMS 圖。 |
| `--export-data` | `None` | INT8 calibration 用資料 YAML；未填時沿用 `--data`。 |
| `--export-fraction` | `1.0` | calibration 使用比例 `(0,1]`，只影響匯出不影響訓練。 |

### QAT 模式參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--qat-loss-mode` | `original` | `original`=官方 loss；`kd-deploy`=KD+deploy 動態平衡 loss。 |
| `--qat-teacher-exported-dir` | `None` | teacher 模型目錄（啟用 distill）。 |
| `--qat-aux-kd-head-label-loss` | `False` | 無 teacher 時保持 KD 分支有效；若未指定 teacher，系統會自動啟用。 |

### 動態平衡參數（`--qat-loss-mode kd-deploy`）

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--qat-balance-strategy` | `grad_norm` | 平衡策略：`grad_norm`（建議）、`dwa`、`ratio`。 |
| `--qat-balance-shared-group` | `head` | `grad_norm` 計算梯度範數所用參數群：`head` 或 `all`。 |
| `--qat-balance-ema-decay` | `0.95` | loss 統計 EMA 平滑係數。 |
| `--qat-balance-update-interval` | `10` | 每幾個 step 更新一次動態權重。 |
| `--qat-balance-warmup-steps` | `0` | 前幾個 step 固定權重不更新。 |
| `--qat-balance-deploy-ramp-steps` | `1000` | deploy 權重漸進拉升步數。 |
| `--qat-balance-min` | `0.2` | `lambda_dep/lambda_kd` 下界，防止某項被壓到 0。 |
| `--qat-balance-max` | `5.0` | `lambda_dep/lambda_kd` 上界，防止單項暴衝。 |
| `--qat-balance-max-step-change` | `1.2` | 單次更新最大倍率變化，抑制震盪。 |
| `--qat-balance-adapt-power` | `0.5` | 權重更新強度指數。 |
| `--qat-balance-renorm-sum` | `2.0` | 每次更新後 `lambda_dep + lambda_kd` 目標總和。 |
| `--qat-balance-eps` | `1e-6` | 數值穩定用 epsilon。 |

---

## 5.6 建議參數組合

### A. 官方對齊優先（建議基準）

用途：先確認資料與流程無誤，最大化與官方 Ultralytics 一致性。

```bash
python train_pose.py \
  --model yolo11n-pose.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --workers 4 \
  --qat-loss-mode original \
  --export-tflite \
  --export-int8
```

### B. KD+Deploy 穩定預設（無 teacher）

用途：使用動態平衡訓練 KD+deploy，兼顧穩定與收斂速度。

```bash
python train_pose.py \
  --model yolo11n-pose.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-ema-decay 0.95 \
  --qat-balance-update-interval 10 \
  --qat-balance-deploy-ramp-steps 1000 \
  --qat-balance-min 0.2 \
  --qat-balance-max 5.0 \
  --qat-balance-max-step-change 1.2 \
  --qat-balance-adapt-power 0.5 \
  --qat-balance-renorm-sum 2.0 \
  --export-tflite \
  --export-int8
```

### C. KD+Deploy + Teacher 蒸餾

用途：有可用 teacher 權重時，提升 student 蒸餾學習效果。

```bash
python train_pose.py \
  --model yolo11n-pose.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-teacher-exported-dir ./teacher_export \
  --qat-balance-strategy grad_norm \
  --qat-balance-shared-group head \
  --qat-balance-ema-decay 0.95 \
  --qat-balance-update-interval 10 \
  --qat-balance-deploy-ramp-steps 1000 \
  --qat-balance-min 0.2 \
  --qat-balance-max 5.0 \
  --export-tflite \
  --export-int8
```

### D. 低記憶體 INT8 匯出（避免 `Killed`）

用途：訓練維持全量資料，匯出 calibration 降載。

```bash
python train_pose.py \
  --model yolo11n-pose.pt \
  --data ./dataset/lanepose-carkeypoint.yaml \
  --task pose \
  --epochs 50 \
  --batch 64 \
  --imgsz 640 640 \
  --device 0 \
  --qat-loss-mode kd-deploy \
  --qat-balance-strategy grad_norm \
  --export-tflite \
  --export-int8 \
  --export-data ./verify/_tmp/bit_parity/data/dataset.yaml \
  --export-fraction 0.25
```

---

## 6. 匯出與檔名

INT8 匯出主檔通常為：

- `best_int8.tflite`

並同步建立相容別名：

- `best_integer_quant.tflite`
- `best_full_integer_quant.tflite`

---

## 7. 驗證流程

## 7.1 基本驗證

```bash
python verify/verify_install.py
python verify/verify_route2_config.py
python verify/verify_kd_dynamic_balancer_smoke.py
python verify/verify_ultralytics_kd_loss_smoke.py
python verify/verify_tflite_contract.py --model <path/to/model.tflite>
```

## 7.2 Ultralytics 對齊驗證（官方 vs train_QAT）

```bash
python verify/verify_ultralytics_bit_parity_smoke.py
```

嚴格位元模式：

```bash
PARITY_STRICT_BYTES=1 python verify/verify_ultralytics_bit_parity_smoke.py
```

`.h5` 路線與 `.pt` 路線的 `.tflite` 語意相似驗證：

```bash
python verify/verify_h5_pt_tflite_semantic_parity.py \
  --h5-tflite <path/to/h5_route.tflite> \
  --pt-tflite <path/to/pt_route.tflite>
```

---

## 8. 已驗證結果（2026-02-10）

已完成並通過：

- `verify_install.py`
- `verify_route2_config.py`
- `verify_ultralytics_bridge_smoke.py`
- `verify_ultralytics_kd_loss_smoke.py`
- `verify_tflite_contract.py --model output/20260210_003941/models/model_quant_fp32.tflite`
- `verify_ultralytics_bit_parity_smoke.py`

Parity 結果（最近一次）：

- `official_sha256=6aac1a05ae63244f7a3193e6fd40d7ca08a71e32b857fc176fedde7432196671`
- `train_qat_sha256=785bca67d0694213bf69be012def304cf5a89677b81fba09d52c496fc976f6cd`
- 結論：`raw_tflite_bytes_differ_but_weights_and_outputs_are_exactly_equal`

---

## 9. 與 `TFlite.h` 對齊

建議每次匯出後執行：

```bash
python verify/verify_tflite_contract.py --model <your_model.tflite>
```

目前已確認可通過的範例：

- `output/20260210_003941/models/model_quant_fp32.tflite`
  - input: `(1, 640, 640, 3)` float32
  - output: `(1, 56, 8400)` float32

---

## 10. 常見問題

### Q1. `train_pose.py` 的 `original` 與 `kd-deploy` 結果會不同嗎？

會。`original` 使用官方 loss；`kd-deploy` 使用組合 loss（`lambda_kd * L_kd + lambda_dep * L_deploy`，且 `lambda` 為動態更新），目標函數不同，權重與指標通常不同。

### Q2. `train_QAT` 與官方 Ultralytics 輸出是否一致？

在 `QAT_LOSS_MODE=original` 下，已可達成功能對齊（權重/輸出一致）。

### Q3. 為什麼 `.tflite` SHA 還是可能不同？

主要是轉換序列化流程非完全 byte-stable。這不代表模型功能不同，請以 parity 腳本的功能一致結果為準。

---

## 11. 官方文件（參考）

- Ultralytics Train mode: `https://docs.ultralytics.com/modes/train/`
- Ultralytics Export mode: `https://docs.ultralytics.com/modes/export/`
- Ultralytics Python usage: `https://docs.ultralytics.com/usage/python/`
