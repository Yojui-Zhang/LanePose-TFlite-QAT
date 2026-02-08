# Document Introduction
    QAT-Re1/
    ├── Dataset/                   # (User Data)
    ├── output/                    # (Generated Logs & Models)
    ├── verify_install.py          # [NEW] Environment verification script
    ├── main.py                    # [MODIFIED] Main entry point with robust checks
    ├── QAT_Refactored/
    │   ├── __init__.py
    │   ├── config/
    │   │   ├── __init__.py
    │   │   └── config.py          # (Keep original, verified in Audit)
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── engine.py          # (Keep original, verified in Audit)
    │   │   └── exporter.py        # (Keep original, verified in Audit)
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── pipeline.py        # (Keep original, logic strengthened via checks.py)
    │   │   ├── parser.py
    │   │   └── transforms.py
    │   ├── losses/
    │   │   ├── __init__.py
    │   │   ├── assigner.py
    │   │   ├── distill_loss.py
    │   │   └── pose_loss.py       # [MODIFIED] Added numerical safety & type hints
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── architecture.py
    │   │   ├── builder.py         # [MODIFIED] Added RepVGGQuantizeConfig logic
    │   │   ├── heads.py
    │   │   ├── layers.py          # [MODIFIED] Added explicit init & serialization
    │   │   └── qat_utils.py       # [NEW] Custom QAT logic for RepVGG
    │   └── utils/
    │       ├── __init__.py
    │       ├── checks.py          # [NEW] Centralized validation logic
    │       ├── device.py
    │       ├── env_setup.py
    │       ├── geometry.py
    │       ├── loading.py
    │       ├── system.py
    └───────└── visualization.py

# How to use

3. 執行說明

步驟 A：驗證（先執行此步驟）
執行獨立工具以確認您的環境已準備就緒。
python verify_install.py
預期輸出：✅ 驗證成功。

步驟 B：配置
編輯 QAT_Refactored/config/config.py 以匹配您的硬體/資料：

# 範例調整
BATCH_SIZE = 16 # 根據顯存大小調整
TRAIN_SUPERVISION = 'label' # 先用 'label'，然後再試試 'distill'
TFLITE_QUANT_MODE = 'int8' # 用於匯出測試

步驟 C：訓練
啟動主訓練循環。
python main.py

步驟 D：驗證結果
檢查日誌：開啟 output/{latest_run}/logs/training_log.csv。
檢查視覺化：查看 output/{latest_run}/plots/。驗證 val_pred_stepX.jpg 是否顯示了繪製在物件上的方塊。
檢查 QAT：
查看控制台輸出：[Builder] 量化層數：...
確保顯示 [Builder] RepVGG 區塊已成功包裝。



# Environment