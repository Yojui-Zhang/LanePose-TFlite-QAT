from ultralytics import YOLO
import torch

def count_parameters(model_path='yolov8n.pt'):
    # 這裡載入您的 YAML
    model = YOLO(model_path) 
    
    # 強制初始化權重以計算參數
    # 注意：不需要真的訓練，只要 build 起來即可
    n_p = sum(x.numel() for x in model.model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.model.parameters() if x.requires_grad)  # gradient parameters
    
    print(f"\nModel: {model_path}")
    print(f"Total Parameters: {n_p / 1e6:.4f} M")
    print(f"GFLOPs: {model.model.info()[1] / 1e9:.4f} G") # 某些版本 info() 返回 (layers, flops, params)

if __name__ == "__main__":
    # 1. 先看原本的 YOLOv8n (作為基準)
    print("--- Baseline YOLOv8n ---")
    try:
        count_parameters('./ultralytics/cfg/models/v8/yolov8.yaml') 
    except:
        print("Standard yolov8n.yaml not found, skipping baseline.")

    # 2. 看您的 CIRA 模型
    print("\n--- CIRA Proposed ---")
    # 請將此處換成您的 yaml 檔名
    count_parameters('./ultralytics/cfg/models/Yojui/yolov8_CIRA-Lite.yaml')
    count_parameters('./ultralytics/cfg/models/Yojui/yolov8_ShuffleNetV2-Lite.yaml')
    count_parameters('./ultralytics/cfg/models/Yojui/yolov8_GhostNetV2-Lite.yaml')
    count_parameters('./ultralytics/cfg/models/Yojui/yolov8_MobileNetV3-Lite.yaml')
