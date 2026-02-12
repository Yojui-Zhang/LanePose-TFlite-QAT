import os
import torch

from ultralytics import YOLO

#model = YOLO('./ultralytics/cfg/models/v8/yolov8n.yaml') 
model = YOLO('./ultralytics/cfg/models/Yojui/yolov8_CIRA-Detect.yaml') 

# Reduce CUDA memory fragmentation risk.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

#results = model.train(data='./dataset/lanepose-carkeypoint.yaml', 
results = model.train(data='./dataset/KITTI.yaml', 
                      epochs=200,
                      batch=64, 
                    #   imgsz=(512, 288), 
                      imgsz=(640, 640), 
                      # Use first CUDA device when available; fallback to CPU.
                      device=0,
                      amp=True,
                      workers=4,
#                       patience=0,
#                       --------------                      
#                       hsv_h = 0.0,
#                       hsv_s = 0.0,
#                       hsv_v = 0.0,
#                       translate = 0.0,
#                       scale = 0.0,
                      fliplr = 0.0,
#                       flipud = 0.0,
#                       --------------
                      
                      cos_lr=True,
#                       pose = 32.0,
#                       box = 6.0,
                      name="test"
                      )
