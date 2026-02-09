import os
import torch

from ultralytics import YOLO

model = YOLO('./ultralytics-8.4.13/ultralytics/cfg/models/v8/yolov8-pose.yaml') 


results = model.train(data='./dataset/lanepose-carkeypoint.yaml', 
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
