from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.pt")  # load an official model
# model = YOLO("runs/Traffic/traffic_pre_v6_1epoc/traffic_pre_v6_1epoc.pt")  # load a custom trained
# model = YOLO("runs/detect/traffic_r1/weights/best.pt")  # load a custom trained
# model = YOLO("runs/detect/test5/weights/best.pt")  # load a custom trained
# model = YOLO("/home/jovyan/datasets/es912-nas/M11013050/BSD/yolov8/runs/detect/BSD/4in1_v2_litev2/weights/best.pt")  # load a custom trained
model = YOLO("./runs/cira_pose_baseline_original/cira_pose_baseline_original.pt")


# Export the model
# model.export(format="tflite", int8=True, imgsz=(288,512), rect=True)

# model.export(format="tflite", int8=True, imgsz=(640,640))
# model.export(format="saved_model", imgsz=640, keras=True)

model.export(format="onnx")

# model.export(format="tflite", int8=True, imgsz=(576,1024))
