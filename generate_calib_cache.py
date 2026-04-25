import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import glob

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size, input_shape, data_dir, cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape # 例如: (3, 640, 640)
        self.cache_file = cache_file
        
        # 抓取資料夾內所有圖片 (支援 jpg, png)
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + \
                           glob.glob(os.path.join(data_dir, "*.png"))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中找不到圖片！")
        
        self.batch_idx = 0
        self.max_batches = len(self.image_paths) // self.batch_size
        
        # 預先分配 GPU 記憶體 (Batch Size * 影像體積 * float32 的 byte 數)
        self.device_input = cuda.mem_alloc(trt.volume(input_shape) * batch_size * np.dtype(np.float32).itemsize)

    def get_batch_size(self):
        return self.batch_size

    def preprocess_image(self, image_path):
        """
        [重要] 這裡的預處理必須與您 PyTorch 訓練時【完全一致】
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1])) # W, H
        
        # 標準化 (若訓練時有 /255.0，這裡也要做)
        img = img.astype(np.float32) / 255.0 
        
        # HWC 轉換為 CHW
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(img)

    def get_batch(self, names):
        if self.batch_idx >= self.max_batches:
            return None # 圖片耗盡，校正結束

        batch_imgs = []
        start_idx = self.batch_idx * self.batch_size
        for i in range(self.batch_size):
            img_path = self.image_paths[start_idx + i]
            batch_imgs.append(self.preprocess_image(img_path))
        
        # 將 batch 轉為連續的 numpy array
        batch_data = np.ascontiguousarray(batch_imgs, dtype=np.float32)
        
        # 將資料複製到 GPU
        cuda.memcpy_htod(self.device_input, batch_data)
        self.batch_idx += 1
        
        # 返回 GPU 記憶體指標
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # 若快取檔已存在，直接讀取以加速 (可選)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        # 將生成的快取寫入硬碟
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def generate_cache(onnx_path, data_dir, cache_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # 解析 ONNX
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 解析失敗")

    # 設定 INT8 與 Calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    
    # 初始化剛才寫的 Calibrator
    # 注意：請確保 input_shape 對應 C, H, W (例如: 3, 640, 640)
    calibrator = ImageCalibrator(
        batch_size=8, # 可依據 VRAM 大小調整
        input_shape=(3, 640, 640), 
        data_dir=data_dir,
        cache_file=cache_file
    )
    config.int8_calibrator = calibrator

    # 建立 Engine (這個過程會觸發圖片讀取與 cache 檔生成)
    print("開始建立 Engine 並生成 Calibration Cache，請耐心等候...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        raise RuntimeError("建立 Engine 失敗")
        
    print(f"成功生成 {cache_file}！")

if __name__ == "__main__":
    ONNX_MODEL = "cira_pose_baseline_original_trt.onnx"
    DATASET_DIR = "/home/Disk/Desktop/AI/dataset/lanepose/acc_datasets/images"
    CACHE_OUTPUT = "CIRA-Baseline-calibration.cache"
    
    generate_cache(ONNX_MODEL, DATASET_DIR, CACHE_OUTPUT)
