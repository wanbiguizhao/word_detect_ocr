import torch
from pathlib import Path

# ===================== 全局路径配置 =====================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 根目录
DATA_DIR = BASE_DIR/"dataset"/"done2"                                      # 数据集目录
PRETRAINED_AE_PATH = BASE_DIR/"ai"/"model_storage"/"feature_model.pth"                        # 预训练自编码器权重
PRETRAINED_CHAR_SEGMENT_MODEL_PATH = BASE_DIR/"ai"/"model_storage"/"char_segment_classifier_0427.pth"  # 预训练分割模型权重
MODEL_SAVE_PATH = BASE_DIR/"char_segment_classifier_new.pth"  # 新训练模型保存路径

# ===================== 训练超参数 =====================
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# ===================== 硬件配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 环境变量 =====================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
