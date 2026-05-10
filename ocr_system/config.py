import os

# 数据路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_ROOT, 'ocr_train_data_balanced')

# 使用相对路径的 CSV 文件
TRAIN_CSV = os.path.join(DATA_ROOT, 'train_rel.csv')
VAL_CSV = os.path.join(DATA_ROOT, 'val_rel.csv')
TEST_CSV = os.path.join(DATA_ROOT, 'test_rel.csv')

# 训练参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10

# 模型参数
def _get_num_classes():
    try:
        from configs.char_mapping import CharMappingManager
        return CharMappingManager().get_stats()["next_custom_id"]
    except Exception:
        return 7000  # GB2312(6763) + 预留(100) + 部分自定义

NUM_CLASSES = _get_num_classes()
IMAGE_SIZE = (64, 64)
NUM_WORKERS = 4

# 模型仓库（集中管理所有模型文件）
MODEL_STORE_DIR = os.path.join(BASE_DIR, 'model_store')

# 模型子目录（统一放到 model_store 下）
PRETRAINED_MODEL_DIR = os.path.join(MODEL_STORE_DIR, 'pretrained')   # 预训练模型（如 ResNet50）
TRAINED_MODEL_DIR = os.path.join(MODEL_STORE_DIR, 'trained')         # 训练输出模型（训练过程中保存）
INFERENCE_MODEL_DIR = os.path.join(MODEL_STORE_DIR, 'inference')     # 推理使用模型（最终发布版本）
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 创建目录
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(INFERENCE_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 模型路径
MODEL_SAVE_PATH = os.path.join(TRAINED_MODEL_DIR, 'ocr_model.pth')           # 训练过程中保存的当前模型
BEST_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'ocr_model_best.pth')       # 训练过程中保存的最佳模型
INFERENCE_MODEL_PATH = os.path.join(INFERENCE_MODEL_DIR, 'ocr_model_best.pth') # 用于推理的最终模型
LOG_PATH = os.path.join(LOG_DIR, 'training.log')