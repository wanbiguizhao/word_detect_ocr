from pathlib import Path
import os

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "bussiness" / "config.json"

DATA_ROOT = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_ROOT / 'ocr_train_data_balanced'

TRAIN_CSV = DATA_ROOT / 'train_rel.csv'
VAL_CSV = DATA_ROOT / 'val_rel.csv'
TEST_CSV = DATA_ROOT / 'test_rel.csv'

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10

def _get_num_classes():
    try:
        from configs.char_mapping import CharMappingManager
        return CharMappingManager().get_stats()["next_custom_id"]
    except Exception:
        return 7000

NUM_CLASSES = _get_num_classes()
IMAGE_SIZE = (64, 64)
NUM_WORKERS = 4

MODEL_STORE_DIR = BASE_DIR / 'model_store'

PRETRAINED_MODEL_DIR = MODEL_STORE_DIR / 'pretrained'
TRAINED_MODEL_DIR = MODEL_STORE_DIR / 'trained'
INFERENCE_MODEL_DIR = MODEL_STORE_DIR / 'inference'
LOG_DIR = BASE_DIR / 'logs'

for dir_path in [PRETRAINED_MODEL_DIR, TRAINED_MODEL_DIR, INFERENCE_MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = TRAINED_MODEL_DIR / 'ocr_model.pth'
BEST_MODEL_PATH = TRAINED_MODEL_DIR / 'ocr_model_best.pth'
INFERENCE_MODEL_PATH = INFERENCE_MODEL_DIR / 'ocr_model_best.pth'
LOG_PATH = LOG_DIR / 'training.log'