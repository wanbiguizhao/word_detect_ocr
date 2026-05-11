from pathlib import Path

def get_project_root(relative_to: str = __file__) -> Path:
    ref_path = Path(relative_to).resolve()
    project_root = ref_path.parents[0]
    
    required_dirs = ["bussiness", "ocr_system", "server"]
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            print(f"⚠️  警告：项目根目录未检测到{dir_name}目录，可能根目录计算错误！")
    
    return project_root

PROJECT_ROOT = get_project_root()

DATASET_ROOT = PROJECT_ROOT / "char_image_dataset"
ANNOTATION_FILE = PROJECT_ROOT / "char_annotation.csv"
MODEL_DIR = PROJECT_ROOT / "models"
VISUAL_DIR = PROJECT_ROOT / "visualizations"

IMG_SIZE = (100, 100)
IMG_CHANNELS = 1

BATCH_SIZE = 32
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 20

def init_project_dirs():
    dirs_to_create = [DATASET_ROOT, MODEL_DIR, VISUAL_DIR]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 目录已创建/存在：{dir_path}")

init_project_dirs()

def to_abs_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path