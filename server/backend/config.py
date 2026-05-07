from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ======================================
# Path Configuration
# ======================================
BASE_DIR = Path(__file__).parent
DATAHOME_DIR = BASE_DIR.parent.parent / "bussiness" / "datahome"
DATASET_ID = "pdf5823"  # 待标注的目标数据集
DATASET_DIR = DATAHOME_DIR / DATASET_ID

# 源数据集（已标注，用于迁移学习）
SOURCE_DATASET_ID = "pdf01"
SOURCE_DATASET_DIR = DATAHOME_DIR / SOURCE_DATASET_ID

RAW_IMAGES_DIR = DATASET_DIR / "pdf_lines"
RULE_JSONS_DIR = DATASET_DIR / "rule_jsons"
MODEL_JSONS_DIR = DATASET_DIR / "model_jsons"
FUSION_JSONS_DIR = DATASET_DIR / "fusion_jsons"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TOP_SAMPLES_PATH = DATASET_DIR / "top_annotate_samples.json"

for dir_path in [RAW_IMAGES_DIR, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ======================================
# Legacy Path for backward compatibility
# ======================================
PROJECT_ROOT = BASE_DIR.parent.parent

# ======================================
# OCR Labeling Configuration
# ======================================
CLUSTERS_DIR = DATASET_DIR / "clusters"
CLUSTERS_JSON = CLUSTERS_DIR / "hog_clusters.json"
LABELS_JSON = CLUSTERS_DIR / "labeling" / "labels.json"

# ======================================
# Migration Data Configuration
# ======================================
MIGRATION_DIR = PROJECT_ROOT / "bussiness" / "migration" / "data"
CHAR_FEATURE_DB_PATH = MIGRATION_DIR / "char_feature_db.json"
CHAR_MIGRATION_MAP_PATH = MIGRATION_DIR / "char_migration_map.json"
CHAR_PRIORITY_LIST_PATH = MIGRATION_DIR / "char_priority_list.json"
GLOBAL_CHAR_REGISTRY_PATH = MIGRATION_DIR / "global_char_registry.json"

# ======================================
# Global Variables
# ======================================
pseudo_label_cache = {}
executor = ThreadPoolExecutor(max_workers=4)
