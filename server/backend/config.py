from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

class ConfigManager:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {
                "dataset": {
                    "current": "pdf5826",
                    "source": "pdf01"
                },
                "labeling": {
                    "high_confidence_threshold": 0.9,
                    "low_confidence_threshold": 0.7
                },
                "clustering": {
                    "default_method": "arcface",
                    "supported_methods": ["arcface", "hog"]
                },
                "ocr": {
                    "model_path": "ocr_system/model_store/inference/ocr_model_best.pth"
                }
            }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def reload(self):
        self._load_config()

config = ConfigManager()

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATASET_ID = config.get("dataset.current", "pdf5826")
SOURCE_DATASET_ID = config.get("dataset.source", "pdf01")

DATAHOME_DIR = PROJECT_ROOT / "bussiness" / "datahome"
DATASET_DIR = DATAHOME_DIR / DATASET_ID
SOURCE_DATASET_DIR = DATAHOME_DIR / SOURCE_DATASET_ID

RAW_IMAGES_DIR = DATASET_DIR / "pdf_lines"
RULE_JSONS_DIR = DATASET_DIR / "rule_jsons"
MODEL_JSONS_DIR = DATASET_DIR / "model_jsons"
FUSION_JSONS_DIR = DATASET_DIR / "fusion_jsons"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TOP_SAMPLES_PATH = DATASET_DIR / "top_annotate_samples.json"

for dir_path in [RAW_IMAGES_DIR, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CLUSTERS_DIR = DATASET_DIR / "clusters"
CLUSTERS_JSON = CLUSTERS_DIR / "hog_clusters.json"
LABELS_JSON = CLUSTERS_DIR / "labeling" / "labels.json"

CLUSTER_RESULTS_DIR = PROJECT_ROOT / "bussiness" / "cluster_results"
ARCFACE_CLUSTERS_DIR = CLUSTER_RESULTS_DIR / "arcface"
HOG_CLUSTERS_DIR = CLUSTER_RESULTS_DIR / "hog"

ARCFACE_CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
HOG_CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

ARCFACE_CLUSTERS_JSON = ARCFACE_CLUSTERS_DIR / f"{DATASET_ID}_clusters.json"
HOG_CLUSTERS_JSON = HOG_CLUSTERS_DIR / f"{DATASET_ID}_clusters.json"

MIGRATION_DIR = PROJECT_ROOT / "bussiness" / "migration" / "data"
CHAR_FEATURE_DB_PATH = MIGRATION_DIR / "char_feature_db.json"
CHAR_MIGRATION_MAP_PATH = MIGRATION_DIR / "char_migration_map.json"
CHAR_PRIORITY_LIST_PATH = MIGRATION_DIR / "char_priority_list.json"
GLOBAL_CHAR_REGISTRY_PATH = MIGRATION_DIR / "global_char_registry.json"

