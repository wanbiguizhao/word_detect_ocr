from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"

RAW_IMAGES_DIR = DATASET_DIR / "raw_images"
RULE_JSONS_DIR = DATASET_DIR / "rule_jsons"
MODEL_JSONS_DIR = DATASET_DIR / "model_jsons"
FUSION_JSONS_DIR = DATASET_DIR / "fusion_jsons"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TOP_SAMPLES_PATH = DATASET_DIR / "top_annotate_samples.json"

for dir_path in [RAW_IMAGES_DIR, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = BASE_DIR.parent.parent
CLUSTERS_DIR = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters"
CLUSTERS_JSON = CLUSTERS_DIR / "hog_clusters.json"
LABELS_JSON = CLUSTERS_DIR / "labeling" / "labels.json"

pseudo_label_cache = {}
executor = ThreadPoolExecutor(max_workers=4)