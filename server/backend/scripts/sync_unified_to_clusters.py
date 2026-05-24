"""将统一标注同步到聚类标注"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_ID

def sync_unified_to_clusters(dataset_id: str = None):
    if dataset_id is None:
        dataset_id = DATASET_ID
    project_root = Path("d:/projects/word_detect_ocr")
    dataset_dir = project_root / "bussiness" / "datahome" / dataset_id
    
    # 路径定义
    unified_path = dataset_dir / "unified_labels.json"
    clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
    labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
    
    print("开始将统一标注同步到聚类标注")
    print("   数据集: {}".format(dataset_id))
    print("=" * 60)
    
    # 读取统一标注
    if not unified_path.exists():
        print("统一标注文件不存在")
        return
    
    with open(unified_path, 'r', encoding='utf-8') as f:
        unified_data = json.load(f)
    
    # 读取聚类数据
    if not clusters_path.exists():
        print("聚类数据文件不存在")
        return
    
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    # 读取聚类标注
    if labels_path.exists():
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
    else:
        labels_data = {}
    
    # 构建 char_id → char 映射
    labeled_chars = {}
    for ann in unified_data.get("annotations", []):
        char_id = ann.get("char_id")
        char = ann.get("char")
        status = ann.get("status")
        if char_id and char and status == "labeled":
            labeled_chars[char_id] = char
    
    print("统一标注中已标注的字符数: {}".format(len(labeled_chars)))
    
    # 构建 char_id → (cluster_id, char_index) 映射
    char_id_to_cluster = {}
    clusters = clusters_data.get("clusters", {})
    for cluster_id, chars in clusters.items():
        for idx, char_info in enumerate(chars):
            char_id = char_info.get("char_id")
            if char_id:
                char_id_to_cluster[char_id] = (cluster_id, idx)
    
    # 更新聚类标注
    updated_clusters = set()
    updated_chars = 0
    
    for char_id, char in labeled_chars.items():
        if char_id in char_id_to_cluster:
            cluster_id, char_index = char_id_to_cluster[char_id]
            
            # 确保聚类标注存在
            if cluster_id not in labels_data:
                labels_data[cluster_id] = {
                    "char": None,
                    "chars": {},
                    "status": "unlabeled",
                    "confidence": None,
                    "alias": "",
                    "char_labels": {}
                }
            
            # 更新字符标注
            char_key = str(char_index)
            if "char_labels" not in labels_data[cluster_id]:
                labels_data[cluster_id]["char_labels"] = {}
            
            # 检查是否需要更新
            existing_label = labels_data[cluster_id]["char_labels"].get(char_key)
            if not existing_label or existing_label.get("char") != char:
                labels_data[cluster_id]["char_labels"][char_key] = {
                    "char": char,
                    "labeled_at": "2024-01-01T00:00:00"  # 简化处理
                }
                updated_chars += 1
                updated_clusters.add(cluster_id)
                
                # 更新聚类统计
                all_chars = [v["char"] for v in labels_data[cluster_id]["char_labels"].values() if v.get("char")]
                if all_chars:
                    char_counts = Counter(all_chars)
                    most_common = char_counts.most_common(1)[0]
                    labels_data[cluster_id]["char"] = most_common[0]
                    labels_data[cluster_id]["confidence"] = most_common[1] / len(all_chars)
                    labels_data[cluster_id]["chars"] = dict(char_counts)
                    labels_data[cluster_id]["status"] = "labeled"
    
    # 保存聚类标注
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    print("同步完成")
    print("   - 更新的聚类数: {}".format(len(updated_clusters)))
    print("   - 更新的字符数: {}".format(updated_chars))

if __name__ == "__main__":
    sync_unified_to_clusters()