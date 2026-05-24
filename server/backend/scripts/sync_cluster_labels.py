"""同步聚类标注到统一标注"""
import json
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_ID

def sync_cluster_labels(dataset_id: str = None):
    if dataset_id is None:
        dataset_id = DATASET_ID
    project_root = Path("d:/projects/word_detect_ocr")
    dataset_dir = project_root / "bussiness" / "datahome" / dataset_id
    
    # 路径定义
    labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
    clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
    unified_path = dataset_dir / "unified_labels.json"
    
    print(f"开始同步聚类标注到统一标注")
    print(f"   数据集: {dataset_id}")
    print("=" * 60)
    
    # 读取聚类标注
    if not labels_path.exists():
        print("聚类标注文件不存在")
        return
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # 读取聚类数据
    if not clusters_path.exists():
        print("聚类数据文件不存在")
        return
    
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    # 读取统一标注
    if unified_path.exists():
        with open(unified_path, 'r', encoding='utf-8') as f:
            unified_data = json.load(f)
    else:
        unified_data = {"annotations": []}
    
    annotations = unified_data.get("annotations", [])
    existing_char_ids = {ann["char_id"] for ann in annotations if ann.get("char_id")}
    
    # 找出需要同步的标注
    synced_count = 0
    skipped_count = 0
    
    for cluster_id, cluster_info in labels_data.items():
        char_labels = cluster_info.get("char_labels", {})
        if not char_labels:
            continue
        
        # 获取该聚类的字符列表
        cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
        
        for char_index_str, label_info in char_labels.items():
            char = label_info.get("char")
            if not char:
                continue
            
            char_index = int(char_index_str)
            if char_index >= len(cluster_chars):
                continue
            
            char_id = cluster_chars[char_index].get("char_id")
            if not char_id:
                continue
            
            # 检查是否已存在
            if char_id in existing_char_ids:
                # 更新已存在的标注
                for ann in annotations:
                    if ann.get("char_id") == char_id:
                        if ann.get("char") != char or ann.get("status") != "labeled":
                            ann["char"] = char
                            ann["status"] = "labeled"
                            ann["updated_at"] = datetime.datetime.now().isoformat()
                            synced_count += 1
                        else:
                            skipped_count += 1
                        break
            else:
                # 添加新标注
                annotations.append({
                    "char_id": char_id,
                    "char": char,
                    "status": "labeled",
                    "created_at": datetime.datetime.now().isoformat(),
                    "updated_at": datetime.datetime.now().isoformat()
                })
                existing_char_ids.add(char_id)
                synced_count += 1
    
    # 保存统一标注
    unified_data["annotations"] = annotations
    with open(unified_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)
    
    print(f"同步完成")
    print(f"   - 新增/更新标注: {synced_count}")
    print(f"   - 已存在跳过: {skipped_count}")

if __name__ == "__main__":
    sync_cluster_labels()