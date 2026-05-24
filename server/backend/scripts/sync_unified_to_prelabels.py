"""将统一标注反向同步到预标注"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_ID

def sync_unified_to_prelabels(dataset_id: str = None):
    if dataset_id is None:
        dataset_id = DATASET_ID
    project_root = Path("d:/projects/word_detect_ocr")
    dataset_dir = project_root / "bussiness" / "datahome" / dataset_id
    
    # 路径定义
    unified_path = dataset_dir / "unified_labels.json"
    prelabels_path = dataset_dir / "pre_labels.json"
    
    print("开始将统一标注反向同步到预标注")
    print("   数据集: {}".format(dataset_id))
    print("=" * 60)
    
    # 读取统一标注
    if not unified_path.exists():
        print("统一标注文件不存在")
        return
    
    with open(unified_path, 'r', encoding='utf-8') as f:
        unified_data = json.load(f)
    
    # 读取预标注
    if not prelabels_path.exists():
        print("预标注文件不存在")
        return
    
    with open(prelabels_path, 'r', encoding='utf-8') as f:
        prelabels_data = json.load(f)
    
    # 构建已标注字符的映射
    labeled_chars = {}
    for ann in unified_data.get("annotations", []):
        char_id = ann.get("char_id")
        char = ann.get("char")
        status = ann.get("status")
        if char_id and char and status == "labeled":
            labeled_chars[char_id] = char
    
    print("统一标注中已标注的字符数: {}".format(len(labeled_chars)))
    
    # 更新预标注状态
    prelabels = prelabels_data.get("prelabels", [])
    updated_count = 0
    skipped_count = 0
    
    for prelabel in prelabels:
        char_id = prelabel.get("char_id")
        if char_id in labeled_chars:
            current_status = prelabel.get("status")
            if current_status != "confirmed":
                prelabel["status"] = "confirmed"
                prelabel["char"] = labeled_chars[char_id]
                updated_count += 1
            else:
                skipped_count += 1
    
    # 保存预标注
    prelabels_data["prelabels"] = prelabels
    with open(prelabels_path, 'w', encoding='utf-8') as f:
        json.dump(prelabels_data, f, ensure_ascii=False, indent=2)
    
    print("同步完成")
    print("   - 更新状态的预标注: {}".format(updated_count))
    print("   - 已确认跳过: {}".format(skipped_count))

if __name__ == "__main__":
    sync_unified_to_prelabels()