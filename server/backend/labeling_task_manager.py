import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class LabelingTaskManager:
    def __init__(self, config):
        self.config = config
        self.dataset_id = config.get("dataset.current", "pdf5826")

        self.project_root = Path(__file__).parent.parent.parent
        self.datahome_dir = self.project_root / "bussiness" / "datahome"
        self.dataset_dir = self.datahome_dir / self.dataset_id

        self.tasks_path = self.dataset_dir / "labeling_tasks.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        self.prelabels_path = self.dataset_dir / "pre_labels.json"

        # 确保统一标注文件存在（不存在则创建）
        if not self.unified_labels_path.exists():
            self._init_unified_labels()
        
    def load_tasks(self):
        if self.tasks_path.exists():
            with open(self.tasks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"tasks": {}}
    
    def save_tasks(self, data):
        with open(self.tasks_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def update_stats(self):
        tasks = self.load_tasks()
        
        if self.dataset_id not in tasks["tasks"]:
            tasks["tasks"][self.dataset_id] = {
                "total_images": 0,
                "labeled_count": 0,
                "prelabeled_count": 0,
                "unlabeled_count": 0,
                "char_stats": {}
            }
        
        total_count = 0
        labeled_count = 0
        prelabeled_count = 0
        unlabeled_count = 0
        pending_count = 0
        char_stats = {}
        
        if self.prelabels_path.exists():
            with open(self.prelabels_path, 'r', encoding='utf-8') as f:
                prelabel_data = json.load(f)
            
            if prelabel_data.get("dataset") == self.dataset_id:
                total_count = prelabel_data.get("stats", {}).get("total", 0)
                prelabeled_count = total_count
                
                char_counts = prelabel_data.get("char_counts", {})
                for char, counts in char_counts.items():
                    total = counts.get("total", 0)
                    if char not in char_stats:
                        char_stats[char] = {"total": 0, "confirmed": 0, "pending": 0}
                    char_stats[char]["total"] = total
                    char_stats[char]["pending"] = total
        
        if self.unified_labels_path.exists():
            with open(self.unified_labels_path, 'r', encoding='utf-8') as f:
                unified_data = json.load(f)
            
            annotations = unified_data.get("annotations", [])
            dataset_annotations = [a for a in annotations if a.get("dataset") == self.dataset_id]
            
            labeled_count = len([a for a in dataset_annotations if a.get("status") == "labeled"])
            
            for ann in dataset_annotations:
                char = ann.get("char", "")
                if char:
                    if char not in char_stats:
                        char_stats[char] = {"total": 0, "confirmed": 0, "pending": 0}
                    if ann.get("status") == "labeled":
                        char_stats[char]["confirmed"] += 1
                        if char_stats[char]["pending"] > 0:
                            char_stats[char]["pending"] -= 1
        
        unlabeled_count = total_count - labeled_count
        pending_count = total_count - labeled_count
        
        tasks["tasks"][self.dataset_id].update({
            "total_images": total_count,
            "labeled_count": labeled_count,
            "prelabeled_count": prelabeled_count,
            "unlabeled_count": unlabeled_count,
            "pending_count": pending_count,
            "char_stats": char_stats
        })
        
        self.save_tasks(tasks)
        return tasks["tasks"][self.dataset_id]
    
    def get_stats(self):
        tasks = self.load_tasks()
        return tasks.get("tasks", {}).get(self.dataset_id, {})
    
    def update_label_status(self, image_path, status, char=None):
        logger.info(f"[LabelingTaskManager] update_label_status called - dataset: {self.dataset_id}, image_path: {image_path}, status: {status}, char: {char}")
        logger.debug(f"[LabelingTaskManager] unified_labels_path: {self.unified_labels_path}")
        
        # 确保统一标注文件存在
        if not self.unified_labels_path.exists():
            logger.debug(f"[LabelingTaskManager] unified_labels.json not found, initializing...")
            self._init_unified_labels()
        
        with open(self.unified_labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"[LabelingTaskManager] Loaded data keys: {list(data.keys())}")
        logger.debug(f"[LabelingTaskManager] annotations count: {len(data.get('annotations', []))}")
        
        annotations = data.get("annotations", [])
        found = False
        
        for ann in annotations:
            if ann.get("image_path") == image_path:
                ann["status"] = status
                if char:
                    ann["char"] = char
                ann["updated_at"] = datetime.now().isoformat()
                found = True
                logger.debug(f"[LabelingTaskManager] Found existing annotation, updated: {ann}")
                break
        
        # 如果不存在，则创建新记录
        if not found:
            new_ann = {
                "char_id": self._extract_char_id(image_path),
                "char": char if char else "",
                "image_path": image_path,
                "dataset": self.dataset_id,
                "status": status,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            annotations.append(new_ann)
            logger.debug(f"[LabelingTaskManager] Created new annotation: {new_ann}")
            data["annotations"] = annotations
        
        with open(self.unified_labels_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if self.prelabels_path.exists():
            with open(self.prelabels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for prelabel in data.get("prelabels", []):
                if prelabel.get("image_path") == image_path:
                    if status == "labeled":
                        prelabel["status"] = "confirmed"
                    else:
                        prelabel["status"] = "rejected"
                    break
            
            with open(self.prelabels_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.update_stats()
    
    def _extract_char_id(self, image_path):
        """从图片路径中提取 char_id"""
        path = Path(image_path)
        return path.stem
    
    def _init_unified_labels(self):
        """初始化统一标注文件"""
        init_data = {
            "dataset": self.dataset_id,
            "total_labeled": 0,
            "char_distribution": {},
            "annotations": []
        }
        with open(self.unified_labels_path, 'w', encoding='utf-8') as f:
            json.dump(init_data, f, ensure_ascii=False, indent=2)
    
    def batch_update_labels(self, updates):
        for update in updates:
            self.update_label_status(
                update["image_path"],
                update["status"],
                update.get("char")
            )

if __name__ == "__main__":
    from server.backend.config import config
    manager = LabelingTaskManager(config._config)
    stats = manager.update_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))