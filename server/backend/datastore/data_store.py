"""统一数据存储抽象层"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

class DataStore:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.dataset_dir = self.project_root / "bussiness" / "datahome" / dataset_id
        
        # 路径配置
        self.prelabels_path = self.dataset_dir / "pre_labels.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        self.clusters_path = self.dataset_dir / "clusters" / "hog_clusters.json"
        self.labels_path = self.dataset_dir / "clusters" / "labeling" / "labels.json"
        
        # 回退路径（兼容旧数据结构）
        self.fallback_unified_labels = self.project_root / "bussiness" / "unified_labels.json"
        self.fallback_prelabels = self.project_root / "bussiness" / "pre_labels.json"
        
        # 缓存
        self._cache = {}
        self._last_modified = {}
    
    def _load_file(self, path: Path, fallback_path: Optional[Path] = None) -> dict:
        """加载 JSON 文件，支持回退路径"""
        cache_key = str(path)
        
        # 检查缓存
        if cache_key in self._cache:
            # 检查文件是否修改
            if path.exists() and self._last_modified.get(cache_key) == path.stat().st_mtime:
                return self._cache[cache_key]
        
        # 尝试加载主路径
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._last_modified[cache_key] = path.stat().st_mtime
            self._cache[cache_key] = data
            return data
        
        # 尝试加载回退路径
        if fallback_path and fallback_path.exists():
            with open(fallback_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[cache_key] = data
            return data
        
        return {}
    
    def _save_file(self, path: Path, data: dict):
        """保存 JSON 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 更新缓存
        cache_key = str(path)
        self._cache[cache_key] = data
        if path.exists():
            self._last_modified[cache_key] = path.stat().st_mtime
    
    def invalidate_cache(self):
        """清除所有缓存"""
        self._cache.clear()
        self._last_modified.clear()
    
    # ==================== 预标注数据 ====================
    
    def get_prelabels(self, char: Optional[str] = None) -> list:
        """获取预标注数据"""
        data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        prelabels = data.get("prelabels", [])
        
        if char:
            return [p for p in prelabels if p.get("predicted_char") == char]
        return prelabels
    
    def get_prelabel_stats(self) -> dict:
        """获取预标注统计"""
        data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        return {
            "total": data.get("stats", {}).get("total", 0),
            "char_counts": data.get("char_counts", {})
        }
    
    def update_prelabel_status(self, char_id: str, status: str, char: str = None):
        """更新预标注状态"""
        data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        prelabels = data.get("prelabels", [])

        for prelabel in prelabels:
            if prelabel.get("char_id") == char_id:
                prelabel["status"] = status
                if char is not None:
                    prelabel["predicted_char"] = char
                break

        self._save_file(self.prelabels_path, data)
    
    # ==================== 统一标注数据 ====================
    
    def get_unified_labels(self, status: Optional[str] = None) -> list:
        """获取统一标注数据"""
        data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)
        annotations = data.get("annotations", [])
        
        if status:
            return [a for a in annotations if a.get("status") == status]
        return annotations
    
    def add_unified_label(self, annotation: dict):
        """添加统一标注"""
        data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)
        
        if "annotations" not in data:
            data["annotations"] = []
        
        data["annotations"].append(annotation)
        self._save_file(self.unified_labels_path, data)
    
    def update_unified_label(self, char_id: str, updates: dict):
        """更新统一标注，如果不存在则添加"""
        data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)
        annotations = data.get("annotations", [])

        found = False
        for ann in annotations:
            if ann.get("char_id") == char_id:
                ann.update(updates)
                found = True
                break

        if not found:
            annotations.append({"char_id": char_id, **updates})

        self._save_file(self.unified_labels_path, data)
    
    def get_confirmed_annotations(self) -> Dict[str, str]:
        """获取已确认的标注（char_id -> char）"""
        annotations = self.get_unified_labels()
        return {
            ann["char_id"]: ann["char"]
            for ann in annotations
            if ann.get("char_id") and ann.get("char") and ann.get("status") == "labeled"
        }
    
    # ==================== 聚类数据 ====================
    
    def get_clusters(self) -> dict:
        """获取聚类数据（包含已确认标注）"""
        clusters_data = self._load_file(self.clusters_path)
        clusters = clusters_data.get("clusters", {})
        
        # 合并已确认的标注
        confirmed = self.get_confirmed_annotations()
        for cluster_id, chars in clusters.items():
            for char in chars:
                char_id = char.get("char_id")
                if char_id in confirmed:
                    char["confirmed"] = True
                    char["confirmed_char"] = confirmed[char_id]
        
        return clusters
    
    def get_cluster_labels(self) -> dict:
        """获取聚类标注"""
        return self._load_file(self.labels_path)
    
    # ==================== 统计数据 ====================
    
    def get_statistics(self) -> dict:
        """获取完整的统计数据"""
        # 预标注统计
        prelabel_stats = self.get_prelabel_stats()
        total_images = prelabel_stats["total"]
        char_counts = prelabel_stats["char_counts"]
        
        # 统一标注统计
        annotations = self.get_unified_labels()
        labeled_count = len([a for a in annotations if a.get("status") == "labeled"])
        
        # 字符级统计
        char_stats = {}
        for char, counts in char_counts.items():
            char_stats[char] = {
                "total": counts.get("total", 0),
                "confirmed": 0,
                "pending": counts.get("total", 0)
            }
        
        for ann in annotations:
            char = ann.get("char")
            if char and char in char_stats:
                if ann.get("status") == "labeled":
                    char_stats[char]["confirmed"] += 1
                    char_stats[char]["pending"] = max(0, char_stats[char]["pending"] - 1)
        
        return {
            "dataset": self.dataset_id,
            "total_images": total_images,
            "labeled_count": labeled_count,
            "unlabeled_count": total_images - labeled_count,
            "char_stats": char_stats
        }