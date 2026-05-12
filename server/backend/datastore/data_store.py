"""统一数据存储抽象层"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

class DataStore:
    def __init__(self, dataset_id: Optional[str] = None):
        if dataset_id is None:
            from config import DATASET_ID
            dataset_id = DATASET_ID
        
        self.dataset_id = dataset_id
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.dataset_dir = self.project_root / "bussiness" / "datahome" / dataset_id
    
        # 路径配置
        self.prelabels_path = self.dataset_dir / "pre_labels.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        self.clusters_path = self.dataset_dir / "clusters" / "hog_clusters.json"
        self.labels_path = self.dataset_dir / "clusters" / "labeling" / "labels.json"
        
        # 多轮聚类路径
        self.multi_clustering_dir = self.dataset_dir / "multi_clustering"
        
        # 回退路径（兼容旧数据结构）
        self.fallback_unified_labels = self.project_root / "bussiness" / "unified_labels.json"
        self.fallback_prelabels = self.project_root / "bussiness" / "pre_labels.json"
        
        # 缓存（按数据集隔离）
        self._cache = {}
        self._last_modified = {}
        
        # 日志记录器
        from .sync_logger import SyncLogger
        self._logger = SyncLogger(dataset_id)
    
    @classmethod
    def from_config(cls):
        """从配置文件创建 DataStore 实例"""
        from config import DATASET_ID
        return cls(DATASET_ID)
    
    def _get_cache_key(self, path: Path) -> str:
        """生成带数据集隔离的缓存键"""
        return f"{self.dataset_id}_{str(path)}"
    
    def _load_file(self, path: Path, fallback_path: Optional[Path] = None) -> dict:
        """加载 JSON 文件，支持回退路径，缓存按数据集隔离，处理编码错误"""
        cache_key = self._get_cache_key(path)
        
        if cache_key in self._cache:
            if path.exists() and self._last_modified.get(cache_key) == path.stat().st_mtime:
                return self._cache[cache_key]
        
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    content = content.replace('\ufffd', '')
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        return {}
            
            self._last_modified[cache_key] = path.stat().st_mtime
            self._cache[cache_key] = data
            return data
        
        if fallback_path and fallback_path.exists():
            try:
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                with open(fallback_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    content = content.replace('\ufffd', '')
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        return {}
            
            self._cache[cache_key] = data
            return data
        
        return {}
    
    def _save_file(self, path: Path, data: dict):
        """保存 JSON 文件，更新缓存"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        cache_key = self._get_cache_key(path)
        self._cache[cache_key] = data
        if path.exists():
            self._last_modified[cache_key] = path.stat().st_mtime
    
    def invalidate_cache(self):
        """清除所有缓存"""
        self._cache.clear()
        self._last_modified.clear()
    
    def _invalidate_dataset_cache(self):
        """仅清除当前数据集的缓存（不影响其他数据集）"""
        keys_to_remove = []
        for key in self._cache.keys():
            if key.startswith(f"{self.dataset_id}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._last_modified:
                del self._last_modified[key]
    
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
    
    def _sync_to_cluster_labels(self, char_id: str, char: str):
        """同步标注到聚类标注文件"""
        clusters = self.get_clusters()
        labels = self.get_cluster_labels()
        
        for cluster_id, chars_in_cluster in clusters.items():
            for idx, char_info in enumerate(chars_in_cluster):
                if char_info.get("char_id") == char_id:
                    if cluster_id not in labels:
                        labels[cluster_id] = {
                            "char": char,
                            "status": "labeled",
                            "char_labels": {}
                        }
                    labels[cluster_id]["char"] = char
                    labels[cluster_id]["status"] = "labeled"
                    if "char_labels" not in labels[cluster_id]:
                        labels[cluster_id]["char_labels"] = {}
                    labels[cluster_id]["char_labels"][str(idx)] = {
                        "char": char
                    }
                    self._save_file(self.labels_path, labels)
                    self._logger.log_sync_to_cluster(char_id, char, cluster_id, success=True)
                    return
        
        self._logger.log_sync_skipped(char_id, char, "字符不在聚类数据中")
    
    def _sync_to_multi_clustering(self, char_id: str, char: str):
        """同步标注到多轮聚类数据"""
        if not self.multi_clustering_dir.exists():
            self._logger.log_sync_skipped(char_id, char, "多轮聚类目录不存在")
            return
        
        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            self._logger.log_sync_skipped(char_id, char, "rounds目录不存在")
            return
        
        found = False
        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            round_num = int(round_dir.name.replace("round_", "")) if "round_" in round_dir.name else 0
            
            labels_path = round_dir / "labeling" / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"
            
            if not clusters_path.exists() or not labels_path.exists():
                continue
            
            clusters_data = self._load_file(clusters_path)
            clusters = clusters_data.get("clusters", {})
            labels = self._load_file(labels_path)
            
            for cluster_id, chars_in_cluster in clusters.items():
                for idx, char_info in enumerate(chars_in_cluster):
                    if char_info.get("char_id") == char_id:
                        if cluster_id not in labels:
                            labels[cluster_id] = {
                                "char": char,
                                "status": "labeled",
                                "char_labels": {}
                            }
                        labels[cluster_id]["char"] = char
                        labels[cluster_id]["status"] = "labeled"
                        if "char_labels" not in labels[cluster_id]:
                            labels[cluster_id]["char_labels"] = {}
                        labels[cluster_id]["char_labels"][str(idx)] = {
                            "char": char
                        }
                        self._save_file(labels_path, labels)
                        self._logger.log_sync_to_multi(char_id, char, round_num, cluster_id, success=True)
                        found = True
                        return
        
        if not found:
            self._logger.log_sync_skipped(char_id, char, "字符不在多轮聚类数据中")
    
    def write_annotation(self, char_id: str, char: str, status: str = "labeled"):
        """统一写入标注，自动同步所有数据源"""
        try:
            updates = {
                "char": char,
                "status": status,
                "dataset": self.dataset_id
            }
            
            self.update_unified_label(char_id, updates)
            self._logger.log_write_annotation(char_id, char, success=True)
            
            self._sync_to_cluster_labels(char_id, char)
            self.update_prelabel_status(char_id, status, char)
            self._logger.log_sync_to_prelabel(char_id, char, success=True)
            self._sync_to_multi_clustering(char_id, char)
            
            self._invalidate_dataset_cache()
            
        except Exception as e:
            self._logger.log_write_annotation(char_id, char, success=False, message=str(e))
            raise
    
    def _count_cluster_labeled(self) -> int:
        """统计聚类标注中的已标注数量"""
        labels = self.get_cluster_labels()
        count = 0
        for cluster_info in labels.values():
            if cluster_info.get("status") == "labeled":
                char_labels = cluster_info.get("char_labels", {})
                count += len([k for k, v in char_labels.items() if v.get("char")])
        return count
    
    def _count_multi_clustering_labeled(self) -> int:
        """统计多轮聚类中的已标注数量"""
        if not self.multi_clustering_dir.exists():
            return 0
        
        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            return 0
        
        total = 0
        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            labels_path = round_dir / "labeling" / "labels.json"
            if not labels_path.exists():
                continue
            
            labels = self._load_file(labels_path)
            for cluster_info in labels.values():
                if cluster_info.get("status") == "labeled":
                    char_labels = cluster_info.get("char_labels", {})
                    total += len([k for k, v in char_labels.items() if v.get("char")])
        
        return total
    
    # ==================== 统计数据 ====================
    
    def get_statistics(self) -> dict:
        """获取完整的统计数据（合并多源数据）"""
        prelabel_stats = self.get_prelabel_stats()
        total_images = prelabel_stats["total"]
        char_counts = prelabel_stats["char_counts"]
        
        annotations = self.get_unified_labels()
        labeled_count = len([a for a in annotations if a.get("status") == "labeled"])
        
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
    
    # ==================== 日志查询 ====================
    
    def get_sync_logs(self, limit: int = 100) -> list:
        """获取最近的同步日志"""
        return self._logger.get_recent_logs(limit)
    
    def get_sync_logs_by_date(self, date_str: str) -> list:
        """按日期获取同步日志"""
        return self._logger.get_logs_by_date(date_str)
    
    def get_today_sync_logs(self) -> list:
        """获取今日同步日志"""
        return self._logger.get_today_logs()
    
    def get_sync_statistics(self) -> dict:
        """获取同步日志统计"""
        return self._logger.get_statistics()