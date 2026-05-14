"""统计数据管理器 - 使用统一数据存储抽象层"""
import logging
from typing import Optional
from .data_store import DataStore

logger = logging.getLogger(__name__)


class StatsManager:
    _instance = None
    
    def __new__(cls, dataset_id: Optional[str] = None):
        if cls._instance is None or (dataset_id and cls._instance.dataset_id != dataset_id):
            cls._instance = super().__new__(cls)
            cls._instance.dataset_id = dataset_id or "pdf5826"
            cls._instance.data_store = DataStore(cls._instance.dataset_id)
        return cls._instance
    
    def get_stats(self) -> dict:
        """获取统计数据"""
        return self.data_store.get_statistics()
    
    def get_char_list(self, page: int = 1, page_size: int = 20, search: Optional[str] = None, 
                      sort_by: Optional[str] = None, sort_order: str = "desc") -> dict:
        """获取字符列表（支持分页、搜索、排序）"""
        stats = self.get_stats()
        char_stats = stats.get("char_stats", {})
        
        char_list = [
            {
                "char": char,
                "total": info["total"],
                "labeled": info["labeled"],
                "pending": info["pending"],
                "skipped": info["skipped"],
                "unlabeled": info["unlabeled"]
            }
            for char, info in char_stats.items()
        ]
        
        if search:
            char_list = [c for c in char_list if search in c["char"]]
        
        if sort_by and sort_by in ["total", "labeled", "pending", "skipped", "unlabeled"]:
            reverse = sort_order == "desc"
            char_list.sort(key=lambda x: x[sort_by], reverse=reverse)
        else:
            char_list.sort(key=lambda x: x["total"], reverse=True)
        
        total = len(char_list)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = char_list[start:end]
        
        return {
            "code": 0,
            "data": paginated,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    def get_cluster_stats(self) -> dict:
        """获取聚类统计"""
        clusters = self.data_store.get_clusters()
        labels = self.data_store.get_cluster_labels()
        
        total_clusters = len(clusters)
        labeled_clusters = 0
        total_chars = 0
        labeled_chars = 0
        
        for cluster_id, chars in clusters.items():
            total_chars += len(chars)
            cluster_labels = labels.get(cluster_id, {})
            
            if cluster_labels.get("status") == "labeled":
                labeled_clusters += 1
            
            char_labels = cluster_labels.get("char_labels", {})
            labeled_chars += sum(1 for v in char_labels.values() if v.get("char"))
            
            for char in chars:
                if char.get("confirmed"):
                    labeled_chars += 1
        
        return {
            "total_clusters": total_clusters,
            "labeled_clusters": labeled_clusters,
            "total_chars": total_chars,
            "labeled_chars": labeled_chars,
            "unlabeled_chars": total_chars - labeled_chars
        }
    
    def confirm_annotation(self, char_id: str, char: str, source: str = "ocr_confirm"):
        """确认标注（通过统一接口更新所有数据源）"""
        #logger.debug(f"[StatsManager] confirm_annotation - dataset: {self.dataset_id}, char_id: {char_id}, char: {char}")
        
        # 使用统一写入接口，自动同步所有数据源
        self.data_store.write_annotation(
            char_id=char_id,
            char=char,
            status="labeled",
            source=source,
            changed_by="user",
            comment="OCR预标注确认"
        )
        
        logger.debug(f"[StatsManager] confirm_annotation completed - char_id: {char_id}")
    
    def batch_confirm_annotations(self, annotations: list) -> dict:
        """批量确认标注（使用批量写入优化性能）"""
        logger.debug(f"[StatsManager] batch_confirm_annotations - dataset: {self.dataset_id}, count: {len(annotations)}")
        
        try:
            # 使用批量写入方法大幅提升性能
            results = self.data_store.batch_write_annotations(
                annotations=annotations,
                source="ocr_confirm",
                changed_by="user",
                comment="OCR预标注确认"
            )
            
            logger.debug(f"[StatsManager] batch_confirm_annotations completed - success: {results['success_count']}/{results['total_count']}")
            return results
            
        except Exception as e:
            logger.error(f"[StatsManager] batch_confirm_annotations failed: {e}")
            return {
                "success_count": 0,
                "total_count": len(annotations)
            }
    
    def refresh_cache(self):
        """刷新缓存"""
        self.data_store.invalidate_cache()
    
    def get_label_history(self, char_id: Optional[str] = None, limit: int = 100) -> list:
        """获取标注历史记录"""
        return self.data_store.get_label_history(char_id, limit)