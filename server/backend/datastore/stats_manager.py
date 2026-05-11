"""统计数据管理器"""
from typing import Optional
from .data_store import DataStore

class StatsManager:
    _instance = None
    
    def __new__(cls, dataset_id: Optional[str] = None):
        if cls._instance is None or (dataset_id and cls._instance.dataset_id != dataset_id):
            cls._instance = super().__new__(cls)
            cls._instance.dataset_id = dataset_id or "pdf5823"
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
        
        # 转换为列表
        char_list = [
            {
                "char": char,
                "total": info["total"],
                "confirmed": info["confirmed"],
                "pending": info["pending"]
            }
            for char, info in char_stats.items()
        ]
        
        # 搜索过滤
        if search:
            char_list = [c for c in char_list if search in c["char"]]
        
        # 排序
        if sort_by and sort_by in ["total", "confirmed", "pending"]:
            reverse = sort_order == "desc"
            char_list.sort(key=lambda x: x[sort_by], reverse=reverse)
        else:
            char_list.sort(key=lambda x: x["total"], reverse=True)
        
        # 分页
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
            
            # 加上已确认的标注
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
    
    def confirm_annotation(self, char_id: str, char: str):
        """确认标注（更新多个数据源）"""
        self.data_store.update_unified_label(char_id, {
            "char": char,
            "status": "labeled"
        })

        self.data_store.update_prelabel_status(char_id, "confirmed", char)

        self.data_store.invalidate_cache()
    
    def batch_confirm_annotations(self, annotations: list):
        """批量确认标注"""
        for ann in annotations:
            self.confirm_annotation(ann["char_id"], ann["char"])
    
    def refresh_cache(self):
        """刷新缓存"""
        self.data_store.invalidate_cache()