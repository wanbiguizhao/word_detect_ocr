"""
标注结果共享模块
支持标注结果的导出、导入和跨数据集共享
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4


class LabelSharingManager:
    """标注共享管理器"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.shared_labels_dir = data_dir / "shared_labels"
        self.shared_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def export_labels(
        self,
        dataset_id: str,
        labels_data: Dict,
        description: str = ""
    ) -> str:
        """
        导出标注结果
        
        Args:
            dataset_id: 数据集ID
            labels_data: 标注数据
            description: 描述信息
        
        Returns:
            导出文件路径
        """
        export_id = str(uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算标注数据的哈希值用于去重
        labels_hash = hashlib.md5(
            json.dumps(labels_data, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()[:16]
        
        export_data = {
            "version": "1.0",
            "export_id": export_id,
            "dataset_id": dataset_id,
            "exported_at": datetime.now().isoformat(),
            "description": description,
            "labels_hash": labels_hash,
            "labels_data": labels_data,
            "metadata": {
                "total_clusters": len(labels_data),
                "labeled_clusters": sum(1 for v in labels_data.values() if v.get("status") == "labeled"),
                "chars_count": self._count_chars(labels_data)
            }
        }
        
        filename = f"labels_{dataset_id}_{timestamp}_{export_id}.json"
        export_path = self.shared_labels_dir / filename
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 标注结果已导出到: {export_path}")
        return str(export_path)
    
    def _count_chars(self, labels_data: Dict) -> int:
        """统计标注的汉字数量"""
        count = 0
        for cluster_data in labels_data.values():
            if "chars" in cluster_data:
                count += sum(cluster_data["chars"].values())
            elif "char" in cluster_data:
                count += 1
            elif "char_labels" in cluster_data:
                count += len(cluster_data["char_labels"])
        return count
    
    def list_exported_files(self) -> List[Dict]:
        """列出所有已导出的标注文件"""
        exported_files = []
        
        for file_path in self.shared_labels_dir.glob("labels_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                exported_files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "export_id": data.get("export_id"),
                    "dataset_id": data.get("dataset_id"),
                    "exported_at": data.get("exported_at"),
                    "description": data.get("description", ""),
                    "metadata": data.get("metadata", {})
                })
            except Exception:
                continue
        
        exported_files.sort(key=lambda x: x["exported_at"], reverse=True)
        return exported_files
    
    def import_labels(
        self,
        source_file_path: str,
        target_dataset_id: str,
        merge_strategy: str = "merge"
    ) -> Dict:
        """
        导入标注结果到目标数据集
        
        Args:
            source_file_path: 源标注文件路径
            target_dataset_id: 目标数据集ID
            merge_strategy: 合并策略
                - "replace": 替换现有标注
                - "merge": 合并（保留现有标注）
                - "overwrite": 覆盖匹配的标注
        
        Returns:
            导入结果统计
        """
        source_path = Path(source_file_path)
        
        if not source_path.exists():
            return {"code": -1, "msg": "源文件不存在"}
        
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)
        except Exception as e:
            return {"code": -1, "msg": f"读取源文件失败: {str(e)}"}
        
        source_labels = source_data.get("labels_data", {})
        if not source_labels:
            return {"code": -1, "msg": "源文件中没有标注数据"}
        
        # 构建字符到聚类的映射
        char_to_clusters = self._build_char_to_clusters_map(source_labels)
        
        # 获取目标数据集的聚类信息
        target_clusters_path = Path(__file__).parent.parent / "datahome" / target_dataset_id / "clusters" / "hog_clusters.json"
        
        if not target_clusters_path.exists():
            return {"code": -1, "msg": "目标数据集的聚类文件不存在"}
        
        try:
            with open(target_clusters_path, "r", encoding="utf-8") as f:
                target_clusters = json.load(f)
        except Exception as e:
            return {"code": -1, "msg": f"读取目标聚类文件失败: {str(e)}"}
        
        # 获取目标数据集的现有标注
        target_labels_path = Path(__file__).parent.parent / "datahome" / target_dataset_id / "clusters" / "labeling" / "labels.json"
        if target_labels_path.exists():
            with open(target_labels_path, "r", encoding="utf-8") as f:
                target_labels = json.load(f)
        else:
            target_labels = {}
        
        # 执行导入
        imported_count = 0
        skipped_count = 0
        updated_count = 0
        
        for cluster_id, cluster_info in target_clusters.get("clusters", {}).items():
            if cluster_id in target_labels:
                if merge_strategy == "merge":
                    skipped_count += len(cluster_info)
                    continue
                elif merge_strategy == "replace":
                    pass
            
            # 尝试找到匹配的字符
            matched_char = self._find_matching_char(cluster_info, char_to_clusters)
            
            if matched_char:
                if cluster_id in target_labels:
                    # 更新现有标注
                    target_labels[cluster_id]["char"] = matched_char
                    target_labels[cluster_id]["status"] = "labeled"
                    updated_count += 1
                else:
                    # 创建新标注
                    target_labels[cluster_id] = {
                        "char": matched_char,
                        "status": "labeled",
                        "confidence": 0.8,
                        "imported_from": source_data.get("export_id"),
                        "imported_at": datetime.now().isoformat()
                    }
                    imported_count += len(cluster_info)
        
        # 保存导入结果
        target_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_labels_path, "w", encoding="utf-8") as f:
            json.dump(target_labels, f, ensure_ascii=False, indent=2)
        
        return {
            "code": 0,
            "msg": "导入完成",
            "imported_count": imported_count,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "source_dataset": source_data.get("dataset_id"),
            "target_dataset": target_dataset_id,
            "merge_strategy": merge_strategy
        }
    
    def _build_char_to_clusters_map(self, labels_data: Dict) -> Dict[str, List]:
        """构建字符到聚类的映射"""
        char_to_clusters = {}
        
        for cluster_id, cluster_data in labels_data.items():
            if "char" in cluster_data:
                char = cluster_data["char"]
                if char not in char_to_clusters:
                    char_to_clusters[char] = []
                char_to_clusters[char].append(cluster_id)
            elif "chars" in cluster_data:
                for char, _ in cluster_data["chars"].items():
                    if char not in char_to_clusters:
                        char_to_clusters[char] = []
                    char_to_clusters[char].append(cluster_id)
        
        return char_to_clusters
    
    def _find_matching_char(self, cluster_info: List, char_to_clusters: Dict) -> Optional[str]:
        """根据聚类信息找到匹配的字符"""
        # 这里可以扩展为更复杂的匹配算法
        # 当前简单实现：返回第一个找到的字符
        for char in char_to_clusters.keys():
            return char
        return None
    
    def share_labels(
        self,
        source_dataset_id: str,
        target_dataset_ids: List[str],
        merge_strategy: str = "merge"
    ) -> Dict:
        """
        跨数据集共享标注
        
        Args:
            source_dataset_id: 源数据集ID
            target_dataset_ids: 目标数据集ID列表
            merge_strategy: 合并策略
        
        Returns:
            共享结果统计
        """
        # 读取源数据集标注
        source_labels_path = Path(__file__).parent.parent / "datahome" / source_dataset_id / "clusters" / "labeling" / "labels.json"
        
        if not source_labels_path.exists():
            return {"code": -1, "msg": "源数据集标注文件不存在"}
        
        with open(source_labels_path, "r", encoding="utf-8") as f:
            source_labels = json.load(f)
        
        results = []
        
        for target_dataset_id in target_dataset_ids:
            result = self.import_labels(
                str(source_labels_path),
                target_dataset_id,
                merge_strategy
            )
            results.append({
                "target_dataset": target_dataset_id,
                **result
            })
        
        return {
            "code": 0,
            "msg": "共享完成",
            "source_dataset": source_dataset_id,
            "results": results
        }
    
    def delete_exported_file(self, filename: str) -> bool:
        """删除已导出的标注文件"""
        file_path = self.shared_labels_dir / filename
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_share_statistics(self) -> Dict:
        """获取共享统计信息"""
        exported_files = self.list_exported_files()
        
        return {
            "total_exported_files": len(exported_files),
            "total_shared_chars": sum(
                item["metadata"].get("chars_count", 0) for item in exported_files
            ),
            "recent_exports": exported_files[:5]
        }