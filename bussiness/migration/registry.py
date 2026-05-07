"""
全局汉字注册表管理
用于管理所有数据集的汉字标注信息，支持自动合并和增量更新
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class GlobalRegistry:
    """全局汉字注册表"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.registry_path = data_dir / "global_char_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表，如果不存在则创建"""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated_at": datetime.now().isoformat(),
                "registry": {},
                "statistics": {
                    "total_unique_chars": 0,
                    "fully_labeled_chars": 0,
                    "partially_labeled_chars": 0,
                    "unlabeled_chars": 0
                }
            }
    
    def _save_registry(self):
        """保存注册表"""
        self.registry["last_updated_at"] = datetime.now().isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def add_char(self, char: str, dataset_id: str, sample_count: int = 1):
        """
        添加汉字到注册表
        
        Args:
            char: 汉字
            dataset_id: 数据集ID
            sample_count: 样本数量
        """
        if char not in self.registry["registry"]:
            self.registry["registry"][char] = {
                "first_seen_dataset": dataset_id,
                "first_seen_at": datetime.now().isoformat(),
                "datasets": [],
                "total_samples_all_datasets": 0,
                "label_status": "unlabeled"
            }
        
        char_info = self.registry["registry"][char]
        
        if dataset_id not in char_info["datasets"]:
            char_info["datasets"].append(dataset_id)
        
        char_info["total_samples_all_datasets"] += sample_count
        
        # 更新标注状态
        char_info["label_status"] = self._determine_label_status(char_info)
        
        self._update_statistics()
        self._save_registry()
    
    def _determine_label_status(self, char_info: Dict) -> str:
        """确定汉字的标注状态"""
        if char_info["total_samples_all_datasets"] == 0:
            return "unlabeled"
        return "fully_labeled"
    
    def get_char_info(self, char: str) -> Optional[Dict]:
        """获取汉字信息"""
        return self.registry["registry"].get(char)
    
    def get_all_chars(self) -> List[str]:
        """获取所有已注册的汉字"""
        return list(self.registry["registry"].keys())
    
    def merge_from_dataset(self, dataset_id: str, char_info: Dict):
        """
        从数据集合并汉字信息
        
        Args:
            dataset_id: 数据集ID
            char_info: 汉字信息字典 {char: sample_count}
        """
        for char, count in char_info.items():
            self.add_char(char, dataset_id, count)
    
    def _update_statistics(self):
        """更新统计信息"""
        total = len(self.registry["registry"])
        fully_labeled = 0
        partially_labeled = 0
        unlabeled = 0
        
        for char_info in self.registry["registry"].values():
            status = char_info["label_status"]
            if status == "fully_labeled":
                fully_labeled += 1
            elif status == "partially_labeled":
                partially_labeled += 1
            else:
                unlabeled += 1
        
        self.registry["statistics"] = {
            "total_unique_chars": total,
            "fully_labeled_chars": fully_labeled,
            "partially_labeled_chars": partially_labeled,
            "unlabeled_chars": unlabeled
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.registry["statistics"]
    
    def clear(self):
        """清空注册表"""
        self.registry = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat(),
            "registry": {},
            "statistics": {
                "total_unique_chars": 0,
                "fully_labeled_chars": 0,
                "partially_labeled_chars": 0,
                "unlabeled_chars": 0
            }
        }
        self._save_registry()