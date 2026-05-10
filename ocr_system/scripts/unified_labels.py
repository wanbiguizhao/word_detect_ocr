"""
统一标注数据加载模块

从 unified_labels.json 加载已标注的数据，提供统一的数据访问接口。

使用示例：
    from unified_labels import load_unified_labels

    data = load_unified_labels()
    print(f"总标注数: {data['total_labeled']}")

    # 获取指定数据集的数据
    pdf01_data = [ann for ann in data['annotations'] if ann['dataset'] == 'pdf01']

    # 获取指定字符的数据
    char_de = [ann for ann in data['annotations'] if ann['char'] == '的']
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class UnifiedLabelsLoader:
    """统一标注数据加载器"""

    def __init__(self, labels_path: str = None):
        """
        初始化加载器

        Args:
            labels_path: unified_labels.json 文件路径，默认为 bussiness/unified_labels.json
        """
        if labels_path is None:
            script_dir = Path(__file__).parent.parent
            self.labels_path = script_dir.parent / 'bussiness' / 'unified_labels.json'
        else:
            self.labels_path = Path(labels_path)

        self._data = None
        self._load()

    def _load(self):
        """加载标注数据"""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.labels_path}")

        with open(self.labels_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

    @property
    def data(self) -> Dict:
        """获取完整数据"""
        return self._data

    @property
    def datasets(self) -> List[str]:
        """获取所有数据集名称"""
        return self._data.get('datasets', [])

    @property
    def total_labeled(self) -> int:
        """获取总标注数"""
        return self._data.get('total_labeled', 0)

    @property
    def char_distribution(self) -> Dict[str, int]:
        """获取字符分布"""
        return self._data.get('char_distribution', {})

    @property
    def annotations(self) -> List[Dict]:
        """获取所有标注"""
        return self._data.get('annotations', [])

    def get_dataset_annotations(self, dataset: str) -> List[Dict]:
        """获取指定数据集的标注"""
        return [ann for ann in self.annotations if ann['dataset'] == dataset]

    def get_char_annotations(self, char: str) -> List[Dict]:
        """获取指定字符的所有标注"""
        return [ann for ann in self.annotations if ann['char'] == char]

    def get_full_image_path(self, annotation: Dict) -> str:
        """
        获取标注对应的完整图片路径

        Args:
            annotation: 标注字典

        Returns:
            完整的图片路径
        """
        # image_path 是相对于 bussiness 目录的路径
        rel_path = annotation['image_path']
        script_dir = Path(__file__).parent.parent.parent
        full_path = script_dir / 'bussiness' / rel_path
        return str(full_path)

    def filter_annotations(
        self,
        datasets: List[str] = None,
        chars: List[str] = None,
        min_samples: int = None
    ) -> List[Dict]:
        """
        过滤标注数据

        Args:
            datasets: 数据集名称列表
            chars: 字符列表
            min_samples: 最少样本数（基于字符）

        Returns:
            过滤后的标注列表
        """
        filtered = self.annotations

        # 按数据集过滤
        if datasets:
            filtered = [ann for ann in filtered if ann['dataset'] in datasets]

        # 按字符过滤
        if chars:
            filtered = [ann for ann in filtered if ann['char'] in chars]

        # 按字符样本数过滤
        if min_samples is not None:
            char_counts = {}
            for ann in filtered:
                char = ann['char']
                char_counts[char] = char_counts.get(char, 0) + 1

            valid_chars = {c for c, count in char_counts.items() if count >= min_samples}
            filtered = [ann for ann in filtered if ann['char'] in valid_chars]

        return filtered

    def get_labeled_chars(self) -> List[str]:
        """获取所有已标注的字符列表"""
        return sorted(self.char_distribution.keys())

    def get_char_count(self, char: str) -> int:
        """获取指定字符的样本数"""
        return self.char_distribution.get(char, 0)

    def print_summary(self):
        """打印数据摘要"""
        print(f"{'='*60}")
        print("统一标注数据摘要")
        print(f"{'='*60}")
        print(f"数据集: {', '.join(self.datasets)}")
        print(f"总标注数: {self.total_labeled}")
        print(f"字符数: {len(self.char_distribution)}")

        print(f"\n字符分布 (前10):")
        sorted_chars = sorted(self.char_distribution.items(),
                           key=lambda x: x[1], reverse=True)[:10]
        for char, count in sorted_chars:
            print(f"  '{char}': {count} 个")


def load_unified_labels(labels_path: str = None) -> Dict:
    """
    便捷函数：加载统一标注数据

    Args:
        labels_path: unified_labels.json 文件路径

    Returns:
        标注数据字典
    """
    loader = UnifiedLabelsLoader(labels_path)
    return loader.data


if __name__ == '__main__':
    # 测试
    loader = UnifiedLabelsLoader()
    loader.print_summary()

    print(f"\n获取 pdf01 数据集的前5条标注:")
    pdf01_anns = loader.get_dataset_annotations('pdf01')[:5]
    for ann in pdf01_anns:
        full_path = loader.get_full_image_path(ann)
        print(f"  字符: {ann['char']}, 图片: {ann['image_path']}")
        print(f"    完整路径: {full_path}")