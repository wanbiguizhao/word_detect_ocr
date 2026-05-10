#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析标注数据分布

使用 unified_labels.json 分析当前标注数据的分布情况。

使用示例：
    python analyze_data_distribution.py

    # 只分析指定数据集
    python analyze_data_distribution.py --datasets pdf01
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_data_distribution(datasets=None):
    """
    分析当前标注数据的分布情况

    Args:
        datasets: 数据集名称列表，None表示所有数据集
    """
    # 加载统一标注数据
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from unified_labels import UnifiedLabelsLoader

    loader = UnifiedLabelsLoader()

    # 按数据集过滤
    if datasets:
        annotations = loader.filter_annotations(datasets=datasets)
    else:
        annotations = loader.annotations

    if not annotations:
        print('错误: 没有找到标注数据！')
        return

    # 统计每个汉字的图片数量
    char_image_counts = Counter()
    for ann in annotations:
        char_image_counts[ann['char']] += 1

    # 统计信息
    total_chars = len(char_image_counts)
    total_images = sum(char_image_counts.values())
    min_count = min(char_image_counts.values()) if char_image_counts else 0
    max_count = max(char_image_counts.values()) if char_image_counts else 0
    avg_count = total_images / total_chars if total_chars > 0 else 0

    print(f"{'='*60}")
    print("数据分布分析报告")
    print(f"{'='*60}")
    print(f"数据集: {', '.join(datasets) if datasets else loader.datasets}")
    print(f"汉字数量: {total_chars} 个")
    print(f"图片总数: {total_images} 张")
    print(f"平均每个汉字: {avg_count:.1f} 张")
    print(f"最少: {min_count} 张")
    print(f"最多: {max_count} 张")
    print(f"极差: {max_count - min_count} 张")
    print(f"标准差: {calculate_std(char_image_counts.values()):.1f}")

    # 统计不同数量级的汉字
    bins = [0, 10, 20, 50, 100, 200, float('inf')]
    bin_labels = ["0-10", "11-20", "21-50", "51-100", "101-200", "200+"]
    bin_counts = [0] * len(bin_labels)

    for count in char_image_counts.values():
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low < count <= high:
                bin_counts[i] += 1
                break

    print("\n汉字数量分布:")
    for label, count in zip(bin_labels, bin_counts):
        print(f"  {label} 张: {count} 个汉字")

    # 显示前20个最多和最少的汉字
    print("\n前10个最多图片的汉字:")
    for char, count in char_image_counts.most_common(10):
        print(f"  {char}: {count} 张")

    print("\n后10个最少图片的汉字:")
    for char, count in char_image_counts.most_common()[-10:]:
        print(f"  {char}: {count} 张")

    # 保存分布数据
    dist_data = {
        "datasets": datasets if datasets else loader.datasets,
        "total_chars": total_chars,
        "total_images": total_images,
        "min_count": min_count,
        "max_count": max_count,
        "avg_count": avg_count,
        "char_distribution": dict(char_image_counts),
        "bin_distribution": dict(zip(bin_labels, bin_counts))
    }

    output_file = PROJECT_ROOT / 'data' / 'data_distribution.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dist_data, f, ensure_ascii=False, indent=2)

    print(f"\n分布数据已保存到: {output_file}")

    return char_image_counts


def calculate_std(values):
    """计算标准差"""
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def main():
    parser = argparse.ArgumentParser(description='分析标注数据分布')
    parser.add_argument('--datasets', nargs='+', help='指定数据集名称（为空则分析所有）')
    args = parser.parse_args()

    analyze_data_distribution(datasets=args.datasets)


if __name__ == '__main__':
    main()