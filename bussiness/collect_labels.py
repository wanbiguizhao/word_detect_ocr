#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标注数据统一收集工具

从 bussiness/datahome 目录下的各个数据集中收集所有已标注的数据，
生成统一的元数据文件。

使用示例：
    # 收集所有标注数据
    python collect_labels.py

    # 只收集指定数据集
    python collect_labels.py pdf01 pdf5823

    # 指定输出文件
    python collect_labels.py -o unified_labels.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict


def collect_dataset_labels(dataset_dir, dataset_name, bussiness_dir):
    """
    收集单个数据集的标注数据

    Args:
        dataset_dir: 数据集目录路径
        dataset_name: 数据集名称
        bussiness_dir: bussiness目录路径（用于计算相对路径）

    Returns:
        list: 标注数据列表
    """
    labels_file = os.path.join(dataset_dir, 'clusters', 'labeling', 'labels.json')
    clusters_file = os.path.join(dataset_dir, 'clusters', 'hog_clusters.json')

    annotations = []

    if not os.path.exists(labels_file):
        print(f"  警告: 标注文件不存在 - {labels_file}")
        return annotations

    if not os.path.exists(clusters_file):
        print(f"  警告: Cluster文件不存在 - {clusters_file}")
        return annotations

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    with open(clusters_file, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)

    clusters = clusters_data.get('clusters', {})

    for cluster_id, label_info in labels_data.items():
        if label_info.get('status') != 'labeled':
            continue

        true_char = label_info.get('char')
        if not true_char:
            continue

        if cluster_id not in clusters:
            continue

        for char_info in clusters[cluster_id]:
            image_path = char_info.get('image_path')
            if not image_path or not os.path.exists(image_path):
                continue

            # 转换为相对于bussiness目录的路径
            rel_path = os.path.relpath(image_path, bussiness_dir)
            rel_path = rel_path.replace('\\', '/')

            annotations.append({
                'dataset': dataset_name,
                'char': true_char,
                'cluster_id': cluster_id,
                'char_id': char_info.get('char_id'),
                'image_path': rel_path
            })

    return annotations


def collect_all_labels(data_home_dir, dataset_names=None):
    """
    收集所有数据集的标注数据

    Args:
        data_home_dir: 数据主目录
        dataset_names: 指定数据集列表，None表示所有数据集

    Returns:
        dict: 统一的标注数据
    """
    if not os.path.exists(data_home_dir):
        print(f"错误: 数据目录不存在 - {data_home_dir}")
        return None

    # 获取bussiness目录
    bussiness_dir = os.path.dirname(data_home_dir.rstrip(os.sep))

    # 确定要处理的数据集
    if dataset_names is None:
        dataset_names = [d for d in os.listdir(data_home_dir)
                        if os.path.isdir(os.path.join(data_home_dir, d))
                        and not d.startswith('.')]
        dataset_names.sort()

    print(f"发现数据集: {dataset_names}")

    all_annotations = []
    dataset_stats = {}

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(data_home_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"  跳过: {dataset_name} (不是目录)")
            continue

        print(f"处理数据集: {dataset_name}...")
        annotations = collect_dataset_labels(dataset_dir, dataset_name, bussiness_dir)
        all_annotations.extend(annotations)

        dataset_stats[dataset_name] = {
            'labeled_count': len(annotations)
        }
        print(f"  已标注数据: {len(annotations)} 个")

    # 统计字符分布
    char_distribution = defaultdict(int)
    for ann in all_annotations:
        char_distribution[ann['char']] += 1

    result = {
        'datasets': dataset_names,
        'total_labeled': len(all_annotations),
        'dataset_stats': dataset_stats,
        'char_distribution': dict(char_distribution),
        'annotations': all_annotations
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='标注数据统一收集工具')
    parser.add_argument('datasets', nargs='*', help='指定数据集名称（为空则处理所有）')
    parser.add_argument('--output', '-o', default='unified_labels.json', help='输出文件路径')
    parser.add_argument('--data_home', '-d', default=None, help='数据主目录路径')
    args = parser.parse_args()

    # 确定数据主目录
    if args.data_home is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_home = os.path.join(script_dir, 'datahome')
    else:
        data_home = args.data_home

    print(f"数据目录: {data_home}")
    print(f"输出文件: {args.output}")
    print("-" * 60)

    # 收集标注数据
    result = collect_all_labels(data_home, args.datasets if args.datasets else None)

    if result is None:
        sys.exit(1)

    # 保存结果
    print("-" * 60)
    print(f"总共收集标注数据: {result['total_labeled']} 个")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {args.output}")

    # 打印字符分布统计（前10）
    if result['char_distribution']:
        print("\n字符分布统计 (前10):")
        sorted_chars = sorted(result['char_distribution'].items(),
                             key=lambda x: x[1], reverse=True)[:10]
        for char, count in sorted_chars:
            print(f"  '{char}': {count} 个")


if __name__ == '__main__':
    main()