#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从统一标注数据生成训练/验证/测试集CSV

使用 unified_labels.json 中的数据构建训练数据集。

使用示例：
    python generate_train_csv.py

    # 只使用部分数据集
    python generate_train_csv.py --datasets pdf01

    # 设置最小样本数
    python generate_train_csv.py --min-samples 10
"""

import os
import sys
import csv
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV

MIN_SAMPLES_PER_CLASS = 10


def generate_csv(datasets=None, min_samples=MIN_SAMPLES_PER_CLASS, train_ratio=0.8):
    """
    从标注数据生成训练/验证/测试集CSV

    Args:
        datasets: 数据集名称列表，None表示所有数据集
        min_samples: 每个类别最少样本数
        train_ratio: 训练集比例
    """
    # 导入统一标注数据加载器
    from unified_labels import UnifiedLabelsLoader

    # 加载数据
    loader = UnifiedLabelsLoader()
    print(f"加载标注数据: {loader.total_labeled} 条")

    # 按数据集过滤
    if datasets:
        annotations = loader.filter_annotations(datasets=datasets)
    else:
        annotations = loader.annotations

    if not annotations:
        print("错误: 没有找到标注数据！")
        return

    # 过滤样本数不足的字符
    char_counter = Counter([ann['char'] for ann in annotations])
    valid_chars = [char for char, count in char_counter.items() if count >= min_samples]
    filtered_annotations = [ann for ann in annotations if ann['char'] in valid_chars]

    print(f"过滤后数据量: {len(filtered_annotations)} (移除 {len(annotations) - len(filtered_annotations)} 条稀有样本)")
    print(f"保留类别数: {len(valid_chars)}")

    if not filtered_annotations:
        print("错误: 没有足够样本的数据！")
        return

    # 准备数据
    images = []
    chars = []

    for ann in filtered_annotations:
        full_path = loader.get_full_image_path(ann)
        if os.path.exists(full_path):
            images.append(full_path)
            chars.append(ann['char'])

    print(f"有效数据: {len(images)} 条")

    # 划分数据集
    # 第一次分割：train_ratio 训练，剩余验证+测试
    train_images, temp_images, train_chars, temp_chars = train_test_split(
        images, chars, test_size=1-train_ratio, random_state=42, stratify=chars
    )

    # 第二次分割：验证集和测试集各占一半
    val_images, test_images, val_chars, test_chars = train_test_split(
        temp_images, temp_chars, test_size=0.5, random_state=42
    )

    # 创建图片目录
    train_img_dir = os.path.join(DATA_ROOT, 'train_images')
    val_img_dir = os.path.join(DATA_ROOT, 'val_images')
    test_img_dir = os.path.join(DATA_ROOT, 'test_images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 写入CSV
    def write_csv(filename, img_list, char_list, img_dir):
        img_dir_name = os.path.basename(img_dir)
        rows = [['image_path', 'char']]

        for img, char in zip(img_list, char_list):
            img_path_obj = Path(img)
            dataset_prefix = img_path_obj.parts[-3] if len(img_path_obj.parts) >= 3 else ''
            filename_only = img_path_obj.name
            name, ext = os.path.splitext(filename_only)
            new_name = f'{name}_{dataset_prefix}{ext}' if dataset_prefix else filename_only

            try:
                shutil.copy2(img, os.path.join(img_dir, new_name))
            except Exception as e:
                print(f'复制失败 {img}: {e}')
                continue
            rel_path = f'./{img_dir_name}/{new_name}'
            rows.append([rel_path, char])

        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return len(rows) - 1

    train_count = write_csv(TRAIN_CSV, train_images, train_chars, train_img_dir)
    val_count = write_csv(VAL_CSV, val_images, val_chars, val_img_dir)
    test_count = write_csv(TEST_CSV, test_images, test_chars, test_img_dir)

    # 统计
    all_chars = set(train_chars + val_chars + test_chars)
    unique_chars = len(all_chars)

    print(f'\n数据集划分完成:')
    print(f'  训练集: {train_count} 条')
    print(f'  验证集: {val_count} 条')
    print(f'  测试集: {test_count} 条')
    print(f'  唯一字符数: {unique_chars}')
    print(f'\nCSV 文件已生成:')
    print(f'  - {TRAIN_CSV}')
    print(f'  - {VAL_CSV}')
    print(f'  - {TEST_CSV}')
    print(f'\n请更新 config.py 中的 NUM_CLASSES = {unique_chars}')


def main():
    parser = argparse.ArgumentParser(description='从统一标注数据生成训练/验证/测试集CSV')
    parser.add_argument('--datasets', nargs='+', help='指定数据集名称（为空则使用所有）')
    parser.add_argument('--min-samples', type=int, default=MIN_SAMPLES_PER_CLASS,
                        help=f'每个类别最少样本数 (默认: {MIN_SAMPLES_PER_CLASS})')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    args = parser.parse_args()

    generate_csv(datasets=args.datasets, min_samples=args.min_samples, train_ratio=args.train_ratio)


if __name__ == '__main__':
    main()