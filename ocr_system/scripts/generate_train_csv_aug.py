#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从统一标注数据生成训练/验证/测试集CSV（带数据增强）

使用 unified_labels.json 中的数据构建训练数据集，并对小样本类别进行数据增强。

使用示例：
    python generate_train_csv_aug.py

    # 设置最小样本数（增强后）
    python generate_train_csv_aug.py --min-samples 20
"""

import os
import sys
import csv
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV
from unified_labels import UnifiedLabelsLoader

MIN_SAMPLES_PER_CLASS = 20


def augment_image(img_path, output_path, aug_type):
    """对图片进行数据增强"""
    img = Image.open(img_path).convert('L')

    if aug_type == 'rotate':
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=255)
    elif aug_type == 'shift':
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)
    elif aug_type == 'noise':
        arr = np.array(img)
        noise = np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif aug_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    elif aug_type == 'blur':
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    img.save(output_path)


def generate_csv_with_augmentation(min_samples=MIN_SAMPLES_PER_CLASS, train_ratio=0.8):
    """
    从标注数据生成训练/验证/测试集CSV（带数据增强）

    Args:
        min_samples: 每个类别最少样本数（增强后）
        train_ratio: 训练集比例
    """
    # 加载数据
    loader = UnifiedLabelsLoader()
    print(f'加载标注数据: {loader.total_labeled} 条')

    annotations = loader.annotations

    if not annotations:
        print('错误: 没有找到标注数据！')
        return

    # 统计每个字符的数量
    char_counter = Counter([ann['char'] for ann in annotations])
    print(f'唯一字符数: {len(char_counter)}')

    # 收集有效数据
    data = []
    for ann in annotations:
        full_path = loader.get_full_image_path(ann)
        if os.path.exists(full_path):
            data.append((full_path, ann['char']))

    print(f'有效数据: {len(data)} 条')

    # 创建图片目录
    train_img_dir = os.path.join(DATA_ROOT, 'train_images')
    val_img_dir = os.path.join(DATA_ROOT, 'val_images')
    test_img_dir = os.path.join(DATA_ROOT, 'test_images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 对小样本类别进行数据增强
    aug_types = ['rotate', 'shift', 'noise', 'contrast', 'blur']
    augmented_data = list(data)

    print(f'\n对小样本类别进行数据增强...')
    for char, count in char_counter.items():
        if count < min_samples:
            need_aug = min_samples - count
            char_samples = [item for item in data if item[1] == char]

            aug_count = 0
            while aug_count < need_aug:
                for img_path, c in char_samples:
                    if aug_count >= need_aug:
                        break
                    aug_type = random.choice(aug_types)
                    base_name = os.path.basename(img_path)
                    name, ext = os.path.splitext(base_name)
                    aug_name = f'{name}_aug_{aug_count}{ext}'
                    aug_path = os.path.join(train_img_dir, aug_name)

                    try:
                        augment_image(img_path, aug_path, aug_type)
                        augmented_data.append((aug_path, c))
                        aug_count += 1
                    except Exception as e:
                        print(f'增强失败 {img_path}: {e}')

            print(f'  {char}: {count} -> {count + aug_count}')

    print(f'\n增强后总数据量: {len(augmented_data)}')

    # 统计增强后的分布
    char_counter_aug = Counter([item[1] for item in augmented_data])
    print(f'增强后类别数: {len(char_counter_aug)}')
    print(f'增强后最少样本数: {min(char_counter_aug.values())}')

    # 划分数据集
    images = [item[0] for item in augmented_data]
    chars = [item[1] for item in augmented_data]

    train_images, temp_images, train_chars, temp_chars = train_test_split(
        images, chars, test_size=1-train_ratio, random_state=42, stratify=chars
    )

    val_images, test_images, val_chars, test_chars = train_test_split(
        temp_images, temp_chars, test_size=0.5, random_state=42
    )

    # 写入CSV
    def write_csv(filename, img_list, char_list, img_dir):
        img_dir_name = os.path.basename(img_dir)
        rows = [['image_path', 'char']]

        for img, char in zip(img_list, char_list):
            img_full_path = Path(img) if not Path(img).is_absolute() else Path(img)
            dataset_prefix = img_full_path.parts[-3] if len(img_full_path.parts) >= 3 else ''
            filename_only = img_full_path.name
            name, ext = os.path.splitext(filename_only)

            if '_aug_' in filename_only:
                aug_name = f'{name}_aug_{dataset_prefix}{ext}' if dataset_prefix else f'{name}_aug{ext}'
                aug_path = os.path.join(img_dir, aug_name)
                if 'train' in img_dir_name:
                    try:
                        shutil.copy2(img, aug_path)
                    except:
                        pass
                    rel_path = f'./{img_dir_name}/{aug_name}'
                    rows.append([rel_path, char])
            else:
                new_name = f'{name}_{dataset_prefix}{ext}' if dataset_prefix else filename_only
                new_path = os.path.join(img_dir, new_name)
                try:
                    shutil.copy2(img, new_path)
                except Exception as e:
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

    print(f'\n数据集划分完成:')
    print(f'  训练集: {train_count} 条')
    print(f'  验证集: {val_count} 条')
    print(f'  测试集: {test_count} 条')
    print(f'  唯一字符数: {len(set(train_chars + val_chars + test_chars))}')
    print(f'\nCSV 文件已生成:')
    print(f'  - {TRAIN_CSV}')
    print(f'  - {VAL_CSV}')
    print(f'  - {TEST_CSV}')
    print(f'\n请更新 config.py 中的 NUM_CLASSES = {len(char_counter_aug)}')


def main():
    parser = argparse.ArgumentParser(description='从统一标注数据生成训练/验证/测试集CSV（带数据增强）')
    parser.add_argument('--min-samples', type=int, default=MIN_SAMPLES_PER_CLASS,
                        help=f'每个类别最少样本数 (默认: {MIN_SAMPLES_PER_CLASS})')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    args = parser.parse_args()

    generate_csv_with_augmentation(min_samples=args.min_samples, train_ratio=args.train_ratio)


if __name__ == '__main__':
    main()