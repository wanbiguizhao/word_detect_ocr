#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据均衡处理

使用 unified_labels.json 中的数据，对字符进行欠采样/过采样，
使得每个字符的样本数量在指定范围内。

使用示例：
    python balance_data.py

    # 设置目标范围
    python balance_data.py --min 30 --max 150
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from unified_labels import UnifiedLabelsLoader


def balance_data(target_min=30, target_max=150, output_dir=None):
    """
    数据均衡处理

    Args:
        target_min: 每个汉字最少图片数
        target_max: 每个汉字最多图片数
        output_dir: 输出目录，默认为 PROJECT_ROOT/data/ocr_train_data_balanced
    """
    # 加载数据
    loader = UnifiedLabelsLoader()
    print(f'加载标注数据: {loader.total_labeled} 条')

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'ocr_train_data_balanced'
    else:
        output_dir = Path(output_dir)

    # 收集所有标注数据
    char_images = {}
    for ann in loader.annotations:
        char = ann['char']
        if char not in char_images:
            char_images[char] = []
        full_path = loader.get_full_image_path(ann)
        if os.path.exists(full_path):
            char_images[char].append({
                'char': char,
                'image_path': full_path,
                'annotation': ann
            })

    print(f'有效汉字数: {len(char_images)}')

    # 创建输出目录
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'before': {},
        'after': {},
        'sampling_info': {}
    }

    for char, images in char_images.items():
        original_count = len(images)
        stats['before'][char] = original_count

        # 打乱顺序
        random.shuffle(images)

        sampling_type = 'none'

        # 欠采样：超过max的截断
        if len(images) > target_max:
            images = images[:target_max]
            sampling_type = 'undersample'

        # 过采样：不足min的复制
        elif len(images) < target_min:
            num_needed = target_min - len(images)
            sampled = images.copy()
            while len(sampled) < target_min:
                sampled.extend(random.sample(images, min(num_needed, len(images))))
                num_needed = target_min - len(sampled)
            images = sampled
            sampling_type = 'oversample'

        stats['after'][char] = len(images)
        stats['sampling_info'][char] = {
            'original': original_count,
            'final': len(images),
            'type': sampling_type
        }

        # 按80/20分割
        train_count = int(len(images) * 0.8)
        train_images = images[:train_count]
        val_images = images[train_count:]

        # 创建字符目录
        train_char_dir = train_dir / char
        val_char_dir = val_dir / char
        train_char_dir.mkdir(exist_ok=True)
        val_char_dir.mkdir(exist_ok=True)

        # 复制训练集
        for i, img_info in enumerate(train_images):
            src_path = img_info['image_path']
            if os.path.exists(src_path):
                src_path_obj = Path(src_path)
                dataset_prefix = src_path_obj.parts[-3] if len(src_path_obj.parts) >= 3 else ''
                new_name = f'{i}_{dataset_prefix}.png' if dataset_prefix else f'{i}.png'
                dst_path = train_char_dir / new_name
                shutil.copy(src_path, dst_path)

        # 复制验证集
        for i, img_info in enumerate(val_images):
            src_path = img_info['image_path']
            if os.path.exists(src_path):
                src_path_obj = Path(src_path)
                dataset_prefix = src_path_obj.parts[-3] if len(src_path_obj.parts) >= 3 else ''
                new_name = f'{i}_{dataset_prefix}.png' if dataset_prefix else f'{i}.png'
                dst_path = val_char_dir / new_name
                shutil.copy(src_path, dst_path)

    # 生成标签文件
    labels = sorted(char_images.keys())
    with open(output_dir / 'labels.txt', 'w', encoding='utf-8') as f:
        for i, char in enumerate(labels):
            f.write(f'{i}\t{char}\n')

    # 生成统计报告
    before_total = sum(stats['before'].values())
    after_total = sum(stats['after'].values())
    undersampled_count = sum(1 for info in stats['sampling_info'].values() if info['type'] == 'undersample')
    oversampled_count = sum(1 for info in stats['sampling_info'].values() if info['type'] == 'oversample')

    with open(output_dir / 'balance_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'parameters': {
                'target_min': target_min,
                'target_max': target_max
            },
            'summary': {
                'total_chars': len(labels),
                'before_total_images': before_total,
                'after_total_images': after_total,
                'undersampled_chars': undersampled_count,
                'oversampled_chars': oversampled_count,
                'unchanged_chars': len(labels) - undersampled_count - oversampled_count
            },
            'details': stats
        }, f, ensure_ascii=False, indent=2)

    # 输出报告
    print(f'{'='*60}')
    print('数据均衡处理完成！')
    print(f'{'='*60}')
    print(f'目标范围: {target_min} - {target_max} 张/汉字')
    print(f'汉字总数: {len(labels)} 个')
    print(f'原始图片数: {before_total} 张')
    print(f'均衡后图片数: {after_total} 张')
    print(f'\n采样统计:')
    print(f'  欠采样（过多->截断）: {undersampled_count} 个汉字')
    print(f'  过采样（不足->复制）: {oversampled_count} 个汉字')
    print(f'  保持不变: {len(labels) - undersampled_count - oversampled_count} 个汉字')

    # 显示处理前后对比（前10个最多的）
    print('\n处理前后对比（前10个）:')
    for char, count in sorted(stats['before'].items(), key=lambda x: -x[1])[:10]:
        after = stats['after'][char]
        change = '↓' if after < count else '↑' if after > count else '='
        print(f'  {char}: {count} {change} {after} ({stats["sampling_info"][char]["type"]})')

    print(f'\n输出目录: {output_dir}')

    return stats


def main():
    parser = argparse.ArgumentParser(description='数据均衡处理')
    parser.add_argument('--min', type=int, default=30, help='每个汉字最少图片数 (默认: 30)')
    parser.add_argument('--max', type=int, default=150, help='每个汉字最多图片数 (默认: 150)')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    balance_data(target_min=args.min, target_max=args.max, output_dir=args.output)


if __name__ == '__main__':
    main()