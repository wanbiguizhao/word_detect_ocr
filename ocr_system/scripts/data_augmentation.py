#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强脚本

对训练数据进行数据增强，生成更多变体图片。

使用示例：
    python data_augmentation.py

    # 指定输入输出目录
    python data_augmentation.py --input-dir data/ocr_train_data --output-dir data/ocr_train_data_augmented
"""

import os
import sys
import cv2
import numpy as np
import json
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def augment_image(image):
    """对单张图片进行数据增强"""
    augmented = []

    # 原始图片
    augmented.append(('original', image))

    # 旋转（小角度）
    rows, cols = image.shape[:2]
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        augmented.append((f'rotate_{angle}', rotated))

    # 平移
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        augmented.append((f'shift_{dx}_{dy}', shifted))

    # 缩放
    for scale in [0.95, 1.05]:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if scale < 1:
            padded = np.ones_like(image) * 255
            start_x = (cols - resized.shape[1]) // 2
            start_y = (rows - resized.shape[0]) // 2
            padded[start_y:start_y+resized.shape[0], start_x:start_x+resized.shape[1]] = resized
            augmented.append((f'scale_{scale}', padded))
        else:
            cropped = resized[(resized.shape[0]-rows)//2:(resized.shape[0]-rows)//2+rows,
                            (resized.shape[1]-cols)//2:(resized.shape[1]-cols)//2+cols]
            augmented.append((f'scale_{scale}', cropped))

    # 添加轻微噪声
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(('noise', noisy))

    return augmented


def run_data_augmentation(input_dir=None, output_dir=None):
    """
    运行数据增强

    Args:
        input_dir: 输入目录，默认为 PROJECT_ROOT/data/ocr_train_data
        output_dir: 输出目录，默认为 PROJECT_ROOT/data/ocr_train_data_augmented
    """
    if input_dir is None:
        input_dir = PROJECT_ROOT / 'data' / 'ocr_train_data'
    else:
        input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'ocr_train_data_augmented'
    else:
        output_dir = Path(output_dir)

    train_input_dir = input_dir / 'train'
    val_input_dir = input_dir / 'val'

    if not train_input_dir.exists():
        print(f'错误: 训练数据目录不存在: {train_input_dir}')
        return

    (output_dir / 'train').mkdir(parents=True, exist_ok=True)

    total_original = 0
    total_augmented = 0
    char_stats = {}

    for char_dir in train_input_dir.iterdir():
        if not char_dir.is_dir():
            continue

        char = char_dir.name
        output_char_dir = output_dir / 'train' / char
        output_char_dir.mkdir(exist_ok=True)

        char_count = 0
        augmented_count = 0

        for img_path in char_dir.glob('*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # 如果是RGBA格式，转换为RGB
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # 数据增强
            augmented = augment_image(img)

            for suffix, aug_img in augmented:
                base_name = img_path.stem
                output_path = output_char_dir / f'{base_name}_{suffix}.png'
                cv2.imwrite(str(output_path), aug_img)
                augmented_count += 1

            char_count += 1
            total_original += 1

        char_stats[char] = {'original': char_count, 'augmented': augmented_count}
        total_augmented += augmented_count
        print(f'处理 {char}: {char_count} -> {augmented_count} 张')

    # 复制验证集
    val_output_dir = output_dir / 'val'
    if val_input_dir.exists():
        if val_output_dir.exists():
            shutil.rmtree(val_output_dir)
        shutil.copytree(val_input_dir, val_output_dir)

    # 复制标签文件
    labels_file = input_dir / 'labels.txt'
    if labels_file.exists():
        shutil.copy(labels_file, output_dir / 'labels.txt')

    # 生成统计报告
    with open(output_dir / 'aug_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_original': total_original,
            'total_augmented': total_augmented,
            'augmentation_factor': total_augmented / total_original if total_original > 0 else 0,
            'char_stats': char_stats
        }, f, ensure_ascii=False, indent=2)

    print(f'\n{"="*60}')
    print('数据增强完成！')
    print(f'{"="*60}')
    print(f'原始图片: {total_original} 张')
    print(f'增强后: {total_augmented} 张')
    print(f'增强倍数: {total_augmented / total_original:.1f}x')
    print(f'输出目录: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='数据增强')
    parser.add_argument('--input-dir', type=str, default=None, help='输入目录')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    run_data_augmentation(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()