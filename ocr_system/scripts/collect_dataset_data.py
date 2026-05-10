#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新字符映射脚本

从 unified_labels.json 提取所有汉字字符，更新到 ocr_system\configs\char_mapping 目录。

使用示例：
    python collect_dataset_data.py
    python collect_dataset_data.py ../../bussiness/unified_labels.json
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_labels import UnifiedLabelsLoader

try:
    from configs.char_mapping import CharMappingManager
except ImportError:
    from ocr_system.configs.char_mapping import CharMappingManager


def main():
    parser = argparse.ArgumentParser(description='更新字符映射')
    parser.add_argument('labels_path', nargs='?', default=None,
                        help='unified_labels.json 文件路径')
    args = parser.parse_args()

    print('START: 开始更新字符映射')

    loader = UnifiedLabelsLoader(labels_path=args.labels_path)
    print(f'加载标注数据: {loader.total_labeled} 条')

    annotations = loader.annotations
    if not annotations:
        print('ERROR: 没有找到已标注的数据')
        return

    chars_in_data = set(ann['char'] for ann in annotations)
    print(f'发现汉字数: {len(chars_in_data)} 个')

    print('INFO: 更新字符映射...')
    manager = CharMappingManager()

    added_count = 0
    for char in chars_in_data:
        if manager.get_label_id(char) is None:
            manager.add_custom_char(char)
            added_count += 1

    print(f'SUCCESS: 字符映射已更新，新增 {added_count} 个字符')
    print(f'  总汉字数: {len(chars_in_data)}')


if __name__ == '__main__':
    main()