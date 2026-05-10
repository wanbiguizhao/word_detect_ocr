#!/usr/bin/env python3
"""
初始化 GB2312-80 标准字符集映射

GB2312-80 字符集包含 6763 个汉字，分为：
- 第一级（常用汉字）：3755 个
- 第二级（不常用汉字）：3008 个
"""

import os
import json
from pathlib import Path


def get_gb2312_chars():
    """
    获取 GB2312-80 所有汉字字符

    GB2312 编码范围：
    - 区号：0xB0-0xF7（高位）
    - 位号：0xA1-0xFE（低位）

    有效字符范围：
    - 第16区-55区：常用汉字（一级汉字）
    - 第56区-87区：不常用汉字（二级汉字）
    """
    chars = []

    for high in range(0xB0, 0xF8):
        for low in range(0xA1, 0xFF):
            try:
                gb2312_bytes = bytes([high, low])
                char = gb2312_bytes.decode('gb2312')
                chars.append(char)
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue

    return chars


def init_gb2312_mapping(config_dir):
    """
    初始化 GB2312 字符映射配置文件
    """
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    print("Getting GB2312-80 character set...")
    chars = get_gb2312_chars()
    print(f"Got {len(chars)} characters")

    char_to_label = {
        "version": "1.0",
        "charset": "GB2312-80",
        "charset_size": len(chars),
        "reserved_ids": 100,
        "custom_start_id": len(chars) + 100,
        "created_at": "2026-05-10T16:00:00",
        "last_updated": "2026-05-10T16:00:00",
        "characters": {},
        "custom_characters": {},
        "next_custom_id": len(chars) + 100
    }

    for idx, char in enumerate(chars):
        char_to_label["characters"][char] = {
            "label_id": idx,
            "added_in_version": "1.0",
            "status": "standard"
        }

    label_to_char = {
        "version": "1.0",
        "charset": "GB2312-80",
        "labels": {}
    }

    for idx, char in enumerate(chars):
        label_to_char["labels"][str(idx)] = {
            "char": char,
            "status": "standard"
        }

    char_properties = {
        "version": "1.0",
        "characters": {}
    }

    for char in chars:
        unicode_val = f"U+{ord(char):04X}"
        char_properties["characters"][char] = {
            "pinyin": "",
            "radical": "",
            "stroke_count": 0,
            "unicode": unicode_val
        }

    dataset_mapping = {
        "version": "1.0",
        "datasets": {}
    }

    char_to_label_path = config_path / "char_to_label.json"
    label_to_char_path = config_path / "label_to_char.json"
    char_properties_path = config_path / "char_properties.json"
    dataset_mapping_path = config_path / "dataset_mapping.json"

    print(f"Saving char_to_label.json...")
    with open(char_to_label_path, "w", encoding="utf-8") as f:
        json.dump(char_to_label, f, ensure_ascii=False, indent=2)

    print(f"Saving label_to_char.json...")
    with open(label_to_char_path, "w", encoding="utf-8") as f:
        json.dump(label_to_char, f, ensure_ascii=False, indent=2)

    print(f"Saving char_properties.json...")
    with open(char_properties_path, "w", encoding="utf-8") as f:
        json.dump(char_properties, f, ensure_ascii=False, indent=2)

    print(f"Saving dataset_mapping.json...")
    with open(dataset_mapping_path, "w", encoding="utf-8") as f:
        json.dump(dataset_mapping, f, ensure_ascii=False, indent=2)

    print(f"\nInitialization complete!")
    print(f"  Total chars: {len(chars)}")
    print(f"  Standard char ID range: 0-{len(chars)-1}")
    print(f"  Custom char start ID: {len(chars)+100}")

    return len(chars)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    config_dir = script_dir

    count = init_gb2312_mapping(config_dir)