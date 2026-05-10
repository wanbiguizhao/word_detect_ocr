#!/usr/bin/env python3
"""
收集数据集数据脚本

从标注系统中提取已标注的汉字图片，生成训练数据CSV文件。
支持指定数据集（如 pdf01）进行数据收集。
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.char_mapping import CharMappingManager
except ImportError:
    # 如果直接运行脚本，尝试另一种导入方式
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from ocr_system.configs.char_mapping import CharMappingManager


def get_labeled_images(dataset_id):
    """
    从数据集目录获取已标注的图片
    """
    # 数据集路径
    datahome_dir = Path(__file__).parent.parent.parent / "bussiness" / "datahome"
    dataset_dir = datahome_dir / dataset_id
    
    if not dataset_dir.exists():
        print(f"ERROR: 数据集目录不存在: {dataset_dir}")
        return []
    
    # 标注文件路径
    labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
    clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
    chars_dir = dataset_dir / "pdf_chars"
    
    if not labels_path.exists():
        print(f"ERROR: 标注文件不存在: {labels_path}")
        return []
    
    if not clusters_path.exists():
        print(f"ERROR: 聚类文件不存在: {clusters_path}")
        return []
    
    if not chars_dir.exists():
        print(f"ERROR: 字符图片目录不存在: {chars_dir}")
        return []
    
    # 加载标注数据
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    # 加载聚类数据
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    
    # 收集已标注的图片路径
    labeled_images = []
    
    for cluster_id, cluster_info in labels_data.items():
        if cluster_info.get("status") != "labeled":
            continue
        
        char_labels = cluster_info.get("char_labels", {})
        if not char_labels:
            continue
        
        # 获取该聚类的所有字符图片
        cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
        
        for char_idx, label_info in char_labels.items():
            char = label_info.get("char")
            if not char:
                continue
            
            # 找到对应的图片路径
            if int(char_idx) < len(cluster_chars):
                char_info = cluster_chars[int(char_idx)]
                char_id = char_info.get("char_id")
                if char_id:
                    img_path = chars_dir / f"{char_id}.png"
                    if img_path.exists():
                        labeled_images.append({
                            "char": char,
                            "char_id": char_id,
                            "img_path": str(img_path),
                            "cluster_id": cluster_id,
                            "char_index": char_idx
                        })
    
    print(f"SUCCESS: 共收集到 {len(labeled_images)} 张已标注图片")
    return labeled_images


def split_train_val(data, train_ratio=0.8):
    """
    按字符分层分割训练集和验证集
    """
    from collections import defaultdict
    import random
    
    # 按字符分组
    char_groups = defaultdict(list)
    for item in data:
        char_groups[item["char"]].append(item)
    
    train_data = []
    val_data = []
    
    for char, items in char_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
    
    # 打乱顺序
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"INFO: 训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张")
    return train_data, val_data


def save_to_csv(data, output_path):
    """
    保存数据到CSV文件
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "char"])
        
        for item in data:
            # 使用相对路径（相对于CSV文件所在目录）
            img_path = Path(item["img_path"])
            rel_path = img_path.as_posix()
            writer.writerow([rel_path, item["char"]])
    
    print(f"SAVE: 数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="收集数据集已标注数据")
    parser.add_argument("--dataset", type=str, default="pdf01", 
                        help="数据集ID（如 pdf01）")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例")
    args = parser.parse_args()
    
    print(f"START: 开始收集数据集: {args.dataset}")
    
    # 获取已标注图片
    labeled_images = get_labeled_images(args.dataset)
    
    if not labeled_images:
        print("ERROR: 没有找到已标注的数据")
        return
    
    # 更新字符映射
    print("INFO: 更新字符映射...")
    manager = CharMappingManager()
    
    for item in labeled_images:
        char = item["char"]
        if manager.get_label_id(char) is None:
            manager.add_custom_char(char)
    
    print(f"SUCCESS: 字符映射已更新")
    
    # 分割训练集和验证集
    train_data, val_data = split_train_val(labeled_images, args.train_ratio)
    
    # 保存到CSV
    output_dir = Path(args.output_dir)
    save_to_csv(train_data, str(output_dir / "train.csv"))
    save_to_csv(val_data, str(output_dir / "val.csv"))
    
    # 输出统计信息
    print("\nDONE: 数据收集完成！")
    print(f"  数据集: {args.dataset}")
    print(f"  总标注数: {len(labeled_images)}")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  覆盖汉字数: {len(set(item['char'] for item in labeled_images))}")


if __name__ == "__main__":
    main()