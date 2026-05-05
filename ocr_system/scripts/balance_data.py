import json
import os
import shutil
from pathlib import Path
from collections import Counter
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLUSTERS_JSON = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "hog_clusters.json"
LABELS_JSON = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "labeling" / "labels.json"
CHARS_DIR = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars"
OUTPUT_DIR = PROJECT_ROOT / "ocr_system" / "data" / "ocr_train_data_balanced"

def load_json_file(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def balance_data(target_min=30, target_max=150):
    """
    数据均衡处理
    
    参数：
    - target_min: 每个汉字最少图片数（默认30）
    - target_max: 每个汉字最多图片数（默认150）
    """
    labels_data = load_json_file(LABELS_JSON)
    clusters_data = load_json_file(CLUSTERS_JSON)
    
    # 收集所有标注图片
    char_images = {}  # char -> list of (char_id, cluster_id, idx)
    
    for cluster_id, cluster_info in labels_data.items():
        if cluster_info.get("status") != "labeled" or not cluster_info.get("char"):
            continue
        
        main_char = cluster_info["char"]
        char_labels = cluster_info.get("char_labels", {})
        cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
        
        if main_char not in char_images:
            char_images[main_char] = []
        
        for idx_str in char_labels.keys():
            idx = int(idx_str)
            if idx < len(cluster_chars):
                char_id = cluster_chars[idx].get("char_id", "")
                if char_id:
                    char_images[main_char].append((char_id, cluster_id, idx))
    
    # 创建输出目录
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "before": {},
        "after": {},
        "sampling_info": {}
    }
    
    for char, images in char_images.items():
        original_count = len(images)
        stats["before"][char] = original_count
        
        # 打乱顺序
        random.shuffle(images)
        
        # 欠采样：超过max的截断
        if len(images) > target_max:
            images = images[:target_max]
            sampling_type = "undersample"
        # 过采样：不足min的复制
        elif len(images) < target_min:
            num_needed = target_min - len(images)
            sampled = images.copy()
            while len(sampled) < target_min:
                sampled.extend(random.sample(images, min(num_needed, len(images))))
                num_needed = target_min - len(sampled)
            images = sampled
            sampling_type = "oversample"
        else:
            sampling_type = "none"
        
        stats["after"][char] = len(images)
        stats["sampling_info"][char] = {
            "original": original_count,
            "final": len(images),
            "type": sampling_type
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
        for i, (char_id, _, _) in enumerate(train_images):
            src_path = CHARS_DIR / f"{char_id}.png"
            if src_path.exists():
                # 添加序号避免重复
                dst_path = train_char_dir / f"{char_id}_{i}.png"
                shutil.copy(src_path, dst_path)
        
        # 复制验证集
        for i, (char_id, _, _) in enumerate(val_images):
            src_path = CHARS_DIR / f"{char_id}.png"
            if src_path.exists():
                dst_path = val_char_dir / f"{char_id}_{i}.png"
                shutil.copy(src_path, dst_path)
    
    # 生成标签文件
    labels = sorted(char_images.keys())
    with open(OUTPUT_DIR / "labels.txt", "w", encoding="utf-8") as f:
        for i, char in enumerate(labels):
            f.write(f"{i}\t{char}\n")
    
    # 生成统计报告
    before_total = sum(stats["before"].values())
    after_total = sum(stats["after"].values())
    undersampled_count = sum(1 for info in stats["sampling_info"].values() if info["type"] == "undersample")
    oversampled_count = sum(1 for info in stats["sampling_info"].values() if info["type"] == "oversample")
    
    with open(OUTPUT_DIR / "balance_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {
                "target_min": target_min,
                "target_max": target_max
            },
            "summary": {
                "total_chars": len(labels),
                "before_total_images": before_total,
                "after_total_images": after_total,
                "undersampled_chars": undersampled_count,
                "oversampled_chars": oversampled_count,
                "unchanged_chars": len(labels) - undersampled_count - oversampled_count
            },
            "details": stats
        }, f, ensure_ascii=False, indent=2)
    
    # 输出报告
    print(f"{'='*60}")
    print("数据均衡处理完成！")
    print(f"{'='*60}")
    print(f"目标范围：{target_min} - {target_max} 张/汉字")
    print(f"汉字总数：{len(labels)} 个")
    print(f"原始图片数：{before_total} 张")
    print(f"均衡后图片数：{after_total} 张")
    print(f"\n采样统计：")
    print(f"  欠采样（过多→截断）：{undersampled_count} 个汉字")
    print(f"  过采样（不足→复制）：{oversampled_count} 个汉字")
    print(f"  保持不变：{len(labels) - undersampled_count - oversampled_count} 个汉字")
    
    # 显示处理前后对比（前10个最多的）
    print("\n处理前后对比（前10个）：")
    for char, count in sorted(stats["before"].items(), key=lambda x: -x[1])[:10]:
        after = stats["after"][char]
        change = "↓" if after < count else "↑" if after > count else "="
        print(f"  {char}: {count} {change} {after} ({stats['sampling_info'][char]['type']})")
    
    print(f"\n输出目录：{OUTPUT_DIR}")
    
    return stats

if __name__ == "__main__":
    # 默认设置：最少30张，最多150张
    balance_data(target_min=30, target_max=150)