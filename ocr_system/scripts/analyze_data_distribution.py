import json
import os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLUSTERS_JSON = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "hog_clusters.json"
LABELS_JSON = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "labeling" / "labels.json"
CHARS_DIR = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars"

def load_json_file(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_data_distribution():
    """分析当前标注数据的分布情况"""
    labels_data = load_json_file(LABELS_JSON)
    clusters_data = load_json_file(CLUSTERS_JSON)
    
    # 统计每个汉字的图片数量
    char_image_counts = Counter()
    
    for cluster_id, cluster_info in labels_data.items():
        if cluster_info.get("status") != "labeled" or not cluster_info.get("char"):
            continue
        
        main_char = cluster_info["char"]
        char_labels = cluster_info.get("char_labels", {})
        
        # 获取聚类中的字符列表
        cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
        
        # 统计标注的图片数量
        for idx_str in char_labels.keys():
            idx = int(idx_str)
            if idx < len(cluster_chars):
                char_image_counts[main_char] += 1
    
    # 统计信息
    total_chars = len(char_image_counts)
    total_images = sum(char_image_counts.values())
    min_count = min(char_image_counts.values()) if char_image_counts else 0
    max_count = max(char_image_counts.values()) if char_image_counts else 0
    avg_count = total_images / total_chars if total_chars > 0 else 0
    
    print(f"{'='*60}")
    print("数据分布分析报告")
    print(f"{'='*60}")
    print(f"汉字数量：{total_chars} 个")
    print(f"图片总数：{total_images} 张")
    print(f"平均每个汉字：{avg_count:.1f} 张")
    print(f"最少：{min_count} 张")
    print(f"最多：{max_count} 张")
    print(f"极差：{max_count - min_count} 张")
    print(f"标准差：{calculate_std(char_image_counts.values()):.1f}")
    
    # 统计不同数量级的汉字
    bins = [0, 10, 20, 50, 100, 200, float('inf')]
    bin_labels = ["0-10", "11-20", "21-50", "51-100", "101-200", "200+"]
    bin_counts = [0] * len(bin_labels)
    
    for count in char_image_counts.values():
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low < count <= high:
                bin_counts[i] += 1
                break
    
    print("\n汉字数量分布：")
    for label, count in zip(bin_labels, bin_counts):
        print(f"  {label} 张：{count} 个汉字")
    
    # 显示前20个最多和最少的汉字
    print("\n前10个最多图片的汉字：")
    for char, count in char_image_counts.most_common(10):
        print(f"  {char}: {count} 张")
    
    print("\n后10个最少图片的汉字：")
    for char, count in char_image_counts.most_common()[-10:]:
        print(f"  {char}: {count} 张")
    
    # 保存分布数据
    dist_data = {
        "total_chars": total_chars,
        "total_images": total_images,
        "min_count": min_count,
        "max_count": max_count,
        "avg_count": avg_count,
        "char_distribution": dict(char_image_counts),
        "bin_distribution": dict(zip(bin_labels, bin_counts))
    }
    
    with open("data_distribution.json", "w", encoding="utf-8") as f:
        json.dump(dist_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n分布数据已保存到：data_distribution.json")
    
    return char_image_counts

def calculate_std(values):
    """计算标准差"""
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

if __name__ == "__main__":
    analyze_data_distribution()