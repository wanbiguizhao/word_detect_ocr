"""
基于HOG特征和余弦相似度的预计算脚本（多进程版本）
运行方式：python run_precompute_hog.py pdf5823 [进程数]
例如：python run_precompute_hog.py pdf5823 4
"""

import json
import sys
import time
import math
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def extract_hog_feature(image_path):
    """从图片提取HOG特征"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img = cv2.resize(img, (64, 64))
        
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(img)
        return features.flatten().tolist()
    
    except Exception as e:
        return None

def compute_avg_feature(samples):
    """计算平均特征向量"""
    if not samples:
        return None
    
    features = [s['feature'] for s in samples if 'feature' in s and s['feature']]
    if not features:
        return None
    
    n = len(features)
    dim = len(features[0])
    avg = [0.0] * dim
    
    for feat in features:
        for i in range(dim):
            avg[i] += feat[i] / n
    
    return avg

def process_char(unlabeled_char, char_features_list, feature_dim):
    """处理单个字符的匹配（用于多进程）"""
    image_path = unlabeled_char["image_path"]
    char_feature = extract_hog_feature(image_path)
    
    if char_feature is None:
        return None
    
    if len(char_feature) != feature_dim:
        return None
    
    best_char = None
    best_sim = 0.0
    
    for char, avg_feature in char_features_list:
        sim = cosine_similarity(char_feature, avg_feature)
        if sim > best_sim:
            best_sim = sim
            best_char = char
    
    if best_char:
        return {
            "char": best_char,
            "match": {
                "char_id": unlabeled_char["char_id"],
                "image_path": unlabeled_char["image_path"],
                "image_filename": unlabeled_char["image_filename"],
                "cluster_id": unlabeled_char["cluster_id"],
                "similarity": round(best_sim, 4)
            }
        }
    
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: python run_precompute_hog.py <目标数据集ID> [进程数]")
        print("例如: python run_precompute_hog.py pdf5823 4")
        sys.exit(1)
    
    target_dataset_id = sys.argv[1]
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print("[INFO] 开始基于HOG特征的预计算（多进程版本）")
    print(f"       目标数据集: {target_dataset_id}")
    print(f"       进程数: {num_processes or '自动检测'}")
    
    # 加载特征库
    feature_db_path = Path(__file__).parent / "bussiness" / "migration" / "data" / "char_feature_db.json"
    
    if not feature_db_path.exists():
        print("[ERROR] 特征库不存在")
        return
    
    with open(feature_db_path, "r", encoding="utf-8") as f:
        feature_db = json.load(f)
    
    # 计算每个汉字的平均特征
    char_features = {}
    for char, data in feature_db['characters'].items():
        avg_feature = compute_avg_feature(data.get('samples', []))
        if avg_feature:
            char_features[char] = avg_feature
    
    print(f"       汉字特征库已加载，共 {len(char_features)} 个汉字")
    
    # 加载目标数据集聚类
    datahome_dir = Path(__file__).parent / "bussiness" / "datahome"
    target_dir = datahome_dir / target_dataset_id
    clusters_path = target_dir / "clusters" / "hog_clusters.json"
    labels_path = target_dir / "clusters" / "labeling" / "labels.json"
    
    if not clusters_path.exists():
        print("[ERROR] 聚类文件不存在")
        return
    
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    
    # 获取已标注聚类ID
    labeled_cluster_ids = set()
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        labeled_cluster_ids = {k for k, v in labels.items() if v.get("status") == "labeled"}
    
    # 收集未标注字符
    unlabeled_chars = []
    clusters = clusters_data.get("clusters", {})
    
    for cluster_id, chars in clusters.items():
        if cluster_id in labeled_cluster_ids:
            continue
        
        for idx, char_info in enumerate(chars):
            image_path = char_info.get("image_path", "")
            if image_path:
                image_filename = Path(image_path).name
                unlabeled_chars.append({
                    "char_id": f"{cluster_id}_{idx}",
                    "image_path": image_path,
                    "image_filename": image_filename,
                    "cluster_id": cluster_id,
                    "index": idx
                })
    
    total_chars = len(unlabeled_chars)
    print(f"       目标数据集未标注字符数: {total_chars}")
    
    # 检查特征维度
    if char_features:
        first_char = list(char_features.keys())[0]
        print(f"       特征维度: {len(char_features[first_char])}")
    
    # 准备数据
    char_features_list = [(char, feat) for char, feat in char_features.items()]
    feature_dim = len(char_features_list[0][1]) if char_features_list else 0
    
    # 多进程处理
    print(f"\n[INFO] 开始计算相似度匹配...")
    start_time = time.time()
    
    # 使用进程池
    with Pool(processes=num_processes) as pool:
        # 创建偏函数
        process_func = partial(process_char, 
                             char_features_list=char_features_list, 
                             feature_dim=feature_dim)
        
        # 使用imap进行迭代，支持进度显示
        results = []
        processed = 0
        last_print_time = 0
        
        for result in pool.imap(process_func, unlabeled_chars):
            results.append(result)
            processed += 1
            
            # 每秒最多打印一次进度
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                progress = (processed / total_chars) * 100
                elapsed = current_time - start_time
                eta = elapsed / (processed / total_chars) - elapsed if processed > 0 else 0
                
                # 清除当前行并打印新进度
                sys.stdout.write(f"\r       进度: [{processed}/{total_chars}] {progress:.1f}% | "
                               f"已用时: {elapsed:.1f}s | ETA: {eta:.1f}s")
                sys.stdout.flush()
                last_print_time = current_time
    
    # 打印换行
    print()
    
    # 汇总结果
    all_matches = {}
    for result in results:
        if result:
            char = result["char"]
            match = result["match"]
            if char not in all_matches:
                all_matches[char] = []
            all_matches[char].append(match)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"       处理完成，耗时: {elapsed:.2f} 秒")
    
    # 对每个汉字的匹配结果按相似度排序
    for char in all_matches:
        all_matches[char].sort(key=lambda x: x["similarity"], reverse=True)
    
    # 保存结果
    result = {
        "version": "2.0",
        "dataset_id": target_dataset_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matches": all_matches
    }
    
    output_path = Path(__file__).parent / "bussiness" / "migration" / "data" / "image_matches.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SUCCESS] 预计算完成！")
    print(f"       匹配结果已保存到: {output_path}")
    print(f"       匹配汉字数: {len(all_matches)}")
    
    total_high_confidence = sum(
        1 for char_matches in all_matches.values()
        for match in char_matches
        if match["similarity"] >= 0.8
    )
    print(f"       高置信度匹配(≥0.8): {total_high_confidence}")

if __name__ == "__main__":
    main()