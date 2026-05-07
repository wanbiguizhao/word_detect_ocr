"""
汉字匹配算法
用于跨数据集的汉字匹配和相似度计算
以图片为单位进行匹配
优化版本：实现双向匹配和置信度过滤
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.optimize import linear_sum_assignment

class CharMatcher:
    """汉字匹配器 - 以图片为单位"""
    
    @staticmethod
    def extract_hog_features(image_path: Path) -> np.ndarray:
        """
        从图片提取HOG特征（优化版本：提高字形相似汉字的区分度）
        
        Args:
            image_path: 图片路径
        
        Returns:
            HOG特征向量
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(576)
            
            # 统一尺寸（稍微增大以保留更多细节）
            img = cv2.resize(img, (64, 64))
            
            # 优化后的HOG参数 - 提高字形相似汉字的区分度
            win_size = (64, 64)
            block_size = (16, 16)      # 保持块大小
            block_stride = (8, 8)       # 块步长
            cell_size = (8, 8)          # 细胞大小
            nbins = 12                  # 增加方向数量，从9增加到12
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(img)
            
            return features.flatten()
        except Exception:
            return np.zeros(576)
    
    @staticmethod
    def extract_hog_features_enhanced(image_path: Path) -> np.ndarray:
        """
        增强版HOG特征提取 - 使用多尺度和多方向组合
        
        Args:
            image_path: 图片路径
        
        Returns:
            增强的HOG特征向量
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(576 * 2)
            
            # 统一尺寸
            img = cv2.resize(img, (64, 64))
            
            # 多方向特征组合
            features_list = []
            
            # 标准方向 (12个方向)
            hog1 = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 12)
            features1 = hog1.compute(img)
            features_list.append(features1.flatten())
            
            # 细粒度方向 (18个方向) - 捕捉更细微的角度变化
            hog2 = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 18)
            features2 = hog2.compute(img)
            features_list.append(features2.flatten())
            
            # 组合特征
            combined_features = np.concatenate(features_list)
            
            return combined_features
        except Exception:
            return np.zeros(576 * 2)
    
    @staticmethod
    def calculate_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
        
        Returns:
            相似度 (0-1)
        """
        try:
            if len(feature1) == 0 or len(feature2) == 0:
                return 0.0
            
            feature1 = np.array(feature1)
            feature2 = np.array(feature2)
            
            if feature1.ndim > 1:
                feature1 = feature1.flatten()
            if feature2.ndim > 1:
                feature2 = feature2.flatten()
            
            norm1 = np.linalg.norm(feature1)
            norm2 = np.linalg.norm(feature2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(feature1, feature2) / (norm1 * norm2))
        except Exception:
            return 0.0
    
    @staticmethod
    def build_char_feature_db(datahome_dir: Path, dataset_ids: List[str]) -> Dict:
        """
        构建汉字特征库（以图片为单位）
        
        Args:
            datahome_dir: 数据集根目录
            dataset_ids: 数据集ID列表
        
        Returns:
            汉字特征库 {char: {"samples": [...], "avg_feature": ...}}
        """
        feature_db = {
            "version": "2.0",
            "datasets": dataset_ids,
            "characters": {}
        }
        
        for dataset_id in dataset_ids:
            dataset_dir = datahome_dir / dataset_id
            labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
            clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
            
            if not labels_path.exists() or not clusters_path.exists():
                continue
            
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            
            with open(clusters_path, "r", encoding="utf-8") as f:
                clusters = json.load(f)
            
            for cluster_id, cluster_data in labels.items():
                if cluster_data.get("status") != "labeled":
                    continue
                
                chars_map = cluster_data.get("chars", {})
                if not chars_map:
                    continue
                
                cluster_chars = clusters.get("clusters", {}).get(cluster_id, [])
                
                for char, count in chars_map.items():
                    if char not in feature_db["characters"]:
                        feature_db["characters"][char] = {
                            "total_samples": 0,
                            "samples": [],
                            "avg_feature": None
                        }
                    
                    features = []
                    for char_info in cluster_chars[:count]:
                        image_path = Path(char_info.get("image_path", ""))
                        if image_path.exists():
                            feature = CharMatcher.extract_hog_features(image_path)
                            sample_entry = {
                                "dataset_id": dataset_id,
                                "cluster_id": cluster_id,
                                "char_id": char_info.get("char_id", ""),
                                "image_path": str(image_path),
                                "feature": feature.tolist()
                            }
                            feature_db["characters"][char]["samples"].append(sample_entry)
                            features.append(feature)
                    
                    feature_db["characters"][char]["total_samples"] += len(features)
                    
                    if features:
                        avg_feature = np.mean(features, axis=0)
                        feature_db["characters"][char]["avg_feature"] = avg_feature.tolist()
        
        return feature_db
    
    @staticmethod
    def find_matching_images(
        feature_db: Dict,
        target_chars: List[Dict],
        char: str,
        top_n: int = 20,
        confidence_threshold: float = 0.0
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        在目标数据集中查找与指定汉字匹配的图片（支持置信度过滤）
        
        Args:
            feature_db: 汉字特征库
            target_chars: 目标数据集的字符列表
            char: 要匹配的汉字
            top_n: 返回前N个匹配结果
            confidence_threshold: 置信度阈值，低于此值的标记为待确认（默认0表示不过滤）
        
        Returns:
            Tuple: (确认匹配结果, 待确认匹配结果)
        """
        confirmed_results = []
        pending_results = []
        
        if char not in feature_db["characters"]:
            return confirmed_results, pending_results
        
        char_data = feature_db["characters"][char]
        avg_feature = np.array(char_data["avg_feature"])
        
        for char_info in target_chars:
            image_path = Path(char_info.get("image_path", ""))
            if not image_path.exists():
                continue
            
            target_feature = CharMatcher.extract_hog_features(image_path)
            similarity = CharMatcher.calculate_similarity(avg_feature, target_feature)
            
            result = {
                "char_id": char_info.get("char_id", ""),
                "image_path": str(image_path),
                "cluster_id": char_info.get("cluster_id", ""),
                "similarity": round(similarity, 4),
                "source_char": char,
                "status": "confirmed" if similarity >= confidence_threshold else "pending"
            }
            
            if similarity >= confidence_threshold:
                confirmed_results.append(result)
            else:
                pending_results.append(result)
        
        confirmed_results.sort(key=lambda x: x["similarity"], reverse=True)
        pending_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return confirmed_results[:top_n], pending_results[:top_n]
    
    @staticmethod
    def calculate_priority(
        source_count: int,
        target_count: int,
        match_confidence: float = 1.0
    ) -> float:
        """
        计算汉字标注优先级
        
        Args:
            source_count: 源数据集中该汉字的样本数
            target_count: 目标数据集中该汉字的样本数
            match_confidence: 匹配置信度
        
        Returns:
            优先级分数 (0-100)
        """
        target_score = max(0, 1 - target_count / 100) * 40
        source_score = min(1, source_count / 50) * 30
        confidence_score = match_confidence * 30
        
        return round(target_score + source_score + confidence_score, 2)
    
    @staticmethod
    def bidirectional_match(
        source_chars: List[Dict],
        target_chars: List[Dict],
        feature_db: Dict = None,
        confidence_threshold: float = 0.5,
        top_n: int = 20
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        双向匹配算法 - 确保每个源汉字都能找到对应的目标字符
        
        Args:
            source_chars: 源汉字列表，每个元素包含char_id, image_path等
            target_chars: 目标汉字列表，每个元素包含char_id, image_path等
            feature_db: 汉字特征库（可选）
            confidence_threshold: 置信度阈值，低于此值的标记为待确认
            top_n: 返回前N个匹配结果
        
        Returns:
            Tuple: (确认匹配结果, 待确认匹配结果)
                   每个结果格式: {source_char: [{target_char_info, similarity, status}]}
        """
        confirmed_matches = {}
        pending_matches = {}
        
        if not source_chars or not target_chars:
            return confirmed_matches, pending_matches
        
        # 构建相似度矩阵
        source_features = {}
        target_features = {}
        
        # 提取源汉字特征
        for source_char in source_chars:
            image_path = Path(source_char.get("image_path", ""))
            if image_path.exists():
                feature = CharMatcher.extract_hog_features(image_path)
                source_features[source_char["char_id"]] = feature
        
        # 提取目标汉字特征
        for target_char in target_chars:
            image_path = Path(target_char.get("image_path", ""))
            if image_path.exists():
                feature = CharMatcher.extract_hog_features(image_path)
                target_features[target_char["char_id"]] = feature
        
        if not source_features or not target_features:
            return confirmed_matches, pending_matches
        
        source_ids = list(source_features.keys())
        target_ids = list(target_features.keys())
        
        # 构建代价矩阵（1 - similarity，因为linear_sum_assignment求最小值）
        cost_matrix = np.zeros((len(source_ids), len(target_ids)))
        
        for i, source_id in enumerate(source_ids):
            for j, target_id in enumerate(target_ids):
                similarity = CharMatcher.calculate_similarity(
                    source_features[source_id],
                    target_features[target_id]
                )
                cost_matrix[i, j] = 1 - similarity
        
        # 使用匈牙利算法进行二分图匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 收集双向匹配结果
        for i, j in zip(row_ind, col_ind):
            source_id = source_ids[i]
            target_id = target_ids[j]
            similarity = 1 - cost_matrix[i, j]
            
            # 找到对应的源汉字和目标汉字信息
            source_info = next((s for s in source_chars if s["char_id"] == source_id), None)
            target_info = next((t for t in target_chars if t["char_id"] == target_id), None)
            
            if source_info and target_info:
                match_result = {
                    "target_char_id": target_id,
                    "target_image_path": target_info.get("image_path", ""),
                    "target_cluster_id": target_info.get("cluster_id", ""),
                    "similarity": round(similarity, 4),
                    "status": "confirmed" if similarity >= confidence_threshold else "pending"
                }
                
                source_char_text = source_info.get("char", source_info.get("char_id", ""))
                
                if similarity >= confidence_threshold:
                    if source_char_text not in confirmed_matches:
                        confirmed_matches[source_char_text] = []
                    confirmed_matches[source_char_text].append(match_result)
                else:
                    if source_char_text not in pending_matches:
                        pending_matches[source_char_text] = []
                    pending_matches[source_char_text].append(match_result)
        
        # 对结果按相似度排序
        for char in confirmed_matches:
            confirmed_matches[char].sort(key=lambda x: x["similarity"], reverse=True)
            confirmed_matches[char] = confirmed_matches[char][:top_n]
        
        for char in pending_matches:
            pending_matches[char].sort(key=lambda x: x["similarity"], reverse=True)
            pending_matches[char] = pending_matches[char][:top_n]
        
        return confirmed_matches, pending_matches
    
    @staticmethod
    def find_matching_images_with_bidirectional(
        feature_db: Dict,
        target_chars: List[Dict],
        char: str,
        top_n: int = 20,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        使用双向匹配算法查找匹配图片
        
        Args:
            feature_db: 汉字特征库
            target_chars: 目标数据集的字符列表
            char: 要匹配的汉字
            top_n: 返回前N个匹配结果
            confidence_threshold: 置信度阈值
        
        Returns:
            Tuple: (确认匹配结果, 待确认匹配结果)
        """
        if char not in feature_db["characters"]:
            return [], []
        
        char_data = feature_db["characters"][char]
        
        # 构建源汉字列表（从特征库中提取该汉字的所有样本）
        source_chars = []
        for sample in char_data.get("samples", []):
            source_chars.append({
                "char_id": sample["char_id"],
                "image_path": sample["image_path"],
                "cluster_id": sample["cluster_id"],
                "char": char
            })
        
        if not source_chars:
            return [], []
        
        confirmed, pending = CharMatcher.bidirectional_match(
            source_chars,
            target_chars,
            feature_db,
            confidence_threshold,
            top_n
        )
        
        return confirmed.get(char, []), pending.get(char, [])