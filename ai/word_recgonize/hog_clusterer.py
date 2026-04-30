"""
HOG特征聚类器
用于对汉字图片进行聚类分析
"""

import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .cluster_config import HOGClusterConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from ai.word_recgonize.cluster_config import HOGClusterConfig

class HOGClusterer:
    """
    HOG特征聚类器
    
    使用方向梯度直方图(HOG)提取图像特征，然后使用K-means进行聚类
    """
    
    def __init__(self, cfg: HOGClusterConfig = None):
        """
        初始化聚类器
        
        Args:
            cfg: 聚类配置，如果为None则使用默认配置
        """
        self.cfg = cfg if cfg else HOGClusterConfig()
        self.clusterer = None
        self.scaler = None
        self.clusters = {}  # {cluster_id: [char_info, ...]}
        self.char_to_cluster = {}  # {char_id: cluster_id}
    
    def extract_hog_features(self, image_path: str) -> np.ndarray:
        """
        从单张图片提取HOG特征
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            HOG特征向量
        """
        # 读取图片（灰度模式）
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 统一图片大小（缩放为固定尺寸）
        target_size = (64, 64)
        img = cv2.resize(img, target_size)
        
        # 计算HOG特征
        hog = cv2.HOGDescriptor(
            _winSize=(target_size[0], target_size[1]),
            _blockSize=(self.cfg.cells_per_block[0] * self.cfg.pixels_per_cell[0],
                        self.cfg.cells_per_block[1] * self.cfg.pixels_per_cell[1]),
            _blockStride=(self.cfg.pixels_per_cell[0], self.cfg.pixels_per_cell[1]),
            _cellSize=(self.cfg.pixels_per_cell[0], self.cfg.pixels_per_cell[1]),
            _nbins=self.cfg.orientations
        )
        
        features = hog.compute(img)
        return features.flatten()
    
    def load_char_images(self) -> List[Dict]:
        """
        加载所有汉字图片信息
        
        Returns:
            汉字信息列表，包含路径和ID
        """
        char_info_list = []
        input_dir = self.cfg.get_input_path()
        
        if not input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        # 遍历所有PNG文件
        for img_file in input_dir.glob("*.png"):
            if "_char_" in img_file.name:
                char_info = {
                    "char_id": img_file.stem,
                    "image_path": str(img_file),
                    "file_name": img_file.name
                }
                char_info_list.append(char_info)
        
        print(f"[INFO] 加载了 {len(char_info_list)} 张汉字图片")
        return char_info_list
    
    def cluster(self, char_info_list: List[Dict]) -> None:
        """
        执行聚类
        
        Args:
            char_info_list: 汉字信息列表
        """
        if len(char_info_list) == 0:
            print("[WARN] 没有可用的汉字图片")
            return
        
        # 提取所有图片的HOG特征
        print(f"[INFO] 正在提取HOG特征...")
        features = []
        valid_char_info = []
        
        for char_info in char_info_list:
            try:
                hog_features = self.extract_hog_features(char_info["image_path"])
                features.append(hog_features)
                valid_char_info.append(char_info)
            except Exception as e:
                print(f"[WARN] 跳过图片 {char_info['file_name']}: {str(e)}")
        
        if len(features) == 0:
            print("[ERROR] 无法提取任何图片特征")
            return
        
        # 标准化特征
        print(f"[INFO] 正在标准化特征...")
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 使用K-means聚类
        print(f"[INFO] 正在执行K-means聚类 (n_clusters={self.cfg.n_clusters})...")
        self.clusterer = KMeans(
            n_clusters=self.cfg.n_clusters,
            random_state=self.cfg.random_state,
            n_init='auto'
        )
        labels = self.clusterer.fit_predict(features_scaled)
        
        # 构建聚类结果
        self.clusters = {}
        self.char_to_cluster = {}
        
        for idx, char_info in enumerate(valid_char_info):
            cluster_id = int(labels[idx])
            char_id = char_info["char_id"]
            
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            
            self.clusters[cluster_id].append({
                "char_id": char_id,
                "image_path": char_info["image_path"],
                "file_name": char_info["file_name"]
            })
            self.char_to_cluster[char_id] = cluster_id
        
        print(f"[INFO] 聚类完成，共生成 {len(self.clusters)} 个聚类")
        
        # 打印聚类大小统计
        cluster_sizes = sorted([(k, len(v)) for k, v in self.clusters.items()], 
                             key=lambda x: x[1], reverse=True)
        print(f"[INFO] 聚类大小分布（前10个）:")
        for cluster_id, size in cluster_sizes[:10]:
            print(f"       聚类 {cluster_id}: {size} 个汉字")
    
    def save_results(self) -> None:
        """
        保存聚类结果到文件
        """
        output_dir = self.cfg.get_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存聚类结果JSON
        result = {
            "config": {
                "orientations": self.cfg.orientations,
                "pixels_per_cell": self.cfg.pixels_per_cell,
                "cells_per_block": self.cfg.cells_per_block,
                "n_clusters": self.cfg.n_clusters,
                "total_chars": sum(len(chars) for chars in self.clusters.values()),
                "total_clusters": len(self.clusters)
            },
            "clusters": self.clusters,
            "char_to_cluster": self.char_to_cluster
        }
        
        json_path = self.cfg.get_clusters_json_path()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 聚类结果已保存到: {json_path}")
    
    def save_cluster_images(self, max_samples_per_cluster: int = 5) -> None:
        """
        保存每个聚类的示例图片
        
        Args:
            max_samples_per_cluster: 每个聚类保存的最大样本数
        """
        cluster_images_dir = self.cfg.get_cluster_images_path()
        cluster_images_dir.mkdir(parents=True, exist_ok=True)
        
        for cluster_id, chars in self.clusters.items():
            # 创建聚类目录
            cluster_dir = cluster_images_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # 复制示例图片
            samples = chars[:max_samples_per_cluster]
            for idx, char_info in enumerate(samples):
                src_path = Path(char_info["image_path"])
                dst_path = cluster_dir / f"{idx}_{char_info['file_name']}"
                
                try:
                    # 读取并保存图片
                    img = cv2.imread(str(src_path))
                    if img is not None:
                        cv2.imwrite(str(dst_path), img)
                except Exception as e:
                    print(f"[WARN] 无法复制图片 {char_info['file_name']}: {str(e)}")
        
        print(f"[INFO] 聚类示例图片已保存到: {cluster_images_dir}")
    
    def run(self) -> None:
        """
        执行完整的聚类流程
        """
        print("=" * 60)
        print("HOG汉字图片聚类")
        print("=" * 60)
        
        try:
            # 1. 加载汉字图片
            char_info_list = self.load_char_images()
            
            # 2. 执行聚类
            self.cluster(char_info_list)
            
            # 3. 保存结果
            self.save_results()
            
            # 4. 保存示例图片
            self.save_cluster_images()
            
            print("=" * 60)
            print("聚类完成！")
            print("=" * 60)
            
        except Exception as e:
            print(f"[ERROR] 聚类失败: {str(e)}")
            raise

if __name__ == "__main__":
    # 创建配置
    cfg = HOGClusterConfig()
    
    # 创建聚类器并执行
    clusterer = HOGClusterer(cfg)
    clusterer.run()