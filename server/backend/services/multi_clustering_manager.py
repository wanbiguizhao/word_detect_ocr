import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import hdbscan
import numpy as np
from .char_pool_manager import CharPoolManager


class MultiClusteringManager:
    """多轮聚类管理器 - 完全独立于现有聚类系统"""
    
    def __init__(self, dataset_id: str = None):
        if dataset_id is None:
            from config import DATASET_ID
            dataset_id = DATASET_ID
        
        self.dataset_id = dataset_id
        # 项目根目录: d:\projects\word_detect_ocr
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.base_dir = self.project_root / "bussiness" / "datahome" / dataset_id / "multi_clustering"
        self.rounds_dir = self.base_dir / "rounds"
        self.config_path = self.base_dir / "config.json"
        self.history_path = self.base_dir / "round_history.json"
        self.unified_labels_path = self.base_dir / "unified_labels.json"
        
        # 确保目录存在
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        
        # 字符池管理器
        self.char_pool = CharPoolManager(dataset_id)
        
        # 数据存储抽象层（用于传播更新）
        from datastore.data_store import DataStore
        self.data_store = DataStore(dataset_id)
        
        # 加载配置
        self.config = self._load_config()
    
    def _load_json(self, path: Path):
        """加载JSON文件"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_json(self, path: Path, data: dict):
        """保存JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_config(self) -> dict:
        """加载配置"""
        config = self._load_json(self.config_path)
        if config:
            return config
        # 默认配置
        return {
            "version": "1.0",
            "clustering": {
                "algorithm": "hog+kmeans",
                "default_n_clusters": 300,
                "similarity_threshold": 0.75
            }
        }
    
    def _extract_hog_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取HOG特征（使用OpenCV实现）"""
        try:
            import cv2
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # 调整大小到固定尺寸 (64x64)
            img = cv2.resize(img, (64, 64))
            
            # 使用真正的 HOG 描述符
            win_size = (64, 64)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(img)
            
            # 归一化
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features.flatten()
        except Exception as e:
            print(f"提取HOG特征失败: {e}")
            return None
    
    def _sort_cluster_by_similarity(self, clusters: Dict[str, List[dict]], 
                                    cluster_indices: Dict[str, List[int]],
                                    features: np.ndarray) -> Dict[str, List[dict]]:
        """
        对每个聚类内部的图片按照相似度排序，使相似的图片尽可能靠近
        
        Args:
            clusters: 原始聚类结果
            cluster_indices: 每个聚类包含的特征索引
            features: 所有字符的特征矩阵
        
        Returns:
            排序后的聚类结果
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        sorted_clusters = {}
        
        for cluster_id_str, cluster_chars in clusters.items():
            indices = cluster_indices[cluster_id_str]
            if len(indices) <= 1:
                # 只有一个元素，无需排序
                sorted_clusters[cluster_id_str] = cluster_chars
                continue
            
            # 获取该聚类的所有特征
            cluster_features = features[indices]
            
            # 计算相似度矩阵
            sim_matrix = cosine_similarity(cluster_features)
            
            # 使用贪心算法进行排序：每次选择与当前序列最相似的元素
            n = len(cluster_chars)
            visited = [False] * n
            order = []
            
            # 从相似度最大的元素开始（作为中心）
            avg_sim = sim_matrix.mean(axis=1)
            start_idx = int(np.argmax(avg_sim))
            order.append(start_idx)
            visited[start_idx] = True
            
            # 贪心选择下一个最相似的元素
            for _ in range(n - 1):
                last_idx = order[-1]
                max_sim = -1
                next_idx = -1
                
                for i in range(n):
                    if not visited[i]:
                        sim = sim_matrix[last_idx, i]
                        if sim > max_sim:
                            max_sim = sim
                            next_idx = i
                
                if next_idx != -1:
                    visited[next_idx] = True
                    order.append(next_idx)
            
            # 根据排序结果重新排列字符
            sorted_chars = [cluster_chars[i] for i in order]
            sorted_clusters[cluster_id_str] = sorted_chars
        
        return sorted_clusters
    
    def _split_large_clusters(self, clusters: Dict[str, List[dict]],
                              cluster_indices: Dict[str, List[int]],
                              features: np.ndarray,
                              max_cluster_size: int) -> tuple:
        """
        拆分超过最大聚类大小限制的聚类
        
        Args:
            clusters: 原始聚类结果
            cluster_indices: 每个聚类包含的特征索引
            features: 所有字符的特征矩阵
            max_cluster_size: 最大聚类大小限制
        
        Returns:
            tuple: (拆分后的聚类结果, 对应的索引信息)
        """
        # 调试日志
        print(f"[_split_large_clusters] 开始拆分超大类")
        print(f"  max_cluster_size: {max_cluster_size}")
        print(f"  原始聚类数: {len(clusters)}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        result_clusters = {}
        result_indices = {}
        new_cluster_id = 1000  # 使用较大的起始ID避免冲突
        
        for cluster_id_str, cluster_chars in clusters.items():
            if len(cluster_chars) <= max_cluster_size:
                # 聚类大小在限制范围内，直接保留
                result_clusters[cluster_id_str] = cluster_chars
                result_indices[cluster_id_str] = cluster_indices[cluster_id_str]
                continue
            
            # 超过限制，需要拆分
            print(f"拆分超大类 {cluster_id_str}，大小: {len(cluster_chars)}")
            
            # 获取该聚类的所有特征
            indices = cluster_indices[cluster_id_str]
            cluster_features = features[indices]
            
            # 计算相似度矩阵
            sim_matrix = cosine_similarity(cluster_features)
            
            n = len(cluster_chars)
            visited = [False] * n
            
            # 贪心算法：每次选择一个中心点，然后选择与它最相似的k个元素形成一个子聚类
            while not all(visited):
                # 找到未访问元素中与其他未访问元素平均相似度最高的作为中心
                max_avg_sim = -1
                center_idx = -1
                
                for i in range(n):
                    if visited[i]:
                        continue
                    
                    # 计算与其他未访问元素的平均相似度
                    unvisited_indices = [j for j in range(n) if not visited[j] and j != i]
                    if not unvisited_indices:
                        avg_sim = 0
                    else:
                        avg_sim = np.mean([sim_matrix[i, j] for j in unvisited_indices])
                    
                    if avg_sim > max_avg_sim:
                        max_avg_sim = avg_sim
                        center_idx = i
                
                if center_idx == -1:
                    break
                
                # 选择与中心最相似的 max_cluster_size 个元素形成一个子聚类
                visited[center_idx] = True
                sub_cluster = [cluster_chars[center_idx]]
                sub_indices = [indices[center_idx]]  # 记录原始特征索引
                
                # 获取未访问元素中与中心的相似度
                candidates = []
                for i in range(n):
                    if not visited[i]:
                        candidates.append((i, sim_matrix[center_idx, i]))
                
                # 按相似度降序排序
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 选择前 max_cluster_size - 1 个（已经包含中心）
                for i, _ in candidates[:max_cluster_size - 1]:
                    visited[i] = True
                    sub_cluster.append(cluster_chars[i])
                    sub_indices.append(indices[i])  # 记录原始特征索引
                
                # 添加到结果中
                result_clusters[str(new_cluster_id)] = sub_cluster
                result_indices[str(new_cluster_id)] = sub_indices
                new_cluster_id += 1
        
        print(f"超大类拆分完成，原始聚类数: {len(clusters)}，拆分后聚类数: {len(result_clusters)}")
        return result_clusters, result_indices
    
    def get_current_round(self) -> int:
        """获取当前最新轮次"""
        history = self._load_json(self.history_path)
        if history and history.get("rounds"):
            return max(r["round"] for r in history["rounds"])
        return 0
    
    def _get_unlabeled_char_ids(self) -> List[str]:
        """获取未标注字符ID列表（含自动初始化）"""
        unlabeled_ids = self.char_pool.get_unlabeled_char_ids()
        
        if not unlabeled_ids:
            try:
                self.char_pool.init_from_lineage()
                unlabeled_ids = self.char_pool.get_unlabeled_char_ids()
            except Exception as e:
                raise ValueError(f"字符池为空且初始化失败: {str(e)}")
        
        return unlabeled_ids

    def _get_low_confidence_char_ids(self, confidence_threshold: float = 0.7) -> List[str]:
        """获取低置信度预测的字符ID列表
        
        从 pre_labels.json 中筛选 confidence < threshold 的字符，
        排除已标注和已跳过的字符，确保数据一致性。
        
        Args:
            confidence_threshold: 置信度阈值，低于此值的字符被选中
        """
        prelabels_path = self.project_root / "bussiness" / "datahome" / self.dataset_id / "pre_labels.json"
        
        if not prelabels_path.exists():
            raise ValueError(f"预标注文件不存在: {prelabels_path}")
        
        with open(prelabels_path, 'r', encoding='utf-8') as f:
            prelabels_data = json.load(f)
        
        prelabels = prelabels_data.get("prelabels", [])
        if not prelabels:
            raise ValueError("预标注数据为空，无法筛选低置信度字符")
        
        labeled_ids = self.char_pool.get_labeled_char_ids()
        
        all_chars = self.char_pool.load_all_chars()
        skipped_ids = {char_id for char_id, info in all_chars.items() if info.get("status") == "skipped"}
        
        low_conf_ids = []
        for p in prelabels:
            char_id = p.get("char_id", "")
            confidence = p.get("confidence", 1.0)
            
            if not char_id:
                continue
            if char_id in labeled_ids:
                continue
            if char_id in skipped_ids:
                continue
            if confidence < confidence_threshold:
                low_conf_ids.append(char_id)
        
        if not low_conf_ids:
            raise ValueError(
                f"没有置信度低于 {confidence_threshold} 的未处理字符 "
                f"(预标注总数: {len(prelabels)}, 已标注: {len(labeled_ids)}, 已跳过: {len(skipped_ids)})"
            )
        
        print(f"[Manager] 低置信度筛选: 阈值={confidence_threshold}, "
              f"预标注总数={len(prelabels)}, 已标注={len(labeled_ids)}, "
              f"已跳过={len(skipped_ids)}, 低置信度未处理={len(low_conf_ids)}")
        
        return low_conf_ids

    def start_new_round(self, n_clusters: Optional[int] = None, 
                        description: str = "",
                        method: str = "hdbscan",
                        min_cluster_size: int = 5,
                        min_samples: int = 2,
                        max_cluster_size: int = 100,
                        data_source: str = "unlabeled",
                        confidence_threshold: float = 0.7) -> int:
        """启动新一轮聚类
        
        Args:
            data_source: 数据源类型
                - "unlabeled": 未标注汉字（默认，现有行为）
                - "low_confidence": 低置信度预测
                - "outlier": 已标注离群检测（Phase 3）
            confidence_threshold: 低置信度阈值，仅 data_source="low_confidence" 时生效
        """
        print(f"[Manager] start_new_round 接收到的参数:")
        print(f"  data_source: {data_source}")
        print(f"  method: {method}")
        print(f"  n_clusters: {n_clusters}")
        print(f"  description: {description}")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  min_samples: {min_samples}")
        print(f"  max_cluster_size: {max_cluster_size}")
        print(f"  confidence_threshold: {confidence_threshold}")
        
        if data_source == "unlabeled":
            char_ids = self._get_unlabeled_char_ids()
        elif data_source == "low_confidence":
            char_ids = self._get_low_confidence_char_ids(confidence_threshold)
        else:
            raise ValueError(f"暂不支持的数据源类型: {data_source}")
        
        if not char_ids:
            raise ValueError("没有可用的字符，无法启动新轮聚类")
        
        features_dict = {}
        valid_char_ids = []
        
        for char_id in char_ids:
            img_path = self.project_root / "bussiness" / "datahome" / self.dataset_id / "pdf_chars" / f"{char_id}.png"
            if img_path.exists():
                feat = self._extract_hog_features(str(img_path))
                if feat is not None:
                    features_dict[char_id] = feat
                    valid_char_ids.append(char_id)
        
        if not valid_char_ids:
            raise ValueError("无法提取任何字符特征")
        
        # 执行聚类
        features = np.array([features_dict[char_id] for char_id in valid_char_ids])
        labels = None
        
        try:
            if method == "hdbscan":
                from scipy.spatial.distance import squareform, pdist
                cosine_dist_matrix = squareform(pdist(features, 'cosine'))
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='precomputed',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(cosine_dist_matrix)
            elif method == "kmeans":
                from sklearn.cluster import KMeans
                
                if n_clusters is None:
                    n_clusters = 20
                
                n_clusters = min(n_clusters, len(valid_char_ids))
                if n_clusters < 1:
                    n_clusters = 1
                
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(features)
            else:
                raise ValueError(f"不支持的聚类方法: {method}")
        except Exception as e:
            raise RuntimeError(f"聚类失败: {e}")
        
        # 构建聚类结果（过滤噪声点，label=-1表示噪声）
        clusters: Dict[str, List[dict]] = {}
        cluster_indices: Dict[str, List[int]] = {}  # 记录每个聚类包含的索引
        
        for idx, char_id in enumerate(valid_char_ids):
            cluster_id = labels[idx]
            if cluster_id == -1:
                continue  # 跳过噪声点
            cluster_id_str = str(cluster_id)
            if cluster_id_str not in clusters:
                clusters[cluster_id_str] = []
                cluster_indices[cluster_id_str] = []
            
            # 获取字符信息
            char_info = self.char_pool.load_all_chars().get(char_id, {})
            clusters[cluster_id_str].append({
                "char_id": char_id,
                "line_name": char_info.get("line_name", ""),
                "col_start": char_info.get("col_start", 0),
                "col_end": char_info.get("col_end", 0)
            })
            cluster_indices[cluster_id_str].append(idx)
        
        # 拆分超大类（超过max_cluster_size限制的聚类）
        if max_cluster_size > 0:
            clusters, cluster_indices = self._split_large_clusters(clusters, cluster_indices, features, max_cluster_size)
        
        # 对每个聚类内部按照相似度排序（相似的图片放在一起）
        clusters = self._sort_cluster_by_similarity(clusters, cluster_indices, features)
        
        # 创建轮次目录
        new_round = self.get_current_round() + 1
        round_dir = self.rounds_dir / f"round_{new_round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存聚类结果
        clusters_data = {
            "version": "1.0",
            "round": new_round,
            "type": "clustering",
            "data_source": data_source,
            "algorithm": method,
            "n_clusters": n_clusters if method == "kmeans" else len(clusters),
            "total_chars": len(valid_char_ids),
            "created_at": datetime.datetime.now().isoformat(),
            "clusters": clusters,
            "params": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "max_cluster_size": max_cluster_size
            } if method == "hdbscan" else {
                "n_clusters": n_clusters
            },
            "data_source_params": {
                "confidence_threshold": confidence_threshold
            } if data_source == "low_confidence" else {}
        }
        self._save_json(round_dir / "hog_clusters.json", clusters_data)
        
        # 初始化标注文件
        labels_data = {
            "version": "1.0",
            "round": new_round,
            "created_at": datetime.datetime.now().isoformat(),
            "labels": {}  # {cluster_id: {"char_labels": {}, "status": "unlabeled"}}
        }
        for cluster_id in clusters:
            labels_data["labels"][cluster_id] = {
                "char": None,
                "chars": {},
                "status": "unlabeled",
                "confidence": None,
                "alias": "",
                "char_labels": {}
            }
        self._save_json(round_dir / "labels.json", labels_data)
        
        # 更新轮次历史
        history = self._load_json(self.history_path) or {"rounds": []}
        history["rounds"].append({
            "round": new_round,
            "date": datetime.datetime.now().isoformat(),
            "type": "clustering",
            "data_source": data_source,
            "confidence_threshold": confidence_threshold if data_source == "low_confidence" else None,
            "algorithm": method,
            "n_clusters": n_clusters if method == "kmeans" else len(clusters),
            "total_chars": len(valid_char_ids),
            "description": description or f"第{new_round}轮聚类"
        })
        self._save_json(self.history_path, history)
        
        return new_round
    
    def get_round_clusters(self, round_num: int) -> Optional[dict]:
        """获取指定轮次的聚类数据"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        clusters_path = round_dir / "hog_clusters.json"
        
        if not clusters_path.exists():
            return None
        
        return self._load_json(clusters_path)
    
    def get_round_labels(self, round_num: int) -> Optional[dict]:
        """获取指定轮次的标注数据"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        labels_path = round_dir / "labels.json"
        
        if not labels_path.exists():
            return None
        
        return self._load_json(labels_path)
    
    def save_label(self, round_num: int, cluster_id: str, 
                   char_index: int, char: str) -> bool:
        """保存标注 - 支持传播更新到所有相关数据源"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        labels_path = round_dir / "labels.json"
        
        if not labels_path.exists():
            return False
        
        labels_data = self._load_json(labels_path)
        
        if labels_data is None:
            labels_data = {"labels": {}}
        
        # 确保labels字段存在
        if "labels" not in labels_data:
            labels_data["labels"] = {}
        
        # 确保cluster_id存在
        if cluster_id not in labels_data["labels"]:
            labels_data["labels"][cluster_id] = {
                "char": None,
                "chars": {},
                "status": "unlabeled",
                "confidence": None,
                "alias": "",
                "char_labels": {}
            }
        
        # 保存标注到多轮聚类本地文件
        char_key = str(char_index)
        labels_data["labels"][cluster_id]["char_labels"][char_key] = {
            "char": char,
            "labeled_at": datetime.datetime.now().isoformat()
        }
        
        # 更新统计
        all_chars = [v["char"] for v in labels_data["labels"][cluster_id]["char_labels"].values() if v.get("char")]
        if all_chars:
            from collections import Counter
            char_counts = Counter(all_chars)
            most_common = char_counts.most_common(1)[0]
            labels_data["labels"][cluster_id]["char"] = most_common[0]
            labels_data["labels"][cluster_id]["confidence"] = most_common[1] / len(all_chars)
            labels_data["labels"][cluster_id]["chars"] = dict(char_counts)
            labels_data["labels"][cluster_id]["status"] = "labeled"
        
        # 保存标注到本地文件
        self._save_json(labels_path, labels_data)
        
        # 获取字符ID
        char_id = None
        try:
            clusters_data = self.get_round_clusters(round_num)
            if clusters_data:
                cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
                if char_index < len(cluster_chars):
                    char_id = cluster_chars[char_index].get("char_id")
        except Exception as e:
            # 忽略获取字符ID的错误，继续保存标注
            pass
        
        if char_id:
            # 更新字符池
            self.char_pool.mark_as_labeled(char_id, char, round_num)
            
            # 使用 DataStore 实现传播更新
            # 这将自动同步到：
            # 1. unified_labels.json（统一标注）
            # 2. clusters/labeling/labels.json（聚类标注）
            # 3. pre_labels.json（预标注状态）
            # 4. multi_clustering/ 中的其他轮次（如果字符存在）
            self.data_store.write_annotation(char_id, char)
        
        return True
    
    def _sync_to_unified(self, char_id: str, char: str):
        """同步到统一标注"""
        unified_data = self._load_json(self.unified_labels_path) or {"annotations": []}
        
        # 检查是否已存在
        found = False
        for ann in unified_data["annotations"]:
            if ann.get("char_id") == char_id:
                ann["char"] = char
                ann["status"] = "labeled"
                ann["updated_at"] = datetime.datetime.now().isoformat()
                found = True
                break
        
        if not found:
            unified_data["annotations"].append({
                "char_id": char_id,
                "char": char,
                "status": "labeled",
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat()
            })
        
        self._save_json(self.unified_labels_path, unified_data)
    
    def save_batch_labels(self, round_num: int, cluster_id: str, 
                          labels: List[Dict[str, Any]]) -> int:
        """批量保存标注 - 优化版本：一次性写入文件和同步"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        labels_path = round_dir / "labels.json"
        
        if not labels_path.exists():
            return 0
        
        # 一次性加载标签数据
        labels_data = self._load_json(labels_path)
        if labels_data is None:
            labels_data = {"labels": {}}
        if "labels" not in labels_data:
            labels_data["labels"] = {}
        
        # 确保cluster_id存在
        if cluster_id not in labels_data["labels"]:
            labels_data["labels"][cluster_id] = {
                "char": None,
                "chars": {},
                "status": "unlabeled",
                "confidence": None,
                "alias": "",
                "char_labels": {}
            }
        
        # 收集需要同步的字符ID
        chars_to_sync = []
        
        # 批量更新标签（支持字典和Pydantic对象）
        for label in labels:
            # 支持字典和Pydantic对象
            if hasattr(label, 'charIndex'):
                char_index = label.charIndex
                char = label.char
            else:
                char_index = label.get("charIndex")
                char = label.get("char")
            
            if char_index is not None and char:
                char_key = str(char_index)
                labels_data["labels"][cluster_id]["char_labels"][char_key] = {
                    "char": char,
                    "labeled_at": datetime.datetime.now().isoformat()
                }
                chars_to_sync.append((char_index, char))
        
        # 更新统计
        all_chars = [v["char"] for v in labels_data["labels"][cluster_id]["char_labels"].values() if v.get("char")]
        if all_chars:
            from collections import Counter
            char_counts = Counter(all_chars)
            most_common = char_counts.most_common(1)[0]
            labels_data["labels"][cluster_id]["char"] = most_common[0]
            labels_data["labels"][cluster_id]["confidence"] = most_common[1] / len(all_chars)
            labels_data["labels"][cluster_id]["chars"] = dict(char_counts)
            labels_data["labels"][cluster_id]["status"] = "labeled"
        
        # 一次性保存到本地文件
        self._save_json(labels_path, labels_data)
        
        # 获取字符ID并批量同步
        saved_count = 0
        try:
            clusters_data = self.get_round_clusters(round_num)
            if clusters_data:
                cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
                
                for char_index, char in chars_to_sync:
                    if char_index < len(cluster_chars):
                        char_id = cluster_chars[char_index].get("char_id")
                        if char_id:
                            # 更新字符池
                            self.char_pool.mark_as_labeled(char_id, char, round_num)
                            # 使用 DataStore 实现传播更新
                            self.data_store.write_annotation(char_id, char)
                            saved_count += 1
        except Exception as e:
            # 同步失败不影响本地保存
            pass
        
        return saved_count
    
    def get_round_history(self) -> dict:
        """获取轮次历史"""
        return self._load_json(self.history_path) or {"rounds": []}
    
    def get_round_progress(self, round_num: int) -> dict:
        """获取轮次标注进度"""
        clusters_data = self.get_round_clusters(round_num)
        labels_data = self.get_round_labels(round_num)
        
        if not clusters_data or not labels_data:
            return {}
        
        total_clusters = len(clusters_data.get("clusters", {}))
        labeled_clusters = 0
        skipped_clusters = 0
        total_chars = clusters_data.get("total_chars", 0)
        labeled_chars = 0
        
        for cluster_id, cluster_info in labels_data.get("labels", {}).items():
            status = cluster_info.get("status", "unlabeled")
            if status == "labeled":
                labeled_clusters += 1
                labeled_chars += len(cluster_info.get("char_labels", {}))
            elif status == "skipped":
                skipped_clusters += 1
        
        return {
            "round": round_num,
            "total_clusters": total_clusters,
            "labeled_clusters": labeled_clusters,
            "skipped_clusters": skipped_clusters,
            "remaining_clusters": total_clusters - labeled_clusters - skipped_clusters,
            "total_chars": total_chars,
            "labeled_chars": labeled_chars
        }
    
    def skip_cluster(self, round_num: int, cluster_id: str) -> bool:
        """跳过聚类"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        labels_path = round_dir / "labels.json"
        
        if not labels_path.exists():
            return False
        
        labels_data = self._load_json(labels_path)
        
        if cluster_id not in labels_data["labels"]:
            labels_data["labels"][cluster_id] = {
                "char": None,
                "chars": {},
                "status": "unlabeled",
                "confidence": None,
                "alias": "",
                "char_labels": {}
            }
        
        labels_data["labels"][cluster_id]["status"] = "skipped"
        labels_data["labels"][cluster_id]["skipped_at"] = datetime.datetime.now().isoformat()
        
        self._save_json(labels_path, labels_data)
        
        # 收集该聚类的所有字符ID
        cluster_char_ids = []
        clusters_data = self.get_round_clusters(round_num)
        if clusters_data:
            cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
            for char_info in cluster_chars:
                char_id = char_info.get("char_id")
                if char_id:
                    cluster_char_ids.append(char_id)
        
        # 同步更新所有数据系统
        self._sync_skip_to_all(cluster_char_ids, round_num)
        
        return True
    
    def _update_prelabels_status(self, char_ids: List[str], status: str):
        """更新 pre_labels.json 中字符的状态
        
        Args:
            char_ids: 字符ID列表
            status: 新状态（"pending", "labeled", "skipped"）
        """
        prelabels_path = self.project_root / "bussiness" / "datahome" / self.dataset_id / "pre_labels.json"
        
        if not prelabels_path.exists():
            return
        
        with open(prelabels_path, 'r', encoding='utf-8') as f:
            prelabels_data = json.load(f)
        
        prelabels = prelabels_data.get("prelabels", [])
        char_id_set = set(char_ids)
        
        updated = False
        for p in prelabels:
            if p.get("char_id") in char_id_set:
                p["status"] = status
                updated = True
        
        if updated:
            with open(prelabels_path, 'w', encoding='utf-8') as f:
                json.dump(prelabels_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Manager] 已更新 pre_labels.json 中 {len(char_id_set)} 个字符的状态为 '{status}'")
    
    def skip_char(self, char_id: str, round_num: int) -> bool:
        """跳过单个字符
        
        Args:
            char_id: 字符ID
            round_num: 轮次编号（用于记录跳过的轮次）
        
        Returns:
            bool: 是否成功跳过
        """
        # 同步更新所有数据系统
        self._sync_skip_to_all([char_id], round_num)
        
        print(f"[Manager] 已跳过字符: {char_id} (轮次: {round_num})")
        return True
    
    def batch_skip_chars(self, char_ids: List[str], round_num: int) -> dict:
        """批量跳过字符（单次文件IO，避免并发冲突）
        
        Args:
            char_ids: 字符ID列表
            round_num: 轮次编号
        
        Returns:
            dict: {"skipped": 跳过数量, "total": 总数量}
        """
        if not char_ids:
            return {"skipped": 0, "total": 0}
        
        valid_ids = [cid for cid in char_ids if cid]
        if not valid_ids:
            return {"skipped": 0, "total": len(char_ids)}
        
        self._sync_skip_to_all(valid_ids, round_num)
        
        print(f"[Manager] 批量跳过: {len(valid_ids)} 个字符 (轮次: {round_num})")
        return {"skipped": len(valid_ids), "total": len(char_ids)}
    
    def _sync_skip_to_all(self, char_ids: List[str], round_num: int):
        """跳过操作同步到所有数据系统（批量优化版）"""
        if not char_ids:
            return
        
        # 1. 聚类系统: 批量更新字符池
        self.char_pool.batch_mark_as_skipped(char_ids, round_num)
        
        # 2. OCR系统: 批量更新 pre_labels.json
        self._update_prelabels_status(char_ids, "skipped")
        
        # 3. OCR系统: 批量更新 prelabel_status.json
        self.data_store.batch_skip_prelabels(char_ids)
        
        # 4. 统一标记: 批量更新 unified_labels.json
        self._update_unified_labels_status(char_ids, "skipped")
        
        print(f"[Manager] 跳过同步完成: {len(char_ids)} 个字符已同步到所有数据系统")
    
    def unskip_char(self, char_id: str, round_num: int) -> bool:
        """撤回跳过单个字符
        
        Args:
            char_id: 字符ID
            round_num: 轮次编号
        
        Returns:
            bool: 是否成功撤回
        """
        self._sync_unskip_to_all([char_id], round_num)
        print(f"[Manager] 已撤回跳过字符: {char_id}")
        return True
    
    def batch_unskip_chars(self, char_ids: List[str], round_num: int) -> dict:
        """批量撤回跳过字符
        
        Args:
            char_ids: 字符ID列表
            round_num: 轮次编号
        
        Returns:
            dict: {"unskipped": 撤回数量, "total": 总数量}
        """
        if not char_ids:
            return {"unskipped": 0, "total": 0}
        
        valid_ids = [cid for cid in char_ids if cid]
        if not valid_ids:
            return {"unskipped": 0, "total": len(char_ids)}
        
        self._sync_unskip_to_all(valid_ids, round_num)
        print(f"[Manager] 批量撤回跳过: {len(valid_ids)} 个字符")
        return {"unskipped": len(valid_ids), "total": len(char_ids)}
    
    def _sync_unskip_to_all(self, char_ids: List[str], round_num: int):
        """撤回跳过操作同步到所有数据系统
        
        确保以下三个系统的一致性：
        1. 聚类系统: char_pool (all_chars.json, labeled.json, unlabeled.json)
        2. OCR系统: pre_labels.json + prelabel_status.json
        3. 统一标记: unified_labels.json
        
        Args:
            char_ids: 要撤回跳过的字符ID列表
            round_num: 轮次编号
        """
        if not char_ids:
            return
        
        # 1. 聚类系统: 更新字符池（重置为未标注状态）
        for char_id in char_ids:
            self.char_pool.reset_char(char_id)
        
        # 2. OCR系统: 更新 pre_labels.json
        self._update_prelabels_status(char_ids, "pending")
        
        # 3. OCR系统: 更新 prelabel_status.json (DataStore 维护)
        for char_id in char_ids:
            self.data_store.reset_prelabel(char_id)
        
        # 4. 统一标记: 更新 unified_labels.json（重置为无状态或删除状态字段）
        self._update_unified_labels_status(char_ids, "pending")
        
        print(f"[Manager] 撤回跳过同步完成: {len(char_ids)} 个字符已同步到所有数据系统")
    
    def _update_unified_labels_status(self, char_ids: List[str], status: str):
        """更新 unified_labels.json 中字符的状态
        
        Args:
            char_ids: 字符ID列表
            status: 新状态（"labeled", "skipped"）
        """
        if not self.unified_labels_path.exists():
            return
        
        unified_data = self._load_json(self.unified_labels_path)
        if not unified_data or "annotations" not in unified_data:
            return
        
        char_id_set = set(char_ids)
        updated = False
        
        for ann in unified_data["annotations"]:
            if ann.get("char_id") in char_id_set:
                ann["status"] = status
                ann["updated_at"] = datetime.datetime.now().isoformat()
                updated = True
        
        if updated:
            self._save_json(self.unified_labels_path, unified_data)
            print(f"[Manager] 已更新 unified_labels.json 中 {len(char_id_set)} 个字符的状态为 '{status}'")