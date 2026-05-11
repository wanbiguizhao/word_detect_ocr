import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import hdbscan
import numpy as np
from .char_pool_manager import CharPoolManager


class MultiClusteringManager:
    """多轮聚类管理器 - 完全独立于现有聚类系统"""
    
    def __init__(self, dataset_id: str = "pdf5823"):
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
    
    def get_current_round(self) -> int:
        """获取当前最新轮次"""
        history = self._load_json(self.history_path)
        if history and history.get("rounds"):
            return max(r["round"] for r in history["rounds"])
        return 0
    
    def start_new_round(self, n_clusters: Optional[int] = None, 
                        description: str = "") -> int:
        """启动新一轮聚类"""
        # 获取未标注字符
        unlabeled_ids = self.char_pool.get_unlabeled_char_ids()
        
        # 如果字符池为空，尝试自动初始化
        if not unlabeled_ids:
            try:
                self.char_pool.init_from_lineage()
                unlabeled_ids = self.char_pool.get_unlabeled_char_ids()
            except Exception as e:
                raise ValueError(f"字符池为空且初始化失败: {str(e)}")
        
        if not unlabeled_ids:
            raise ValueError("没有未标注字符，无法启动新轮聚类")
        
        # 设置聚类参数
        if n_clusters is None:
            n_clusters = self.config["clustering"]["default_n_clusters"]
        
        # 限制聚类数量不超过字符数量
        n_clusters = min(n_clusters, len(unlabeled_ids))
        
        if n_clusters < 1:
            raise ValueError("聚类数量必须大于0")
        
        # 获取字符特征
        features_dict = {}
        valid_char_ids = []
        
        for char_id in unlabeled_ids:
            img_path = self.project_root / "bussiness" / "datahome" / self.dataset_id / "pdf_chars" / f"{char_id}.png"
            if img_path.exists():
                feat = self._extract_hog_features(str(img_path))
                if feat is not None:
                    features_dict[char_id] = feat
                    valid_char_ids.append(char_id)
        
        if not valid_char_ids:
            raise ValueError("无法提取任何字符特征")
        
        # 执行HDBSCAN聚类
        features = np.array([features_dict[char_id] for char_id in valid_char_ids])
        
        try:
            # 计算余弦距离矩阵（1 - 余弦相似度）
            from scipy.spatial.distance import squareform, pdist
            cosine_dist_matrix = squareform(pdist(features, 'cosine'))
            
            # HDBSCAN 参数说明：
            # min_cluster_size: 最小聚类大小
            # min_samples: 每个核心点周围的最小样本数
            # metric: 使用预计算的距离矩阵
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                metric='precomputed',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(cosine_dist_matrix)
        except Exception as e:
            raise RuntimeError(f"聚类失败: {e}")
        
        # 构建聚类结果（过滤噪声点，label=-1表示噪声）
        clusters: Dict[str, List[dict]] = {}
        for idx, char_id in enumerate(valid_char_ids):
            cluster_id = labels[idx]
            if cluster_id == -1:
                continue  # 跳过噪声点
            cluster_id_str = str(cluster_id)
            if cluster_id_str not in clusters:
                clusters[cluster_id_str] = []
            
            # 获取字符信息
            char_info = self.char_pool.load_all_chars().get(char_id, {})
            clusters[cluster_id_str].append({
                "char_id": char_id,
                "line_name": char_info.get("line_name", ""),
                "col_start": char_info.get("col_start", 0),
                "col_end": char_info.get("col_end", 0)
            })
        
        # 创建轮次目录
        new_round = self.get_current_round() + 1
        round_dir = self.rounds_dir / f"round_{new_round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存聚类结果
        clusters_data = {
            "version": "1.0",
            "round": new_round,
            "algorithm": self.config["clustering"]["algorithm"],
            "n_clusters": n_clusters,
            "total_chars": len(valid_char_ids),
            "created_at": datetime.datetime.now().isoformat(),
            "clusters": clusters
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
            "algorithm": self.config["clustering"]["algorithm"],
            "n_clusters": n_clusters,
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
        """保存标注"""
        round_dir = self.rounds_dir / f"round_{round_num}"
        labels_path = round_dir / "labels.json"
        
        if not labels_path.exists():
            return False
        
        labels_data = self._load_json(labels_path)
        
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
        
        # 保存标注
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
        
        # 保存标注
        self._save_json(labels_path, labels_data)
        
        # 获取字符ID并更新字符池
        clusters_data = self.get_round_clusters(round_num)
        if clusters_data:
            cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
            if char_index < len(cluster_chars):
                char_id = cluster_chars[char_index].get("char_id")
                if char_id:
                    self.char_pool.mark_as_labeled(char_id, char, round_num)
                    
                    # 同步到统一标注
                    if self.config.get("sync", {}).get("auto_sync_to_unified", True):
                        self._sync_to_unified(char_id, char)
        
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
        """批量保存标注"""
        saved_count = 0
        for label in labels:
            char_index = label.get("charIndex")
            char = label.get("char")
            if char_index is not None and char:
                if self.save_label(round_num, cluster_id, char_index, char):
                    saved_count += 1
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
        
        # 将该聚类的所有字符标记为跳过
        clusters_data = self.get_round_clusters(round_num)
        if clusters_data:
            cluster_chars = clusters_data.get("clusters", {}).get(cluster_id, [])
            for char_info in cluster_chars:
                char_id = char_info.get("char_id")
                if char_id:
                    self.char_pool.mark_as_skipped(char_id, round_num)
        
        return True