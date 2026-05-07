"""
迁移管理器 - 核心迁移逻辑
支持多源数据集标注迁移，以图片为单位进行处理
优化版本：添加预计算和缓存机制，支持标注共享
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from .registry import GlobalRegistry
from .matching import CharMatcher
from .sharing import LabelSharingManager


class MigrationManager:
    """标注迁移管理器 - 以图片为单位，支持预计算和缓存，集成标注共享"""
    
    def __init__(self):
        self.migration_dir = Path(__file__).parent
        self.data_dir = self.migration_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.datahome_dir = self.migration_dir.parent / "datahome"
        
        self.registry = GlobalRegistry(self.data_dir)
        self.sharing_manager = LabelSharingManager(self.data_dir)
        
        self.feature_db_path = self.data_dir / "char_feature_db.json"
        self.migration_map_path = self.data_dir / "char_migration_map.json"
        self.priority_list_path = self.data_dir / "char_priority_list.json"
        self.image_matches_path = self.data_dir / "image_matches.json"
        
        # 内存缓存
        self._cached_feature_db = None
        self._cached_image_matches = {}  # {target_dataset_id: {char: [matches]}}
    
    def _discover_labeled_datasets(self) -> List[str]:
        """发现所有已标注的数据集"""
        labeled_datasets = []
        
        if not self.datahome_dir.exists():
            return labeled_datasets
        
        for item in self.datahome_dir.iterdir():
            if item.is_dir() and item.name.startswith("pdf"):
                labels_path = item / "clusters" / "labeling" / "labels.json"
                if labels_path.exists():
                    labeled_datasets.append(item.name)
        
        return labeled_datasets
    
    def _extract_char_info_from_dataset(self, dataset_id: str) -> Dict[str, int]:
        """从单个数据集提取汉字信息"""
        char_info = {}
        dataset_dir = self.datahome_dir / dataset_id
        labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
        
        if not labels_path.exists():
            return char_info
        
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        for cluster_data in labels.values():
            if cluster_data.get("status") != "labeled":
                continue
            
            if "chars" in cluster_data:
                for char, count in cluster_data["chars"].items():
                    char_info[char] = char_info.get(char, 0) + count
            elif "char" in cluster_data:
                char = cluster_data["char"]
                char_info[char] = char_info.get(char, 0) + 1
        
        return char_info
    
    def build_feature_db(self, source_dataset_ids: Optional[List[str]] = None) -> Dict:
        """构建汉字特征库（以图片为单位）"""
        if source_dataset_ids is None:
            source_dataset_ids = self._discover_labeled_datasets()
        
        print(f"\n[INFO] 开始构建汉字特征库")
        print(f"       源数据集: {source_dataset_ids}")
        
        feature_db = CharMatcher.build_char_feature_db(self.datahome_dir, source_dataset_ids)
        
        with open(self.feature_db_path, "w", encoding="utf-8") as f:
            json.dump(feature_db, f, ensure_ascii=False, indent=2)
        
        # 更新内存缓存
        self._cached_feature_db = feature_db
        
        print(f"[SUCCESS] 汉字特征库已保存，共 {len(feature_db['characters'])} 个汉字")
        return feature_db
    
    def _load_feature_db(self) -> Dict:
        """加载汉字特征库（带缓存）"""
        if self._cached_feature_db is not None:
            return self._cached_feature_db
        
        if self.feature_db_path.exists():
            with open(self.feature_db_path, "r", encoding="utf-8") as f:
                self._cached_feature_db = json.load(f)
                return self._cached_feature_db
        
        return {"version": "2.0", "datasets": [], "characters": {}}
    
    def _get_target_unlabeled_chars(self, target_dataset_id: str) -> List[Dict]:
        """获取目标数据集所有未标注的字符图片"""
        unlabeled_chars = []
        
        dataset_dir = self.datahome_dir / target_dataset_id
        clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
        labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
        
        if not clusters_path.exists():
            return unlabeled_chars
        
        with open(clusters_path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
        
        # 加载已标注的字符信息
        labeled_cluster_ids = set()
        labeled_char_indices = {}  # {cluster_id: set of labeled indices}
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            
            for cluster_id, label_info in labels.items():
                # 检查整个聚类是否已标注
                if label_info.get("status") == "labeled":
                    labeled_cluster_ids.add(cluster_id)
                # 记录已标注的单个字符索引
                char_labels = label_info.get("char_labels", {})
                if char_labels:
                    labeled_char_indices[cluster_id] = set(char_labels.keys())
        
        for cluster_id, chars in clusters.get("clusters", {}).items():
            if cluster_id in labeled_cluster_ids:
                continue
            
            cluster_labeled_indices = labeled_char_indices.get(cluster_id, set())
            
            for idx, char_info in enumerate(chars):
                # 跳过已标注的单个字符
                if str(idx) in cluster_labeled_indices:
                    continue
                
                unlabeled_chars.append({
                    "char_id": char_info.get("char_id", ""),
                    "image_path": char_info.get("image_path", ""),
                    "cluster_id": cluster_id,
                    "lineage": char_info.get("lineage", {})
                })
        
        return unlabeled_chars
    
    def _load_image_matches(self, target_dataset_id: str) -> Optional[Dict]:
        """加载预计算的图片匹配结果"""
        if target_dataset_id in self._cached_image_matches:
            return self._cached_image_matches[target_dataset_id]
        
        if self.image_matches_path.exists():
            try:
                with open(self.image_matches_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "matches" in data:
                        self._cached_image_matches[target_dataset_id] = data["matches"]
                        return data["matches"]
                    else:
                        self._cached_image_matches[target_dataset_id] = data
                        return data
            except:
                pass
        
        return None
    
    def find_image_matches(self, target_dataset_id: str, char: str, top_n: int = 20, confidence_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
        """
        为指定汉字查找目标数据集中匹配的图片（优先使用缓存）
        
        Args:
            target_dataset_id: 目标数据集ID
            char: 要匹配的汉字
            top_n: 返回前N个匹配结果
            confidence_threshold: 置信度阈值，低于此值的标记为待确认
        
        Returns:
            Tuple: (确认匹配结果, 待确认匹配结果)
        """
        # 获取已标注的字符ID集合
        labeled_char_ids = self._get_labeled_char_ids(target_dataset_id)
        
        # 优先从缓存获取
        cached_matches = self._load_image_matches(target_dataset_id)
        if cached_matches and char in cached_matches:
            # 过滤已标注的字符
            all_results = [r for r in cached_matches[char] if r.get("char_id") not in labeled_char_ids][:top_n]
            confirmed = [r for r in all_results if r.get("similarity", 0) >= confidence_threshold]
            pending = [r for r in all_results if r.get("similarity", 0) < confidence_threshold]
            return confirmed, pending
        
        # 如果没有缓存，实时计算
        feature_db = self._load_feature_db()
        
        if char not in feature_db["characters"]:
            return [], []
        
        target_chars = self._get_target_unlabeled_chars(target_dataset_id)
        confirmed, pending = CharMatcher.find_matching_images(feature_db, target_chars, char, top_n, confidence_threshold)
        
        return confirmed, pending
    
    def _get_labeled_char_ids(self, target_dataset_id: str) -> set:
        """获取目标数据集中所有已标注的字符ID"""
        labeled_char_ids = set()
        
        dataset_dir = self.datahome_dir / target_dataset_id
        labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
        
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            
            for cluster_id, label_info in labels.items():
                char_labels = label_info.get("char_labels", {})
                if char_labels:
                    # 需要获取已标注字符的char_id
                    clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
                    if clusters_path.exists():
                        with open(clusters_path, "r", encoding="utf-8") as f:
                            clusters = json.load(f)
                        
                        cluster_chars = clusters.get("clusters", {}).get(cluster_id, [])
                        for idx_str in char_labels.keys():
                            idx = int(idx_str)
                            if idx < len(cluster_chars):
                                char_id = cluster_chars[idx].get("char_id", "")
                                if char_id:
                                    labeled_char_ids.add(char_id)
        
        return labeled_char_ids
    
    def precompute_all_image_matches(self, target_dataset_id: str, top_n: int = 10, confidence_threshold: float = 0.5):
        """
        预计算所有汉字的图片匹配结果（离线批处理）
        
        Args:
            target_dataset_id: 目标数据集ID
            top_n: 每个汉字返回前N个匹配结果
            confidence_threshold: 置信度阈值，低于此值的标记为待确认
        """
        print(f"\n[INFO] 开始预计算图片匹配（目标数据集: {target_dataset_id}）")
        
        feature_db = self._load_feature_db()
        target_chars = self._get_target_unlabeled_chars(target_dataset_id)
        
        print(f"       目标数据集未标注字符数: {len(target_chars)}")
        print(f"       待匹配汉字数: {len(feature_db['characters'])}")
        print(f"       置信度阈值: {confidence_threshold}")
        
        all_matches = {}
        total_chars = len(feature_db["characters"])
        processed = 0
        total_confirmed = 0
        total_pending = 0
        
        for char in feature_db["characters"]:
            confirmed, pending = CharMatcher.find_matching_images(feature_db, target_chars, char, top_n, confidence_threshold)
            all_results = confirmed + pending
            if all_results:
                all_matches[char] = all_results
                total_confirmed += len(confirmed)
                total_pending += len(pending)
            
            processed += 1
            if processed % 50 == 0:
                print(f"       已处理: {processed}/{total_chars} ({(processed/total_chars*100):.1f}%)")
        
        # 保存预计算结果（带数据集ID和时间戳）
        result = {
            "version": "2.0",
            "dataset_id": target_dataset_id,
            "generated_at": datetime.now().isoformat(),
            "confidence_threshold": confidence_threshold,
            "matches": all_matches,
            "stats": {
                "total_matched_chars": len(all_matches),
                "total_confirmed": total_confirmed,
                "total_pending": total_pending
            }
        }
        
        with open(self.image_matches_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 更新内存缓存
        self._cached_image_matches[target_dataset_id] = all_matches
        
        print(f"[SUCCESS] 预计算完成！匹配结果已保存到: {self.image_matches_path}")
        print(f"       匹配汉字数: {len(all_matches)}")
        print(f"       确认匹配数: {total_confirmed}")
        print(f"       待确认匹配数: {total_pending}")
        
        return all_matches
    
    def find_all_image_matches(self, target_dataset_id: str, top_n: int = 10) -> Dict:
        """
        为所有汉字查找目标数据集中匹配的图片（优先使用预计算）
        
        Args:
            target_dataset_id: 目标数据集ID
            top_n: 每个汉字返回前N个匹配结果
        
        Returns:
            {char: [matches]}
        """
        # 优先使用预计算结果
        cached_matches = self._load_image_matches(target_dataset_id)
        if cached_matches:
            return cached_matches
        
        # 如果没有预计算，实时计算（较慢）
        return self.precompute_all_image_matches(target_dataset_id, top_n)
    
    def _generate_priority_list(self, target_dataset_id: str):
        """生成优先级列表"""
        feature_db = self._load_feature_db()
        
        target_char_counts = self._get_target_char_counts(target_dataset_id)
        
        priorities = []
        for char, char_data in feature_db["characters"].items():
            source_count = char_data["total_samples"]
            target_count = target_char_counts.get(char, 0)
            
            priority_score = CharMatcher.calculate_priority(source_count, target_count)
            
            reason = ""
            action = ""
            if target_count == 0:
                if source_count < 5:
                    reason = "源数据集中样本极少，目标数据集未标注"
                    action = "优先标注"
                else:
                    reason = "源数据集中样本充足，目标数据集未标注"
                    action = "自动迁移"
            else:
                reason = "目标数据集已有少量标注"
                action = "补充标注"
            
            priorities.append({
                "char": char,
                "source_count": source_count,
                "target_count": target_count,
                "priority_score": priority_score,
                "reason": reason,
                "action": action
            })
        
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)
        
        priority_list = {
            "version": "2.0",
            "dataset_id": target_dataset_id,
            "generated_at": datetime.now().isoformat(),
            "priorities": priorities
        }
        
        with open(self.priority_list_path, "w", encoding="utf-8") as f:
            json.dump(priority_list, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 优先级列表已保存到: {self.priority_list_path}")
    
    def _get_target_char_counts(self, dataset_id: str) -> Dict[str, int]:
        """获取目标数据集的汉字计数（从预计算的匹配结果中获取）"""
        char_counts = {}
        
        # 首先从预计算的匹配结果中获取目标字符数量
        matches = self._load_image_matches(dataset_id)
        if matches:
            for char, char_matches in matches.items():
                char_counts[char] = len(char_matches)
        
        # 如果匹配结果为空，尝试从labels.json中获取已标注数量
        if not char_counts:
            dataset_dir = self.datahome_dir / dataset_id
            labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
            
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                
                for cluster_data in labels.values():
                    if cluster_data.get("status") != "labeled":
                        continue
                    
                    if "chars" in cluster_data:
                        for char, count in cluster_data["chars"].items():
                            char_counts[char] = char_counts.get(char, 0) + count
                    elif "char" in cluster_data:
                        char = cluster_data["char"]
                        char_counts[char] = char_counts.get(char, 0) + 1
        
        return char_counts
    
    def migrate_labels(self, target_dataset_id: str, source_dataset_ids: Optional[List[str]] = None):
        """执行标注迁移（包含预计算）"""
        print(f"\n[INFO] 开始执行标注迁移")
        print(f"       目标数据集: {target_dataset_id}")
        
        if source_dataset_ids is None:
            source_dataset_ids = self._discover_labeled_datasets()
            print(f"       自动发现源数据集: {source_dataset_ids}")
        else:
            print(f"       指定源数据集: {source_dataset_ids}")
        
        if not source_dataset_ids:
            print("[WARN] 未找到任何源数据集")
            return
        
        print(f"\n[STEP 1/4] 构建汉字特征库...")
        self.build_feature_db(source_dataset_ids)
        
        print(f"\n[STEP 2/4] 预计算图片匹配...")
        self.precompute_all_image_matches(target_dataset_id)
        
        print(f"\n[STEP 3/4] 更新全局注册表...")
        for dataset_id in source_dataset_ids:
            char_info = self._extract_char_info_from_dataset(dataset_id)
            self.registry.merge_from_dataset(dataset_id, char_info)
        
        print(f"\n[STEP 4/4] 生成优先级列表...")
        self._generate_priority_list(target_dataset_id)
        
        print(f"\n[SUCCESS] 标注迁移完成！")
    
    def generate_report(self, target_dataset_id: str) -> str:
        """生成迁移报告"""
        feature_db = self._load_feature_db()
        
        report = f"""# 标注迁移报告

## 概览
- 目标数据集: {target_dataset_id}
- 生成时间: {datetime.now().isoformat()}

## 源数据集
{chr(10).join([f"- {ds}" for ds in feature_db.get("datasets", [])])}

## 统计
| 项目 | 数量 |
|-----|-----|
| 源数据集汉字总数 | {len(feature_db.get("characters", {}))} |

## 优先级排名（前10）
"""
        
        if self.priority_list_path.exists():
            with open(self.priority_list_path, "r", encoding="utf-8") as f:
                priority_list = json.load(f)
            
            report += """
| 排名 | 汉字 | 源样本数 | 目标样本数 | 优先级 | 操作建议 |
|-----|-----|---------|-----------|-------|---------|
"""
            
            for i, item in enumerate(priority_list["priorities"][:10], 1):
                report += f"| {i} | {item['char']} | {item['source_count']} | {item['target_count']} | {item['priority_score']} | {item['action']} |\n"
        
        return report
    
    # ==================== 标注共享方法 ====================
    
    def share_labels(self, source_dataset_id: str, target_dataset_ids: List[str], merge_strategy: str = "merge") -> Dict:
        """
        跨数据集共享标注（整合版）
        
        Args:
            source_dataset_id: 源数据集ID
            target_dataset_ids: 目标数据集ID列表
            merge_strategy: 合并策略 ("replace", "merge", "overwrite")
        
        Returns:
            共享结果统计
        """
        print(f"\n[INFO] 开始跨数据集标注共享")
        print(f"       源数据集: {source_dataset_id}")
        print(f"       目标数据集: {target_dataset_ids}")
        print(f"       合并策略: {merge_strategy}")
        
        # 先确保源数据集标注已注册到全局注册表
        source_char_info = self._extract_char_info_from_dataset(source_dataset_id)
        if source_char_info:
            self.registry.merge_from_dataset(source_dataset_id, source_char_info)
            print(f"[INFO] 已更新全局注册表")
        
        # 执行共享
        result = self.sharing_manager.share_labels(source_dataset_id, target_dataset_ids, merge_strategy)
        
        if result.get("code") == 0:
            print(f"[SUCCESS] 标注共享完成")
            for target_result in result.get("results", []):
                print(f"         目标 {target_result['target_dataset']}: 导入 {target_result.get('imported_count', 0)} 个")
        
        return result
    
    def export_labels(self, dataset_id: str, description: str = "") -> str:
        """
        导出标注结果
        
        Args:
            dataset_id: 数据集ID
            description: 描述信息
        
        Returns:
            导出文件路径
        """
        print(f"\n[INFO] 导出标注结果 (数据集: {dataset_id})")
        
        # 读取标注数据
        labels_path = self.datahome_dir / dataset_id / "clusters" / "labeling" / "labels.json"
        
        if not labels_path.exists():
            print(f"[WARN] 标注文件不存在: {labels_path}")
            return ""
        
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        
        export_path = self.sharing_manager.export_labels(dataset_id, labels_data, description)
        print(f"[SUCCESS] 标注已导出到: {export_path}")
        
        return export_path
    
    def import_labels(self, source_file_path: str, target_dataset_id: str, merge_strategy: str = "merge") -> Dict:
        """
        导入标注结果到目标数据集
        
        Args:
            source_file_path: 源标注文件路径
            target_dataset_id: 目标数据集ID
            merge_strategy: 合并策略
        
        Returns:
            导入结果统计
        """
        print(f"\n[INFO] 导入标注结果")
        print(f"       源文件: {source_file_path}")
        print(f"       目标数据集: {target_dataset_id}")
        print(f"       合并策略: {merge_strategy}")
        
        result = self.sharing_manager.import_labels(source_file_path, target_dataset_id, merge_strategy)
        
        if result.get("code") == 0:
            print(f"[SUCCESS] 标注导入完成")
            print(f"         新增: {result.get('imported_count', 0)} 个")
            print(f"         更新: {result.get('updated_count', 0)} 个")
            print(f"         跳过: {result.get('skipped_count', 0)} 个")
        
        return result
    
    def get_all_datasets(self) -> List[str]:
        """获取所有数据集列表"""
        datasets = []
        
        if not self.datahome_dir.exists():
            return datasets
        
        for item in self.datahome_dir.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        
        return datasets
    
    def get_dataset_label_stats(self, dataset_id: str) -> Dict:
        """获取数据集标注统计"""
        dataset_dir = self.datahome_dir / dataset_id
        labels_path = dataset_dir / "clusters" / "labeling" / "labels.json"
        clusters_path = dataset_dir / "clusters" / "hog_clusters.json"
        
        total_clusters = 0
        labeled_clusters = 0
        total_chars = 0
        
        if clusters_path.exists():
            with open(clusters_path, "r", encoding="utf-8") as f:
                clusters_data = json.load(f)
                total_clusters = len(clusters_data.get("clusters", {}))
        
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
                for cluster_id, cluster_data in labels_data.items():
                    if cluster_data.get("status") == "labeled":
                        labeled_clusters += 1
                        if "chars" in cluster_data:
                            total_chars += sum(cluster_data["chars"].values())
                        elif "char" in cluster_data:
                            total_chars += 1
        
        return {
            "dataset_id": dataset_id,
            "total_clusters": total_clusters,
            "labeled_clusters": labeled_clusters,
            "labeled_ratio": round(labeled_clusters / max(total_clusters, 1) * 100, 2),
            "total_chars": total_chars
        }