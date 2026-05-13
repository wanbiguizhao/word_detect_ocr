import json
import datetime
from pathlib import Path
from typing import List, Set, Dict, Optional


class CharPoolManager:
    """字符池管理器 - 完全独立于现有聚类系统"""
    
    def __init__(self, dataset_id: str = "pdf5826"):
        self.dataset_id = dataset_id
        # 项目根目录: d:\projects\word_detect_ocr
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.base_dir = self.project_root / "bussiness" / "datahome" / dataset_id / "multi_clustering"
        self.char_pool_dir = self.base_dir / "char_pool"
        
        # 确保目录存在
        self.char_pool_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.all_chars_path = self.char_pool_dir / "all_chars.json"
        self.labeled_path = self.char_pool_dir / "labeled.json"
        self.unlabeled_path = self.char_pool_dir / "unlabeled.json"
        
        # 缓存
        self._all_chars_cache: Optional[Dict[str, dict]] = None
        self._labeled_cache: Optional[Set[str]] = None
        self._unlabeled_cache: Optional[List[str]] = None
    
    def _load_json(self, path: Path):
        """加载JSON文件（处理编码错误）"""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except UnicodeDecodeError:
                # 尝试使用替换模式处理编码错误
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    # 移除无效字符
                    content = content.replace('\ufffd', '')
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return None
        return None
    
    def _save_json(self, path: Path, data: dict):
        """保存JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_all_chars(self) -> Dict[str, dict]:
        """加载所有字符信息"""
        if self._all_chars_cache is None:
            data = self._load_json(self.all_chars_path)
            self._all_chars_cache = data.get("chars", {}) if data else {}
        return self._all_chars_cache
    
    def get_labeled_char_ids(self) -> Set[str]:
        """获取已标注的字符ID集合"""
        if self._labeled_cache is None:
            data = self._load_json(self.labeled_path)
            self._labeled_cache = set(data.get("char_ids", [])) if data else set()
        return self._labeled_cache
    
    def get_unlabeled_char_ids(self) -> List[str]:
        """获取未标注的字符ID列表（候选池）"""
        if self._unlabeled_cache is None:
            data = self._load_json(self.unlabeled_path)
            self._unlabeled_cache = data.get("char_ids", []) if data else []
        return self._unlabeled_cache
    
    def init_from_lineage(self, lineage_path: Optional[Path] = None):
        """从lineage.json初始化字符池，并排除已标注的字符"""
        if lineage_path is None:
            lineage_path = self.project_root / "bussiness" / "datahome" / self.dataset_id / "lineage.json"
        
        if not lineage_path.exists():
            raise FileNotFoundError(f"lineage.json 不存在: {lineage_path}")
        
        with open(lineage_path, 'r', encoding='utf-8') as f:
            lineage_data = json.load(f)
        
        # 获取已标注的字符ID集合（从现有聚类系统中）
        labeled_char_ids = self._get_existing_labeled_chars()
        
        chars = lineage_data.get("chars", {})
        
        # 构建字符信息，排除已标注的
        all_chars = {}
        unlabeled_ids = []
        
        for char_id, char_info in chars.items():
            all_chars[char_id] = {
                "char_id": char_id,
                "line_name": char_info.get("line_name", ""),
                "col_start": char_info.get("col_start", 0),
                "col_end": char_info.get("col_end", 0),
                "status": "labeled" if char_id in labeled_char_ids else "unlabeled",
                "labeled_round": None if char_id not in labeled_char_ids else 0,
                "labeled_char": labeled_char_ids.get(char_id) if char_id in labeled_char_ids else None,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # 只添加未标注的字符到候选池
            if char_id not in labeled_char_ids:
                unlabeled_ids.append(char_id)
        
        # 保存
        self._save_json(self.all_chars_path, {
            "version": "1.0",
            "total_count": len(all_chars),
            "labeled_count": len(all_chars) - len(unlabeled_ids),
            "unlabeled_count": len(unlabeled_ids),
            "created_at": datetime.datetime.now().isoformat(),
            "chars": all_chars
        })
        
        self._save_json(self.labeled_path, {
            "version": "1.0",
            "count": len(all_chars) - len(unlabeled_ids),
            "char_ids": list(labeled_char_ids.keys())
        })
        
        self._save_json(self.unlabeled_path, {
            "version": "1.0",
            "count": len(unlabeled_ids),
            "char_ids": unlabeled_ids
        })
        
        # 更新缓存
        self._all_chars_cache = all_chars
        self._labeled_cache = set(labeled_char_ids.keys())
        self._unlabeled_cache = unlabeled_ids
        
        return len(all_chars)
    
    def _get_existing_labeled_chars(self) -> Dict[str, str]:
        """从现有聚类系统和统一标注中获取已标注的字符映射 {char_id: char}"""
        labeled_chars = {}
        
        # 1. 从统一标注加载（优先级最高）
        try:
            from datastore.data_store import DataStore
            ds = DataStore(self.dataset_id)
            confirmed = ds.get_confirmed_annotations()
            labeled_chars.update(confirmed)
            print(f"从统一标注加载已标注字符: {len(confirmed)}")
        except Exception as e:
            print(f"从统一标注加载失败: {e}")
        
        # 2. 从聚类系统加载（作为补充）
        clusters_dir = self.project_root / "bussiness" / "datahome" / self.dataset_id / "clusters"
        hog_clusters_path = clusters_dir / "hog_clusters.json"
        labels_path = clusters_dir / "labeling" / "labels.json"
        
        if hog_clusters_path.exists() and labels_path.exists():
            try:
                with open(hog_clusters_path, 'r', encoding='utf-8') as f:
                    clusters_data = json.load(f)
                
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
                
                clusters = clusters_data.get("clusters", {})
                cluster_labeled_count = 0
                
                for cluster_id, cluster_chars in clusters.items():
                    label_info = labels_data.get(cluster_id, {})
                    if label_info.get("status") == "labeled" and label_info.get("char_labels"):
                        cluster_label = label_info.get("char", "")
                        if not cluster_label:
                            continue
                        
                        for idx, char_label_info in label_info["char_labels"].items():
                            if char_label_info.get("char"):
                                cluster_label = char_label_info["char"]
                            
                            try:
                                idx_int = int(idx)
                                if idx_int < len(cluster_chars):
                                    char_id = cluster_chars[idx_int].get("char_id", "")
                                    if char_id and char_id not in labeled_chars:
                                        labeled_chars[char_id] = cluster_label
                                        cluster_labeled_count += 1
                            except (ValueError, IndexError):
                                continue
                
                print(f"从聚类系统加载已标注字符: {cluster_labeled_count}")
            except Exception as e:
                print(f"从聚类系统加载失败: {e}")
        
        return labeled_chars
    
    def mark_as_labeled(self, char_id: str, char: str, round_num: int):
        """标记字符为已标注"""
        # 更新 all_chars
        all_chars = self.load_all_chars()
        if char_id in all_chars:
            all_chars[char_id]["status"] = "labeled"
            all_chars[char_id]["labeled_round"] = round_num
            all_chars[char_id]["labeled_char"] = char
            all_chars[char_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # 更新 labeled
        labeled_ids = self.get_labeled_char_ids()
        labeled_ids.add(char_id)
        
        # 更新 unlabeled
        unlabeled_ids = self.get_unlabeled_char_ids()
        if char_id in unlabeled_ids:
            unlabeled_ids.remove(char_id)
        
        # 保存
        self._save_json(self.all_chars_path, {
            "version": "1.0",
            "total_count": len(all_chars),
            "updated_at": datetime.datetime.now().isoformat(),
            "chars": all_chars
        })
        
        self._save_json(self.labeled_path, {
            "version": "1.0",
            "count": len(labeled_ids),
            "updated_at": datetime.datetime.now().isoformat(),
            "char_ids": list(labeled_ids)
        })
        
        self._save_json(self.unlabeled_path, {
            "version": "1.0",
            "count": len(unlabeled_ids),
            "updated_at": datetime.datetime.now().isoformat(),
            "char_ids": unlabeled_ids
        })
        
        return True
    
    def mark_as_skipped(self, char_id: str, round_num: int):
        """标记字符为跳过"""
        all_chars = self.load_all_chars()
        if char_id in all_chars:
            all_chars[char_id]["status"] = "skipped"
            all_chars[char_id]["labeled_round"] = round_num
            all_chars[char_id]["updated_at"] = datetime.datetime.now().isoformat()
            
            # 保存
            self._save_json(self.all_chars_path, {
                "version": "1.0",
                "total_count": len(all_chars),
                "updated_at": datetime.datetime.now().isoformat(),
                "chars": all_chars
            })
            
            # 从未标注列表移除
            unlabeled_ids = self.get_unlabeled_char_ids()
            if char_id in unlabeled_ids:
                unlabeled_ids.remove(char_id)
                self._save_json(self.unlabeled_path, {
                    "version": "1.0",
                    "count": len(unlabeled_ids),
                    "updated_at": datetime.datetime.now().isoformat(),
                    "char_ids": unlabeled_ids
                })
        
        return True
    
    def reset_char(self, char_id: str):
        """重置字符状态为未标注"""
        all_chars = self.load_all_chars()
        if char_id in all_chars:
            all_chars[char_id]["status"] = "unlabeled"
            all_chars[char_id]["labeled_round"] = None
            all_chars[char_id]["labeled_char"] = None
            all_chars[char_id]["updated_at"] = datetime.datetime.now().isoformat()
            
            # 保存
            self._save_json(self.all_chars_path, {
                "version": "1.0",
                "total_count": len(all_chars),
                "updated_at": datetime.datetime.now().isoformat(),
                "chars": all_chars
            })
            
            # 更新 labeled 和 unlabeled
            labeled_ids = self.get_labeled_char_ids()
            if char_id in labeled_ids:
                labeled_ids.remove(char_id)
                self._save_json(self.labeled_path, {
                    "version": "1.0",
                    "count": len(labeled_ids),
                    "updated_at": datetime.datetime.now().isoformat(),
                    "char_ids": list(labeled_ids)
                })
            
            unlabeled_ids = self.get_unlabeled_char_ids()
            if char_id not in unlabeled_ids:
                unlabeled_ids.append(char_id)
                self._save_json(self.unlabeled_path, {
                    "version": "1.0",
                    "count": len(unlabeled_ids),
                    "updated_at": datetime.datetime.now().isoformat(),
                    "char_ids": unlabeled_ids
                })
        
        return True
    
    def get_stats(self) -> dict:
        """获取字符池统计信息"""
        all_chars = self.load_all_chars()
        labeled_ids = self.get_labeled_char_ids()
        unlabeled_ids = self.get_unlabeled_char_ids()
        
        status_counts = {}
        for char_info in all_chars.values():
            status = char_info.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total": len(all_chars),
            "labeled": len(labeled_ids),
            "unlabeled": len(unlabeled_ids),
            "status_counts": status_counts
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._all_chars_cache = None
        self._labeled_cache = None
        self._unlabeled_cache = None