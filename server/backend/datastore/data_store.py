"""统一数据存储抽象层 - 支持数据集隔离、事务机制、双向同步"""
import json
import logging
import datetime
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, dataset_id: Optional[str] = None):
        if dataset_id is None:
            from config import DATASET_ID
            dataset_id = DATASET_ID
        
        self.dataset_id = dataset_id
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.dataset_dir = self.project_root / "bussiness" / "datahome" / dataset_id
        
        # 核心路径配置
        self.prelabels_path = self.dataset_dir / "pre_labels.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        self.label_history_path = self.dataset_dir / "label_history.json"
        
        # 索引路径
        self.meta_dir = self.dataset_dir / ".meta"
        self.prelabel_status_path = self.meta_dir / "prelabel_status.json"
        self.char_to_images_path = self.meta_dir / "index" / "char_to_images.json"
        self.image_to_char_path = self.meta_dir / "index" / "image_to_char.json"
        self.char_to_prelabels_path = self.meta_dir / "index" / "char_to_prelabels.json"
        self.char_id_to_prelabel_path = self.meta_dir / "index" / "char_id_to_prelabel.json"
        
        # WAL日志路径
        self.wal_dir = self.meta_dir / "wal"
        
        # 多轮聚类路径
        self.multi_clustering_dir = self.dataset_dir / "multi_clustering"
        
        # 缓存（按数据集隔离）- 必须在 _init_structure 之前初始化
        self._cache = {}
        self._last_modified = {}
        
        # 初始化必要的目录和文件
        self._init_structure()
        
        # 日志记录器
        from .sync_logger import SyncLogger
        self._logger = SyncLogger(dataset_id)
        
        # 预加载索引到内存（只读）
        self._preload_indexes()
    
    @classmethod
    def from_config(cls):
        """从配置文件创建 DataStore 实例"""
        from config import DATASET_ID
        return cls(DATASET_ID)
    
    def _init_structure(self):
        """初始化数据存储结构"""
        # 创建目录
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        (self.meta_dir / "index").mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统一标注文件
        self._init_unified_labels()
        
        # 初始化标注历史文件
        self._init_label_history()
        
        # 如果预标注文件存在，初始化状态和索引文件
        if self.prelabels_path.exists():
            self._init_prelabel_status()
            self._build_prelabel_indexes()
    
    def _init_unified_labels(self):
        """初始化数据集的统一标注文件"""
        if not self.unified_labels_path.exists():
            init_data = {
                "dataset": self.dataset_id,
                "total_labeled": 0,
                "char_distribution": {},
                "annotations": []
            }
            self._save_file(self.unified_labels_path, init_data)
    
    def _init_label_history(self):
        """初始化标注变更历史文件"""
        if not self.label_history_path.exists():
            init_data = {
                "dataset": self.dataset_id,
                "history": []
            }
            self._save_file(self.label_history_path, init_data)
    
    def _init_prelabel_status(self):
        """初始化预标注状态文件"""
        if not self.prelabel_status_path.exists():
            init_data = {
                "dataset": self.dataset_id,
                "version": "1.0",
                "updated_at": datetime.datetime.now().isoformat(),
                "status": {},
                "corrected_chars": {}
            }
            self._save_file(self.prelabel_status_path, init_data)
    
    def _build_prelabel_indexes(self):
        """构建预标注索引（只在首次运行或重建时调用）"""
        # 如果索引已存在且最新，跳过
        if self.char_to_prelabels_path.exists() and self.char_id_to_prelabel_path.exists():
            return
        
        prelabels = self._load_file(self.prelabels_path).get("prelabels", [])
        
        # 构建 char -> char_ids 索引
        char_to_prelabels = defaultdict(list)
        # 构建 char_id -> index 索引
        char_id_to_prelabel = {}
        
        for idx, prelabel in enumerate(prelabels):
            char_id = prelabel.get("char_id")
            predicted_char = prelabel.get("predicted_char")
            
            if char_id:
                char_id_to_prelabel[char_id] = idx
            
            if predicted_char and char_id:
                char_to_prelabels[predicted_char].append(char_id)
        
        # 保存索引
        self._save_file(self.char_to_prelabels_path, dict(char_to_prelabels))
        self._save_file(self.char_id_to_prelabel_path, char_id_to_prelabel)
    
    def _preload_indexes(self):
        """预加载索引到内存（只读）"""
        # 预加载 pre_labels.json（只读，不修改）
        if self.prelabels_path.exists():
            self._prelabels = self._load_file(self.prelabels_path)
            self._prelabels_list = self._prelabels.get("prelabels", [])
        else:
            self._prelabels = {}
            self._prelabels_list = []
        
        # 预加载索引
        self._char_to_prelabels = self._load_file(self.char_to_prelabels_path)
        self._char_id_to_prelabel = self._load_file(self.char_id_to_prelabel_path)
        
        # 加载状态（可写）
        self._prelabel_status = self._load_file(self.prelabel_status_path)
        self._prelabel_status_dirty = False
    
    def _get_cache_key(self, path: Path) -> str:
        """生成带数据集隔离的缓存键"""
        return f"{self.dataset_id}_{str(path)}"
    
    def _load_file(self, path: Path, fallback_path: Optional[Path] = None, default: dict = None) -> dict:
        """加载 JSON 文件，支持回退路径，缓存按数据集隔离"""
        if default is None:
            default = {}
            
        cache_key = self._get_cache_key(path)
        
        if cache_key in self._cache:
            if path.exists() and self._last_modified.get(cache_key) == path.stat().st_mtime:
                return self._cache[cache_key]
        
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 处理空文件
                    if not content.strip():
                        return default
                    data = json.loads(content)
            except UnicodeDecodeError:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    content = content.replace('\ufffd', '')
                    if not content.strip():
                        return default
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        return default
            except (json.JSONDecodeError, StopIteration):
                return default
            
            self._last_modified[cache_key] = path.stat().st_mtime
            self._cache[cache_key] = data
            return data
        
        if fallback_path and fallback_path.exists():
            try:
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        return default
                    data = json.loads(content)
            except UnicodeDecodeError:
                with open(fallback_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    content = content.replace('\ufffd', '')
                    if not content.strip():
                        return default
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        return default
            except (json.JSONDecodeError, StopIteration):
                return default
            
            self._cache[cache_key] = data
            return data
        
        return default
    
    def _save_file(self, path: Path, data: dict, use_lock: bool = True):
        """保存 JSON 文件，更新缓存（带文件锁保护）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        lock_path = path.parent / f".{path.name}.lock"
        acquired = False
        
        try:
            # 跨平台文件锁
            if use_lock:
                max_wait = 10  # 最大等待10秒
                wait_interval = 0.1
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    try:
                        # 使用独占模式打开锁文件
                        lock_file = open(lock_path, 'x')
                        lock_file.write(str(os.getpid()))
                        lock_file.close()
                        acquired = True
                        break
                    except FileExistsError:
                        # 检查锁文件是否过期（超过30秒）
                        try:
                            if lock_path.exists():
                                lock_age = time.time() - lock_path.stat().st_mtime
                                if lock_age > 30:
                                    # 锁文件已过期，尝试删除
                                    lock_path.unlink()
                                    logger.warning(f"检测到过期锁文件并删除: {lock_path}")
                        except Exception as e:
                            pass
                        time.sleep(wait_interval)
                else:
                    raise TimeoutError(f"无法获取文件锁: {lock_path}")
            
            # 使用原子写入：先写临时文件，再重命名
            tmp_path = path.parent / f".{path.name}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Windows 上 os.replace 可能因为文件被占用而失败，添加重试和回退
            max_retries = 3
            retry_delay = 0.1
            replaced = False
            
            for attempt in range(max_retries):
                try:
                    os.replace(str(tmp_path), str(path))
                    replaced = True
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        # 最后一次尝试失败，使用直接写入作为回退
                        logger.warning(f"原子写入失败，使用直接写入回退: {e}")
                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        # 删除临时文件
                        if tmp_path.exists():
                            tmp_path.unlink()
            
            cache_key = self._get_cache_key(path)
            self._cache[cache_key] = data
            if path.exists():
                self._last_modified[cache_key] = path.stat().st_mtime
        finally:
            if acquired and lock_path.exists():
                try:
                    lock_path.unlink()
                except Exception as e:
                    logger.warning(f"无法删除锁文件: {lock_path}, 错误: {e}")
    
    # ==================== 原子事务机制 ====================
    
    def _write_wal(self, operation: str, data: dict):
        """写入WAL日志（用于恢复）"""
        wal_entry = {
            "id": f"wal_{datetime.datetime.now().timestamp()}",
            "operation": operation,
            "data": data,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": self.dataset_id
        }
        
        wal_path = self.wal_dir / f"wal_{int(datetime.datetime.now().timestamp())}.json"
        with open(wal_path, 'w', encoding='utf-8') as f:
            json.dump(wal_entry, f, ensure_ascii=False)
    
    def _atomic_commit(self, file_pairs: List[tuple]):
        """原子提交（一次性提交所有变更）"""
        import os
        
        for tmp_path, target_path in file_pairs:
            tmp_path = Path(tmp_path)
            target_path = Path(target_path)
            
            if tmp_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # 使用原子重命名
                os.replace(str(tmp_path), str(target_path))
    
    def _cleanup_wal(self):
        """清理WAL日志"""
        for wal_file in self.wal_dir.iterdir():
            if wal_file.name.startswith("wal_"):
                wal_file.unlink()
    
    def _write_tmp(self, name: str, data: dict) -> str:
        """写入临时文件，返回临时文件路径"""
        tmp_path = self.meta_dir / f"{name}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(tmp_path)
    
    # ==================== 标注历史记录 ====================
    
    def _add_history_record(self, char_id: str, old_char: Optional[str], new_char: str, 
                            changed_by: str, change_source: str, comment: str = ""):
        """添加标注变更历史记录"""
        history_data = self._load_file(self.label_history_path)
        
        new_record = {
            "id": f"hist_{len(history_data.get('history', [])) + 1}",
            "char_id": char_id,
            "old_char": old_char,
            "new_char": new_char,
            "changed_by": changed_by,
            "change_source": change_source,
            "timestamp": datetime.datetime.now().isoformat(),
            "ip": "127.0.0.1",
            "comment": comment
        }
        
        if "history" not in history_data:
            history_data["history"] = []
        history_data["history"].append(new_record)
        
        self._save_file(self.label_history_path, history_data)
        return new_record
    
    def get_label_history(self, char_id: Optional[str] = None, limit: int = 100) -> list:
        """获取标注变更历史"""
        history_data = self._load_file(self.label_history_path)
        history = history_data.get("history", [])
        
        if char_id:
            history = [h for h in history if h.get("char_id") == char_id]
        
        return history[-limit:]
    
    # ==================== 预标注数据（优化方案：只读预测 + 单独状态）====================
    
    def get_prelabels_by_char(self, char: str) -> list:
        """按汉字查询预标注（毫秒级，使用索引）"""
        char_ids = self._char_to_prelabels.get(char, [])
        result = []
        
        for char_id in char_ids:
            idx = self._char_id_to_prelabel.get(char_id)
            if idx is not None and idx < len(self._prelabels_list):
                prelabel = self._prelabels_list[idx]
                status = self._prelabel_status.get("status", {}).get(char_id, "pending")
                corrected_char = self._prelabel_status.get("corrected_chars", {}).get(char_id)
                
                result.append({
                    **prelabel,
                    "status": status,
                    "corrected_char": corrected_char
                })
        
        return result
    
    def get_prelabel_by_char_id(self, char_id: str) -> Optional[dict]:
        """按 char_id 查询预标注"""
        idx = self._char_id_to_prelabel.get(char_id)
        if idx is not None and idx < len(self._prelabels_list):
            prelabel = self._prelabels_list[idx]
            status = self._prelabel_status.get("status", {}).get(char_id, "pending")
            corrected_char = self._prelabel_status.get("corrected_chars", {}).get(char_id)
            
            return {
                **prelabel,
                "status": status,
                "corrected_char": corrected_char
            }
        return None
    
    def batch_confirm_prelabels(self, char_ids: list):
        """批量确认预标注（仅更新状态文件，几百KB IO）"""
        for char_id in char_ids:
            if "status" not in self._prelabel_status:
                self._prelabel_status["status"] = {}
            self._prelabel_status["status"][char_id] = "confirmed"
        
        self._prelabel_status["updated_at"] = datetime.datetime.now().isoformat()
        self._prelabel_status_dirty = True
        self._flush_prelabel_status()
    
    def correct_prelabel_char(self, char_id: str, new_char: str):
        """修正OCR预测字符"""
        if "corrected_chars" not in self._prelabel_status:
            self._prelabel_status["corrected_chars"] = {}
        self._prelabel_status["corrected_chars"][char_id] = new_char
        self._prelabel_status["updated_at"] = datetime.datetime.now().isoformat()
        self._prelabel_status_dirty = True
        self._flush_prelabel_status()
    
    def _flush_prelabel_status(self):
        """只刷状态文件（小！）"""
        self._save_file(self.prelabel_status_path, self._prelabel_status)
        self._prelabel_status_dirty = False
    
    def get_prelabel_stats(self) -> dict:
        """获取预标注统计"""
        if not self.prelabels_path.exists():
            return {"total": 0, "char_counts": {}}
        
        stats = self._prelabels.get("stats", {})
        char_counts = self._prelabels.get("char_counts", {})
        
        # 计算已确认数量
        confirmed_count = sum(1 for v in self._prelabel_status.get("status", {}).values() 
                             if v == "confirmed")
        
        return {
            "total": stats.get("total", 0),
            "confirmed": confirmed_count,
            "pending": stats.get("total", 0) - confirmed_count,
            "char_counts": char_counts
        }
    
    # ==================== 统一标注数据 ====================
    
    def get_unified_labels(self, status: Optional[str] = None) -> list:
        """获取统一标注数据"""
        data = self._load_file(self.unified_labels_path)
        annotations = data.get("annotations", [])
        
        if status:
            return [a for a in annotations if a.get("status") == status]
        return annotations
    
    def add_unified_label(self, annotation: dict):
        """添加统一标注"""
        data = self._load_file(self.unified_labels_path)
        
        if "annotations" not in data:
            data["annotations"] = []
        
        data["annotations"].append(annotation)
        self._save_file(self.unified_labels_path, data)
    
    def update_unified_label(self, char_id: str, updates: dict):
        """更新统一标注，如果不存在则添加"""
        data = self._load_file(self.unified_labels_path)
        
        if "annotations" not in data:
            data["annotations"] = []
        
        annotations = data["annotations"]
        found = False
        old_char = None
        
        for ann in annotations:
            if ann.get("char_id") == char_id:
                old_char = ann.get("char")
                ann.update(updates)
                if "dataset" not in ann:
                    ann["dataset"] = self.dataset_id
                found = True
                break
        
        if not found:
            new_ann = {"char_id": char_id, "dataset": self.dataset_id, **updates}
            annotations.append(new_ann)
        
        self._save_file(self.unified_labels_path, data)
        return old_char
    
    def get_confirmed_annotations(self) -> Dict[str, str]:
        """获取已确认的标注（char_id -> char）"""
        annotations = self.get_unified_labels()
        return {
            ann["char_id"]: ann["char"]
            for ann in annotations
            if ann.get("char_id") and ann.get("char") and ann.get("status") == "labeled"
        }
    
    # ==================== 双向索引 ====================
    
    def get_char_to_images(self) -> Dict[str, List[str]]:
        """获取汉字→图片索引"""
        return self._load_file(self.char_to_images_path, {})
    
    def get_image_to_char(self) -> Dict[str, str]:
        """获取图片→汉字索引"""
        return self._load_file(self.image_to_char_path, {})
    
    def _update_indexes(self, char_id: str, old_char: Optional[str], new_char: str):
        """更新双向索引"""
        # 加载当前索引
        char_to_images = self._load_file(self.char_to_images_path, defaultdict(list))
        image_to_char = self._load_file(self.image_to_char_path, {})
        
        # 如果有旧字符，从索引中移除
        if old_char and char_id in image_to_char and image_to_char[char_id] == old_char:
            if old_char in char_to_images and char_id in char_to_images[old_char]:
                char_to_images[old_char].remove(char_id)
                if not char_to_images[old_char]:
                    del char_to_images[old_char]
            del image_to_char[char_id]
        
        # 添加新字符到索引
        if new_char:
            if new_char not in char_to_images:
                char_to_images[new_char] = []
            if char_id not in char_to_images[new_char]:
                char_to_images[new_char].append(char_id)
            image_to_char[char_id] = new_char
        
        # 保存索引
        self._save_file(self.char_to_images_path, dict(char_to_images))
        self._save_file(self.image_to_char_path, image_to_char)
    
    def _validate_update(self, char_id: str, new_char: str) -> bool:
        """验证更新：防止一张图片对应多个汉字"""
        image_to_char = self.get_image_to_char()
        current_char = image_to_char.get(char_id)
        
        if current_char and current_char != new_char:
            raise ValueError(
                f"图片 {char_id} 已有标签 '{current_char}', 不能改为 '{new_char}'"
            )
        
        return True
    
    # ==================== 双向数据同步 ====================
    
    def _sync_to_cluster_labels(self, char_id: str, char: str):
        """同步标注到聚类标注文件"""
        clusters_path = self.dataset_dir / "clusters" / "hog_clusters.json"
        labels_path = self.dataset_dir / "clusters" / "labeling" / "labels.json"
        
        if not clusters_path.exists():
            self._logger.log_sync_skipped(char_id, char, "聚类数据文件不存在")
            return
        
        clusters_data = self._load_file(clusters_path)
        clusters = clusters_data.get("clusters", {})
        labels = self._load_file(labels_path)
        
        for cluster_id, chars_in_cluster in clusters.items():
            for idx, char_info in enumerate(chars_in_cluster):
                if char_info.get("char_id") == char_id:
                    if cluster_id not in labels:
                        labels[cluster_id] = {
                            "char": char,
                            "status": "labeled",
                            "char_labels": {}
                        }
                    labels[cluster_id]["char"] = char
                    labels[cluster_id]["status"] = "labeled"
                    if "char_labels" not in labels[cluster_id]:
                        labels[cluster_id]["char_labels"] = {}
                    labels[cluster_id]["char_labels"][str(idx)] = {"char": char}
                    self._save_file(labels_path, labels)
                    self._logger.log_sync_to_cluster(char_id, char, cluster_id, success=True)
                    return
        
        self._logger.log_sync_skipped(char_id, char, "字符不在聚类数据中")
    
    def _sync_to_multi_clustering(self, char_id: str, char: str):
        """同步标注到多轮聚类数据"""
        if not self.multi_clustering_dir.exists():
            self._logger.log_sync_skipped(char_id, char, "多轮聚类目录不存在")
            return
        
        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            self._logger.log_sync_skipped(char_id, char, "rounds目录不存在")
            return
        
        found = False
        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            round_num = int(round_dir.name.replace("round_", "")) if "round_" in round_dir.name else 0
            
            labels_path = round_dir / "labeling" / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"
            
            if not clusters_path.exists() or not labels_path.exists():
                continue
            
            clusters_data = self._load_file(clusters_path)
            clusters = clusters_data.get("clusters", {})
            labels = self._load_file(labels_path)
            
            for cluster_id, chars_in_cluster in clusters.items():
                for idx, char_info in enumerate(chars_in_cluster):
                    if char_info.get("char_id") == char_id:
                        if cluster_id not in labels:
                            labels[cluster_id] = {
                                "char": char,
                                "status": "labeled",
                                "char_labels": {}
                            }
                        labels[cluster_id]["char"] = char
                        labels[cluster_id]["status"] = "labeled"
                        if "char_labels" not in labels[cluster_id]:
                            labels[cluster_id]["char_labels"] = {}
                        labels[cluster_id]["char_labels"][str(idx)] = {"char": char}
                        self._save_file(labels_path, labels)
                        self._logger.log_sync_to_multi(char_id, char, round_num, cluster_id, success=True)
                        found = True
                        return
        
        if not found:
            self._logger.log_sync_skipped(char_id, char, "字符不在多轮聚类数据中")
    
    def write_annotation(self, char_id: str, char: str, status: str = "labeled", 
                         source: str = "manual", changed_by: str = "user", comment: str = ""):
        """统一写入标注，自动同步所有数据源（带事务）"""
        try:
            # 1. 写入WAL日志
            self._write_wal("update", {
                "char_id": char_id,
                "old_char": self.get_confirmed_annotations().get(char_id),
                "new_char": char,
                "source": source
            })
            
            # 2. 验证更新
            self._validate_update(char_id, char)
            
            # 3. 更新统一标注并获取旧字符
            old_char = self.update_unified_label(char_id, {
                "char": char,
                "status": status,
                "source": source,
                "updated_at": datetime.datetime.now().isoformat()
            })
            
            # 4. 添加历史记录
            self._add_history_record(char_id, old_char, char, changed_by, source, comment)
            
            # 5. 更新索引
            self._update_indexes(char_id, old_char, char)
            
            # 6. 同步到其他数据源
            self._sync_to_cluster_labels(char_id, char)
            self.batch_confirm_prelabels([char_id])
            self._logger.log_sync_to_prelabel(char_id, char, success=True)
            self._sync_to_multi_clustering(char_id, char)
            
            # 7. 原子提交（清理WAL）
            self._cleanup_wal()
            
            # 8. 清除缓存
            self._invalidate_dataset_cache()
            
            self._logger.log_write_annotation(char_id, char, success=True)
            
        except Exception as e:
            self._logger.log_write_annotation(char_id, char, success=False, message=str(e))
            raise
    
    def batch_write_annotations(self, annotations: list, source: str = "ocr_confirm", changed_by: str = "user", comment: str = ""):
        """批量写入标注，大幅提升性能"""
        if not annotations:
            return {"success_count": 0, "total_count": 0}
        
        start_time = time.time()
        success_count = 0
        total_count = len(annotations)
        
        try:
            # 1. 批量验证所有更新
            for ann in annotations:
                char_id = ann.get("char_id") if isinstance(ann, dict) else ann.char_id
                char = ann.get("char") if isinstance(ann, dict) else ann.char
                self._validate_update(char_id, char)
            
            # 2. 一次性读取所有需要的数据
            unified_data = self._load_file(self.unified_labels_path)
            history_data = self._load_file(self.label_history_path)
            char_to_images = self._load_file(self.char_to_images_path, default={})
            image_to_char = self._load_file(self.image_to_char_path, default={})
            prelabel_status = self._load_file(self.prelabel_status_path, default={"status": {}, "corrected_chars": {}})
            
            # 3. 在内存中批量更新
            for ann in annotations:
                try:
                    char_id = ann.get("char_id") if isinstance(ann, dict) else ann.char_id
                    char = ann.get("char") if isinstance(ann, dict) else ann.char
                    
                    # 更新统一标注
                    old_char = self._update_unified_label_in_memory(unified_data, char_id, {
                        "char": char,
                        "status": "labeled",
                        "source": source,
                        "updated_at": datetime.datetime.now().isoformat()
                    })
                    
                    # 添加历史记录
                    self._add_history_record_in_memory(history_data, char_id, old_char, char, changed_by, source, comment)
                    
                    # 更新索引
                    self._update_indexes_in_memory(char_to_images, image_to_char, char_id, old_char, char)
                    
                    # 更新预标注状态
                    if "status" in prelabel_status:
                        prelabel_status["status"][char_id] = "confirmed"
                    
                    success_count += 1
                except Exception as e:
                    self._logger.log_write_annotation(char_id, char, success=False, message=str(e))
            
            # 4. 一次性写入所有文件
            self._save_file(self.unified_labels_path, unified_data)
            self._save_file(self.label_history_path, history_data)
            self._save_file(self.char_to_images_path, char_to_images)
            self._save_file(self.image_to_char_path, image_to_char)
            self._save_file(self.prelabel_status_path, prelabel_status)
            
            # 5. 清理WAL和缓存
            self._cleanup_wal()
            self._invalidate_dataset_cache()
            
            elapsed = time.time() - start_time
            self._logger.log_batch_write_annotations(success_count, total_count, elapsed)
            
            return {"success_count": success_count, "total_count": total_count}
            
        except Exception as e:
            self._logger.log_write_annotation("", "", success=False, message=f"批量写入失败: {e}")
            raise
    
    def _update_unified_label_in_memory(self, unified_data: dict, char_id: str, updates: dict) -> str:
        """在内存中更新统一标注"""
        annotations = unified_data.setdefault("annotations", [])
        old_char = None
        
        for ann in annotations:
            if ann.get("char_id") == char_id:
                old_char = ann.get("char")
                ann.update(updates)
                break
        else:
            # 创建新标注
            new_ann = {
                "char_id": char_id,
                "char": updates.get("char"),
                "dataset": self.dataset_id,
                "status": updates.get("status", "labeled"),
                "source": updates.get("source", "manual"),
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": updates.get("updated_at", datetime.datetime.now().isoformat())
            }
            annotations.append(new_ann)
        
        # 更新统计
        unified_data["total_labeled"] = len([a for a in annotations if a.get("status") == "labeled"])
        char_dist = {}
        for ann in annotations:
            char = ann.get("char")
            if char:
                char_dist[char] = char_dist.get(char, 0) + 1
        unified_data["char_distribution"] = char_dist
        
        return old_char
    
    def _add_history_record_in_memory(self, history_data: dict, char_id: str, old_char: str, new_char: str, 
                                      changed_by: str, source: str, comment: str):
        """在内存中添加历史记录"""
        history = history_data.setdefault("history", [])
        history.append({
            "id": f"hist_{len(history) + 1}",
            "char_id": char_id,
            "old_char": old_char,
            "new_char": new_char,
            "changed_by": changed_by,
            "change_source": source,
            "timestamp": datetime.datetime.now().isoformat(),
            "comment": comment
        })
    
    def _update_indexes_in_memory(self, char_to_images: dict, image_to_char: dict, 
                                  char_id: str, old_char: str, new_char: str):
        """在内存中更新索引"""
        # 移除旧字符的索引
        if old_char and old_char in char_to_images:
            if char_id in char_to_images[old_char]:
                char_to_images[old_char].remove(char_id)
                if not char_to_images[old_char]:
                    del char_to_images[old_char]
        
        # 添加新字符的索引
        if new_char:
            if new_char not in char_to_images:
                char_to_images[new_char] = []
            if char_id not in char_to_images[new_char]:
                char_to_images[new_char].append(char_id)
            image_to_char[char_id] = new_char
    
    def invalidate_cache(self):
        """清除所有缓存"""
        self._cache.clear()
        self._last_modified.clear()
    
    def _invalidate_dataset_cache(self):
        """仅清除当前数据集的缓存"""
        keys_to_remove = []
        for key in self._cache.keys():
            if key.startswith(f"{self.dataset_id}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._last_modified:
                del self._last_modified[key]
    
    # ==================== 统计数据 ====================
    
    def get_statistics(self) -> dict:
        """获取完整的统计数据"""
        prelabel_stats = self.get_prelabel_stats()
        total_images = prelabel_stats["total"]
        char_counts = prelabel_stats["char_counts"]
        
        annotations = self.get_unified_labels()
        dataset_annotations = [a for a in annotations if a.get("dataset") == self.dataset_id or not a.get("dataset")]
        labeled_count = len([a for a in dataset_annotations if a.get("status") == "labeled"])
        
        char_stats = {}
        for char, counts in char_counts.items():
            char_stats[char] = {
                "total": counts.get("total", 0),
                "confirmed": 0,
                "pending": counts.get("total", 0)
            }
        
        # 使用 prelabel_status.json 中的状态来统计已确认数（与预标注确认页面保持一致）
        prelabel_status = self._load_file(self.prelabel_status_path)
        status_map = prelabel_status.get("status", {})
        
        # 构建 char_id -> char 的映射
        char_id_to_char = {}
        for char, info in char_stats.items():
            char_ids = self._char_to_prelabels.get(char, [])
            for char_id in char_ids:
                char_id_to_char[char_id] = char
        
        # 根据状态统计
        for char_id, status in status_map.items():
            if status == "confirmed" and char_id in char_id_to_char:
                char = char_id_to_char[char_id]
                if char in char_stats:
                    char_stats[char]["confirmed"] += 1
                    char_stats[char]["pending"] = max(0, char_stats[char]["pending"] - 1)
        
        return {
            "dataset": self.dataset_id,
            "total_images": total_images,
            "labeled_count": labeled_count,
            "unlabeled_count": total_images - labeled_count,
            "char_stats": char_stats
        }
    
    # ==================== 日志查询 ====================
    
    def get_sync_logs(self, limit: int = 100) -> list:
        """获取最近的同步日志"""
        return self._logger.get_recent_logs(limit)
    
    def get_sync_logs_by_date(self, date_str: str) -> list:
        """按日期获取同步日志"""
        return self._logger.get_logs_by_date(date_str)
    
    def get_today_sync_logs(self) -> list:
        """获取今日同步日志"""
        return self._logger.get_today_logs()
    
    def get_sync_statistics(self) -> dict:
        """获取同步日志统计"""
        return self._logger.get_statistics()