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
    
    def skip_prelabel(self, char_id: str):
        """标记预标注为跳过"""
        if "status" not in self._prelabel_status:
            self._prelabel_status["status"] = {}
        self._prelabel_status["status"][char_id] = "skipped"
        self._prelabel_status["updated_at"] = datetime.datetime.now().isoformat()
        self._prelabel_status_dirty = True
        self._flush_prelabel_status()
    
    def reset_prelabel(self, char_id: str):
        """重置预标注状态为 pending"""
        if "status" not in self._prelabel_status:
            self._prelabel_status["status"] = {}
        self._prelabel_status["status"][char_id] = "pending"
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
        
        # 从文件读取最新状态（可能被 batch_write_annotations 等操作更新）
        prelabel_status = self._load_file(self.prelabel_status_path)
        status_map = prelabel_status.get("status", {})
        confirmed_count = sum(1 for v in status_map.values() if v == "confirmed")
        skipped_count = sum(1 for v in status_map.values() if v == "skipped")
        pending_count = sum(1 for v in status_map.values() if v == "pending")
        
        return {
            "total": stats.get("total", 0),
            "confirmed": confirmed_count,
            "skipped": skipped_count,
            "pending": pending_count,
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
        """验证更新：允许修改已有标签（覆盖模式）"""
        image_to_char = self.get_image_to_char()
        current_char = image_to_char.get(char_id)
        
        if current_char and current_char != new_char:
            logger.info(f"[validate_update] 覆盖已有标签: {char_id}, '{current_char}' -> '{new_char}'")
        
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
            logger.debug(f"[sync_to_mc] 跳过: 多轮聚类目录不存在, char_id={char_id}")
            return
        
        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            logger.debug(f"[sync_to_mc] 跳过: rounds目录不存在, char_id={char_id}")
            return
        
        found = False
        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            round_num = int(round_dir.name.replace("round_", "")) if "round_" in round_dir.name else 0
            
            labels_path = round_dir / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"
            
            if not clusters_path.exists():
                logger.debug(f"[sync_to_mc] round_{round_num}: hog_clusters.json 不存在")
                continue
            if not labels_path.exists():
                logger.debug(f"[sync_to_mc] round_{round_num}: labels.json 不存在")
                continue
            
            clusters_data = self._load_file(clusters_path)
            clusters = clusters_data.get("clusters", {})
            labels_data = self._load_file(labels_path)
            
            if "labels" not in labels_data:
                labels_data["labels"] = {}
            labels = labels_data["labels"]
            
            round_found = False
            for cluster_id, chars_in_cluster in clusters.items():
                for idx, char_info in enumerate(chars_in_cluster):
                    if char_info.get("char_id") == char_id:
                        logger.debug(f"[sync_to_mc] 找到字符: char_id={char_id}, char={char}, round={round_num}, cluster={cluster_id}, idx={idx}")
                        
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
                        labels[cluster_id]["char_labels"][str(idx)] = {
                            "char": char,
                            "labeled_at": datetime.datetime.now().isoformat()
                        }
                        logger.info(f"[sync_to_mc] 同步成功: char_id={char_id}, char={char}, round={round_num}, cluster={cluster_id}")
                        found = True
                        round_found = True
            
            if round_found:
                self._save_file(labels_path, labels_data)
        
        if not found:
            logger.debug(f"[sync_to_mc] 字符不在多轮聚类数据中: char_id={char_id}, char={char}")
    
    def write_annotation(self, char_id: str, char: str, status: str = "labeled", 
                         source: str = "manual", changed_by: str = "user", comment: str = ""):
        """统一写入标注，自动同步所有数据源（带事务）"""
        logger.info(f"[write_annotation] 开始: char_id={char_id}, char={char}, source={source}")
        try:
            logger.info(f"[write_annotation] 步骤1: 写入WAL日志")
            self._write_wal("update", {
                "char_id": char_id,
                "old_char": self.get_confirmed_annotations().get(char_id),
                "new_char": char,
                "source": source
            })
            
            logger.info(f"[write_annotation] 步骤2: 验证更新, char='{char}'")
            self._validate_update(char_id, char)
            
            logger.info(f"[write_annotation] 步骤3: 更新统一标注")
            old_char = self.update_unified_label(char_id, {
                "char": char,
                "status": status,
                "source": source,
                "updated_at": datetime.datetime.now().isoformat()
            })
            logger.info(f"[write_annotation] 步骤3完成: old_char='{old_char}'")
            
            logger.info(f"[write_annotation] 步骤4: 添加历史记录")
            self._add_history_record(char_id, old_char, char, changed_by, source, comment)
            
            logger.info(f"[write_annotation] 步骤5: 更新索引")
            self._update_indexes(char_id, old_char, char)
            
            logger.info(f"[write_annotation] 步骤6: 同步到其他数据源")
            self._sync_to_cluster_labels(char_id, char)
            logger.info(f"[write_annotation] 步骤6a: 同步到聚类标注完成")
            
            self.batch_confirm_prelabels([char_id])
            logger.info(f"[write_annotation] 步骤6b: 确认预标注完成")
            
            self.correct_prelabel_char(char_id, char)
            logger.info(f"[write_annotation] 步骤6c: 修正预标注字符完成")
            
            self._logger.log_sync_to_prelabel(char_id, char, success=True)
            self._sync_to_multi_clustering(char_id, char)
            logger.info(f"[write_annotation] 步骤6d: 同步到多轮聚类完成")
            
            self._sync_to_char_pool_labeled(char_id, char)
            logger.info(f"[write_annotation] 步骤6e: 同步到字符池完成")
            
            self._cleanup_wal()
            self._invalidate_dataset_cache()
            
            logger.info(f"[write_annotation] 全部完成: char_id={char_id}, char='{char}'")
            self._logger.log_write_annotation(char_id, char, success=True)
            
        except Exception as e:
            logger.error(f"[write_annotation] 失败: char_id={char_id}, char='{char}', error={e}", exc_info=True)
            self._logger.log_write_annotation(char_id, char, success=False, message=str(e))
            raise
    
    def batch_write_annotations(self, annotations: list, source: str = "ocr_confirm", changed_by: str = "user", comment: str = ""):
        """批量写入标注，大幅提升性能"""
        if not annotations:
            return {"success_count": 0, "total_count": 0}
        
        logger.info(f"[batch_write] 开始批量写入: count={len(annotations)}, source={source}")
        start_time = time.time()
        success_count = 0
        total_count = len(annotations)
        
        try:
            # 2. 一次性读取所有需要的数据
            unified_data = self._load_file(self.unified_labels_path)
            history_data = self._load_file(self.label_history_path)
            char_to_images = self._load_file(self.char_to_images_path, default={})
            image_to_char = self._load_file(self.image_to_char_path, default={})
            prelabel_status = self._load_file(self.prelabel_status_path, default={"status": {}, "corrected_chars": {}})
            
            # 3. 在内存中批量更新（逐项处理，覆盖旧数据）
            for ann in annotations:
                try:
                    char_id = ann.get("char_id") if isinstance(ann, dict) else ann.char_id
                    char = ann.get("char") if isinstance(ann, dict) else ann.char
                    
                    # 如果已有不同标注，记录日志并覆盖
                    current_char = image_to_char.get(char_id)
                    if current_char and current_char != char:
                        logger.info(f"覆盖冲突标注: {char_id}, '{current_char}' -> '{char}'")
                    
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
                    if "corrected_chars" not in prelabel_status:
                        prelabel_status["corrected_chars"] = {}
                    prelabel_status["corrected_chars"][char_id] = char
                    
                    success_count += 1
                except Exception as e:
                    self._logger.log_write_annotation(char_id, char, success=False, message=str(e))
            
            # 4. 一次性写入所有文件
            self._save_file(self.unified_labels_path, unified_data)
            self._save_file(self.label_history_path, history_data)
            self._save_file(self.char_to_images_path, char_to_images)
            self._save_file(self.image_to_char_path, image_to_char)
            self._save_file(self.prelabel_status_path, prelabel_status)
            
            # 4.1 同步更新内存中的 _prelabel_status
            self._prelabel_status = prelabel_status
            self._prelabel_status_dirty = False
            
            # 5. 同步到字符池和多轮聚类
            for ann in annotations:
                try:
                    char_id = ann.get("char_id") if isinstance(ann, dict) else ann.char_id
                    char = ann.get("char") if isinstance(ann, dict) else ann.char
                    logger.info(f"[batch_write] 同步到字符池和聚类: char_id={char_id}, char={char}")
                    self._sync_to_char_pool_labeled(char_id, char)
                    self._sync_to_multi_clustering(char_id, char)
                except Exception as e:
                    logger.warning(f"同步到字符池/聚类失败: {char_id}, {e}")
            
            # 6. 清理WAL和缓存
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
        """获取完整的统计数据
        四个互斥口径：labeled + pending + skipped + unlabeled = total
        """
        prelabel_stats = self.get_prelabel_stats()
        total_images = prelabel_stats["total"]
        char_counts = prelabel_stats["char_counts"]
        
        # labeled 来自 unified_labels.json（所有渠道的最终标注）
        annotations = self.get_unified_labels()
        dataset_annotations = [a for a in annotations if a.get("dataset") == self.dataset_id or not a.get("dataset")]
        labeled_count = len([a for a in dataset_annotations if a.get("status") == "labeled"])
        labeled_char_ids = {a.get("char_id") for a in dataset_annotations if a.get("status") == "labeled" and a.get("char_id")}
        
        # prelabel_status.json：确认和跳过记录
        prelabel_status = self._load_file(self.prelabel_status_path)
        status_map = prelabel_status.get("status", {})
        skipped_count = sum(1 for v in status_map.values() if v == "skipped")
        skipped_char_ids = {cid for cid, v in status_map.items() if v == "skipped"}
        
        # pending = 有预标 且 未被标注 且 未被跳过
        # 所有有预标的图片 = pre_labels.json 中的所有条目
        all_prelabel_char_ids = set()
        for p in self._prelabels_list:
            cid = p.get("char_id")
            if cid:
                all_prelabel_char_ids.add(cid)
        
        pending_char_ids = all_prelabel_char_ids - labeled_char_ids - skipped_char_ids
        pending_count = len(pending_char_ids)
        
        # unlabeled = 完全没有预标（一般应接近 0）
        unlabeled_count = total_images - labeled_count - pending_count - skipped_count
        
        # 逐字统计
        char_stats = {}
        for char, counts in char_counts.items():
            char_stats[char] = {
                "total": counts.get("total", 0),
                "labeled": 0,
                "skipped": 0,
                "pending": 0,
                "unlabeled": counts.get("total", 0)
            }
        
        # 构建 char_id -> char 映射
        char_id_to_char = {}
        for char in char_stats:
            char_ids = self._char_to_prelabels.get(char, [])
            for char_id in char_ids:
                char_id_to_char[char_id] = char
        
        # 标记 labeled/skipped 状态
        for char_id in labeled_char_ids:
            char = char_id_to_char.get(char_id)
            if char and char in char_stats:
                char_stats[char]["labeled"] += 1
                char_stats[char]["unlabeled"] = max(0, char_stats[char]["unlabeled"] - 1)
        
        for char_id in skipped_char_ids:
            char = char_id_to_char.get(char_id)
            if char and char in char_stats:
                char_stats[char]["skipped"] += 1
                char_stats[char]["unlabeled"] = max(0, char_stats[char]["unlabeled"] - 1)
        
        # 标记 pending 状态（有预标但未被覆盖的）
        for char_id in pending_char_ids:
            char = char_id_to_char.get(char_id)
            if char and char in char_stats and char_stats[char]["unlabeled"] > 0:
                char_stats[char]["pending"] += 1
                char_stats[char]["unlabeled"] = max(0, char_stats[char]["unlabeled"] - 1)
        
        return {
            "dataset": self.dataset_id,
            "total_images": total_images,
            "labeled_count": labeled_count,
            "pending_count": pending_count,
            "skipped_count": skipped_count,
            "unlabeled_count": unlabeled_count,
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
    
    # ==================== 修改/撤回/跳过操作 ====================
    
    def modify_annotation(self, char_id: str, old_char: Optional[str], new_char: str, 
                         source: str = "manual", changed_by: str = "user", comment: str = "修改标注"):
        """修改已有的标注（替换字符）"""
        try:
            # 1. 写入WAL日志
            self._write_wal("modify", {
                "char_id": char_id,
                "old_char": old_char,
                "new_char": new_char,
                "source": source
            })
            
            # 2. 更新统一标注
            updated_old_char = self.update_unified_label(char_id, {
                "char": new_char,
                "status": "labeled",
                "source": source,
                "updated_at": datetime.datetime.now().isoformat()
            })
            
            # 3. 添加历史记录
            self._add_history_record(char_id, updated_old_char, new_char, changed_by, source, comment)
            
            # 4. 更新索引
            self._update_indexes(char_id, updated_old_char, new_char)
            
            # 5. 更新 prelabel_status 中的修正字符
            self._update_prelabel_corrected(char_id, new_char)
            
            # 6. 同步到其他数据源
            self._sync_to_cluster_labels(char_id, new_char)
            self._sync_to_multi_clustering(char_id, new_char)
            
            # 7. 原子提交（清理WAL）
            self._cleanup_wal()
            
            # 8. 清除缓存
            self._invalidate_dataset_cache()
            
            self._logger.log_write_annotation(char_id, new_char, success=True)
            
        except Exception as e:
            self._logger.log_write_annotation(char_id, new_char, success=False, message=str(e))
            raise
    
    def revoke_annotation(self, char_id: str, changed_by: str = "user", comment: str = "撤回确认"):
        """撤回已确认的标注（恢复为待确认状态）"""
        try:
            # 1. 获取旧字符
            confirmed_anns = self.get_confirmed_annotations()
            old_char = confirmed_anns.get(char_id)
            
            if old_char is None:
                logger.warning(f"尝试撤回未确认的标注: {char_id}")
                return
            
            # 2. 写入WAL日志
            self._write_wal("revoke", {
                "char_id": char_id,
                "old_char": old_char,
                "source": "revoke"
            })
            
            # 3. 从统一标注中删除
            self._remove_unified_label(char_id)
            
            # 4. 添加历史记录
            self._add_history_record(char_id, old_char, None, changed_by, "revoke", comment)
            
            # 5. 从索引中移除
            self._update_indexes(char_id, old_char, None)
            
            # 6. 更新 prelabel_status 为 pending
            self._update_prelabel_status(char_id, "pending")
            
            # 7. 从其他数据源移除
            self._remove_from_cluster_labels(char_id)
            
            # 8. 同步到字符池和多轮聚类（恢复为未标注）
            self._sync_to_char_pool_reset(char_id)
            self._sync_reset_to_multi_clustering(char_id)
            
            # 9. 原子提交（清理WAL）
            self._cleanup_wal()
            
            # 9. 清除缓存
            self._invalidate_dataset_cache()
            
            logger.info(f"撤回确认: {char_id} - {old_char}")
            
        except Exception as e:
            logger.error(f"撤回确认失败: {char_id}, 错误: {e}")
            raise
    
    def skip_prelabel(self, char_id: str, changed_by: str = "user"):
        """跳过单个预标注"""
        try:
            # 更新 prelabel_status 为 skipped
            self._update_prelabel_status(char_id, "skipped")
            
            # 添加历史记录
            self._add_history_record(
                char_id, None, None, changed_by, "skip", "跳过该预标注"
            )
            
            # 同步到统一标注、字符池和多轮聚类
            try:
                self.update_unified_label(char_id, {
                    "status": "skipped",
                    "updated_at": datetime.datetime.now().isoformat()
                })
                self._sync_to_char_pool_skipped(char_id)
                self._sync_skip_to_multi_clustering(char_id)
            except Exception as e:
                logger.warning(f"跳过同步失败: {char_id}, {e}")
            
            logger.info(f"跳过预标注: {char_id}")
            
        except Exception as e:
            logger.error(f"跳过预标注失败: {char_id}, 错误: {e}")
            raise
    
    def batch_skip_prelabels(self, char_ids: List[str], changed_by: str = "user", sync_downstream: bool = True):
        """批量跳过预标注
        
        Args:
            char_ids: 字符ID列表
            changed_by: 操作者
            sync_downstream: 是否同步到下游（unified_labels, char_pool, multi_clustering）
                             当由 MultiClusteringManager 调用时设为 False，由 Manager 统一调度同步
        """
        if not char_ids:
            return
        
        try:
            status_data = self._load_file(self.prelabel_status_path, default={
                "status": {}, "corrected_chars": {}
            })
            
            for char_id in char_ids:
                status_data["status"][char_id] = "skipped"
                
                self._add_history_record(
                    char_id, None, None, changed_by, "skip", "批量跳过"
                )
            
            self._save_file(self.prelabel_status_path, status_data)
            
            self._prelabel_status = status_data
            self._prelabel_status_dirty = False
            
            if sync_downstream:
                for char_id in char_ids:
                    try:
                        self.update_unified_label(char_id, {
                            "status": "skipped",
                            "updated_at": datetime.datetime.now().isoformat()
                        })
                        self._sync_to_char_pool_skipped(char_id)
                        self._sync_skip_to_multi_clustering(char_id)
                    except Exception as e:
                        logger.warning(f"跳过同步失败: {char_id}, {e}")
            
            logger.info(f"批量跳过预标注: {len(char_ids)} 条, sync_downstream={sync_downstream}")
            
        except Exception as e:
            logger.error(f"批量跳过预标注失败, 错误: {e}")
            raise
    
    def batch_modify_prelabels(self, char_updates: dict, changed_by: str = "user"):
        """批量修改预标注（优化版，减少文件IO次数）
        
        Args:
            char_updates: {char_id: new_char} 的字典
            changed_by: 修改者
            
        Returns:
            成功修改的数量
        """
        if not char_updates:
            return 0
            
        try:
            # 确保缓存已加载
            if not self._prelabel_status:
                self._prelabel_status = self._load_file(self.prelabel_status_path, default={
                    "status": {}, "corrected_chars": {}
                })
            
            # 确保键存在
            if "status" not in self._prelabel_status:
                self._prelabel_status["status"] = {}
            if "corrected_chars" not in self._prelabel_status:
                self._prelabel_status["corrected_chars"] = {}
            
            # 批量更新缓存
            success_count = 0
            for char_id, new_char in char_updates.items():
                self._prelabel_status["corrected_chars"][char_id] = new_char
                self._prelabel_status["status"][char_id] = "confirmed"
                
                # 添加历史记录
                self._add_history_record(
                    char_id, None, new_char, changed_by, "modify", "批量修改"
                )
                success_count += 1
            
            self._prelabel_status["updated_at"] = datetime.datetime.now().isoformat()
            
            # 保存到文件（同时也更新缓存）
            self._flush_prelabel_status()
            
            # 同步到统一标注、字符池和多轮聚类
            for char_id, new_char in char_updates.items():
                try:
                    self.update_unified_label(char_id, {
                        "char": new_char,
                        "status": "labeled",
                        "source": "ocr_modify",
                        "updated_at": datetime.datetime.now().isoformat()
                    })
                    self._sync_to_char_pool_labeled(char_id, new_char)
                    self._sync_to_multi_clustering(char_id, new_char)
                except Exception as e:
                    logger.warning(f"修改同步失败: {char_id}, {e}")
            
            logger.info(f"批量修改预标注: {success_count}/{len(char_updates)} 条成功")
            return success_count
            
        except Exception as e:
            logger.error(f"批量修改预标注失败, 错误: {e}")
            raise
    
    def unskip_prelabel(self, char_id: str, changed_by: str = "user"):
        """取消跳过预标注（恢复为待确认状态）"""
        try:
            self._update_prelabel_status(char_id, "pending")
            
            # 添加历史记录
            self._add_history_record(
                char_id, None, None, changed_by, "unskip", "取消跳过"
            )
            
            # 同步到统一标注、字符池和多轮聚类
            try:
                self.update_unified_label(char_id, {
                    "status": "pending",
                    "updated_at": datetime.datetime.now().isoformat()
                })
                self._sync_to_char_pool_reset(char_id)
                self._sync_reset_to_multi_clustering(char_id)
            except Exception as e:
                logger.warning(f"取消跳过同步失败: {char_id}, {e}")
            
            logger.info(f"取消跳过预标注: {char_id}")
            
        except Exception as e:
            logger.error(f"取消跳过预标注失败: {char_id}, 错误: {e}")
            raise
    
    def modify_prelabel(self, char_id: str, new_char: str, changed_by: str = "user"):
        """单独修改预标注（直接修改prelabel_status）
        
        Args:
            char_id: 字符ID
            new_char: 新的字符
            changed_by: 修改者
            
        Returns:
            是否成功
        """
        try:
            # 确保缓存已加载
            if not self._prelabel_status:
                self._prelabel_status = self._load_file(self.prelabel_status_path, default={
                    "status": {}, "corrected_chars": {}
                })
            
            # 确保键存在
            if "status" not in self._prelabel_status:
                self._prelabel_status["status"] = {}
            if "corrected_chars" not in self._prelabel_status:
                self._prelabel_status["corrected_chars"] = {}
            
            # 更新缓存
            self._prelabel_status["corrected_chars"][char_id] = new_char
            self._prelabel_status["status"][char_id] = "confirmed"
            self._prelabel_status["updated_at"] = datetime.datetime.now().isoformat()
            
            # 添加历史记录
            self._add_history_record(
                char_id, None, new_char, changed_by, "modify", "单独修改"
            )
            
            # 保存到文件（同时也更新缓存）
            self._flush_prelabel_status()
            
            # 同步到统一标注、字符池和多轮聚类
            try:
                self.update_unified_label(char_id, {
                    "char": new_char,
                    "status": "labeled",
                    "source": "ocr_modify",
                    "updated_at": datetime.datetime.now().isoformat()
                })
                self._sync_to_char_pool_labeled(char_id, new_char)
                self._sync_to_multi_clustering(char_id, new_char)
            except Exception as e:
                logger.warning(f"修改同步失败: {char_id}, {e}")
            
            logger.info(f"单独修改预标注: {char_id} -> {new_char} 成功")
            return True
            
        except Exception as e:
            logger.error(f"单独修改预标注失败: {char_id}, 错误: {e}")
            raise
    
    # ==================== 内部辅助方法 ====================
    
    def _remove_unified_label(self, char_id: str):
        """从统一标注中删除一个标注"""
        data = self._load_file(self.unified_labels_path)
        annotations = data.get("annotations", [])
        
        # 查找并删除
        new_annotations = []
        for ann in annotations:
            if ann.get("char_id") != char_id:
                new_annotations.append(ann)
        
        data["annotations"] = new_annotations
        
        # 更新统计
        data["total_labeled"] = len([a for a in new_annotations if a.get("status") == "labeled"])
        
        # 更新字符分布
        char_dist = {}
        for ann in new_annotations:
            char = ann.get("char")
            if char:
                char_dist[char] = char_dist.get(char, 0) + 1
        data["char_distribution"] = char_dist
        
        self._save_file(self.unified_labels_path, data)
    
    def _update_prelabel_status(self, char_id: str, status: str):
        """更新预标注状态"""
        status_data = self._load_file(self.prelabel_status_path, default={
            "status": {}, "corrected_chars": {}
        })
        status_data["status"][char_id] = status
        self._save_file(self.prelabel_status_path, status_data)
        self._prelabel_status = status_data
        self._prelabel_status_dirty = False
    
    def _update_prelabel_corrected(self, char_id: str, corrected_char: str):
        """更新预标注的修正字符，并同时设置状态为confirmed"""
        status_data = self._load_file(self.prelabel_status_path, default={
            "status": {}, "corrected_chars": {}
        })
        status_data["corrected_chars"][char_id] = corrected_char
        status_data["status"][char_id] = "confirmed"
        self._save_file(self.prelabel_status_path, status_data)
        self._prelabel_status = status_data
        self._prelabel_status_dirty = False
    
    def _remove_from_cluster_labels(self, char_id: str):
        """从聚类标注中移除（如果有）"""
        pass

    def _sync_to_char_pool_labeled(self, char_id: str, char: str):
        """同步标注到字符池（标记为已标注）"""
        all_chars_path = self.multi_clustering_dir / "char_pool" / "all_chars.json"
        unlabeled_path = self.multi_clustering_dir / "char_pool" / "unlabeled.json"
        labeled_path = self.multi_clustering_dir / "char_pool" / "labeled.json"

        if not all_chars_path.exists():
            logger.debug(f"[sync_char_pool_labeled] all_chars.json 不存在")
            return

        all_chars_data = self._load_file(all_chars_path)
        chars = all_chars_data.get("chars", {})

        if char_id not in chars:
            logger.debug(f"[sync_char_pool_labeled] char_id={char_id} 不在字符池中")
            return

        now = datetime.datetime.now().isoformat()
        chars[char_id]["status"] = "labeled"
        chars[char_id]["labeled_char"] = char
        chars[char_id]["updated_at"] = now

        all_chars_data["chars"] = chars
        all_chars_data["updated_at"] = now
        self._save_file(all_chars_path, all_chars_data)
        logger.info(f"[sync_char_pool_labeled] 成功: char_id={char_id}, char={char}")

        if unlabeled_path.exists():
            unlabeled_data = self._load_file(unlabeled_path)
            char_ids = unlabeled_data.get("char_ids", [])
            if char_id in char_ids:
                char_ids.remove(char_id)
                unlabeled_data["char_ids"] = char_ids
                unlabeled_data["count"] = len(char_ids)
                unlabeled_data["updated_at"] = now
                self._save_file(unlabeled_path, unlabeled_data)

        if labeled_path.exists():
            labeled_data = self._load_file(labeled_path)
            labeled_ids = set(labeled_data.get("char_ids", []))
            labeled_ids.add(char_id)
            labeled_data["char_ids"] = list(labeled_ids)
            labeled_data["count"] = len(labeled_ids)
            labeled_data["updated_at"] = now
            self._save_file(labeled_path, labeled_data)

    def _sync_to_char_pool_skipped(self, char_id: str):
        """同步跳过状态到字符池"""
        all_chars_path = self.multi_clustering_dir / "char_pool" / "all_chars.json"
        unlabeled_path = self.multi_clustering_dir / "char_pool" / "unlabeled.json"
        labeled_path = self.multi_clustering_dir / "char_pool" / "labeled.json"

        if not all_chars_path.exists():
            logger.debug(f"[sync_char_pool_skipped] all_chars.json 不存在")
            return

        all_chars_data = self._load_file(all_chars_path)
        chars = all_chars_data.get("chars", {})

        if char_id not in chars:
            logger.debug(f"[sync_char_pool_skipped] char_id={char_id} 不在字符池中")
            return

        now = datetime.datetime.now().isoformat()
        chars[char_id]["status"] = "skipped"
        chars[char_id]["updated_at"] = now

        all_chars_data["chars"] = chars
        all_chars_data["updated_at"] = now
        self._save_file(all_chars_path, all_chars_data)
        logger.info(f"[sync_char_pool_skipped] 成功: char_id={char_id}")

        if unlabeled_path.exists():
            unlabeled_data = self._load_file(unlabeled_path)
            char_ids = unlabeled_data.get("char_ids", [])
            if char_id in char_ids:
                char_ids.remove(char_id)
                unlabeled_data["char_ids"] = char_ids
                unlabeled_data["count"] = len(char_ids)
                unlabeled_data["updated_at"] = now
                self._save_file(unlabeled_path, unlabeled_data)

        if labeled_path.exists():
            labeled_data = self._load_file(labeled_path)
            labeled_ids = set(labeled_data.get("char_ids", []))
            if char_id in labeled_ids:
                labeled_ids.remove(char_id)
                labeled_data["char_ids"] = list(labeled_ids)
                labeled_data["count"] = len(labeled_ids)
                labeled_data["updated_at"] = now
                self._save_file(labeled_path, labeled_data)

    def _sync_to_char_pool_reset(self, char_id: str):
        """同步重置状态到字符池（恢复为未标注）"""
        all_chars_path = self.multi_clustering_dir / "char_pool" / "all_chars.json"
        unlabeled_path = self.multi_clustering_dir / "char_pool" / "unlabeled.json"
        labeled_path = self.multi_clustering_dir / "char_pool" / "labeled.json"

        if not all_chars_path.exists():
            logger.debug(f"[sync_char_pool_reset] all_chars.json 不存在")
            return

        all_chars_data = self._load_file(all_chars_path)
        chars = all_chars_data.get("chars", {})

        if char_id not in chars:
            logger.debug(f"[sync_char_pool_reset] char_id={char_id} 不在字符池中")
            return

        now = datetime.datetime.now().isoformat()
        chars[char_id]["status"] = "unlabeled"
        chars[char_id]["labeled_char"] = None
        chars[char_id]["labeled_round"] = None
        chars[char_id]["updated_at"] = now

        all_chars_data["chars"] = chars
        all_chars_data["updated_at"] = now
        self._save_file(all_chars_path, all_chars_data)
        logger.info(f"[sync_char_pool_reset] 成功: char_id={char_id}")

        if labeled_path.exists():
            labeled_data = self._load_file(labeled_path)
            labeled_ids = set(labeled_data.get("char_ids", []))
            if char_id in labeled_ids:
                labeled_ids.remove(char_id)
                labeled_data["char_ids"] = list(labeled_ids)
                labeled_data["count"] = len(labeled_ids)
                labeled_data["updated_at"] = now
                self._save_file(labeled_path, labeled_data)

        if unlabeled_path.exists():
            unlabeled_data = self._load_file(unlabeled_path)
            char_ids = unlabeled_data.get("char_ids", [])
            if char_id not in char_ids:
                char_ids.append(char_id)
                unlabeled_data["char_ids"] = char_ids
                unlabeled_data["count"] = len(char_ids)
                unlabeled_data["updated_at"] = now
                self._save_file(unlabeled_path, unlabeled_data)

    def _sync_skip_to_multi_clustering(self, char_id: str):
        """同步跳过状态到多轮聚类标注"""
        if not self.multi_clustering_dir.exists():
            return

        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            return

        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue

            round_num = int(round_dir.name.replace("round_", "")) if "round_" in round_dir.name else 0
            labels_path = round_dir / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"

            if not clusters_path.exists() or not labels_path.exists():
                continue

            clusters_data = self._load_file(clusters_path)
            clusters = clusters_data.get("clusters", {})
            labels_data = self._load_file(labels_path)

            if "labels" not in labels_data:
                labels_data["labels"] = {}
            labels = labels_data["labels"]

            round_found = False
            for cluster_id, chars_in_cluster in clusters.items():
                for idx, char_info in enumerate(chars_in_cluster):
                    if char_info.get("char_id") == char_id:
                        if cluster_id not in labels:
                            labels[cluster_id] = {"status": "unlabeled", "char_labels": {}}
                        if "char_labels" not in labels[cluster_id]:
                            labels[cluster_id]["char_labels"] = {}
                        labels[cluster_id]["char_labels"][str(idx)] = {"status": "skipped"}
                        logger.info(f"[sync_skip_to_mc] 同步跳过: char_id={char_id}, round={round_num}, cluster={cluster_id}")
                        round_found = True

            if round_found:
                self._save_file(labels_path, labels_data)

    def _sync_reset_to_multi_clustering(self, char_id: str):
        """同步重置状态到多轮聚类标注（恢复为未标注）"""
        if not self.multi_clustering_dir.exists():
            return

        rounds_dir = self.multi_clustering_dir / "rounds"
        if not rounds_dir.exists():
            return

        for round_dir in rounds_dir.iterdir():
            if not round_dir.is_dir():
                continue

            round_num = int(round_dir.name.replace("round_", "")) if "round_" in round_dir.name else 0
            labels_path = round_dir / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"

            if not clusters_path.exists() or not labels_path.exists():
                continue

            clusters_data = self._load_file(clusters_path)
            clusters = clusters_data.get("clusters", {})
            labels_data = self._load_file(labels_path)

            if "labels" not in labels_data:
                labels_data["labels"] = {}
            labels = labels_data["labels"]

            round_found = False
            for cluster_id, chars_in_cluster in clusters.items():
                for idx, char_info in enumerate(chars_in_cluster):
                    if char_info.get("char_id") == char_id:
                        char_key = str(idx)
                        if cluster_id in labels and "char_labels" in labels[cluster_id]:
                            if char_key in labels[cluster_id]["char_labels"]:
                                del labels[cluster_id]["char_labels"][char_key]
                                if not labels[cluster_id]["char_labels"]:
                                    labels[cluster_id]["status"] = "unlabeled"
                                    labels[cluster_id]["char"] = None
                                logger.info(f"[sync_reset_to_mc] 同步重置: char_id={char_id}, round={round_num}, cluster={cluster_id}")
                                round_found = True

            if round_found:
                self._save_file(labels_path, labels_data)