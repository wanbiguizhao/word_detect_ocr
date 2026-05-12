"""数据同步日志系统"""
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class SyncLogger:
    """同步日志记录器"""
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.log_dir = self.project_root / "bussiness" / "logs" / dataset_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log_path = None
        self._update_log_path()
    
    def _update_log_path(self):
        """更新当前日志文件路径（按日期滚动）"""
        today = datetime.datetime.now().strftime("%Y%m%d")
        self.current_log_path = self.log_dir / f"sync_{today}.log"
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.datetime.now().isoformat()
    
    def log(self, operation: str, char_id: str, char: str = "", 
            target: str = "", status: str = "success", 
            message: str = "", details: Optional[Dict[str, Any]] = None):
        """
        记录同步日志
        
        Args:
            operation: 操作类型 (write_annotation, sync_to_cluster, sync_to_multi, sync_to_prelabel)
            char_id: 字符ID
            char: 标注的字符
            target: 同步目标
            status: 操作状态 (success, failed, skipped)
            message: 附加消息
            details: 详细信息
        """
        self._update_log_path()
        
        log_entry = {
            "timestamp": self._get_timestamp(),
            "dataset": self.dataset_id,
            "operation": operation,
            "char_id": char_id,
            "char": char,
            "target": target,
            "status": status,
            "message": message,
            "details": details or {}
        }
        
        with open(self.current_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_write_annotation(self, char_id: str, char: str, success: bool = True, message: str = ""):
        """记录标注写入操作"""
        self.log(
            operation="write_annotation",
            char_id=char_id,
            char=char,
            status="success" if success else "failed",
            message=message
        )
    
    def log_sync_to_cluster(self, char_id: str, char: str, cluster_id: str = "", 
                           success: bool = True, message: str = ""):
        """记录同步到聚类标注"""
        self.log(
            operation="sync_to_cluster",
            char_id=char_id,
            char=char,
            target=cluster_id,
            status="success" if success else "failed",
            message=message,
            details={"cluster_id": cluster_id}
        )
    
    def log_sync_to_multi(self, char_id: str, char: str, round_num: int = 0, 
                         cluster_id: str = "", success: bool = True, message: str = ""):
        """记录同步到多轮聚类"""
        self.log(
            operation="sync_to_multi_clustering",
            char_id=char_id,
            char=char,
            target=f"round_{round_num}",
            status="success" if success else "failed",
            message=message,
            details={"round": round_num, "cluster_id": cluster_id}
        )
    
    def log_sync_to_prelabel(self, char_id: str, char: str, success: bool = True, message: str = ""):
        """记录同步到预标注"""
        self.log(
            operation="sync_to_prelabel",
            char_id=char_id,
            char=char,
            target="pre_labels.json",
            status="success" if success else "failed",
            message=message
        )
    
    def log_sync_skipped(self, char_id: str, char: str, reason: str = ""):
        """记录跳过的同步操作"""
        self.log(
            operation="sync_skipped",
            char_id=char_id,
            char=char,
            status="skipped",
            message=reason
        )
    
    def get_today_logs(self) -> list:
        """获取今日日志"""
        self._update_log_path()
        logs = []
        if self.current_log_path.exists():
            with open(self.current_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return logs
    
    def get_logs_by_date(self, date_str: str) -> list:
        """按日期获取日志"""
        log_path = self.log_dir / f"sync_{date_str}.log"
        logs = []
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return logs
    
    def get_recent_logs(self, limit: int = 100) -> list:
        """获取最近的日志"""
        logs = self.get_today_logs()
        return logs[-limit:]
    
    def get_statistics(self) -> dict:
        """获取日志统计"""
        logs = self.get_today_logs()
        stats = {
            "total": len(logs),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "operations": defaultdict(int)
        }
        
        for log in logs:
            stats["operations"][log.get("operation", "unknown")] += 1
            status = log.get("status", "unknown")
            if status == "success":
                stats["success"] += 1
            elif status == "failed":
                stats["failed"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
        
        return stats

from collections import defaultdict
