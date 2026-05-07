"""
迁移学习模块 - 支持多数据集标注迁移

功能：
    1. 从多个源数据集提取汉字标注信息
    2. 构建全局汉字注册表
    3. 执行跨数据集标注迁移
    4. 生成标注优先级列表

使用方法：
    from bussiness.migration import MigrationManager
    
    # 创建管理器
    manager = MigrationManager()
    
    # 执行标注迁移
    manager.migrate_labels(target_dataset_id="pdf5823")
    
    # 或指定源数据集
    manager.migrate_labels(
        target_dataset_id="pdf5823",
        source_dataset_ids=["pdf01", "pdf5824"]
    )
"""

from .manager import MigrationManager
from .registry import GlobalRegistry

__all__ = ["MigrationManager", "GlobalRegistry"]