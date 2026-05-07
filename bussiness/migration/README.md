# 迁移学习模块

支持多数据集标注迁移的核心模块，以汉字为单位进行处理。

## 功能特性

- 从多个源数据集提取汉字标注信息
- 构建全局汉字注册表（自动合并所有数据集的汉字）
- 执行跨数据集标注迁移
- 生成标注优先级列表

## 目录结构

```
bussiness/migration/
├── __init__.py               # 模块初始化
├── manager.py                # 迁移管理器（核心逻辑）
├── registry.py               # 全局汉字注册表管理
├── matching.py               # 汉字匹配算法
├── data/                     # 数据文件目录
│   ├── char_feature_db.json      # 汉字特征库
│   ├── char_migration_map.json   # 迁移映射表
│   ├── char_priority_list.json   # 优先级列表
│   └── global_char_registry.json # 全局汉字注册表
├── test_migration.py         # 测试脚本
└── README.md                 # 说明文档
```

## 使用方法

### 基本用法

```python
from bussiness.migration import MigrationManager

# 创建迁移管理器
manager = MigrationManager()

# 执行标注迁移（自动发现所有已标注数据集作为源）
manager.migrate_labels(target_dataset_id="pdf5823")

# 或指定源数据集
manager.migrate_labels(
    target_dataset_id="pdf5823",
    source_dataset_ids=["pdf01", "pdf5824"]
)

# 生成迁移报告
report = manager.generate_report(target_dataset_id="pdf5823")
print(report)
```

### 命令行测试

```bash
cd bussiness/migration
python test_migration.py
```

## 数据结构

### char_feature_db.json（汉字特征库）

```json
{
  "version": "1.0",
  "datasets": ["pdf01", "pdf5824"],
  "characters": {
    "路": {
      "total_samples": 120,
      "sources": [
        {"dataset_id": "pdf01", "cluster_ids": [483], "sample_count": 46},
        {"dataset_id": "pdf5824", "cluster_ids": [200, 201], "sample_count": 74}
      ]
    }
  }
}
```

### char_priority_list.json（优先级列表）

```json
{
  "version": "1.0",
  "dataset_id": "pdf5823",
  "priorities": [
    {
      "char": "璁",
      "source_count": 1,
      "target_count": 0,
      "priority_score": 100,
      "reason": "源数据集中样本极少，目标数据集未标注",
      "action": "优先标注"
    }
  ]
}
```

## 优先级计算

优先级 = 目标样本因子(40%) + 源样本因子(30%) + 置信度因子(30%)

- **目标样本因子**：目标数据集样本越少，优先级越高
- **源样本因子**：源数据集样本越多，优先级越高
- **置信度因子**：汉字匹配的置信度

## 注意事项

1. 确保 `bussiness/datahome/` 目录下存在数据集文件夹（如 pdf01, pdf5823）
2. 源数据集需要有已标注的 `labels.json` 文件
3. 迁移前建议备份目标数据集的标注文件