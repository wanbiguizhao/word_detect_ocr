# OCR训练系统

## 目录结构

```
ocr_system/
├── configs/                # 配置文件
│   └── char_mapping/       # 字符映射配置
│       ├── char_to_label.json      # 字符到标签ID映射（GB2312-80全量）
│       ├── label_to_char.json      # 标签ID到字符映射
│       ├── char_properties.json    # 汉字属性配置（含Unicode码点）
│       ├── dataset_mapping.json    # 数据集映射配置
│       ├── manager.py              # 映射管理器
│       └── init_gb2312.py          # GB2312字符集初始化脚本
├── datasets/               # 数据集模块
│   └── char_dataset.py     # 字符数据集类
├── evaluation/              # 评估模块
│   ├── evaluator.py         # 评估器基类
│   └── pdf_evaluator.py    # PDF数据集评估器
├── models/                 # 模型模块
│   └── resnet_ocr.py       # ResNet OCR模型
├── data/                   # 训练数据
│   ├── train.csv           # 训练集
│   └── val.csv             # 验证集
├── scripts/                # 脚本文件
│   ├── unified_labels.py            # 统一标注数据加载模块
│   ├── collect_dataset_data.py      # 更新字符映射
│   ├── generate_train_csv.py         # 生成训练/验证/测试集CSV
│   ├── generate_train_csv_aug.py     # 带数据增强生成数据集
│   ├── analyze_data_distribution.py  # 数据分布分析
│   ├── balance_data.py              # 数据均衡处理
│   └── data_augmentation.py          # 数据增强
├── train.py                # 训练脚本
├── infer.py                # 推理脚本
├── evaluate.py             # 统一评估脚本
└── config.py               # 配置文件
```

## 核心特性

### 1. 统一数据管理
- 所有脚本统一使用 `bussiness/unified_labels.json` 作为数据源
- 支持多数据集（pdf01、pdf5823等）统一管理
- 图片路径使用相对于 `bussiness` 目录的路径

### 2. 字符映射系统
- 使用 GB2312-80 标准字符集（6763个汉字）
- 稳定的标签ID分配：标准汉字ID固定（0-6762），自定义字符从6863开始追加
- 支持自动注册新汉字，保证向后兼容

### 3. 数据处理
- 支持多数据集统一管理
- 数据均衡和增强功能
- 训练/验证/测试集自动分割（80%/10%/10%）
- 自动处理同名图片冲突（添加数据集前缀）

## 快速开始

### 1. 初始化字符映射（首次使用）
```bash
python configs/char_mapping/init_gb2312.py
```

### 2. 更新字符映射
从 `bussiness/unified_labels.json` 读取所有汉字字符，更新到字符映射文件。
```bash
# 使用默认路径
python scripts/collect_dataset_data.py

# 指定unified_labels.json路径
python scripts/collect_dataset_data.py ../../bussiness/unified_labels.json
```

### 3. 生成训练数据集
```bash
# 简单版 - 生成训练/验证/测试CSV
python scripts/generate_train_csv.py

# 带数据增强版
python scripts/generate_train_csv_aug.py

# 指定数据集过滤
python scripts/generate_train_csv.py --datasets pdf01 --min-samples 10
```

### 4. 数据分析
```bash
# 分析所有数据分布
python scripts/analyze_data_distribution.py

# 只分析指定数据集
python scripts/analyze_data_distribution.py --datasets pdf01
```

### 5. 数据均衡（可选）
```bash
python scripts/balance_data.py --min 30 --max 150
```

### 6. 数据增强（可选）
```bash
python scripts/data_augmentation.py
```

### 7. 模型训练
```bash
python train.py
```

### 8. 推理预测
```bash
# 单张图片预测
python infer.py --image test.png

# 批量预测
python infer.py --batch img1.png img2.png img3.png
```

## 统一标注数据格式

`bussiness/unified_labels.json` 结构：
```json
{
  "version": "1.0",
  "datasets": ["pdf01", "pdf5823"],
  "total_labeled": 12345,
  "char_distribution": {"的": 100, "了": 95},
  "annotations": [
    {
      "dataset": "pdf01",
      "char": "的",
      "cluster_id": "xxx",
      "char_id": 1,
      "image_path": "datahome/pdf01/clusters/xxx/char_001.png"
    }
  ]
}
```

**路径说明**：`image_path` 是相对于 `bussiness` 目录的相对路径。

## 脚本使用示例

### 更新字符映射
```bash
$ python scripts/collect_dataset_data.py

START: 开始更新字符映射
加载标注数据: 12345 条
发现汉字数: 1164 个
INFO: 更新字符映射...
SUCCESS: 字符映射已更新，新增 1164 个字符
  总汉字数: 1164
```

### 生成训练数据集
```bash
$ python scripts/generate_train_csv.py --datasets pdf01 --min-samples 10

加载标注数据: 12345 条
筛选后数据量: 12000 (移除 345 条稀有样本)
保留类别数: 1100
有效数据: 12000 条

数据集划分完成:
  训练集: 9600 条
  验证集: 1200 条
  测试集: 1200 条
  唯一字符数: 1100

CSV 文件已生成:
  - data/train.csv
  - data/val.csv
  - data/test.csv

请更新 config.py 中的 NUM_CLASSES = 1100
```

### 分析数据分布
```bash
$ python scripts/analyze_data_distribution.py --datasets pdf01

============================================================
数据分布分析报告
============================================================
数据集: pdf01
汉字数量: 1164 个
图片总数: 23711 张
平均每个汉字: 20.4 张
最少: 10 张
最多: 150 张
...
```

## 字符映射系统

### 字符ID分配规则

| ID范围 | 用途 | 说明 |
|--------|------|------|
| 0-6762 | GB2312标准汉字 | 固定映射，永不改变 |
| 6763-6862 | 预留ID | 特殊用途 |
| 6863+ | 自定义扩展汉字 | 按顺序追加 |

### 映射管理器使用

```python
from configs.char_mapping import CharMappingManager

manager = CharMappingManager()

# 获取字符ID
label_id = manager.get_label_id("中")  # 返回 3619
label_id = manager.get_label_id("国")  # 返回 935

# 获取汉字
char = manager.get_char(3619)  # 返回 "中"

# 添加新字符（自动分配ID）
success, label_id = manager.add_custom_char("龢", "hé", "龠", 22, "U+9F92")

# 获取统计信息
stats = manager.get_stats()
print(f"标准字符数: {stats['standard_chars_count']}")  # 6763
print(f"自定义字符数: {stats['custom_chars_count']}")
print(f"总字符数: {stats['total_chars_count']}")
```

### 初始化脚本

```bash
# 初始化GB2312-80标准字符集
python configs/char_mapping/init_gb2312.py

# 输出示例：
# Getting GB2312-80 character set...
# Got 6763 characters
# Initialization complete!
#   Total chars: 6763
#   Standard char ID range: 0-6762
#   Custom char start ID: 6863
```

## 配置文件说明

### char_to_label.json
```json
{
  "version": "1.0",
  "charset": "GB2312-80",
  "charset_size": 6763,
  "custom_start_id": 6863,
  "characters": {
    "中": {"label_id": 3619, "status": "standard"},
    "国": {"label_id": 935, "status": "standard"}
  },
  "custom_characters": {
    "笙": {"label_id": 6863, "status": "custom"}
  }
}
```

### char_properties.json
```json
{
  "characters": {
    "中": {
      "pinyin": "zhōng",
      "radical": "丨",
      "stroke_count": 4,
      "unicode": "U+4E2D"
    }
  }
}
```

## 训练配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 训练集比例 | 80% | 训练/验证/测试分割比例 |
| 验证集比例 | 10% | - |
| 测试集比例 | 10% | - |
| 数据均衡目标 | 30-150张/汉字 | 均衡后的样本数范围 |
| 图像尺寸 | 64x64 | 输入模型前的图像大小 |
| 训练轮次 | 自定义 | 通过 EPOCHS 配置 |
| 批次大小 | 自定义 | 通过 BATCH_SIZE 配置 |
