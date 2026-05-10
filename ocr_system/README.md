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
├── models/                 # 模型模块
│   └── resnet_ocr.py       # ResNet OCR模型
├── data/                   # 训练数据
│   ├── train.csv           # 训练集
│   └── val.csv             # 验证集
├── scripts/                # 脚本文件
│   ├── collect_dataset_data.py     # 数据集数据收集
│   ├── export_train_data.py        # 导出标注数据
│   ├── balance_data.py             # 数据均衡处理
│   ├── data_augmentation.py        # 数据增强
│   └── analyze_data_distribution.py # 数据分布分析
├── train.py                # 训练脚本
├── infer.py                # 推理脚本
└── config.py               # 配置文件
```

## 核心特性

### 1. 字符映射系统
- 使用 GB2312-80 标准字符集（6763个汉字）
- 稳定的标签ID分配：标准汉字ID固定（0-6762），自定义字符从6863开始追加
- 支持自动注册新汉字，保证向后兼容
- 新汉字不影响现有模型

### 2. 数据管理
- 支持多数据集统一管理
- 自动从标注系统导入已标注数据
- 数据均衡和增强功能
- 训练/验证集自动分割（80%/20%）

### 3. 模型训练
- 基于 ResNet50 的 OCR 模型
- 支持单通道灰度图像输入
- 自动获取类别数量

## 快速开始

### 1. 初始化字符映射（首次使用）
```bash
python configs/char_mapping/init_gb2312.py
```

### 2. 收集数据集数据
```bash
# 收集指定数据集的标注数据
python scripts/collect_dataset_data.py --dataset pdf01

# 收集多个数据集
python scripts/collect_dataset_data.py --dataset pdf5823
```

### 3. 数据均衡（可选）
```bash
python scripts/balance_data.py
```

### 4. 数据增强（可选）
```bash
python scripts/data_augmentation.py
```

### 5. 模型训练
```bash
python train.py
```

### 6. 推理预测
```bash
# 单张图片预测
python infer.py --image test.png

# 批量预测
python infer.py --batch img1.png img2.png img3.png
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

## 数据收集示例

```bash
# 收集pdf01数据集的标注数据
$ python scripts/collect_dataset_data.py --dataset pdf01

START: 开始收集数据集: pdf01
SUCCESS: 共收集到 23711 张已标注图片
INFO: 更新字符映射...
SUCCESS: 字符映射已更新
INFO: 训练集: 18404 张, 验证集: 5307 张
SAVE: 数据已保存到: data\train.csv
SAVE: 数据已保存到: data\val.csv

DONE: 数据收集完成！
  数据集: pdf01
  总标注数: 23711
  训练集: 18404
  验证集: 5307
  覆盖汉字数: 1164
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
| 训练集比例 | 80% | 训练/验证分割比例 |
| 验证集比例 | 20% | - |
| 数据均衡目标 | 30-150张/汉字 | 均衡后的样本数范围 |
| 图像尺寸 | 64x64 | 输入模型前的图像大小 |
| 训练轮次 | 自定义 | 通过 EPOCHS 配置 |
| 批次大小 | 自定义 | 通过 BATCH_SIZE 配置 |