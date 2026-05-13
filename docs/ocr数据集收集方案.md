# OCR数据集收集方案

## 1. 方案概述

本方案旨在将 `pdf01` 和 `pdf5823` 两个数据集的标注数据统一收集，构建一个高质量的汉字OCR训练数据集。

**目标**：
- 统一两个数据集的标注格式
- 去重合并标注数据
- 构建标准化的数据集目录结构
- 支持后续OCR模型训练

## 2. 当前数据结构分析

### 2.1 pdf01 数据集
```
pdf01/
├── clusters/
│   ├── hog_clusters.json    # 聚类数据
│   └── labeling/
│       └── labels.json      # 标注数据
├── pdf_images/              # PDF页面图片
├── pdf_lines/               # 行图片
└── lineage.json             # 血缘数据
```

### 2.2 pdf5823 数据集
```
pdf5823/
├── clusters/
│   ├── hog_clusters.json    # 聚类数据
│   └── labeling/
│       └── labels.json      # 标注数据
├── model_infer/             # 模型推理结果
├── model_jsons/             # 模型JSON文件
├── unified_labels.json      # 统一标注（5818条）✓ 推荐作为主数据源
├── evaluation_results.txt   # 评估结果
├── labeling_tasks.json      # 标注任务
└── lineage.json             # 血缘数据
```

### 2.3 统一标注格式（unified_labels.json）

`pdf5823` 已有的统一标注格式非常完善：

```json
{
  "datasets": ["pdf5823"],
  "total_labeled": 5818,
  "dataset_stats": {
    "pdf5823": {"labeled_count": 5818}
  },
  "char_distribution": {
    "的": 226,
    "年": 85,
    "蚕": 81,
    ...
  },
  "annotations": [
    {
      "char_id": "page_10_line_10_char_10",
      "char": "型",
      "image_path": "datahome\\pdf5823\\pdf_chars\\page_10_line_10_char_10.png",
      "dataset": "pdf5823",
      "status": "labeled",
      "cluster_id": ""
    }
  ]
}
```

## 3. 数据集合并方案

### 3.1 合并策略

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | pdf5823/unified_labels.json | 最完整的标注数据，5818条 |
| 2 | pdf5823/clusters/labeling/labels.json | 聚类标注作为补充 |
| 3 | pdf01/clusters/labeling/labels.json | pdf01的标注数据 |

### 3.2 去重规则

1. **按 char_id 去重**：相同 `char_id` 只保留一条记录
2. **按 image_path 去重**：相同图片路径只保留一条记录
3. **冲突解决**：优先保留 `pdf5823` 的标注

### 3.3 目标数据集结构

```
ocr_dataset/
├── metadata.json           # 数据集元信息
├── char_distribution.json  # 字符分布统计
├── annotations/            # 标注记录
│   └── all_labels.json     # 所有标注
├── images/                 # 字符图片目录
│   ├── pdf01/              # pdf01的字符图片
│   └── pdf5823/            # pdf5823的字符图片
└── splits/                 # 数据集划分
    ├── train.txt           # 训练集
    ├── val.txt             # 验证集
    └── test.txt            # 测试集
```

## 4. 数据集格式规范

### 4.1 标注记录格式

```json
{
  "id": "唯一标识符",
  "char_id": "page_X_line_Y_char_Z",
  "char": "汉字",
  "image_path": "images/pdf5823/page_X_line_Y_char_Z.png",
  "dataset": "pdf5823",
  "page": X,
  "line": Y,
  "char_index": Z,
  "status": "labeled",
  "cluster_id": "",
  "source": "unified_labels"
}
```

### 4.2 元信息格式

```json
{
  "name": "Chinese OCR Dataset",
  "version": "1.0",
  "description": "汉字OCR标注数据集，包含pdf01和pdf5823两个数据源",
  "total_samples": 0,
  "total_chars": 0,
  "unique_chars": 0,
  "datasets": ["pdf01", "pdf5823"],
  "dataset_stats": {
    "pdf01": {"samples": 0},
    "pdf5823": {"samples": 0}
  },
  "created_at": "2026-05-12",
  "updated_at": "2026-05-12"
}
```

## 5. 实现脚本

### 5.1 数据集收集脚本

创建 `scripts/collect_dataset.py`：

```python
import os
import json
import shutil
from pathlib import Path

class DatasetCollector:
    def __init__(self, datahome_path, output_path):
        self.datahome = Path(datahome_path)
        self.output = Path(output_path)
        self.output.mkdir(parents=True, exist_ok=True)
        
        self.annotations = []
        self.char_distribution = {}
        self.seen_char_ids = set()
        self.seen_image_paths = set()
    
    def load_unified_labels(self, dataset_name):
        """加载统一标注文件"""
        unified_path = self.datahome / dataset_name / "unified_labels.json"
        if not unified_path.exists():
            print(f"跳过: {unified_path} 不存在")
            return
        
        print(f"加载 {dataset_name} 的统一标注...")
        with open(unified_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for ann in data.get("annotations", []):
            self.add_annotation(ann, dataset_name, source="unified_labels")
    
    def load_cluster_labels(self, dataset_name):
        """加载聚类标注文件"""
        labels_path = self.datahome / dataset_name / "clusters" / "labeling" / "labels.json"
        clusters_path = self.datahome / dataset_name / "clusters" / "hog_clusters.json"
        
        if not labels_path.exists() or not clusters_path.exists():
            print(f"跳过: {labels_path} 或 {clusters_path} 不存在")
            return
        
        print(f"加载 {dataset_name} 的聚类标注...")
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        
        clusters = clusters_data.get("clusters", {})
        
        for cluster_id, label_info in labels_data.items():
            char_labels = label_info.get("char_labels", {})
            cluster_chars = clusters.get(cluster_id, [])
            
            for idx_str, char_label_info in char_labels.items():
                char = char_label_info.get("char")
                if not char:
                    continue
                
                try:
                    idx = int(idx_str)
                    if idx < len(cluster_chars):
                        char_info = cluster_chars[idx]
                        char_id = char_info.get("char_id", f"{dataset_name}_cluster_{cluster_id}_char_{idx}")
                        
                        self.add_annotation({
                            "char_id": char_id,
                            "char": char,
                            "image_path": f"datahome/{dataset_name}/pdf_chars/{char_id}.png",
                            "dataset": dataset_name,
                            "status": "labeled",
                            "cluster_id": cluster_id
                        }, dataset_name, source="cluster_labels")
                except (ValueError, IndexError):
                    continue
    
    def add_annotation(self, ann, dataset_name, source):
        """添加标注记录（自动去重）"""
        char_id = ann.get("char_id")
        image_path = ann.get("image_path", "")
        
        # 去重检查
        if char_id in self.seen_char_ids:
            return
        if image_path in self.seen_image_paths:
            return
        
        # 解析路径信息
        page, line, char_idx = self.parse_char_id(char_id)
        
        # 构建标准化记录
        record = {
            "id": f"{dataset_name}_{char_id}",
            "char_id": char_id,
            "char": ann.get("char", ""),
            "image_path": f"images/{dataset_name}/{char_id}.png",
            "dataset": dataset_name,
            "page": page,
            "line": line,
            "char_index": char_idx,
            "status": ann.get("status", "labeled"),
            "cluster_id": ann.get("cluster_id", ""),
            "source": source
        }
        
        self.annotations.append(record)
        self.seen_char_ids.add(char_id)
        self.seen_image_paths.add(image_path)
        
        # 更新字符分布
        char = ann.get("char", "")
        if char:
            self.char_distribution[char] = self.char_distribution.get(char, 0) + 1
    
    def parse_char_id(self, char_id):
        """从char_id解析页面、行、字符索引"""
        parts = char_id.split("_")
        page = None
        line = None
        char_idx = None
        
        try:
            for i, part in enumerate(parts):
                if part == "page" and i + 1 < len(parts):
                    page = int(parts[i+1])
                elif part == "line" and i + 1 < len(parts):
                    line = int(parts[i+1])
                elif part == "char" and i + 1 < len(parts):
                    char_idx = int(parts[i+1])
        except ValueError:
            pass
        
        return page, line, char_idx
    
    def copy_images(self):
        """复制字符图片到输出目录"""
        images_dir = self.output / "images"
        images_dir.mkdir(exist_ok=True)
        
        for ann in self.annotations:
            dataset = ann["dataset"]
            char_id = ann["char_id"]
            
            # 源路径
            src_path = self.datahome / dataset / "pdf_chars" / f"{char_id}.png"
            
            # 目标路径
            dst_dir = images_dir / dataset
            dst_dir.mkdir(exist_ok=True)
            dst_path = dst_dir / f"{char_id}.png"
            
            if src_path.exists() and not dst_path.exists():
                shutil.copy(src_path, dst_path)
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """划分训练/验证/测试集"""
        import random
        
        splits_dir = self.output / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        indices = list(range(len(self.annotations)))
        random.shuffle(indices)
        
        total = len(indices)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        def write_split(name, indices_list):
            with open(splits_dir / f"{name}.txt", 'w', encoding='utf-8') as f:
                for idx in indices_list:
                    ann = self.annotations[idx]
                    f.write(f"{ann['image_path']}\t{ann['char']}\n")
        
        write_split("train", train_indices)
        write_split("val", val_indices)
        write_split("test", test_indices)
        
        print(f"数据集划分完成: 训练集{len(train_indices)} 验证集{len(val_indices)} 测试集{len(test_indices)}")
    
    def save(self):
        """保存数据集"""
        # 保存所有标注
        annotations_dir = self.output / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        with open(annotations_dir / "all_labels.json", 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        
        # 保存字符分布
        with open(self.output / "char_distribution.json", 'w', encoding='utf-8') as f:
            json.dump(self.char_distribution, f, ensure_ascii=False, indent=2)
        
        # 保存元信息
        metadata = {
            "name": "Chinese OCR Dataset",
            "version": "1.0",
            "description": "汉字OCR标注数据集，包含pdf01和pdf5823两个数据源",
            "total_samples": len(self.annotations),
            "total_chars": sum(self.char_distribution.values()),
            "unique_chars": len(self.char_distribution),
            "datasets": ["pdf01", "pdf5823"],
            "dataset_stats": {
                "pdf01": {"samples": sum(1 for a in self.annotations if a["dataset"] == "pdf01")},
                "pdf5823": {"samples": sum(1 for a in self.annotations if a["dataset"] == "pdf5823")}
            },
            "created_at": "2026-05-12",
            "updated_at": "2026-05-12"
        }
        with open(self.output / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"数据集保存完成！共 {len(self.annotations)} 条标注，{len(self.char_distribution)} 个不同汉字")
    
    def run(self):
        """执行完整的数据集收集流程"""
        print("开始收集数据集...")
        
        # 加载pdf5823的统一标注（优先级最高）
        self.load_unified_labels("pdf5823")
        
        # 加载pdf5823的聚类标注（补充）
        self.load_cluster_labels("pdf5823")
        
        # 加载pdf01的聚类标注
        self.load_cluster_labels("pdf01")
        
        # 复制图片
        self.copy_images()
        
        # 划分数据集
        self.split_dataset()
        
        # 保存结果
        self.save()

if __name__ == "__main__":
    collector = DatasetCollector(
        datahome_path="bussiness/datahome",
        output_path="datasets/ocr_dataset"
    )
    collector.run()
```

## 6. 执行步骤

### 6.1 运行脚本

```bash
cd d:\projects\word_detect_ocr
mkdir -p scripts
# 将上述脚本保存到 scripts/collect_dataset.py
python scripts/collect_dataset.py
```

### 6.2 预期输出

```
开始收集数据集...
加载 pdf5823 的统一标注...
加载 pdf5823 的聚类标注...
加载 pdf01 的聚类标注...
数据集划分完成: 训练集XXX 验证集XXX 测试集XXX
数据集保存完成！共 XXXX 条标注，XXX 个不同汉字
```

### 6.3 输出目录结构

```
datasets/
└── ocr_dataset/
    ├── metadata.json           # 数据集元信息
    ├── char_distribution.json  # 字符分布统计
    ├── annotations/
    │   └── all_labels.json     # 所有标注记录
    ├── images/
    │   ├── pdf01/              # pdf01字符图片
    │   │   └── page_*.png
    │   └── pdf5823/            # pdf5823字符图片
    │       └── page_*.png
    └── splits/
        ├── train.txt           # 训练集列表
        ├── val.txt             # 验证集列表
        └── test.txt            # 测试集列表
```

## 7. 数据集使用说明

### 7.1 训练集格式

`splits/train.txt` 格式：
```
images/pdf5823/page_10_line_10_char_10.png	型
images/pdf5823/page_10_line_10_char_12.png	的
images/pdf01/page_1_line_0_char_0.png	中
```

### 7.2 加载数据集（PyTorch示例）

```python
import os

class OCRDataset(Dataset):
    def __init__(self, split_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        self.samples.append((parts[0], parts[1]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, char = self.samples[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('L')
        
        if self.transform:
            img = self.transform(img)
        
        return img, char

# 使用示例
train_dataset = OCRDataset(
    split_file='datasets/ocr_dataset/splits/train.txt',
    root_dir='datasets/ocr_dataset',
    transform=transforms.ToTensor()
)
```

## 8. 后续优化建议

### 8.1 数据清洗
- 过滤低质量标注（模糊、遮挡的图片）
- 检查标注一致性（同一图片多次标注是否一致）
- 移除标点符号和特殊字符

### 8.2 数据增强
- 旋转、平移、缩放变换
- 噪声添加
- 对比度调整

### 8.3 质量评估
- 计算标注准确率
- 检查字符分布均衡性
- 生成混淆矩阵分析

---

**方案总结**：本方案通过统一标注格式、自动去重、标准化目录结构，将两个数据集合并为一个高质量的OCR训练数据集，便于后续模型训练和评估。