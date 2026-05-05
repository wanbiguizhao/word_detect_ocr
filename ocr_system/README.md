# OCR训练系统

## 目录结构

```
ocr_system/
├── scripts/          # 脚本文件
│   ├── export_train_data.py      # 导出标注数据
│   ├── balance_data.py           # 数据均衡处理
│   ├── data_augmentation.py      # 数据增强
│   ├── train_ocr.py              # 模型训练
│   ├── run_ocr_training.py       # 一键执行
│   ├── analyze_data_distribution.py  # 数据分布分析
│   ├── smart_label.py            # 智能标注
│   └── batch_auto_label.py       # 批量标注
├── data/             # 训练数据
│   ├── ocr_train_data/           # 原始训练数据
│   ├── ocr_train_data_balanced/  # 均衡后数据
│   └── ocr_train_data_augmented/ # 增强后数据
├── models/           # 训练好的模型
└── reports/          # 分析报告
```

## 使用流程

1. **数据导出**
   ```bash
   python scripts/export_train_data.py
   ```

2. **数据均衡**
   ```bash
   python scripts/balance_data.py
   ```

3. **数据增强（可选）**
   ```bash
   python scripts/data_augmentation.py
   ```

4. **模型训练**
   ```bash
   python scripts/train_ocr.py
   ```

5. **一键执行**
   ```bash
   python scripts/run_ocr_training.py
   ```

## 配置说明

- 数据均衡目标：30-150张/汉字
- 训练/验证分割：80%/20%
- 数据增强倍数：约10倍
