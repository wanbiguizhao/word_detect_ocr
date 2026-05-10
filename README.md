# OCR汉字识别系统

基于深度学习的汉字识别系统，支持从图像中检测和识别汉字。

## 功能特性

- ✅ 汉字字符级识别
- ✅ 支持多数据集训练
- ✅ 迁移学习（基于ResNet50）
- ✅ 模型评估工具
- ✅ 批量推理支持

## 项目结构

```
word_detect_ocr/
├── ocr_system/                    # OCR核心模块
│   ├── configs/                   # 配置文件
│   │   └── char_mapping/          # 字符映射配置
│   │       ├── char_to_label.json
│   │       ├── label_to_char.json
│   │       └── manager.py
│   ├── data/                      # 训练/验证数据
│   │   ├── train_rel.csv
│   │   └── val_rel.csv
│   ├── datasets/                  # 数据集处理
│   │   └── char_dataset.py
│   ├── evaluation/                # 评估模块
│   │   ├── evaluator.py
│   │   └── pdf_evaluator.py
│   ├── model_store/               # 模型存储
│   │   ├── inference/             # 推理模型
│   │   ├── pretrained/            # 预训练模型
│   │   └── trained/               # 训练模型
│   ├── models/                    # 模型定义
│   │   └── resnet_ocr.py
│   ├── infer.py                   # 推理模块
│   ├── train.py                   # 训练脚本
│   └── evaluate.py                # 评估脚本
├── bussiness/                     # 业务数据
│   └── datahome/                  # 数据集目录
│       ├── pdf01/                 # PDF数据集示例
│       └── pdf5823/               # PDF数据集示例
├── evaluate.py                    # 统一评估入口
└── README.md
```

## 安装依赖

```bash
# 创建虚拟环境
python -m venv .ocr_venv

# 激活虚拟环境
# Windows
.ocr_venv\Scripts\Activate.ps1
# Linux/Mac
source .ocr_venv/bin/activate

# 安装依赖
pip install torch torchvision pillow tqdm pandas
```

## 数据准备

### 数据集格式

训练数据需要CSV文件，格式如下：

```csv
image_path,char
images/001.png,中
images/002.png,国
images/003.png,人
```

### 字符映射

字符映射文件位于 `ocr_system/configs/char_mapping/`：
- `char_to_label.json`: 字符到标签ID的映射
- `label_to_char.json`: 标签ID到字符的映射

## 训练

```bash
cd ocr_system
python train.py --epochs 50 --batch_size 64 --lr 0.001
```

训练策略：
- 前10个epoch：冻结ResNet50 backbone，只训练分类头
- 第10个epoch后：解冻backbone，使用更低学习率进行微调

## 推理

### 单张图像推理

```python
from ocr_system.infer import load_model, get_char_mapping, predict
from torchvision import transforms

# 获取字符映射
idx_to_char, _ = get_char_mapping()

# 加载模型
model = load_model('ocr_system/model_store/inference/ocr_model_best.pth')

# 定义变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 预测
pred_char, confidence, label_id = predict('test.png', model, transform, idx_to_char)
print(f"预测字符: {pred_char}, 置信度: {confidence:.4f}")
```

## 评估

### 评估标注数据

```bash
# 评估PDF数据集的标注数据
python evaluate.py pdf5823 --mode labeled

# 评估所有数据（标注+未标注）
python evaluate.py pdf5823 --mode all

# 评估验证集
python evaluate.py --mode val
```

### 评估模式说明

| 模式 | 说明 |
|------|------|
| `labeled` | 只评估已标注的数据，计算准确率 |
| `all` | 评估所有数据（标注+未标注） |
| `mixed` | 评估所有数据，标记标注状态 |
| `val` | 评估验证集 |

## 模型导出

训练完成后，最佳模型会保存到：
- `ocr_system/model_store/trained/ocr_model_best.pth`

用于推理的模型需要复制到：
- `ocr_system/model_store/inference/ocr_model_best.pth`

## 技术栈

- **框架**: PyTorch 2.x
- **模型**: ResNet50 + 自定义分类头
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss（带类别权重）

## 注意事项

1. 确保训练数据中字符图像尺寸统一
2. 字符映射需要包含所有训练数据中的字符
3. 训练前确保预训练模型已下载
4. 推理时使用 `inference` 目录下的模型

## 许可证

MIT License