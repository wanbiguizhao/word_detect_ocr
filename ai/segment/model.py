import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from ai.feature.model import UNetAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# 导入你之前写的数据集（必须同目录）
from ai.segment import config 

class CharSegmentClassifier(nn.Module):
    def __init__(self, pretrained_ae_path):
        super().__init__()
        # 1. 加载你训练好的 UNet 自编码器
        autoencoder = UNetAutoEncoder()
        # 加载预训练权重（关键！）
        autoencoder.load_state_dict(torch.load(pretrained_ae_path, map_location='cpu'))
        
        # 2. 提取编码器，作为特征提取器
        self.encoder = autoencoder.encoder
        
        # 3. 🔥 冻结编码器所有参数：不训练、不更新权重
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()  # 设为评估模式，固定BN层参数

        # --------------------- 分类头（仅这部分会被训练） ---------------------
        # 编码器输出 bottleneck 形状: [B, 256, 6, 12]
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 展平: 256*6*12 = 18432
            nn.Linear(18432, 512),         # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),               # 防止过拟合
            nn.Linear(512, 1),             # 二分类输出
            nn.Sigmoid()                   # 输出 0~1 概率
        )

    def forward(self, x):
        # 前向传播：只使用编码器的 bottleneck 特征
        _, _, _, bottleneck = self.encoder(x)
        # 分类头输出结果
        out = self.classifier(bottleneck)
        return out


if __name__ == "__main__":
    # 训练分类器
    # 测试模型加载补全这里代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型
    model = CharSegmentClassifier(config.PRETRAINED_AE_PATH).to(device)
    model.eval()  # 推理模式
    print("✅ 模型初始化成功！")

    # 3. 生成【伪输入张量】Fake Tensor (核心！模拟数据集的输入格式)
    # 输入维度：[batch_size, channels, height, width]
    # 对应你的数据：batch=2, 3通道, 高55, 宽101
    fake_input = torch.randn(2, 3, 55, 101).to(device)
    print(f"✅ 伪输入形状: {fake_input.shape}")

    # 4. 模型前向传播（推理）
    with torch.no_grad():  # 关闭梯度，纯推理
        fake_output = model(fake_input)

    # 5. 打印输出结果，验证维度
    print(f"✅ 模型输出形状: {fake_output.shape}")
    print(f"✅ 模型输出值:\n{fake_output}")
    print("\n🎉 模型输入输出测试通过！结构完全正常！")