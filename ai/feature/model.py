import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- 基础卷积块 ---------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ===================== 【编码器】 下采样，压缩特征 =====================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(128, 256)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        bottleneck = self.bottleneck(self.pool3(x3))
        
        return x1, x2, x3, bottleneck

# ===================== 【解码器】 上采样，还原图像 =====================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_conv1 = ConvBlock(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Tanh()
        )

    # 🔥 核心修复：自动对齐特征图尺寸，解决奇数高宽报错
    def align_and_concat(self, x1, x2):
        # 裁剪/填充，让x1和x2的H、W完全一致
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        return torch.cat([x1, x2], dim=1)

    def forward(self, x1, x2, x3, bottleneck):
        # 瓶颈 → 上采样 → 对齐拼接
        x = self.up3(bottleneck)
        x = self.align_and_concat(x, x3)  # ✅ 修复点
        x = self.up_conv3(x)

        x = self.up2(x)
        x = self.align_and_concat(x, x2)  # ✅ 修复点
        x = self.up_conv2(x)

        x = self.up1(x)
        x = self.align_and_concat(x, x1)  # ✅ 修复点
        x = self.up_conv1(x)

        return self.out(x)

# ===================== 完整自编码器 = 编码器 + 解码器 =====================
class UNetAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # 编码
        x1, x2, x3, bottleneck = self.encoder(x)
        # 解码
        out = self.decoder(x1, x2, x3, bottleneck)
        return out

# --------------------- 测试 ---------------------
if __name__ == '__main__':
    model = UNetAutoEncoder()
    # 你的输入尺寸 3x55x101（奇数高度，完美运行）
    test_input = torch.randn(1, 3, 55, 101)
    test_output = model(test_input)
    print(f"输入: {test_input.shape}")
    print(f"输出: {test_output.shape}")  # [1, 1, 55, 101]