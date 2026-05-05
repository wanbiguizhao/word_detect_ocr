import torch
import torch.nn as nn
import os
from torchvision import models
from torchvision.models import ResNet50_Weights

def load_resnet50_from_local(pretrained_model_dir):
    """尝试从本地预训练模型目录加载 ResNet50"""
    # ResNet50 预训练模型文件名
    model_filename = "resnet50-0676ba61.pth"
    local_model_path = os.path.join(pretrained_model_dir, model_filename)
    
    if os.path.exists(local_model_path):
        print(f"📦 从本地加载预训练模型: {local_model_path}")
        # 创建模型结构
        model = models.resnet50(weights=None)
        # 加载本地权重
        state_dict = torch.load(local_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model
    else:
        print(f"🔄 本地模型不存在，使用默认方式加载: {local_model_path}")
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

class ResNetOCR(nn.Module):
    def __init__(self, num_classes=1156, pretrained=True, pretrained_model_dir=None):
        super(ResNetOCR, self).__init__()
        if pretrained:
            if pretrained_model_dir:
                self.resnet = load_resnet50_from_local(pretrained_model_dir)
            else:
                self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # 修改第一层卷积以支持单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 解冻最后一层全连接
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """解冻骨干网络参数"""
        for param in self.resnet.parameters():
            param.requires_grad = True