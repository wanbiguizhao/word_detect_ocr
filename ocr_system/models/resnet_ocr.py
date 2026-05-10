import torch
import torch.nn as nn
import os
import sys
from torchvision import models
from torchvision.models import ResNet50_Weights

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_resnet50_from_local(pretrained_model_dir):
    """尝试从本地预训练模型目录加载 ResNet50"""
    model_filename = "resnet50-0676ba61.pth"
    local_model_path = os.path.join(pretrained_model_dir, model_filename)
    
    if os.path.exists(local_model_path):
        print("Loading pretrained model from local:", local_model_path)
        model = models.resnet50(weights=None)
        state_dict = torch.load(local_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model
    else:
        print("Local model not found, loading with default method:", local_model_path)
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


def get_default_num_classes():
    """从全局字符映射获取类别数量"""
    try:
        from configs.char_mapping import CharMappingManager
        manager = CharMappingManager()
        return manager.get_stats()["next_custom_id"]
    except ImportError:
        try:
            from ocr_system.configs.char_mapping import CharMappingManager
            manager = CharMappingManager()
            return manager.get_stats()["next_custom_id"]
        except Exception as e:
            print("Warning: Cannot load char mapping, using default:", e)
            return 7000  # GB2312(6763) + 预留(100) + 部分自定义


class ResNetOCR(nn.Module):
    def __init__(self, num_classes=None, pretrained=True, pretrained_model_dir=None):
        super(ResNetOCR, self).__init__()
        
        # 如果未指定类别数，从全局映射获取
        if num_classes is None:
            num_classes = get_default_num_classes()
        
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
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """解冻骨干网络参数"""
        for param in self.resnet.parameters():
            param.requires_grad = True