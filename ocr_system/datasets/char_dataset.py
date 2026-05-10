import os
import sys
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.char_mapping import CharMappingManager
except ImportError:
    from ocr_system.configs.char_mapping import CharMappingManager


class CharDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_train=True, use_global_mapping=True):
        self.data = []
        self.use_global_mapping = use_global_mapping
        
        # 记录 CSV 文件所在目录（用于解析相对路径）
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))
        
        # 使用全局字符映射
        if self.use_global_mapping:
            self.mapping_manager = CharMappingManager()
            self.num_classes = self.mapping_manager.get_stats()["next_custom_id"]
        else:
            self.char_to_idx = {}
            self.idx_to_char = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                img_path, char = row[0], row[1]
                
                if self.use_global_mapping:
                    label_id = self.mapping_manager.get_label_id(char)
                    if label_id is None:
                        label_id = self.mapping_manager.add_custom_char(char)[1]
                else:
                    if char not in self.char_to_idx:
                        self.char_to_idx[char] = len(self.idx_to_char)
                        self.idx_to_char.append(char)
                    label_id = self.char_to_idx[char]
                
                self.data.append((img_path, label_id))
        
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.csv_dir, img_path)
        
        img = Image.open(img_path).convert('L')  # 转为灰度图
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_char_count(self):
        if self.use_global_mapping:
            return self.mapping_manager.get_stats()["total_chars_count"]
        return len(self.idx_to_char)
    
    def get_label_id(self, char):
        if self.use_global_mapping:
            return self.mapping_manager.get_label_id(char)
        return self.char_to_idx.get(char)
    
    def get_char(self, label_id):
        if self.use_global_mapping:
            return self.mapping_manager.get_char(label_id)
        if label_id < len(self.idx_to_char):
            return self.idx_to_char[label_id]
        return None


def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])