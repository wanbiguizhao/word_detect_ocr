import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CharDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_train=True):
        self.data = []
        self.char_to_idx = {}
        self.idx_to_char = []
        
        # 记录 CSV 文件所在目录（用于解析相对路径）
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                img_path, char = row[0], row[1]
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = len(self.idx_to_char)
                    self.idx_to_char.append(char)
                self.data.append((img_path, self.char_to_idx[char]))
        
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
        return len(self.idx_to_char)

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