import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

from config import *
from datasets.char_dataset import CharDataset, get_transform
from models.resnet_ocr import ResNetOCR
from configs.char_mapping import CharMappingManager

# 全局字符映射管理器
mapping_manager = CharMappingManager()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )

def calculate_class_weights(dataset, num_classes):
    """计算类别权重以处理数据不均衡"""
    labels = [item[1] for item in dataset.data]
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # 保存原始计数用于后续判断
    original_counts = class_counts.copy()
    
    # 仅对有样本的类别计算权重
    class_counts = np.maximum(class_counts, 1e-5)
    weights = 1.0 / class_counts
    
    # 将没有样本的类别权重设为0（使用原始计数判断）
    weights[original_counts == 0] = 0.0
    
    # 归一化
    weights = weights / (weights.sum() + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    logging.info(f'Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, epoch=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于记录预测详情的列表
    prediction_details = []
    max_log_samples = 10  # 每个epoch最多记录10个样本

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录预测详情（只记录前max_log_samples个）
            if len(prediction_details) < max_log_samples:
                for i in range(len(labels)):
                    if len(prediction_details) >= max_log_samples:
                        break
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    true_char = mapping_manager.get_char(true_label)
                    pred_char = mapping_manager.get_char(pred_label)
                    prediction_details.append({
                        'true_char': true_char,
                        'true_label': true_label,
                        'pred_char': pred_char,
                        'pred_label': pred_label,
                        'correct': true_label == pred_label
                    })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 记录详细的预测结果
    logging.info(f'Valid - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    logging.info('--- 预测详情 ---')
    for i, detail in enumerate(prediction_details):
        status = '✓' if detail['correct'] else '✗'
        logging.info(f'  [{status}] 真实: "{detail["true_char"]}"(ID:{detail["true_label"]}) | '
                     f'预测: "{detail["pred_char"]}"(ID:{detail["pred_label"]})')
    logging.info('--- 预测详情结束 ---')
    
    return epoch_loss, epoch_acc

def main():
    setup_logging()
    logging.info('Starting training...')

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 加载数据集
    train_transform = get_transform(is_train=True)
    val_transform = get_transform(is_train=False)

    train_dataset = CharDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = CharDataset(VAL_CSV, transform=val_transform)

    logging.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    logging.info(f'Number of classes: {train_dataset.get_char_count()}')

    # 计算类别权重
    class_weights = calculate_class_weights(train_dataset, NUM_CLASSES).to(device)
    logging.info(f'Class weights calculated for {NUM_CLASSES} classes (active: {int((class_weights > 0).sum().item())})')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 创建模型（冻结部分层以防止过拟合）
    model = ResNetOCR(num_classes=NUM_CLASSES, pretrained_model_dir=PRETRAINED_MODEL_DIR).to(device)
    
    # 冻结前几层
    model.freeze_backbone()
    logging.info('Model backbone frozen')

    # 损失函数（使用类别权重）和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # 训练循环
    best_val_acc = 0.0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        # 解冻策略：在第10个epoch后解冻部分层
        if epoch == 9:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=LR/10, weight_decay=WEIGHT_DECAY)
            logging.info('Model backbone unfrozen, continuing training with lower LR')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logging.info(f'Best model saved with val acc: {best_val_acc:.4f}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                logging.info(f'Early stopping after {epoch+1} epochs')
                break

        # 保存当前模型
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    logging.info(f'Training finished. Best val acc: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()