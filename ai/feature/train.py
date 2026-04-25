# -*- coding: utf-8 -*-
"""
汉字切割U-Net自编码器 - 抽象化训练脚本
功能：数据加载、模型训练、最优模型保存、混合精度加速
"""
import os
import sys
from pathlib import Path

# 修复环境冲突（必须放在最顶部）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- 1. 路径配置（全局统一） --------------------------
# 自动获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))  # Windows必须转字符串

# 导入项目依赖
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 导入项目模块
from ai.feature.dataset import create_dataloader
from ai.feature.model import UNetAutoEncoder
from ai.feature.loss import CombinedLoss

# -------------------------- 2. 核心配置（抽象集中管理） --------------------------
class TrainConfig:
    """训练配置类：所有超参数集中管理，修改更方便"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0  # Windows固定为0
    # 路径配置
    IMAGE_DIR = BASE_DIR / "images" / "filtered_images"
    SAVE_MODEL_PATH = Path(__file__).parent / "best_model.pth"

# -------------------------- 3. 工具函数（抽象复用） --------------------------
def build_model() -> torch.nn.Module:
    """构建并返回模型"""
    model = UNetAutoEncoder().to(TrainConfig.DEVICE)
    return model

def build_loss() -> torch.nn.Module:
    """构建损失函数"""
    return CombinedLoss().to(TrainConfig.DEVICE)

def build_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    """构建优化器"""
    return optim.AdamW(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY
    )

def build_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    """构建学习率调度器"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.EPOCHS)

def build_data_loader() -> tuple:
    """构建数据加载器"""
    return create_dataloader(
        TrainConfig.IMAGE_DIR,
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=TrainConfig.NUM_WORKERS
    )

# -------------------------- 4. 核心训练逻辑（抽象封装） --------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    dataset
) -> float:
    """
    单轮训练逻辑抽象
    :return: 本轮平均损失
    """
    model.train()
    total_loss = 0.0
    # 每轮重新采样数据
    dataset.generate_sample_indices()

    pbar = tqdm(loader, desc=f"训练中")
    for input_img, gt_img in pbar:
        # 数据迁移到设备
        input_img = input_img.to(TrainConfig.DEVICE)
        gt_img = gt_img.to(TrainConfig.DEVICE)

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            pred_img = model(input_img)
            loss = criterion(pred_img, gt_img)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计损失
        batch_loss = loss.item()
        total_loss += batch_loss
        pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

    return total_loss / len(loader)

def save_best_model(model: torch.nn.Module, save_path: Path):
    """保存最优模型"""
    torch.save(model.state_dict(), save_path)
    print(f"💾 最优模型已保存: {save_path}")

# -------------------------- 5. 主训练流程（极简清晰） --------------------------
def main():
    # 1. 初始化组件
    train_loader, train_dataset = build_data_loader()
    model = build_model()
    criterion = build_loss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler()

    # 2. 训练参数初始化
    best_loss = float("inf")
    print(f"🚀 训练启动 | 设备: {TrainConfig.DEVICE} | 总轮次: {TrainConfig.EPOCHS}")

    # 3. 循环训练
    for epoch in range(TrainConfig.EPOCHS):
        print(f"\n📌 Epoch {epoch + 1}/{TrainConfig.EPOCHS}")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, train_dataset)
        
        # 更新学习率
        scheduler.step()
        
        # 打印日志
        print(f"✅ Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_model(model, TrainConfig.SAVE_MODEL_PATH)

    print("\n🎉 训练任务全部完成！")

# 程序入口
if __name__ == "__main__":
    main()