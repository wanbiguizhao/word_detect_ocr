import sys
from pathlib import Path
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 修复matplotlib无界面服务器运行问题
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from ai.segment import config 
import torch
import torch.nn as nn
from tqdm import tqdm
from ai.segment.dataset import create_dataloader,split_dataset
from ai.segment.model import CharSegmentClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

# ===================== FocalLoss 不变 =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = targets * inputs + (1 - targets) * (1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ===================== 优化版可视化函数（带浅绿色边框） =====================
def visualize_inference_samples(model, dataloader, device, epoch, step, sample_num=20):
    """
    训练中动态生成可视化热力图 + 浅绿色边框
    :param epoch: 当前训练轮次
    :param step: 当前训练步数
    """
    model.eval()  # 推理模式
    dataset = dataloader.dataset
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(sample_num, len(dataset)))
    
    # 自动创建temp文件夹，防止保存失败
    save_dir = Path(__file__).resolve().parent / "temp"
    save_dir.mkdir(exist_ok=True)
    
    samples = []
    with torch.no_grad():
        for idx in sample_indices:
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)
            pred_prob = model(img).item()
            pred_label = 1 if pred_prob > 0.5 else 0
            img_np = img.squeeze(0).permute(1,2,0).cpu().numpy()
            img_np = (img_np * 127.5 + 127.5).astype(np.uint8)
            gray_img = img_np.mean(axis=2).astype(np.uint8)
            
            samples.append({
                "img": gray_img,
                "true_label": int(label.item()),
                "pred_prob": round(pred_prob, 3),
                "pred_label": pred_label
            })

    plt.figure(figsize=(20, 16))
    # 浅绿色边框颜色
    border_color = "lightgreen"
    for i, sample in enumerate(samples):
        plt.subplot(4, 5, i+1)
        plt.imshow(sample["img"], cmap="gray")
        
        true_lb = sample["true_label"]
        pred_lb = sample["pred_label"]
        prob = sample["pred_prob"]
        color = "green" if true_lb == pred_lb else "red"
        plt.title(f"True:{true_lb}\nProb:{prob}\nPred:{pred_lb}", color=color, fontsize=10)
        
        # ========== 🔥 新增：浅绿色边框 ==========
        # 隐藏刻度，保留边框
        plt.xticks([])
        plt.yticks([])
        # 设置所有边框为浅绿色
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)  # 边框宽度
        # ======================================

    plt.tight_layout()
    # 动态命名保存
    save_path = save_dir / f"train_visualize_epoch{epoch}_step{step}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n🎨 训练可视化已保存：{save_path}")

# ===================== 训练主函数（无改动） =====================
def train_classifier():
    all_img_paths = []
    all_label_paths = []
    for img_path in Path(config.DATA_DIR).glob("*.png"):
        label_path = Path(config.DATA_DIR) / f"{img_path.stem}.json"
        if label_path.exists():
            all_img_paths.append(img_path)
            all_label_paths.append(label_path)

    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = split_dataset(
        all_img_paths, all_label_paths, 
        train_ratio=0.7, val_ratio=0.15
    )

    train_loader, _ = create_dataloader(train_imgs, train_labels, split="train", 
                                        batch_size=config.BATCH_SIZE, white_skip_prob=0.85)
    val_loader, _ = create_dataloader(val_imgs, val_labels, split="val", 
                                    batch_size=config.BATCH_SIZE, white_skip_prob=0.85)

    model = CharSegmentClassifier(config.PRETRAINED_AE_PATH).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.LR)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    best_val_f1 = 0

    # 每200个训练样本可视化一次
    VISUALIZE_INTERVAL = 200*32
    sample_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        model.encoder.eval()
        train_loss = 0
        train_TP = train_FP = train_FN = train_TN = 0
        
        train_loader.dataset.generate_sample_indices()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} Train")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE).float().unsqueeze(1)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = (outputs > 0.5).float()
            train_TP += ((pred == 1) & (labels == 1)).sum().item()
            train_FP += ((pred == 1) & (labels == 0)).sum().item()
            train_FN += ((pred == 0) & (labels == 1)).sum().item()
            train_TN += ((pred == 0) & (labels == 0)).sum().item()
            
            train_loss += loss.item() * imgs.size(0)
            batch_acc = ((pred == labels).sum().item()) / labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=batch_acc)

            # 累计样本数，每200个生成一次热力图
            sample_counter += imgs.size(0)
            if sample_counter >= VISUALIZE_INTERVAL:
                visualize_inference_samples(model, val_loader, config.DEVICE, epoch+1, batch_idx+1)
                sample_counter = 0
                model.train()
                model.encoder.eval()

        # 训练集指标计算
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = (train_TP + train_TN) / (train_TP + train_FP + train_FN + train_TN + 1e-6)
        train_precision = train_TP / (train_TP + train_FP + 1e-6)
        train_recall = train_TP / (train_TP + train_FN + 1e-6)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-6)

        # 验证阶段
        model.eval()
        val_loss = val_TP = val_FP = val_FN = val_TN = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE).float().unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                pred = (outputs > 0.5).float()
                val_TP += ((pred == 1) & (labels == 1)).sum().item()
                val_FP += ((pred == 1) & (labels == 0)).sum().item()
                val_FN += ((pred == 0) & (labels == 1)).sum().item()
                val_TN += ((pred == 0) & (labels == 0)).sum().item()
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = (val_TP + val_TN) / (val_TP + val_FP + val_FN + val_TN + 1e-6)
        val_precision = val_TP / (val_TP + val_FP + 1e-6)
        val_recall = val_TP / (val_TP + val_FN + 1e-6)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-6)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train -> Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✅ 最优模型已保存 | Val F1: {best_val_f1:.4f}")

    print("🎯 训练完成！")

if __name__ == "__main__":
    train_classifier()