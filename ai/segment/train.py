import sys
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg')
import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from ai.segment import config
import torch
import torch.nn as nn
from tqdm import tqdm
from ai.segment.dataset import create_dataloader, split_dataset
from ai.segment.model import CharSegmentClassifier
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

# ===================== FocalLoss =====================
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

# ===================== 可视化函数 =====================
def visualize_inference_samples(model, dataloader, device, epoch, step, sample_num=20):
    model.eval()
    dataset = dataloader.dataset
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(sample_num, len(dataset)))

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
    border_color = "lightgreen"
    for i, sample in enumerate(samples):
        plt.subplot(4, 5, i+1)
        plt.imshow(sample["img"], cmap="gray")

        true_lb = sample["true_label"]
        pred_lb = sample["pred_label"]
        prob = sample["pred_prob"]
        color = "green" if true_lb == pred_lb else "red"
        plt.title(f"True:{true_lb}\nProb:{prob}\nPred:{pred_lb}", color=color, fontsize=10)
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    save_path = save_dir / f"train_visualize_epoch{epoch}_step{step}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n🎨 训练可视化已保存: {save_path}")

# ===================== 训练主函数 =====================
def train_classifier():
    log_dir = Path(__file__).resolve().parent / "runs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"segment_training_{timestamp}"
    writer = SummaryWriter(log_dir=str(run_dir))

    print("="*60)
    print("🚀 训练开始")
    print(f"📁 数据目录: {config.DATA_DIR}")
    print(f"📁 模型保存路径: {config.MODEL_SAVE_PATH}")
    print(f"📁 TensorBoard日志: {run_dir}")
    print(f"🔧 批次大小: {config.BATCH_SIZE}")
    print(f"🔧 训练轮数: {config.EPOCHS}")
    print(f"🔧 学习率: {config.LR}")
    print(f"🖥️  设备: {config.DEVICE}")
    print("="*60)
    print(f"\n💡 TensorBoard查看命令: tensorboard --logdir={log_dir}")
    print("="*60)

    # 加载数据
    print("\n📂 加载数据集...")
    all_img_paths = []
    all_label_paths = []
    for img_path in Path(config.DATA_DIR).glob("*.png"):
        label_path = Path(config.DATA_DIR) / f"{img_path.stem}.json"
        if label_path.exists():
            all_img_paths.append(img_path)
            all_label_paths.append(label_path)

    print(f"✅ 找到 {len(all_img_paths)} 个训练样本")

    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = split_dataset(
        all_img_paths, all_label_paths,
        train_ratio=0.7, val_ratio=0.15
    )

    print(f"📊 数据划分: 训练集 {len(train_imgs)} | 验证集 {len(val_imgs)} | 测试集 {len(test_imgs)}")

    train_loader, _ = create_dataloader(train_imgs, train_labels, split="train",
                                        batch_size=config.BATCH_SIZE, white_skip_prob=0.85)
    val_loader, _ = create_dataloader(val_imgs, val_labels, split="val",
                                    batch_size=config.BATCH_SIZE, white_skip_prob=0.85)

    # 初始化模型
    print("\n🧠 初始化模型...")
    model = CharSegmentClassifier(config.PRETRAINED_AE_PATH).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.LR)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    best_val_f1 = 0
    best_epoch = 0

    VISUALIZE_INTERVAL = 200 * 32
    sample_counter = 0

    # 早停参数
    EARLY_STOPPING_PATIENCE = 10
    epochs_without_improvement = 0
    convergence_epoch = None

    print(f"⏰ 早停耐心值: {EARLY_STOPPING_PATIENCE} 轮")
    print("="*60)

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"📍 Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*60}")

        model.train()
        model.encoder.eval()
        train_loss = 0
        train_TP = train_FP = train_FN = train_TN = 0

        train_loader.dataset.generate_sample_indices()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE).float().unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=5.0)

            optimizer.step()

            pred = (outputs > 0.5).float()
            train_TP += ((pred == 1) & (labels == 1)).sum().item()
            train_FP += ((pred == 1) & (labels == 0)).sum().item()
            train_FN += ((pred == 0) & (labels == 1)).sum().item()
            train_TN += ((pred == 0) & (labels == 0)).sum().item()

            train_loss += loss.item() * imgs.size(0)
            batch_acc = ((pred == labels).sum().item()) / labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=batch_acc)

            sample_counter += imgs.size(0)
            if sample_counter >= VISUALIZE_INTERVAL:
                visualize_inference_samples(model, val_loader, config.DEVICE, epoch, batch_idx+1)
                sample_counter = 0
                model.train()
                model.encoder.eval()

        # 训练集指标
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

        # ========== TensorBoard 记录标量 ==========
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Best_F1/val', best_val_f1, epoch)

        # 记录TP/FP/FN/TN
        writer.add_scalar('Stats/train_TP', train_TP, epoch)
        writer.add_scalar('Stats/train_FP', train_FP, epoch)
        writer.add_scalar('Stats/train_FN', train_FN, epoch)
        writer.add_scalar('Stats/train_TN', train_TN, epoch)
        writer.add_scalar('Stats/val_TP', val_TP, epoch)
        writer.add_scalar('Stats/val_FP', val_FP, epoch)
        writer.add_scalar('Stats/val_FN', val_FN, epoch)
        writer.add_scalar('Stats/val_TN', val_TN, epoch)

        # ========== 打印Epoch报告 ==========
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch}/{config.EPOCHS} 训练报告")
        print(f"{'='*60}")
        print(f"  训练集: Loss={avg_train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}")
        print(f"  验证集: Loss={avg_val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f}")
        print(f"{'-'*60}")
        print(f"  � 历史最佳: Val F1={best_val_f1:.4f} (Epoch {best_epoch})")
        print(f"  📉 未改善轮数: {epochs_without_improvement}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  ✅ Val F1 提升至 {best_val_f1:.4f}！已保存最佳模型")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️ Val F1 未提升 ({epochs_without_improvement}轮)")

        # 学习率调度
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  📉 当前学习率: {current_lr:.6f}")

        # 收敛检测
        if len(writer.all_writers) > 0 and epoch >= 6:
            recent_f1s = []
            # 检查收敛
            if convergence_epoch is None and epochs_without_improvement >= 5:
                if val_f1 > 0.9:  # 假设F1>0.9为收敛阈值
                    convergence_epoch = epoch
                    print(f"  🎯 模型已收敛！")

        # 早停
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏰ 早停触发，训练结束！")
            break

    # 训练结束
    print("\n" + "="*60)
    print("🎯 训练完成！")
    print(f"📊 最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"📁 模型保存路径: {config.MODEL_SAVE_PATH}")
    print(f"📁 TensorBoard日志: {run_dir}")
    print(f"\n💡 查看训练曲线: tensorboard --logdir={run_dir}")
    print("="*60)

    writer.close()
    return best_val_f1, best_epoch

if __name__ == "__main__":
    best_f1, best_epoch = train_classifier()
