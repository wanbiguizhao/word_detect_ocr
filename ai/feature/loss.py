import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def gradient_loss(self, pred, target):
        # 文字边缘损失
        pred_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_h = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_v = target[:, :, 1:, :] - target[:, :, :-1, :]
        return torch.abs(pred_h - target_h).mean() + torch.abs(pred_v - target_v).mean()

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_grad = self.gradient_loss(pred, target)
        return 0.7 * loss_l1 + 0.3 * loss_grad