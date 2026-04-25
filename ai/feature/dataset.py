import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
# ===================== 【只需修改这2个路径】 =====================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ====================== 【核心数据集类】 ======================
class OCRLineAutoEncoderDataset(Dataset):
    def __init__(
        self,
        line_img_dir,        # 长行图片所在文件夹
        crop_w=101,          # 裁剪宽度（固定你的101）
        target_h=55,         # 目标高度（固定你的55）
        samples_per_line=200 # 每一行采样多少个小图
    ):
        self.line_img_dir = line_img_dir
        self.crop_w = crop_w
        self.target_h = target_h
        self.samples_per_line = samples_per_line

        # 加载所有长行图片路径
        self.img_paths = [
            os.path.join(line_img_dir, f)
            for f in os.listdir(line_img_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.img_paths) == 0:
            raise Exception("未找到任何图片！请检查图片文件夹路径")

        # 第一次生成采样索引（每行随机采200个位置）
        self.generate_sample_indices()

    # ------------------- 核心：每Epoch重新采样 + 打乱 -------------------
    def generate_sample_indices(self):
        """
        每一轮训练重新采样：
        1. 对每一张长图，随机采200个裁剪起始位置
        2. 全部样本跨行打乱
        """
        self.samples = []
        for img_path in self.img_paths:
            # 打开图片获取宽度
            with Image.open(img_path) as img:
                img_w = img.width

            # 长图必须比裁剪窗口宽
            max_x = img_w - self.crop_w
            if max_x <= 0:
                continue

            # 从这一行随机采 200 个位置
            for _ in range(self.samples_per_line):
                x = random.randint(0, max_x)
                self.samples.append((img_path, x))

        # ✅ 关键：所有样本全局打乱
        random.shuffle(self.samples)

    # ------------------- 过滤全白样本（文字占比 < 5% 丢弃） -------------------
    def is_invalid_patch(self, img_pil):
        img_np = np.array(img_pil) / 255.0
        text_ratio = np.mean(img_np < 0.95)  # 非白色像素比例
        return text_ratio < 0.05             # 文字太少 = 无效样本

    # ===================== 【修复】正确的单通道灰度图填充（无报错、无黑边） =====================
    def pad_height(self, img_pil):
        w, h = img_pil.size
        if h >= self.target_h:
            # 超出高度直接裁剪
            return img_pil.crop((0, 0, w, self.target_h))

        # 计算填充高度
        pad_top = (self.target_h - h) // 2
        pad_bottom = self.target_h - h - pad_top

        # 创建新画布（单通道灰度图）
        new_img = Image.new('L', (w, self.target_h))
        # 粘贴原始图片到中间
        new_img.paste(img_pil, (0, pad_top))

        # 边缘复制填充顶部
        top_edge = img_pil.crop((0, 0, w, 1))
        for i in range(pad_top):
            new_img.paste(top_edge, (0, i))

        # 边缘复制填充底部
        bottom_edge = img_pil.crop((0, h-1, w, h))
        for i in range(pad_bottom):
            new_img.paste(bottom_edge, (0, pad_top + h + i))

        return new_img

    # ------------------- 中间画黄线（输入图带线，GT不带线） -------------------
    def draw_yellow_line(self, gray_pil):
        # 灰度 → RGB
        img = np.array(gray_pil)
        img_rgb = np.stack([img, img, img], axis=-1)
        mid_x = self.crop_w // 2
        img_rgb[:, mid_x, 0] = 255  # R
        img_rgb[:, mid_x, 1] = 255  # G
        img_rgb[:, mid_x, 2] = 0    # B → 黄线
        return Image.fromarray(img_rgb)

    # ------------------- 归一化到 [-1, 1]（适配U-Net的Tanh输出） -------------------
    def norm(self, x):
        return x / 127.5 - 1.0

    # ====================== Dataset 必须实现 ======================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 获取采样的 图片路径 + 裁剪位置
        img_path, x_start = self.samples[idx]

        # 2. 打开灰度图
        img = Image.open(img_path).convert("L")

        # 3. 裁剪 101 宽度
        patch = img.crop((x_start, 0, x_start + self.crop_w, img.height))

        # 4. 过滤无效全白样本（重采）
        if self.is_invalid_patch(patch):
            return self.__getitem__(random.randint(0, len(self)-1))

        # 5. 高度填充到 55
        patch = self.pad_height(patch)

        # 6. 构建：输入图（带黄线） + 目标图（GT，不带线）
        input_pil = self.draw_yellow_line(patch)
        gt_pil = patch

        # 7. 转张量 + 归一化
        input_tensor = self.norm(torch.FloatTensor(np.array(input_pil)).permute(2, 0, 1))  # [3,55,101]
        gt_tensor = self.norm(torch.FloatTensor(np.array(gt_pil)).unsqueeze(0))            # [1,55,101]

        return input_tensor, gt_tensor


# ====================== 【训练/验证集划分】 ======================
def split_train_val_dir(original_dir, train_ratio=0.9):
    """
    按 行图 划分训练集/验证集（绝对不能切分行！）
    """
    all_paths = [
        os.path.join(original_dir, f)
        for f in os.listdir(original_dir)
        if f.endswith(('.png', '.jpg'))
    ]
    random.shuffle(all_paths)
    split_idx = int(len(all_paths) * train_ratio)
    return all_paths[:split_idx], all_paths[split_idx:]


# ====================== 【创建 DataLoader】 ======================
def create_dataloader(
    img_dir,
    batch_size=32,
    shuffle=False,
    num_workers=0  # 【修复】Windows强制设为0，多进程必报错
):
    dataset = OCRLineAutoEncoderDataset(img_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 已修复
        pin_memory=True
    )
    return loader, dataset


# ====================== 【使用示例】 ======================
if __name__ == '__main__':
    # ========== 只需要改这里：你的长行图片文件夹 ==========
    LINE_IMG_FOLDER = Path(PROJECT_DIR,"images","filtered_images")

    # 1. 创建数据集 & 加载器
    train_loader, train_dataset = create_dataloader(
        str(LINE_IMG_FOLDER),
        batch_size=32,
        num_workers=8  # 【修复】Windows必须=0
    )

    # 2. 测试一轮
    for batch_idx, (input_img, gt_img) in enumerate(train_loader):
        print(f"输入图形状: {input_img.shape}")  # [32, 3, 55, 101] 带黄线
        print(f"目标图形状: {gt_img.shape}")    # [32, 1, 55, 101] 原图GT
        break

    print("数据集加载成功！")