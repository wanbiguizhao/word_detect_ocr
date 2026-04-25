import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import random
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

# 线段树实现（完全保留，无修改）
class SegmentTree:
    def __init__(self, intervals):
        self.intervals = intervals
        if not intervals:
            self.max_val = 0
            self.tree = []
            return
        self.max_val = max(interval[1] for interval in intervals)
        self.tree = [False] * (4 * (self.max_val + 1))
        for start, end in intervals:
            self._update(0, 0, self.max_val, start, end, True)

    def _update(self, node, node_left, node_right, update_left, update_right, value):
        if update_right < node_left or update_left > node_right:
            return
        if update_left <= node_left and node_right <= update_right:
            self.tree[node] = value
            return
        mid = (node_left + node_right) // 2
        self._update(2*node+1, node_left, mid, update_left, update_right, value)
        self._update(2*node+2, mid+1, node_right, update_left, update_right, value)
        self.tree[node] = self.tree[2*node+1] or self.tree[2*node+2]

    def query(self, point):
        if not self.intervals or point > self.max_val or point < 0:
            return False
        return self._query_recursive(0, 0, self.max_val, point)

    def _query_recursive(self, node, node_left, node_right, point):
        if self.tree[node]:
            return True
        if node_left == node_right:
            return False
        mid = (node_left + node_right) // 2
        if point <= mid:
            return self._query_recursive(2*node+1, node_left, mid, point)
        else:
            return self._query_recursive(2*node+2, mid+1, node_right, point)

# ====================== 配置参数 ======================
CROP_WIDTH = 101
TARGET_HEIGHT = 55
MID_COL = CROP_WIDTH // 2
STEP = 1
# 纯空白patch跳过概率（你的核心逻辑，保留）
WHITE_SKIP_PROB = 0.95


# ====================== 修复：全局一次性划分数据，杜绝重合 ======================
def split_dataset(img_paths, label_paths, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    全局只调用1次：固定随机种子，一次性划分 train/val/test
    返回划分好的文件列表，永远不重复、不重合
    """
    random.seed(seed)  # 固定种子，保证结果可复现
    combined = list(zip(img_paths, label_paths))
    random.shuffle(combined)
    
    total = len(combined)
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    
    # 固定划分
    train_files = combined[:train_num]
    val_files = combined[train_num:train_num+val_num]
    test_files = combined[train_num+val_num:]
    
    # 解包返回
    return (
        list(zip(*train_files)) if train_files else ([], []),
        list(zip(*val_files)) if val_files else ([], []),
        list(zip(*test_files)) if test_files else ([], []),
    )
class CharSegmentDataset(Dataset):
    # 🔥 修改 __init__：接收预划分的文件，不再内部重新划分
    def __init__(self, img_paths, label_paths, split="train", balance_ratio=0.25, white_skip_prob=0.95):
        self.split = split
        self.balance_ratio = balance_ratio
        self.white_skip_prob = white_skip_prob
        self.target_height = TARGET_HEIGHT
        self.crop_width = CROP_WIDTH
        self.step = STEP
        self.MID_COL = CROP_WIDTH // 2

        # 直接使用预划分的文件，无重复、无重合
        self.img_paths = list(img_paths)
        self.label_paths = list(label_paths)

        # 预解析样本
        self.all_samples = self._preparse_all_samples()
        # 仅训练集平衡
        self.balanced_samples = self._balance_samples() if self.split == "train" else self.all_samples


    # 纯空白判断（你的逻辑，完全保留，正确！）
    def _is_pure_white(self, gray_patch):
        extrema = gray_patch.getextrema()
        return extrema == (255, 255)

    def _preparse_all_samples(self):
        """你的核心逻辑：仅过滤纯纯白patch，间隙样本全部保留"""
        all_samples = []
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            img = Image.open(img_path).convert("L")
            W, H = img.size
            with open(label_path, encoding="utf-8") as f:
                char_regions = json.load(f).get("chars", [])
            
            char_intervals = [(c["col_start"], c["col_end"]) for c in char_regions]
            #st = SegmentTree(char_intervals)

            x_start = 0
            while x_start + self.crop_width <= W:
                mid = x_start + MID_COL
                label = 0
                for char_interval_pair in char_intervals:
                    if mid < char_interval_pair[0]:
                        continue
                    elif mid>= char_interval_pair[0] and mid<=char_interval_pair[1]:
                        label=1
                    elif char_interval_pair[0]>mid:
                        break
                # ====================== 你的核心逻辑：不动！ ======================
                if label == 0:
                    patch = img.crop((x_start, 0, x_start+self.crop_width, H))
                    if self._is_pure_white(patch):
                        if random.random() < self.white_skip_prob:
                            x_start += self.step
                            continue
                # =================================================================
                all_samples.append((str(img_path), x_start, label))
                x_start += self.step
        return all_samples

    # ====================== 核心修改2：优化平衡采样（不强制1:1） ======================
    def _balance_samples(self):
        """
        原逻辑：强制1:1 → 负样本太少，模型不会识别间隙
        新逻辑：正样本:负样本 = 3:7 → 保留大量间隙/空白样本
        """
        neg_samples = [s for s in self.all_samples if s[2] == 0]  # 空白/间隙
        pos_samples = [s for s in self.all_samples if s[2] == 1]  # 汉字

        if not pos_samples or not neg_samples:
            return self.all_samples

        # 关键：负样本数量 = 正样本数量 / 0.3 → 负样本更多
        target_neg_num = int(len(pos_samples) / self.balance_ratio)
        target_pos_num = len(pos_samples)

        # 负样本过多则降采样
        if len(neg_samples) > target_neg_num:
            neg_samples = random.sample(neg_samples, target_neg_num)
        
        # 正样本不足则重复（不破坏特征）
        if len(pos_samples) < target_pos_num:
            pos_samples = pos_samples * (target_pos_num // len(pos_samples) + 1)
            pos_samples = pos_samples[:target_pos_num]

        balanced = pos_samples + neg_samples
        random.shuffle(balanced)
        return balanced

    # 预处理（完全保留，与推理100%对齐）
    def _process_patch(self, img_path, x_start):
        img = Image.open(img_path).convert("L")
        patch = img.crop((x_start, 0, x_start + self.crop_width, img.height))

        # 高度对齐
        if patch.height < self.target_height:
            pad = self.target_height - patch.height
            patch = ImageOps.expand(patch, (0, pad//2, 0, pad-pad//2), fill=255)
        elif patch.height > self.target_height:
            patch = patch.crop((0, 0, self.crop_width, self.target_height))

        # 黄线绘制（与推理完全一致）
        patch_rgb = patch.convert("RGB")
        patch_np = torch.ByteTensor(torch.ByteStorage.from_buffer(patch_rgb.tobytes())).view(
            self.target_height, self.crop_width, 3)
        patch_np[:, MID_COL] = torch.tensor([255, 255, 0], dtype=torch.uint8)

        # 归一化
        patch_tensor = patch_np.float() / 127.5 - 1.0
        return patch_tensor.permute(2, 0, 1)

    def __getitem__(self, idx):
        img_path, x_start, label = self.balanced_samples[idx]
        return self._process_patch(img_path, x_start), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.balanced_samples)

    def generate_sample_indices(self):
        if self.split == "train":
            self.balanced_samples = self._balance_samples()
            random.shuffle(self.balanced_samples)

# -------------------------- 工具函数 --------------------------
def create_dataloader(img_paths, label_paths, split="train", batch_size=32, white_skip_prob=None):
    dataset = CharSegmentDataset(
        img_paths=img_paths,
        label_paths=label_paths,
        split=split,
        white_skip_prob=white_skip_prob
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=0
    ), dataset

# ====================== 🔥 修复完成：测试主函数 ======================
if __name__ == "__main__":
    # 1. 定义数据集路径
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_DIR / "dataset" / "done"

    # 2. 扫描所有图片和对应的标签文件（配对）
    all_img_paths = []
    all_label_paths = []
    for img_file in DATA_DIR.glob("*.png"):
        # 图片同名json标签
        label_file = DATA_DIR / f"{img_file.stem}_chars_marked.json"
        if label_file.exists():
            all_img_paths.append(img_file)
            all_label_paths.append(label_file)

    # 3. 全局一次性划分数据集（无重叠、无泄露）
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = split_dataset(
        all_img_paths, all_label_paths,
        train_ratio=0.7, val_ratio=0.15, seed=42
    )

    # 4. 创建DataLoader（参数正确，无报错）
    train_loader, train_dataset = create_dataloader(
        train_imgs, train_labels, split="train", batch_size=32, white_skip_prob=0.95
    )
    val_loader, val_dataset = create_dataloader(
        val_imgs, val_labels, split="val", batch_size=32, white_skip_prob=0.95
    )
    test_loader, test_dataset = create_dataloader(
        test_imgs, test_labels, split="test", batch_size=32, white_skip_prob=0.95
    )

    # 5. 打印结果
    print(f"✅ 训练集样本总数: {len(train_dataset)}")
    print(f"✅ 验证集样本总数: {len(val_dataset)}")
    print(f"✅ 测试集样本总数: {len(test_dataset)}")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")