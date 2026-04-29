import math
import os
import sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import torch
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

from ai.segment.model import CharSegmentClassifier
from ai.segment import config

# 完全对齐训练代码的常量
CROP_WIDTH = 101
TARGET_HEIGHT = 55
MID_COL = CROP_WIDTH // 2

class CharSegmentInfer:
    def __init__(self, batch_size=32):
        self.device = config.DEVICE
        self.model = CharSegmentClassifier(config.PRETRAINED_AE_PATH).to(self.device)
        self.model.load_state_dict(torch.load(config.PRETRAINED_CHAR_SEGMENT_MODEL_PATH, map_location=self.device))
        self.model.eval()
        print("✅ 模型加载成功！")

        self.crop_width = CROP_WIDTH
        self.target_height = TARGET_HEIGHT
        self.mid_col = MID_COL
        self.step = 1
        self.threshold = 0.65  # 二分类阈值
        self.batch_size = batch_size
        # 概率可视化相关配置
        self.prob_height = 51  # 概率展示区高度（51行）
        self.prob_max_pixel = 50  # 概率条最大高度（对应100%概率）

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert("L")
        W, H = img.size

        if H < self.target_height:
            pad_total = self.target_height - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            img = ImageOps.expand(img, (0, pad_top, 0, pad_bottom), fill=255)
        elif H > self.target_height:
            img = img.crop((0, 0, W, self.target_height))
        return img, W

    def patch_to_tensor(self, img_patch_gray):
        # 1. 转RGB
        patch_rgb = img_patch_gray.convert("RGB")

        # 2. 极速转换：PIL -> bytes -> PyTorch Tensor
        patch_tensor = torch.frombuffer(patch_rgb.tobytes(), dtype=torch.uint8)
        patch_tensor = patch_tensor.view(self.target_height, self.crop_width, 3)

        # 3. 训练同款黄线：纯张量操作
        patch_tensor[:, self.mid_col, 0] = 255
        patch_tensor[:, self.mid_col, 1] = 255
        patch_tensor[:, self.mid_col, 2] = 0

        # 4. 完全对齐训练的归一化
        patch_tensor = patch_tensor.float() / 127.5 - 1.0
        patch_tensor = patch_tensor.permute(2, 0, 1)  # HWC -> CHW
        return patch_tensor

    def predict_batch(self, img_path):
        img_gray, W = self.preprocess_image(img_path)
        all_tensors = []
        all_positions = []

        x_start = 0
        total_steps = W - self.crop_width + 1

        # 裁剪patch
        with tqdm(total=total_steps, desc="裁剪patch") as pbar:
            while x_start + self.crop_width <= W:
                patch = img_gray.crop((x_start, 0, x_start + self.crop_width, self.target_height))
                tensor = self.patch_to_tensor(patch)
                all_tensors.append(tensor)
                all_positions.append(x_start + self.mid_col)
                x_start += self.step
                pbar.update(1)

        # 批量推理：返回原始概率值（关键修改）
        pred_probs = []  # 存储每列的原始预测概率（0~1）
        predictions = []  # 存储二分类结果（0/1）
        with torch.no_grad():
            for i in tqdm(range(0, len(all_tensors), self.batch_size), desc="批量推理"):
                batch = torch.stack(all_tensors[i:i+self.batch_size]).to(self.device)
                outputs = self.model(batch)
                # 提取原始概率（0~1）
                batch_probs = outputs.squeeze(-1).cpu().numpy()
                pred_probs.extend(batch_probs)
                # 二分类结果（用于后续边界提取）
                batch_preds = (batch_probs > self.threshold).tolist()
                predictions.extend(batch_preds)

        # 预测平滑（仅用于边界提取，不影响概率可视化）
        smoothed_preds = predictions#self.smooth_predictions(predictions, window=3)
        # 返回：原图、(列位置, 二分类结果)、(列位置, 原始概率)
        return img_gray, list(zip(all_positions, smoothed_preds)), list(zip(all_positions, pred_probs))

    def smooth_predictions(self, preds, window=3):
        smoothed = []
        length = len(preds)
        for i in range(length):
            left = max(0, i - window)
            right = min(length, i + window + 1)
            avg = sum(preds[left:right]) / (right - left)
            smoothed.append(1 if avg > 0.5 else 0)
        return smoothed

    def get_char_boundaries(self, predictions, min_width=5):
        boxes = []
        start_x = None
        for col, pred in predictions:
            if pred == 1 and start_x is None:
                start_x = col
            elif pred == 0 and start_x is not None:
                if col - start_x >= min_width:
                    boxes.append((start_x, col))
                start_x = None
        if start_x is not None:
            boxes.append((start_x, predictions[-1][0]))
        return boxes

    def draw_boundaries_with_prob(self, img_gray, boxes, pred_probs, save_path):
        """
        绘制字符边界 + 概率可视化
        :param img_gray: 预处理后的灰度图
        :param boxes: 字符边界框
        :param pred_probs: 列表，每个元素是(列位置, 概率值)
        :param save_path: 保存路径
        """
        # 1. 拼接原图和概率展示区（下方新增51像素白色区域）
        W = img_gray.width
        H = img_gray.height
        # 创建新画布：宽度=原图宽度，高度=原图高度+51（概率区）
        new_img = Image.new("RGB", (W, H + self.prob_height), color=(255, 255, 255))
        new_img.paste(img_gray.convert("RGB"), (0, 0))  # 粘贴原图到顶部
        draw = ImageDraw.Draw(new_img)

        # 2. 绘制字符边界（原有逻辑）
        for x_start, x_end in boxes:
            draw.line([(x_start, 0), (x_start, H)], fill=(255, 0, 0), width=1)
            draw.line([(x_end, 0), (x_end, H)], fill=(0, 255, 0), width=1)

        # 3. 绘制概率可视化（核心逻辑）
        # 先构建「列位置→概率值」的映射（处理可能的列缺失）
        prob_dict = {col: prob for col, prob in pred_probs}
        for col in range(W):  # 遍历每一列
            prob = prob_dict.get(col, 0.0)  # 无预测的列概率记为0
            prob_percent = prob * 100  # 转百分比（0~100）

            # 计算黄色条高度：96-100%→50像素，91-95%→49像素，以此类推
            # 公式：高度 = 50 - ((100 - 概率值) // 5)，最小为0
            bar_height = math.ceil(prob_percent*0.5)
            if bar_height > 0:
                # 黄色条绘制范围：y从H到H+bar_height-1（因为是闭区间）
                y_start = H+self.prob_height-bar_height
                y_end = H + self.prob_height
                # 绘制黄色竖线（每列的概率条）
                draw.line([(col, y_start), (col, y_end)], fill=(255, 255, 0), width=1)

        # 4. 绘制红色分隔线（第51行，即H+50的位置）
        red_line_y = H    # 概率区最后一行（第51行）
        draw.line([(0, red_line_y), (W, red_line_y)], fill=(255, 0, 0), width=1)

        # 5. 保存图片
        new_img.save(save_path)
        print(f"✅ 带概率可视化的结果已保存：{save_path}")

    def infer_whole_line(self, line_img_path, save_path="result_line.png"):
        # 修改：获取概率值
        img_gray, predictions, pred_probs = self.predict_batch(line_img_path)
        boxes = self.get_char_boundaries(predictions, min_width=5)
        # 调用新的绘制函数
        self.draw_boundaries_with_prob(img_gray, boxes, pred_probs, save_path)
        return boxes, pred_probs  # 返回边界和概率，方便后续分析


if __name__ == "__main__":
    infer = CharSegmentInfer(batch_size=64)

    input_img = Path(__file__).parent / "page_28.png_line_15.png"
    output_img = Path(__file__).parent/"temp" / "my_segment_result.png"

    char_boxes, pred_probs = infer.infer_whole_line(str(input_img), str(output_img))
    print(f"识别到 {len(char_boxes)} 个字符: {char_boxes}")
    # 可选：打印前10列的概率，方便调试
    print("前10列的预测概率（示例）：")
    for col, prob in pred_probs[:100]:
        print(f"列{col:3d} → 概率{prob*100:.1f}% → 黄色条高度{math.ceil(prob*100*0.5)}像素")
