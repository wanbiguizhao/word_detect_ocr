import math
import os
import json
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

# 全局常量（仅定义1次，无重复）
CROP_WIDTH = 101
TARGET_HEIGHT = 55
MID_COL = CROP_WIDTH // 2

# 动态导入项目配置（兼容你的项目结构）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(BASE_DIR))

from ai.segment.model import CharSegmentClassifier
from ai.segment import config

class CharSegmentInfer:
    """
    汉字分割模型推理类（全功能合并版）
    整合：模型推理、概率计算、边界提取、可视化、批量推理、JSON保存
    无重复代码，单一职责，易维护
    """
    def __init__(self, batch_size: int = 32):
        # 模型初始化
        self.device = config.DEVICE
        self.model = CharSegmentClassifier(config.PRETRAINED_AE_PATH).to(self.device)
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
        self.model.eval()
        print("✅ 模型加载成功！")

        # 核心参数
        self.crop_width = CROP_WIDTH
        self.target_height = TARGET_HEIGHT
        self.mid_col = MID_COL
        self.step = 1
        self.threshold = 0.65
        self.batch_size = batch_size

        # 概率可视化参数
        self.prob_height = 51
        self.prob_max_pixel = 50

    # ====================== 基础工具：图像预处理（唯一实现） ======================
    def preprocess_image(self, img_path: str) -> Tuple[Image.Image, int]:
        """统一图像预处理：缩放/填充到目标尺寸"""
        img = Image.open(img_path).convert("L")
        W, H = img.size

        # 高度适配
        if H < self.target_height:
            pad_total = self.target_height - H
            pad_top, pad_bottom = pad_total // 2, pad_total - pad_total // 2
            img = ImageOps.expand(img, (0, pad_top, 0, pad_bottom), fill=255)
        elif H > self.target_height:
            img = img.crop((0, 0, W, self.target_height))
        return img, W

    # ====================== 核心：Patch转换（唯一最优实现） ======================
    def patch_to_tensor(self, img_patch_gray: Image.Image) -> torch.Tensor:
        """极速Patch转张量，完全对齐训练代码"""
        patch_rgb = img_patch_gray.convert("RGB")
        patch_tensor = torch.frombuffer(patch_rgb.tobytes(), dtype=torch.uint8)
        patch_tensor = patch_tensor.view(self.target_height, self.crop_width, 3)
        
        # 训练同款黄线标记
        patch_tensor[:, self.mid_col, 0] = 255
        patch_tensor[:, self.mid_col, 1] = 255
        patch_tensor[:, self.mid_col, 2] = 0

        # 归一化 + 维度转换
        patch_tensor = patch_tensor.float() / 127.5 - 1.0
        return patch_tensor.permute(2, 0, 1)  # HWC -> CHW

    # ====================== 核心：批量推理 + 概率补0（你的要求） ======================
    def predict_batch(self, img_path: str) -> Tuple[Image.Image, List[Tuple[int, int]], List[float]]:
        """
        统一推理入口
        🔥 核心修复：图片前50列概率默认填充0，完美解决起始像素无概率问题
        返回：预处理图、(列, 预测值)、全列概率数组(长度=图片宽度)
        """
        img_gray, W = self.preprocess_image(img_path)
        all_tensors, all_positions = [], []

        # 裁剪Patch
        x_start = 0
        total_steps = W - self.crop_width + 1
        with tqdm(total=total_steps, desc="裁剪patch") as pbar:
            while x_start + self.crop_width <= W:
                patch = img_gray.crop((x_start, 0, x_start + self.crop_width, self.target_height))
                all_tensors.append(self.patch_to_tensor(patch))
                all_positions.append(x_start + self.mid_col)
                x_start += self.step
                pbar.update(1)

        # 模型批量推理
        probabilities = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_tensors), self.batch_size), desc="批量推理"):
                batch = torch.stack(all_tensors[i:i+self.batch_size]).to(self.device)
                outputs = self.model(batch)
                probs = torch.sigmoid(outputs.squeeze(-1)).cpu().tolist()
                probabilities.extend([probs] if not isinstance(probs, list) else probs)

        # ====================== 你的核心需求：前50列概率填充0 ======================
        full_probs = [0.0] * W  # 初始化全图概率为0
        for pos, prob in zip(all_positions, probabilities):
            if pos < W:
                full_probs[pos] = prob  # 仅填充有推理结果的列

        # 概率平滑 + 二分类预测
        smoothed_probs = self.smooth_predictions(full_probs)
        predictions = [(col, 1 if p > self.threshold else 0) for col, p in enumerate(smoothed_probs)]

        return img_gray, predictions, full_probs

    # ====================== 工具：概率平滑（唯一实现） ======================
    def smooth_predictions(self, probs: List[float], window: int = 3) -> List[float]:
        """概率平滑滤波"""
        smoothed = []
        length = len(probs)
        for i in range(length):
            left = max(0, i - window)
            right = min(length, i + window + 1)
            smoothed.append(sum(probs[left:right]) / (right - left))
        return smoothed

    # ====================== 工具：字符边界提取（唯一实现） ======================
    def get_char_boundaries(self, predictions: List[Tuple[int, int]], min_width: int = 5) -> List[Tuple[int, int]]:
        """提取字符左右边界"""
        boxes, start_x = [], None
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

    # ====================== 可视化：绘制边界+概率条（合并版，唯一实现） ======================
    def draw_and_save(self, img_gray: Image.Image, boxes: List[Tuple[int, int]], probs: List[float], save_path: str):
        """绘制字符边界 + 概率黄色条，统一保存"""
        W, H = img_gray.width, img_gray.height
        # 拼接画布：原图 + 概率展示区
        new_img = Image.new("RGB", (W, H + self.prob_height), color=(255, 255, 255))
        new_img.paste(img_gray.convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(new_img)

        # 绘制字符边界
        for x_start, x_end in boxes:
            draw.line([(x_start, 0), (x_start, H)], fill=(255, 0, 0), width=1)
            draw.line([(x_end, 0), (x_end, H)], fill=(0, 255, 0), width=1)

        # 绘制概率黄色条
        for col in range(W):
            prob = probs[col]
            bar_height = math.ceil(prob * 100 * 0.5)
            if bar_height > 0:
                y_start = H + self.prob_height - bar_height
                draw.line([(col, y_start), (col, H + self.prob_height)], fill=(255, 255, 0), width=1)

        # 分隔线
        draw.line([(0, H), (W, H)], fill=(255, 0, 0), width=1)
        new_img.save(save_path)

    # ====================== 输出：保存JSON结果（唯一实现） ======================
    def save_results_to_json(self, boxes: List[Tuple[int, int]], probabilities: List[float], save_path: str):
        """保存分割结果+全列概率到JSON"""
        result = {
            "threshold": round(self.threshold, 2),
            "probabilities": [round(p, 4) for p in probabilities],
            "chars": [{"col_start": s, "col_end": e, "width": e-s} for s, e in boxes]
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ====================== 推理接口：单张图片（合并版） ======================
    def infer_single_image(self, img_path: str, save_dir: str = None, save_vis: bool = True) -> Tuple[List[Tuple[int, int]], str]:
        """单张图片推理：可视化 + JSON保存"""
        img_gray, predictions, probs = self.predict_batch(img_path)
        boxes = self.get_char_boundaries(predictions)
        
        # 路径处理
        img_stem = Path(img_path).stem
        save_dir = Path(save_dir) or Path(img_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存可视化
        if save_vis:
            vis_path = save_dir / f"{img_stem}_seg.png"
            self.draw_and_save(img_gray, boxes, probs, str(vis_path))

        # 保存JSON
        json_path = save_dir / f"{img_stem}_model.json"
        self.save_results_to_json(boxes, probs, str(json_path))
        return boxes, str(json_path)

    # ====================== 推理接口：批量推理（合并版） ======================
    def batch_infer_folder(self, folder_path: str, save_dir: str = None, save_vis: bool = True):
        """文件夹批量推理"""
        folder = Path(folder_path)
        img_suffix = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        img_files = [f for f in folder.iterdir() if f.suffix.lower() in img_suffix]

        if not img_files:
            print("❌ 未找到图片")
            return

        print(f"\n🚀 批量推理：{len(img_files)} 张图片")
        for img_file in tqdm(img_files, desc="处理中"):
            try:
                self.infer_single_image(str(img_file), save_dir, save_vis)
            except Exception as e:
                print(f"\n❌ 处理失败 {img_file.name}: {str(e)}")
        print(f"\n✅ 批量推理完成！")

    # ====================== 推理接口：单行测试（兼容旧代码） ======================
    def infer_whole_line(self, line_img_path: str, save_path: str = "result_line.png") -> List[Tuple[int, int]]:
        """单行快速测试（仅可视化，无JSON）"""
        img_gray, predictions, probs = self.predict_batch(line_img_path)
        boxes = self.get_char_boundaries(predictions)
        self.draw_and_save(img_gray, boxes, probs, save_path)
        return boxes


# ====================== 主函数：一键调用 ======================
if __name__ == "__main__":
    infer = CharSegmentInfer(batch_size=64)

    # --------------- 用法1：单行测试 ---------------
    # input_img = Path(__file__).parent / "page_28.png_line_15.png"
    # infer.infer_whole_line(str(input_img), "my_segment_result.png")

    # --------------- 用法2：批量推理（推荐） ---------------
    INPUT_FOLDER = BASE_DIR / "dataset" / "unlable"
    SAVE_FOLDER = BASE_DIR / "dataset" / "infer_results"
    infer.batch_infer_folder(INPUT_FOLDER, SAVE_FOLDER, save_vis=True)