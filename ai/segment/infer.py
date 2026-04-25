import os
import sys
import json
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
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
        self.model.eval()
        print("✅ 模型加载成功！")

        self.crop_width = CROP_WIDTH
        self.target_height = TARGET_HEIGHT
        self.mid_col = MID_COL
        self.step = 1
        self.threshold = 0.65  # 汉字判断阈值
        self.batch_size = batch_size

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

    # =======================
    # 🔥 核心极速优化：去掉双层Python循环，100%对齐训练
    # =======================
    def patch_to_tensor(self, img_patch_gray):
        # 1. 转RGB
        patch_rgb = img_patch_gray.convert("RGB")
        
        # 2. 🔥 极速转换：PIL -> bytes -> PyTorch Tensor (底层C++级别，无Python循环)
        patch_tensor = torch.frombuffer(patch_rgb.tobytes(), dtype=torch.uint8)
        patch_tensor = patch_tensor.view(self.target_height, self.crop_width, 3)
        
        # 3. 🔥 训练同款黄线：纯张量操作
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
        
        # 🔥 裁剪patch现在会瞬间完成
        with tqdm(total=total_steps, desc="裁剪patch") as pbar:
            while x_start + self.crop_width <= W:
                patch = img_gray.crop((x_start, 0, x_start + self.crop_width, self.target_height))
                tensor = self.patch_to_tensor(patch)
                all_tensors.append(tensor)
                all_positions.append(x_start + self.mid_col)
                x_start += self.step
                pbar.update(1)

        # 批量推理：返回原始概率值（非0/1）
        probabilities = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_tensors), self.batch_size), desc="批量推理"):
                batch = torch.stack(all_tensors[i:i+self.batch_size]).to(self.device)
                outputs = self.model(batch)
                # 获取sigmoid概率值，保存原始预测概率
                probs = torch.sigmoid(outputs.squeeze(-1)).cpu().tolist()
                if not isinstance(probs, list):
                    probs = [probs]
                probabilities.extend(probs)

        # 预测平滑（基于概率）
        smoothed_probs = self.smooth_predictions(probabilities, window=3)
        # 转换为0/1预测
        predictions = [1 if p > self.threshold else 0 for p in smoothed_probs]
        
        return img_gray, list(zip(all_positions, predictions)), smoothed_probs

    def smooth_predictions(self, preds, window=3):
        smoothed = []
        length = len(preds)
        for i in range(length):
            left = max(0, i - window)
            right = min(length, i + window + 1)
            avg = sum(preds[left:right]) / (right - left)
            smoothed.append(avg)
        return smoothed

    def get_char_boundaries(self, predictions, min_width=15):
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

    def draw_boundaries(self, img_gray, boxes, save_path):
        img_rgb = img_gray.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        H = img_rgb.height
        for x_start, x_end in boxes:
            draw.line([(x_start, 0), (x_start, H)], fill=(255, 0, 0), width=1)
            draw.line([(x_end, 0), (x_end, H)], fill=(0, 255, 0), width=1)
        img_rgb.save(save_path)

    # =======================
    # 🆕 保存结果为JSON文件
    # =======================
    def save_results_to_json(self, boxes, probabilities, save_path):
        # 构建JSON格式数据
        result = {
            "threshold": round(self.threshold, 2),
            "probabilities": [round(p, 4) for p in probabilities],  # 每列像素预测概率
            "chars": [
                {
                    "col_start": int(start),
                    "col_end": int(end),
                    "width": int(end - start)
                } for start, end in boxes
            ]
        }
        # 写入JSON文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # =======================
    # 🆕 单张图片推理（支持自定义保存路径）
    # =======================
    def infer_single_image(self, img_path, save_dir=None, save_vis=True):
        img_gray, predictions, probs = self.predict_batch(img_path)
        boxes = self.get_char_boundaries(predictions, min_width=15)
        
        # 文件名（无后缀）
        img_stem = Path(img_path).stem
        # 保存目录：未指定则默认原图片目录，指定则用自定义目录
        save_dir = Path(save_dir) if save_dir else Path(img_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存可视化结果
        if save_vis:
            vis_path = save_dir / f"{img_stem}_seg.png"
            self.draw_boundaries(img_gray, boxes, str(vis_path))
        
        # 保存JSON结果
        json_path = save_dir / f"{img_stem}_model.json"
        self.save_results_to_json(boxes, probs, str(json_path))
        
        return boxes, json_path

    # =======================
    # 🆕 文件夹批量推理（保存路径参数化）
    # =======================
    def batch_infer_folder(self, folder_path, save_dir=None, save_vis=True):
        """
        批量推理文件夹
        :param folder_path: 输入图片文件夹路径
        :param save_dir: 结果保存文件夹（参数化，可自定义）
        :param save_vis: 是否保存可视化图片
        """
        folder = Path(folder_path)
        # 支持的图片格式
        img_suffix = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        img_files = [f for f in folder.iterdir() if f.suffix.lower() in img_suffix]
        
        if not img_files:
            print("❌ 文件夹中未找到图片！")
            return
        
        print(f"\n🚀 开始批量推理，共 {len(img_files)} 张图片")
        print(f"📂 输入目录：{folder_path}")
        print(f"📂 输出目录：{save_dir if save_dir else '原图片目录'}")
        
        for img_file in tqdm(img_files, desc="批量处理中"):
            try:
                # 传入自定义保存路径
                self.infer_single_image(str(img_file), save_dir=save_dir, save_vis=save_vis)
            except Exception as e:
                print(f"\n❌ 处理失败 {img_file.name}: {str(e)}")
        
        print(f"\n✅ 批量推理完成！所有结果保存在：{save_dir if save_dir else folder}")

    def infer_whole_line(self, line_img_path, save_path="result_line.png"):
        # 原有功能保留
        img_gray, predictions, _ = self.predict_batch(line_img_path)
        boxes = self.get_char_boundaries(predictions, min_width=5)
        self.draw_boundaries(img_gray, boxes, save_path)
        return boxes


if __name__ == "__main__":
    infer = CharSegmentInfer(batch_size=64)
    # ============== 可自由配置的参数 ==============
    # 输入图片文件夹
    INPUT_FOLDER = BASE_DIR / "dataset" / "unlable"
    # 结果保存文件夹（参数化，随意修改）
    SAVE_FOLDER = BASE_DIR / "dataset" / "infer_results"
    # ==============================================
    
    # 批量推理（传入自定义保存路径）
    infer.batch_infer_folder(
        folder_path=INPUT_FOLDER,
        save_dir=SAVE_FOLDER,
        save_vis=True
    )