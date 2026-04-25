# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import torch
from PIL import Image, ImageOps
from ai.feature.model import UNetAutoEncoder

# ===================== 配置 =====================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = Path(__file__).parent / "best_model.pth"
CROP_WIDTH   = 101
TARGET_HEIGHT= 55
STEP         = CROP_WIDTH // 2  # 滑动步长

# ===================== 纯 PyTorch/PIL 转换工具（无 NumPy） =====================
def pil_to_tensor(pic):
    """PIL 转 Tensor [0, 1]，完全不用 torchvision 和 numpy"""
    # 获取 PIL 底层字节数据
    img_bytes = pic.tobytes()
    h, w = pic.size[1], pic.size[0]
    # 用 bytearray 包装防止 PyTorch 报只读缓冲区错误
    tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8).reshape(h, w)
    return tensor.to(torch.float32) / 255.0

def tensor_to_pil(tensor):
    """Tensor 转 PIL，完全不用 torchvision 和 numpy"""
    # 如果是 3 通道，取第一个通道转灰度图
    if tensor.dim() == 3:
        tensor = tensor[0] 
    
    # 🔥 必须先记录宽高，因为展平后形状信息就丢失了
    h, w = tensor.shape
    
    # 确保内存连续、类型正确，并展平为1维 (核心修复)
    tensor = tensor.clamp(0, 255).to(torch.uint8).cpu().contiguous().view(-1)
    
    # 将1维张量转为 bytes（tolist()此时返回一维整数列表，bytes()可正常解析）
    img_bytes = bytes(tensor.tolist())
    return Image.frombytes('L', (w, h), img_bytes)

# ===================== 加载模型 =====================
def load_trained_model():
    model = UNetAutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ===================== 单块推理 =====================
def infer_patch(model, patch_pil):
    # 1. 使用自定义方法转张量 [0, 1]
    patch_tensor = pil_to_tensor(patch_pil) * 255.0  # (H, W)

    # 2. 构造3通道RGB + 画黄线
    input_rgb = torch.stack([patch_tensor, patch_tensor, patch_tensor], dim=0)  # (3, H, W)
    mid_x = CROP_WIDTH // 2
    input_rgb[0, :, mid_x] = 255.0
    input_rgb[1, :, mid_x] = 255.0
    input_rgb[2, :, mid_x] = 0.0

    # 3. 归一化
    tensor = (input_rgb / 127.5) - 1.0
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # 4. 推理
    with torch.no_grad():
        out = model(tensor)

    # 5. 使用自定义方法转回 PIL 图片（直接输出单通道 L 模式）
    out = out.squeeze(0)
    out = (out + 1.0) * 127.5
    return tensor_to_pil(out)

# ===================== 整行推理（滑动窗口） =====================
def infer_whole_line(line_img_path, save_path="result_line.png"):
    model = load_trained_model()
    img = Image.open(line_img_path).convert("L")
    W, H = img.size

    # 空白画布
    result = Image.new("L", (W, H), 255)
    x = 0

    while x < W:
        # 1. 计算实际裁切宽度和右边距
        actual_width = min(CROP_WIDTH, W - x)
        right_pad = CROP_WIDTH - actual_width

        # 2. 裁切并做右边距填充（修复右侧边缘丢失）
        patch = img.crop((x, 0, x + actual_width, H))
        if right_pad > 0:
            patch = ImageOps.expand(patch, (0, 0, right_pad, 0), fill=255)

        # 3. 上下高度填充
        pad_top = 0
        if patch.height < TARGET_HEIGHT:
            pad_total = TARGET_HEIGHT - patch.height
            pad_top = pad_total // 2
            patch = ImageOps.expand(patch, (0, pad_top, 0, pad_total - pad_top), fill=255)

        # 4. 推理
        cleaned_img = infer_patch(model, patch)

        # 5. 裁掉上下填充
        cleaned_img = cleaned_img.crop((0, pad_top, CROP_WIDTH, pad_top + H))

        # 6. 裁掉右侧填充，恢复真实宽度
        if right_pad > 0:
            cleaned_img = cleaned_img.crop((0, 0, actual_width, H))

        # 7. 粘贴
        result.paste(cleaned_img, (x, 0))

        # 更新步长
        x += STEP if actual_width == CROP_WIDTH else W

    result.save(save_path)
    print(f"✅ 推理完成，已保存：{save_path}")

# ===================== 运行 =====================
if __name__ == "__main__":
    INPUT_LINE_IMAGE = Path(__file__).parent / "page_33.png_line_9.png"
    infer_whole_line(INPUT_LINE_IMAGE, Path(__file__).parent / "result_clean_line.png")