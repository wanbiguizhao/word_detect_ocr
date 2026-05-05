import os
import cv2
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "ocr_system" / "data" / "ocr_train_data" / "train"
OUTPUT_DIR = PROJECT_ROOT / "ocr_system" / "data" / "ocr_train_data_augmented"

def augment_image(image):
    """对单张图片进行数据增强"""
    augmented = []
    
    # 原始图片
    augmented.append(("original", image))
    
    # 旋转（小角度）
    rows, cols = image.shape[:2]
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        augmented.append((f"rotate_{angle}", rotated))
    
    # 平移
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        augmented.append((f"shift_{dx}_{dy}", shifted))
    
    # 缩放
    for scale in [0.95, 1.05]:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if scale < 1:
            padded = np.ones_like(image) * 255
            start_x = (cols - resized.shape[1]) // 2
            start_y = (rows - resized.shape[0]) // 2
            padded[start_y:start_y+resized.shape[0], start_x:start_x+resized.shape[1]] = resized
            augmented.append((f"scale_{scale}", padded))
        else:
            cropped = resized[(resized.shape[0]-rows)//2:(resized.shape[0]-rows)//2+rows,
                            (resized.shape[1]-cols)//2:(resized.shape[1]-cols)//2+cols]
            augmented.append((f"scale_{scale}", cropped))
    
    # 添加轻微噪声
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(("noise", noisy))
    
    return augmented

def run_data_augmentation():
    """运行数据增强"""
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    
    total_original = 0
    total_augmented = 0
    char_stats = {}
    
    for char_dir in INPUT_DIR.iterdir():
        if not char_dir.is_dir():
            continue
        
        char = char_dir.name
        output_char_dir = OUTPUT_DIR / "train" / char
        output_char_dir.mkdir(exist_ok=True)
        
        char_count = 0
        augmented_count = 0
        
        for img_path in char_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            
            # 如果是RGBA格式，转换为RGB
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # 数据增强
            augmented = augment_image(img)
            
            for suffix, aug_img in augmented:
                base_name = img_path.stem
                output_path = output_char_dir / f"{base_name}_{suffix}.png"
                cv2.imwrite(str(output_path), aug_img)
                augmented_count += 1
            
            char_count += 1
            total_original += 1
        
        char_stats[char] = {"original": char_count, "augmented": augmented_count}
        total_augmented += augmented_count
        print(f"处理 {char}: {char_count} → {augmented_count} 张")
    
    # 复制验证集
    val_dir = PROJECT_ROOT / "ocr_system" / "data" / "ocr_train_data" / "val"
    output_val_dir = OUTPUT_DIR / "val"
    if val_dir.exists():
        import shutil
        if output_val_dir.exists():
            shutil.rmtree(output_val_dir)
        shutil.copytree(val_dir, output_val_dir)
    
    # 复制标签文件
    labels_file = PROJECT_ROOT / "ocr_system" / "data" / "ocr_train_data" / "labels.txt"
    if labels_file.exists():
        shutil.copy(labels_file, OUTPUT_DIR / "labels.txt")
    
    # 生成统计报告
    with open(OUTPUT_DIR / "aug_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_original": total_original,
            "total_augmented": total_augmented,
            "augmentation_factor": total_augmented / total_original if total_original > 0 else 0,
            "char_stats": char_stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("数据增强完成！")
    print(f"{'='*60}")
    print(f"原始图片：{total_original} 张")
    print(f"增强后：{total_augmented} 张")
    print(f"增强倍数：{total_augmented / total_original:.1f}x")
    print(f"输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    run_data_augmentation()