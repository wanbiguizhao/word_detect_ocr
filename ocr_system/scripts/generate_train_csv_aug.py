import os
import sys
import json
import csv
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

MIN_SAMPLES_PER_CLASS = 20  # 每个类别最少样本数（增强后）

def augment_image(img_path, output_path, aug_type):
    """对图片进行数据增强"""
    img = Image.open(img_path).convert('L')
    
    if aug_type == 'rotate':
        # 随机旋转 -5 到 5 度
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=255)
    elif aug_type == 'shift':
        # 随机平移
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)
    elif aug_type == 'noise':
        # 添加噪声
        arr = np.array(img)
        noise = np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif aug_type == 'contrast':
        # 对比度调整
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    elif aug_type == 'blur':
        # 轻微模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    img.save(output_path)

def generate_csv_with_augmentation():
    """从标注数据生成训练/验证/测试集CSV（带数据增强）"""
    # 读取标注数据
    labels_path = os.path.join(BASE_DIR, '../bussiness/datahome/pdf01/clusters/labeling/labels.json')
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    # 读取聚类数据
    clusters_path = os.path.join(BASE_DIR, '../bussiness/datahome/pdf01/clusters/hog_clusters.json')
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    # 收集所有标注数据
    data = []

    for cluster_id, cluster_data in labels.items():
        if 'char_labels' in cluster_data:
            cluster_items = clusters['clusters'].get(cluster_id, [])
            for idx, label_info in cluster_data['char_labels'].items():
                char = label_info.get('char')
                if char:
                    if int(idx) < len(cluster_items):
                        item = cluster_items[int(idx)]
                        img_path = item.get('image_path')
                        if img_path and os.path.exists(img_path):
                            data.append((img_path, char))

    print(f'找到 {len(data)} 条标注数据')

    if len(data) == 0:
        print('没有找到标注数据！')
        return

    # 统计每个字符的数量
    char_counter = Counter([item[1] for item in data])
    print(f'唯一字符数: {len(char_counter)}')

    # 创建图片目录
    train_img_dir = os.path.join(DATA_ROOT, 'train_images')
    val_img_dir = os.path.join(DATA_ROOT, 'val_images')
    test_img_dir = os.path.join(DATA_ROOT, 'test_images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 对小样本类别进行数据增强
    aug_types = ['rotate', 'shift', 'noise', 'contrast', 'blur']
    augmented_data = list(data)
    
    print(f'\n对小样本类别进行数据增强...')
    for char, count in char_counter.items():
        if count < MIN_SAMPLES_PER_CLASS:
            # 计算需要增强的数量
            need_aug = MIN_SAMPLES_PER_CLASS - count
            char_samples = [item for item in data if item[1] == char]
            
            aug_count = 0
            while aug_count < need_aug:
                for img_path, c in char_samples:
                    if aug_count >= need_aug:
                        break
                    # 生成增强图片
                    aug_type = random.choice(aug_types)
                    base_name = os.path.basename(img_path)
                    name, ext = os.path.splitext(base_name)
                    aug_name = f'{name}_aug_{aug_count}{ext}'
                    aug_path = os.path.join(train_img_dir, aug_name)
                    
                    try:
                        augment_image(img_path, aug_path, aug_type)
                        augmented_data.append((aug_path, c))
                        aug_count += 1
                    except Exception as e:
                        print(f'增强失败 {img_path}: {e}')
            
            print(f'  {char}: {count} -> {count + aug_count}')

    print(f'\n增强后总数据量: {len(augmented_data)}')

    # 统计增强后的分布
    char_counter_aug = Counter([item[1] for item in augmented_data])
    print(f'增强后类别数: {len(char_counter_aug)}')
    print(f'增强后最少样本数: {min(char_counter_aug.values())}')

    # 划分数据集
    images = [item[0] for item in augmented_data]
    chars = [item[1] for item in augmented_data]

    # 第一次分割：80% 训练，20% 临时（验证+测试）
    train_images, temp_images, train_chars, temp_chars = train_test_split(
        images, chars, test_size=0.2, random_state=42, stratify=chars
    )

    # 第二次分割：50% 验证，50% 测试
    val_images, test_images, val_chars, test_chars = train_test_split(
        temp_images, temp_chars, test_size=0.5, random_state=42
    )

    # 写入CSV（复制图片并使用相对路径）
    def write_csv(filename, images, chars, img_dir):
        img_dir_name = os.path.basename(img_dir)
        rows = [['image_path', 'char']]
        
        for img, char in zip(images, chars):
            filename_only = os.path.basename(img)
            # 如果是增强图片，已经在 train_img_dir 中
            if '_aug_' in filename_only:
                # 增强图片只用于训练集
                if 'train' in img_dir_name:
                    try:
                        shutil.copy2(img, os.path.join(img_dir, filename_only))
                    except:
                        pass
                    rel_path = f'./{img_dir_name}/{filename_only}'
                    rows.append([rel_path, char])
            else:
                try:
                    shutil.copy2(img, os.path.join(img_dir, filename_only))
                except Exception as e:
                    continue
                rel_path = f'./{img_dir_name}/{filename_only}'
                rows.append([rel_path, char])
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        return len(rows) - 1

    train_count = write_csv(TRAIN_CSV, train_images, train_chars, train_img_dir)
    val_count = write_csv(VAL_CSV, val_images, val_chars, val_img_dir)
    test_count = write_csv(TEST_CSV, test_images, test_chars, test_img_dir)

    print(f'\n数据集划分完成:')
    print(f'  训练集: {train_count} 条')
    print(f'  验证集: {val_count} 条')
    print(f'  测试集: {test_count} 条')
    print(f'  唯一字符数: {len(set(train_chars + val_chars + test_chars))}')
    print(f'\nCSV 文件已生成:')
    print(f'  - {TRAIN_CSV}')
    print(f'  - {VAL_CSV}')
    print(f'  - {TEST_CSV}')
    
    print(f'\n请更新 config.py 中的 NUM_CLASSES = {len(char_counter_aug)}')

if __name__ == '__main__':
    generate_csv_with_augmentation()