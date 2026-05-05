import os
import sys
import json
import csv
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

MIN_SAMPLES_PER_CLASS = 10  # 每个类别最少样本数

def generate_csv():
    """从标注数据生成训练/验证/测试集CSV（使用相对路径）"""
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
                    # 获取图片路径 - hog_clusters 中已经有完整路径
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

    # 过滤掉样本数不足的字符
    valid_chars = [char for char, count in char_counter.items() if count >= MIN_SAMPLES_PER_CLASS]
    filtered_data = [item for item in data if item[1] in valid_chars]
    print(f'过滤后数据量: {len(filtered_data)} (移除了 {len(data) - len(filtered_data)} 条稀有样本)')
    print(f'保留类别数: {len(valid_chars)}')

    # 划分数据集
    images = [item[0] for item in filtered_data]
    chars = [item[1] for item in filtered_data]

    # 第一次分割：80% 训练，20% 临时（验证+测试）
    train_images, temp_images, train_chars, temp_chars = train_test_split(
        images, chars, test_size=0.2, random_state=42, stratify=chars
    )

    # 第二次分割：50% 验证，50% 测试（不使用分层抽样，避免样本太少的问题）
    val_images, test_images, val_chars, test_chars = train_test_split(
        temp_images, temp_chars, test_size=0.5, random_state=42
    )

    # 创建图片目录
    train_img_dir = os.path.join(DATA_ROOT, 'train_images')
    val_img_dir = os.path.join(DATA_ROOT, 'val_images')
    test_img_dir = os.path.join(DATA_ROOT, 'test_images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 写入CSV（复制图片并使用相对路径）
    def write_csv(filename, images, chars, img_dir):
        img_dir_name = os.path.basename(img_dir)
        rows = [['image_path', 'char']]
        
        for img, char in zip(images, chars):
            # 获取文件名
            filename_only = os.path.basename(img)
            # 复制图片
            try:
                shutil.copy2(img, os.path.join(img_dir, filename_only))
            except Exception as e:
                print(f'复制失败 {img}: {e}')
                continue
            # 使用相对路径
            rel_path = f'./{img_dir_name}/{filename_only}'
            rows.append([rel_path, char])
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        return len(rows) - 1  # 返回写入的行数（减去表头）

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
    
    # 更新 config.py 中的 NUM_CLASSES
    print(f'\n请更新 config.py 中的 NUM_CLASSES = {len(valid_chars)}')

if __name__ == '__main__':
    generate_csv()