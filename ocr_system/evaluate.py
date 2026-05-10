import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms
import csv
import argparse
from tqdm import tqdm

# 导入现有的推理模块
from infer import load_model, get_char_mapping, predict


def evaluate_from_csv(csv_path, model, transform, idx_to_char, char_to_idx):
    """从CSV文件评估模型"""
    correct = 0
    total = 0
    results = []
    
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        
        for row in tqdm(reader, desc=f'Evaluating {os.path.basename(csv_path)}'):
            if len(row) >= 2:
                img_path, true_char = row[0], row[1]
                
                # 获取绝对路径
                if not os.path.isabs(img_path):
                    full_img_path = os.path.join(csv_dir, img_path)
                else:
                    full_img_path = img_path
                
                if not os.path.exists(full_img_path):
                    print(f"警告: 图像文件不存在: {full_img_path}")
                    continue
                
                # 推理
                pred_char, confidence, label_id = predict(full_img_path, model, transform, idx_to_char)
                
                # 判断是否正确
                is_correct = (pred_char == true_char)
                if is_correct:
                    correct += 1
                total += 1
                
                # 记录结果
                results.append({
                    'image_path': img_path,
                    'true_char': true_char,
                    'pred_char': pred_char,
                    'confidence': confidence,
                    'correct': is_correct
                })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description='OCR Model Evaluation')
    parser.add_argument('--model', type=str, default=None, help='Path to model')
    parser.add_argument('--csv', type=str, help='Path to evaluation CSV file')
    parser.add_argument('--dataset', type=str, choices=['train', 'val'], help='Use train or val dataset')
    args = parser.parse_args()
    
    # 获取字符映射
    idx_to_char, _ = get_char_mapping()
    
    # 确定模型路径
    from config import INFERENCE_MODEL_PATH, TRAIN_CSV, VAL_CSV
    model_path = args.model if args.model else INFERENCE_MODEL_PATH
    
    # 确定CSV路径
    if args.csv:
        csv_path = args.csv
    elif args.dataset == 'train':
        csv_path = TRAIN_CSV
    elif args.dataset == 'val':
        csv_path = VAL_CSV
    else:
        print("请提供 --csv 或 --dataset 参数")
        return
    
    # 加载模型
    num_classes = max(idx_to_char.keys()) + 1 if idx_to_char else 7000
    model = load_model(model_path, num_classes=num_classes)
    print(f'模型加载自: {model_path}')
    print(f'类别数: {num_classes}')
    
    # 变换（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建字符到ID的映射
    char_to_idx = {char: idx for idx, char in idx_to_char.items()}
    
    # 评估
    print(f'\n开始评估 {csv_path}...')
    accuracy, results = evaluate_from_csv(csv_path, model, transform, idx_to_char, char_to_idx)
    
    # 输出结果
    print(f'\n{"="*60}')
    print('评估结果')
    print(f'{"="*60}')
    print(f'总样本数: {len(results)}')
    print(f'正确数: {sum(1 for r in results if r["correct"])}')
    print(f'准确率: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # 输出部分错误样本
    print(f'\n{"="*60}')
    print('错误样本示例（前10个）')
    print(f'{"="*60}')
    errors = [r for r in results if not r['correct']]
    for i, r in enumerate(errors[:10]):
        print(f'{i+1}. 图像: {os.path.basename(r["image_path"])}')
        print(f'   真实: "{r["true_char"]}" | 预测: "{r["pred_char"]}" | 置信度: {r["confidence"]:.4f}')
    
    # 保存详细结果
    output_file = os.path.join(os.path.dirname(csv_path), 'evaluation_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'评估文件: {csv_path}\n')
        f.write(f'模型路径: {model_path}\n')
        f.write(f'总样本数: {len(results)}\n')
        f.write(f'正确数: {sum(1 for r in results if r["correct"])}\n')
        f.write(f'准确率: {accuracy:.4f}\n')
        f.write('\n详细结果:\n')
        f.write('='*60 + '\n')
        for r in results:
            status = '正确' if r['correct'] else '错误'
            f.write(f'{status} | 图像: {r["image_path"]} | 真实: "{r["true_char"]}" | 预测: "{r["pred_char"]}" | 置信度: {r["confidence"]:.4f}\n')
    
    print(f'\n详细结果已保存到: {output_file}')


if __name__ == '__main__':
    main()
