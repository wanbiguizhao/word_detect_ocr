#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的OCR模型评估工具

支持评估：
1. PDF标注数据（标注的数据）
2. PDF所有数据（标注+未标注）
3. CSV验证集数据

使用示例：
    # 评估pdf5823的标注数据
    python evaluate.py pdf5823 --mode labeled
    
    # 评估pdf01的所有数据
    python evaluate.py pdf01 --mode all
    
    # 评估验证集
    python evaluate.py --mode val
    
    # 指定模型路径
    python evaluate.py pdf5823 --mode labeled --model path/to/model.pth
"""

import os
import sys
import argparse
import json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_system'))

from ocr_system.evaluation.pdf_evaluator import PDFEvaluator
from ocr_system.evaluation.evaluator import Evaluator


def evaluate_pdf(data_dir, mode='labeled', model_path=None):
    """
    评估PDF数据集
    
    Args:
        data_dir: 数据集名称（如 pdf5823）或完整路径
        mode: 'labeled' - 只评估标注数据, 'all' - 评估所有数据, 'mixed' - 评估所有数据并标记标注状态
        model_path: 模型路径
    """
    # 确定数据目录（使用相对路径，基于脚本所在目录）
    if not os.path.isabs(data_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, 'bussiness', 'datahome', data_dir)
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    print(f"正在评估: {data_dir}")
    print(f"评估模式: {mode}")
    
    # 创建评估器
    evaluator = PDFEvaluator(data_dir, model_path)
    
    # 根据模式进行评估
    if mode == 'labeled':
        results = evaluator.evaluate_labeled()
    elif mode == 'all':
        results = evaluator.evaluate_all()
    elif mode == 'mixed':
        results = evaluator.evaluate_with_mixed()
    else:
        print(f"未知模式: {mode}")
        return
    
    if results is None:
        return
    
    # 输出结果
    print_results(results, mode)
    
    # 保存结果
    output_file = os.path.join(data_dir, f'evaluation_{mode}_results.json')
    evaluator.save_results(results, output_file)
    print(f"\n详细结果已保存到: {output_file}")


def evaluate_val(csv_path=None, model_path=None):
    """
    评估CSV验证集
    
    Args:
        csv_path: CSV文件路径
        model_path: 模型路径
    """
    if csv_path is None:
        csv_path = 'ocr_system/data/val_rel.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    print(f"正在评估验证集: {csv_path}")
    
    # 创建评估器
    evaluator = Evaluator(model_path)
    
    # 读取CSV数据
    import csv
    image_paths = []
    true_chars = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                full_path = os.path.join('ocr_system/data', row[0])
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                    true_chars.append(row[1])
    
    print(f"验证集样本数: {len(image_paths)}")
    
    # 评估
    results = evaluator.evaluate_with_ground_truth(image_paths, true_chars)
    
    # 输出结果
    print_results(results, 'val')
    
    # 保存结果
    output_file = 'val_evaluation_results.json'
    evaluator.save_results(results, output_file)
    print(f"\n详细结果已保存到: {output_file}")


def print_results(results, mode):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    if mode == 'labeled' or mode == 'mixed':
        print(f"总样本数: {results['total']}")
        if 'correct' in results:
            print(f"正确预测数: {results['correct']}")
            print(f"准确率: {results['accuracy'] * 100:.2f}%")
        if mode == 'mixed':
            print(f"\n标注数据统计:")
            print(f"  已标注样本数: {results['labeled_total']}")
            print(f"  标注数据正确数: {results['labeled_correct']}")
            print(f"  标注数据准确率: {results['labeled_accuracy'] * 100:.2f}%")
    
    elif mode == 'all':
        print(f"总样本数: {results['total']}")
    
    elif mode == 'val':
        print(f"有效样本数: {results['total']}")
        print(f"正确预测数: {results['correct']}")
        print(f"准确率: {results['accuracy'] * 100:.2f}%")
    
    # 打印预测分布
    if 'pred_distribution' in results:
        print("\n" + "=" * 60)
        print("预测字符分布 (前10个)")
        print("=" * 60)
        pred_dist = sorted(results['pred_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for char, count in pred_dist:
            print(f"  '{char}': {count} 次")
    
    # 打印错误示例
    if mode in ['labeled', 'val'] and 'errors' in results and len(results['errors']) > 0:
        print("\n" + "=" * 60)
        print(f"错误预测示例 ({min(5, len(results['errors']))}条)")
        print("=" * 60)
        for i, error in enumerate(results['errors'][:5]):
            img_name = os.path.basename(error['image_path'])
            print(f"{i+1}. 图像: {img_name}")
            print(f"   真实: '{error['true_char']}', 预测: '{error['pred_char']}' (置信度: {error['confidence']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='OCR模型评估工具')
    parser.add_argument('dataset', nargs='?', help='数据集名称（如 pdf5823, pdf01）')
    parser.add_argument('--mode', '-m', choices=['labeled', 'all', 'mixed', 'val'], 
                        default='labeled', help='评估模式')
    parser.add_argument('--model', '-p', help='模型路径')
    args = parser.parse_args()
    
    # 如果没有指定数据集且模式是val，评估验证集
    if args.dataset is None:
        if args.mode == 'val':
            evaluate_val(model_path=args.model)
        else:
            parser.print_help()
            sys.exit(1)
    else:
        # 评估PDF数据集
        evaluate_pdf(args.dataset, mode=args.mode, model_path=args.model)


if __name__ == '__main__':
    main()