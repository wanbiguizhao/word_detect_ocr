import os
import sys

# 添加项目路径（必须在其他导入之前）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from torchvision import transforms
import argparse

from config import *
from datasets.char_dataset import CharDataset
from models.resnet_ocr import ResNetOCR

def load_model(model_path, num_classes=1156):
    """加载训练好的模型"""
    model = ResNetOCR(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path, model, transform, idx_to_char):
    """预测单张图片"""
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    
    return idx_to_char[predicted.item()], confidence

def batch_predict(image_paths, model, transform, idx_to_char):
    """批量预测"""
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')
        img = transform(img)
        images.append(img)
    
    images = torch.stack(images)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        confidences = torch.softmax(outputs, dim=1)[range(len(predictions)), predictions].tolist()
    
    results = []
    for path, pred, conf in zip(image_paths, predictions, confidences):
        results.append({
            'image_path': path,
            'prediction': idx_to_char[pred.item()],
            'confidence': conf
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='OCR Inference')
    parser.add_argument('--model', type=str, default=INFERENCE_MODEL_PATH, help='Path to model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    args = parser.parse_args()
    
    # 加载字符映射
    val_dataset = CharDataset(VAL_CSV, transform=None)
    idx_to_char = val_dataset.idx_to_char
    
    # 加载模型
    model = load_model(args.model, num_classes=len(idx_to_char))
    print(f'Model loaded from {args.model}')
    
    # 变换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    if args.image:
        # 单张图片预测
        char, confidence = predict(args.image, model, transform, idx_to_char)
        print(f'Image: {args.image}')
        print(f'Prediction: {char}')
        print(f'Confidence: {confidence:.4f}')
    
    elif args.batch:
        # 批量预测
        results = batch_predict(args.batch, model, transform, idx_to_char)
        for result in results:
            print(f'{result["image_path"]}: {result["prediction"]} (confidence: {result["confidence"]:.4f})')
    
    else:
        print('Please provide --image or --batch argument')

if __name__ == '__main__':
    main()