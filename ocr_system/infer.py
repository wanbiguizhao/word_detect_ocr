import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from torchvision import transforms
import argparse

from config import *
from datasets.char_dataset import CharDataset
from models.resnet_ocr import ResNetOCR, get_default_num_classes


def load_model(model_path, num_classes=None):
    """加载训练好的模型"""
    if num_classes is None:
        num_classes = get_default_num_classes()
    model = ResNetOCR(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def get_char_mapping():
    """获取字符到ID的映射和ID到字符的映射"""
    try:
        from configs.char_mapping import CharMappingManager
        manager = CharMappingManager()
        stats = manager.get_stats()
        idx_to_char = {}
        
        for label_id in range(stats["next_custom_id"]):
            char = manager.get_char(label_id)
            if char:
                idx_to_char[label_id] = char
        
        return idx_to_char, manager
    except ImportError:
        try:
            from ocr_system.configs.char_mapping import CharMappingManager
            manager = CharMappingManager()
            stats = manager.get_stats()
            idx_to_char = {}
            
            for label_id in range(stats["next_custom_id"]):
                char = manager.get_char(label_id)
                if char:
                    idx_to_char[label_id] = char
            
            return idx_to_char, manager
        except Exception as e:
            print("Warning: Cannot load global char mapping, using dataset mapping:", e)
            val_dataset = CharDataset(VAL_CSV, transform=None)
            idx_to_char = {i: c for i, c in enumerate(val_dataset.idx_to_char)}
            return idx_to_char, None


def predict(image_path, model, transform, idx_to_char):
    """预测单张图片"""
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    
    char = idx_to_char.get(predicted.item(), f'UNKNOWN_{predicted.item()}')
    return char, confidence, predicted.item()


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
        char = idx_to_char.get(pred.item(), f'UNKNOWN_{pred.item()}')
        results.append({
            'image_path': path,
            'prediction': char,
            'confidence': conf,
            'label_id': pred.item()
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='OCR Inference')
    parser.add_argument('--model', type=str, default=INFERENCE_MODEL_PATH, help='Path to model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    args = parser.parse_args()
    
    # 获取字符映射
    idx_to_char, _ = get_char_mapping()
    num_classes = max(idx_to_char.keys()) + 1 if idx_to_char else NUM_CLASSES
    
    # 加载模型
    model = load_model(args.model, num_classes=num_classes)
    print(f'Model loaded from {args.model}')
    print(f'Number of classes: {num_classes}')
    
    # 变换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    if args.image:
        char, confidence, label_id = predict(args.image, model, transform, idx_to_char)
        print(f'Image: {args.image}')
        print(f'Prediction: {char} (ID: {label_id})')
        print(f'Confidence: {confidence:.4f}')
    
    elif args.batch:
        results = batch_predict(args.batch, model, transform, idx_to_char)
        for result in results:
            print(f'{result["image_path"]}: {result["prediction"]} (ID: {result["label_id"]}, confidence: {result["confidence"]:.4f})')
    
    else:
        print('Please provide --image or --batch argument')


if __name__ == '__main__':
    main()