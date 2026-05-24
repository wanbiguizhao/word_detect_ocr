import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "ocr_system"))

os.chdir(project_root)


def generate_prelabels(dataset_name: str, output_dir: str = None, regenerate: bool = False):
    """
    生成预标注数据

    Args:
        dataset_name: 数据集名称（如 'pdf5823'）
        output_dir: 输出目录，默认 None 表示输出到数据集目录下
        regenerate: 是否重新生成（覆盖已有结果）
    """
    dataset_dir = project_root / "bussiness" / "datahome" / dataset_name
    pdf_chars_dir = dataset_dir / "pdf_chars"
    lineage_path = dataset_dir / "lineage.json"

    if output_dir:
        output_path = Path(output_dir) / "pre_labels.json"
    else:
        output_path = dataset_dir / "pre_labels.json"

    if output_path.exists() and not regenerate:
        print(f"[INFO] Pre-label file already exists: {output_path}")
        print(f"[INFO] Use --regenerate to overwrite")
        return None

    char_mapping_path = project_root / "ocr_system" / "configs" / "char_mapping" / "label_to_char.json"
    if char_mapping_path.exists():
        with open(char_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "labels" in data:
            idx_to_char = {}
            for k, v in data["labels"].items():
                idx_to_char[int(k)] = v["char"]
        else:
            idx_to_char = {int(k): v for k, v in data.items()}
    else:
        print("[ERR] Char mapping file not found")
        return None

    lineage_data = {}
    if lineage_path.exists():
        with open(lineage_path, 'r', encoding='utf-8') as f:
            lineage_data = json.load(f)
        print(f"[INFO] Loaded lineage data from {lineage_path}")
    else:
        print(f"[WARN] Lineage file not found: {lineage_path}")

    high_threshold = 0.9
    low_threshold = 0.7

    png_files = list(pdf_chars_dir.glob("*.png"))
    print(f"[INFO] Found {len(png_files)} character images in {pdf_chars_dir}")

    if len(png_files) == 0:
        print("[ERR] No character images found")
        return None

    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        model_path = project_root / "ocr_system" / "model_store" / "inference" / "ocr_model_best.pth"
        if not model_path.exists():
            print(f"[ERR] Model file not found: {model_path}")
            return None

        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        fc_key = None
        for key in state_dict.keys():
            if 'fc.4' in key and 'weight' in key:
                fc_key = key
                break

        if fc_key:
            num_classes = state_dict[fc_key].shape[0]
            print(f"[INFO] Model num_classes from checkpoint: {num_classes}")
        else:
            num_classes = 6893
            print(f"[WARN] Could not determine num_classes, using default: {num_classes}")

        from models.resnet_ocr import ResNetOCR
        model = ResNetOCR(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"[OK] Model loaded from {model_path}")

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        prelabels = []
        high_count = 0
        low_count = 0
        batch_size = 32

        chars_info = lineage_data.get("chars", {})

        print(f"[INFO] Starting OCR prediction...")

        for i in range(0, len(png_files), batch_size):
            batch_files = png_files[i:i+batch_size]
            images = []

            for png_file in batch_files:
                try:
                    img = Image.open(png_file).convert('L')
                    img = transform(img)
                    images.append(img)
                except Exception as e:
                    print(f"[WARN] Failed to load {png_file}: {e}")
                    continue

            if not images:
                continue

            images = torch.stack(images)

            with torch.no_grad():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)

            for j, png_file in enumerate(batch_files):
                try:
                    pred_idx = predictions[j].item()
                    conf = confidences[j].item()
                    pred_char = idx_to_char.get(pred_idx, f"UNKNOWN_{pred_idx}")

                    if conf >= high_threshold:
                        level = "high"
                        high_count += 1
                    elif conf >= low_threshold:
                        level = "medium"
                    else:
                        level = "low"
                        low_count += 1

                    relative_path = str(png_file.relative_to(project_root / "bussiness"))

                    char_id = png_file.stem

                    lineage = {}
                    if char_id in chars_info:
                        char_info = chars_info[char_id]
                        lineage = {
                            "line_name": char_info.get("line_name", ""),
                            "char_idx": char_info.get("char_idx", 0),
                            "page": char_info.get("page_num", 0),
                            "line": char_info.get("line_idx", 0),
                            "col_start": char_info.get("col_start"),
                            "col_end": char_info.get("col_end"),
                            "width": char_info.get("width")
                        }

                    prelabel_entry = {
                        "image_path": relative_path,
                        "char_id": char_id,
                        "predicted_char": pred_char,
                        "confidence": round(conf, 4),
                        "confidence_level": level,
                        "status": "pending"
                    }

                    if lineage:
                        prelabel_entry["lineage"] = lineage

                    prelabels.append(prelabel_entry)

                except Exception as e:
                    print(f"[WARN] Failed to process {png_file}: {e}")

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(png_files):
                print(f"[INFO] Processed {min(i + batch_size, len(png_files))}/{len(png_files)} images")

        char_counts = defaultdict(lambda: {"total": 0, "high": 0, "low": 0, "medium": 0})
        for prelabel in prelabels:
            char = prelabel["predicted_char"]
            char_counts[char]["total"] += 1
            char_counts[char][prelabel["confidence_level"]] += 1

        result = {
            "version": "1.0",
            "dataset": dataset_name,
            "generated_at": datetime.now().isoformat(),
            "stats": {
                "total": len(prelabels),
                "high_confidence": high_count,
                "low_confidence": low_count
            },
            "char_counts": dict(char_counts),
            "prelabels": prelabels
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] Pre-labeling complete!")
        print(f"     Dataset: {dataset_name}")
        print(f"     Total: {len(prelabels)}")
        print(f"     High confidence (>={high_threshold}): {high_count}")
        print(f"     Medium confidence (>={low_threshold}): {len(prelabels) - high_count - low_count}")
        print(f"     Low confidence (<{low_threshold}): {low_count}")
        print(f"     Output: {output_path}")

        return result

    except Exception as e:
        print(f"[ERR] Failed to generate prelabels: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成预标注数据")
    parser.add_argument("--dataset", type=str, default="pdf5824", help="数据集名称")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认输出到数据集目录下")
    parser.add_argument("--regenerate", action="store_true", help="重新生成（覆盖已有结果）")

    args = parser.parse_args()

    generate_prelabels(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        regenerate=args.regenerate
    )