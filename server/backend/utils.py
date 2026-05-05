import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List
from fastapi import HTTPException
from config import LABELS_JSON


def load_json_file(file_path: Path) -> Optional[dict]:
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_hog_features(image_path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        target_size = (64, 64)
        img = cv2.resize(img, target_size)
        hog = cv2.HOGDescriptor(
            _winSize=(target_size[0], target_size[1]),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        features = hog.compute(img)
        return features.flatten()
    except Exception:
        return None


def get_labeled_clusters_info():
    if not LABELS_JSON.exists():
        return {}
    labels_data = load_json_file(LABELS_JSON) or {}
    result = {}
    for cluster_id, label_info in labels_data.items():
        if label_info.get("status") == "labeled" and label_info.get("char"):
            result[cluster_id] = {
                "char": label_info["char"],
                "char_labels": label_info.get("char_labels", {}),
                "confidence": label_info.get("confidence", 0)
            }
    return result


def chars_to_lines(chars: List[dict], image_width: int) -> List[dict]:
    line_dict = {}
    for char in chars:
        col_start = char.get("col_start")
        if col_start is not None and col_start != 0:
            line_dict[col_start] = "red"
    for char in chars:
        col_end = char.get("col_end")
        if col_end is not None and col_end != 0:
            line_dict[col_end] = "green"

    lines = [{"pos": pos, "color": color} for pos, color in line_dict.items()]
    lines.sort(key=lambda x: x["pos"])
    return lines


def lines_to_chars(lines: List[dict], image_width: int) -> List[dict]:
    positions = sorted([item["pos"] for item in lines])
    chars = []
    for i in range(0, len(positions), 2):
        if i + 1 >= len(positions):
            break
        left = positions[i]
        right = positions[i + 1]
        chars.append({
            "col_start": left,
            "col_end": right,
            "width": right - left
        })
    return chars


def get_image_width(image_id: str, raw_images_dir: Path) -> int:
    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = raw_images_dir / f"{image_id}{ext}"
        if img_file.exists():
            with Image.open(img_file) as img:
                return img.width
    raise HTTPException(status_code=404, detail="获取图片宽度失败")