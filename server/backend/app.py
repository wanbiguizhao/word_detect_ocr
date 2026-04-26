import json
import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image

# ======================================
# 应用初始化
# ======================================
app = FastAPI(title="主动学习汉字分割标注系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ======================================
# 路径配置
# ======================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"

RAW_IMAGES_DIR = DATASET_DIR / "raw_images"
RULE_JSONS_DIR = DATASET_DIR / "rule_jsons"
MODEL_JSONS_DIR = DATASET_DIR / "model_jsons"
FUSION_JSONS_DIR = DATASET_DIR / "fusion_jsons"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TOP_SAMPLES_PATH = DATASET_DIR / "top_annotate_samples.json"

for dir_path in [RAW_IMAGES_DIR, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ======================================
# 数据模型
# ======================================
class Line(BaseModel):
    pos: int
    color: str

class AnnotationSubmit(BaseModel):
    lines: List[Line]

# ======================================
# 工具函数
# ======================================
def load_json_file(file_path: Path) -> Optional[dict]:
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

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

def get_image_width(image_id: str) -> int:
    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = RAW_IMAGES_DIR / f"{image_id}{ext}"
        if img_file.exists():
            with Image.open(img_file) as img:
                return img.width
    raise HTTPException(status_code=404, detail="获取图片宽度失败")

# ======================================
# 1. 列表接口（POST筛选，修复完成）
# ======================================
@app.post("/api/images")
async def get_images(request: Request):
    data = await request.json()
    is_annotated = data.get("is_annotated")
    print("📌 后端收到参数 is_annotated =", is_annotated)

    sample_list = load_json_file(TOP_SAMPLES_PATH)
    items = []
    target = None
    if is_annotated == "true":
        target = True
    elif is_annotated == "false":
        target = False

    for sample in sample_list or []:
        img_name = sample.get("img_name", "")
        total_score = sample.get("total_score", 0.0)
        if not img_name:
            continue

        anno_file = ANNOTATIONS_DIR / f"{img_name}.json"
        anno_data = load_json_file(anno_file)
        annotated = anno_data.get("is_annotated", False) if anno_data else False
        updated_at = anno_data.get("updated_at", "") if anno_data else ""

        if target is not None and annotated != target:
            continue

        items.append({
            "id": img_name,
            "image_name": f"{img_name}.png",
            "score": round(total_score, 4),
            "is_annotated": annotated,
            "updated_at": updated_at
        })
    return {"code": 0, "msg": "success", "data": items}

# ======================================
# 🔥 2. 【核心修复】恢复详情接口 + 支持特殊字符（解决404）
# ======================================
@app.get("/api/images/{image_id:path}/detail")
def get_detail(image_id: str):
    img_path = None
    # 匹配图片后缀
    for ext in [".png", ".jpg", ".jpeg"]:
        target_img = RAW_IMAGES_DIR / f"{image_id}{ext}"
        if target_img.exists():
            img_path = target_img
            break
    if not img_path:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    img_width = get_image_width(image_id)

    # 加载所有切割数据
    rule_data = load_json_file(RULE_JSONS_DIR / f"{image_id}_chars.json") or {}
    rule_lines = chars_to_lines(rule_data.get("chars", []), img_width)

    model_data = load_json_file(MODEL_JSONS_DIR / f"{image_id}_model.json") or {}
    model_lines = chars_to_lines(model_data.get("chars", []), img_width)

    fusion_data = load_json_file(FUSION_JSONS_DIR / f"{image_id}_fusion.json") or {}
    fusion_lines = chars_to_lines(fusion_data.get("chars", []), img_width)

    # 🔥 加载你的标注数据（恢复显示）
    anno_data = load_json_file(ANNOTATIONS_DIR / f"{image_id}.json") or {}
    annotation_lines = chars_to_lines(anno_data.get("chars", []), img_width)

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "id": image_id,
            "image_name": img_path.name,
            "image_url": f"/api/images/{image_id}/raw",
            "rule_lines": rule_lines,
            "model_lines": model_lines,
            "fusion_lines": fusion_lines,
            "annotation": {"lines": annotation_lines}
        }
    }

# ======================================
# 3. 保存标注接口
# ======================================
@app.post("/api/images/{image_id:path}/annotate")
def save_anno(image_id: str, body: AnnotationSubmit):
    lines_list = [line.model_dump() for line in body.lines]
    img_width = get_image_width(image_id)
    chars_list = lines_to_chars(lines_list, img_width)

    anno_data = {
        "image_name": f"{image_id}.png",
        "is_annotated": True,
        "chars": chars_list,
        "updated_at": datetime.datetime.now().isoformat()
    }

    save_path = ANNOTATIONS_DIR / f"{image_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "保存成功"}

# ======================================
# 4. 图片预览接口
# ======================================
@app.get("/api/images/{image_id:path}/raw")
async def get_raw(image_id: str):
    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = RAW_IMAGES_DIR / f"{image_id}{ext}"
        if img_file.exists():
            with open(img_file, "rb") as f:
                content = f.read()
            
            response = Response(content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Content-Type"] = f"image/{ext.strip('.')}"
            return response

    raise HTTPException(status_code=404, detail="图片不存在")