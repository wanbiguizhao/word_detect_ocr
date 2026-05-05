import json
import datetime
from fastapi import APIRouter, HTTPException, Request, Response
from pathlib import Path
from config import RAW_IMAGES_DIR, ANNOTATIONS_DIR, TOP_SAMPLES_PATH, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR
from models import AnnotationSubmit, BatchLabelSave
from utils import load_json_file, chars_to_lines, lines_to_chars, get_image_width

router = APIRouter()


@router.post("/api/images")
async def get_images(request: Request):
    data = await request.json()
    is_annotated = data.get("is_annotated")

    sample_list = load_json_file(TOP_SAMPLES_PATH)
    items = []
    target = None
    postpone_target = None

    if is_annotated == "true":
        target = True
    elif is_annotated == "false":
        target = False
    elif is_annotated == "pending":
        postpone_target = True

    for sample in sample_list or []:
        img_name = sample.get("img_name", "")
        total_score = sample.get("total_score", 0.0)
        if not img_name:
            continue

        anno_file = ANNOTATIONS_DIR / f"{img_name}.json"
        anno_data = load_json_file(anno_file)
        annotated = anno_data.get("is_annotated", False) if anno_data else False
        is_postponed = anno_data.get("is_postponed", False) if anno_data else False
        updated_at = anno_data.get("updated_at", "") if anno_data else ""

        if postpone_target is not None:
            if not is_postponed:
                continue
        elif target is not None:
            if annotated != target or is_postponed:
                continue

        items.append({
            "id": img_name,
            "image_name": f"{img_name}.png",
            "score": round(total_score, 4),
            "is_annotated": annotated,
            "is_postponed": is_postponed,
            "updated_at": updated_at
        })

    return {"code": 0, "msg": "success", "data": items, "total": len(items)}


@router.get("/api/images/{image_id:path}/detail")
def get_detail(image_id: str):
    img_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        target_img = RAW_IMAGES_DIR / f"{image_id}{ext}"
        if target_img.exists():
            img_path = target_img
            break
    if not img_path:
        raise HTTPException(status_code=404, detail="图片不存在")

    img_width = get_image_width(image_id, RAW_IMAGES_DIR)

    rule_data = load_json_file(RULE_JSONS_DIR / f"{image_id}_chars.json") or {}
    rule_lines = chars_to_lines(rule_data.get("chars", []), img_width)

    model_data = load_json_file(MODEL_JSONS_DIR / f"{image_id}_model.json") or {}
    model_lines = chars_to_lines(model_data.get("chars", []), img_width)

    fusion_data = load_json_file(FUSION_JSONS_DIR / f"{image_id}_fusion.json") or {}
    fusion_lines = chars_to_lines(fusion_data.get("chars", []), img_width)

    anno_data = load_json_file(ANNOTATIONS_DIR / f"{image_id}.json") or {}
    annotation_lines = chars_to_lines(anno_data.get("chars", []), img_width)
    is_annotated = anno_data.get("is_annotated", False) if anno_data else False
    is_postponed = anno_data.get("is_postponed", False) if anno_data else False

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "id": image_id,
            "image_name": img_path.name,
            "image_url": f"/api/images/{image_id}/raw",
            "is_annotated": is_annotated,
            "is_postponed": is_postponed,
            "rule_lines": rule_lines,
            "model_lines": model_lines,
            "fusion_lines": fusion_lines,
            "annotation": {"lines": annotation_lines}
        }
    }


@router.post("/api/images/{image_id:path}/annotate")
def save_anno(image_id: str, body: AnnotationSubmit):
    lines_list = [line.model_dump() for line in body.lines]
    img_width = get_image_width(image_id, RAW_IMAGES_DIR)
    chars_list = lines_to_chars(lines_list, img_width)

    anno_data = {
        "image_name": f"{image_id}.png",
        "is_annotated": True,
        "is_postponed": False,
        "chars": chars_list,
        "updated_at": datetime.datetime.now().isoformat()
    }

    save_path = ANNOTATIONS_DIR / f"{image_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "保存成功"}


@router.post("/api/images/{image_id:path}/postpone")
def postpone_anno(image_id: str):
    save_path = ANNOTATIONS_DIR / f"{image_id}.json"

    if save_path.exists():
        anno_data = load_json_file(save_path)
        if anno_data:
            anno_data["is_annotated"] = True
            anno_data["is_postponed"] = True
            anno_data["updated_at"] = datetime.datetime.now().isoformat()
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(anno_data, f, ensure_ascii=False, indent=2)
    else:
        anno_data = {
            "image_name": f"{image_id}.png",
            "is_annotated": True,
            "is_postponed": True,
            "chars": [],
            "updated_at": datetime.datetime.now().isoformat()
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(anno_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "已标记为暂不标注"}


@router.get("/api/images/{image_id:path}/raw")
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


@router.get("/api/char-images/{image_name:path}")
async def get_char_image(image_name: str):
    from config import PROJECT_ROOT
    char_dir = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars"

    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = char_dir / f"{image_name}{ext}" if not image_name.endswith(ext) else char_dir / image_name
        if img_file.exists():
            with open(img_file, "rb") as f:
                content = f.read()

            response = Response(content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Content-Type"] = "image/png"
            return response

    raise HTTPException(status_code=404, detail="图片不存在")


@router.get("/api/line-images/{line_path:path}")
async def get_line_image(line_path: str):
    from config import PROJECT_ROOT
    line_dir = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_lines"
    line_file = line_dir / f"{line_path}.png"

    if not line_file.exists():
        raise HTTPException(status_code=404, detail="行图片不存在")

    with open(line_file, "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Content-Type"] = "image/png"
    return response