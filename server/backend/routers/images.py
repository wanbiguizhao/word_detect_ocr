import json
import datetime
from fastapi import APIRouter, HTTPException, Request, Response
from pathlib import Path
from config import RAW_IMAGES_DIR, ANNOTATIONS_DIR, TOP_SAMPLES_PATH, RULE_JSONS_DIR, MODEL_JSONS_DIR, FUSION_JSONS_DIR, SOURCE_DATASET_ID, PROJECT_ROOT
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
    from config import PROJECT_ROOT, DATASET_DIR
    
    # FastAPI会自动URL解码，所以这里需要处理解码后的路径
    # 统一使用正斜杠路径
    image_path = image_name.replace("\\", "/")
    
    # 提取纯文件名（去掉路径和扩展名）
    if "/" in image_path:
        filename = image_path.split("/")[-1]
    else:
        filename = image_path
    
    # 去掉扩展名
    base_name = filename
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            break
    
    # 1. 尝试相对路径（从bussiness目录开始）
    relative_path = PROJECT_ROOT / "bussiness" / image_path
    print(f"[DEBUG] Trying path: {relative_path}")
    if relative_path.exists():
        with open(relative_path, "rb") as f:
            content = f.read()
        response = Response(content)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Content-Type"] = "image/png"
        return response
    
    # 2. 尝试不带扩展名的相对路径
    for ext in [".png", ".jpg", ".jpeg"]:
        relative_path_with_ext = PROJECT_ROOT / "bussiness" / f"{image_path}{ext}"
        if relative_path_with_ext.exists():
            with open(relative_path_with_ext, "rb") as f:
                content = f.read()
            response = Response(content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Content-Type"] = "image/png"
            return response
    
    # 3. 尝试完整路径（处理绝对路径）
    if "\\" in image_name or len(image_name) > 50:
        full_path = Path(image_name)
        if full_path.exists():
            with open(full_path, "rb") as f:
                content = f.read()
            response = Response(content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Content-Type"] = "image/png"
            return response
    
    # 4. 尝试多个可能的字符图片目录
    possible_dirs = [
        DATASET_DIR / "pdf_chars",
        PROJECT_ROOT / "bussiness" / "datahome" / SOURCE_DATASET_ID / "pdf_chars",
        PROJECT_ROOT / "bussiness" / "datahome" / "pdf5823" / "pdf_chars",
        DATASET_DIR / "clusters" / "char_images",
        DATASET_DIR,
        PROJECT_ROOT / "bussiness" / "datahome" / SOURCE_DATASET_ID,
        PROJECT_ROOT / "bussiness" / "datahome" / "pdf5823",
    ]

    for char_dir in possible_dirs:
        for ext in [".png", ".jpg", ".jpeg"]:
            img_file = char_dir / f"{base_name}{ext}"
            if img_file.exists():
                with open(img_file, "rb") as f:
                    content = f.read()
                response = Response(content)
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Content-Type"] = "image/png"
                return response
    
    # 3. 从行图片中裁剪字符（最后的尝试）
    try:
        # 从 char_id 中解析出行信息
        # 格式: page_X_line_Y_char_Z
        parts = base_name.split("_")
        if len(parts) >= 4 and parts[0] == "page":
            page_num = parts[1]
            line_idx = parts[3]
            line_name = f"page_{page_num}_line_{line_idx}"
            
            # 查找行图片
            line_dir = DATASET_DIR / "pdf_lines"
            line_file = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = line_dir / f"{line_name}{ext}"
                if candidate.exists():
                    line_file = candidate
                    break
            
            if line_file:
                # 尝试从聚类数据中获取字符坐标
                clusters_path = DATASET_DIR / "clusters" / "hog_clusters.json"
                if clusters_path.exists():
                    import json
                    with open(clusters_path, "r", encoding="utf-8") as f:
                        clusters_data = json.load(f)
                    
                    # 搜索所有聚类找到匹配的字符
                    for cluster_id, chars in clusters_data.get("clusters", {}).items():
                        for char_info in chars:
                            if char_info.get("char_id") == base_name:
                                # 找到字符信息，提取坐标
                                lineage = char_info.get("lineage", {})
                                col_start = lineage.get("col_start", 0)
                                col_end = lineage.get("col_end", 100)
                                width = lineage.get("width", col_end - col_start)
                                
                                # 裁剪行图片
                                from PIL import Image
                                with Image.open(line_file) as img:
                                    # 计算裁剪区域
                                    left = col_start
                                    top = 0
                                    right = min(col_end, img.width)
                                    bottom = img.height
                                    
                                    cropped = img.crop((left, top, right, bottom))
                                    import io
                                    buffer = io.BytesIO()
                                    cropped.save(buffer, format="PNG")
                                    content = buffer.getvalue()
                                
                                response = Response(content)
                                response.headers["Access-Control-Allow-Origin"] = "*"
                                response.headers["Content-Type"] = "image/png"
                                return response
    except Exception as e:
        print(f"Error cropping char from line image: {e}")

    raise HTTPException(status_code=404, detail=f"图片不存在: {image_name}")


@router.get("/api/line-images/{line_path:path}")
async def get_line_image(line_path: str):
    line_dir = PROJECT_ROOT / "bussiness" / "datahome" / SOURCE_DATASET_ID / "pdf_lines"
    line_file = line_dir / f"{line_path}.png"

    if not line_file.exists():
        raise HTTPException(status_code=404, detail="行图片不存在")

    with open(line_file, "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Content-Type"] = "image/png"
    return response


@router.get("/api/image/pdf_chars/{char_id}")
async def get_char_image_from_pdf_chars(char_id: str):
    """获取PDF字符图片（支持多轮聚类页面）"""
    from config import DATASET_DIR
    
    # 尝试多个可能的字符图片目录
    possible_dirs = [
        DATASET_DIR / "pdf_chars",
        PROJECT_ROOT / "bussiness" / "datahome" / SOURCE_DATASET_ID / "pdf_chars",
        PROJECT_ROOT / "bussiness" / "datahome" / "pdf5823" / "pdf_chars",
    ]

    for char_dir in possible_dirs:
        for ext in [".png", ".jpg", ".jpeg"]:
            img_file = char_dir / f"{char_id}{ext}"
            if img_file.exists():
                with open(img_file, "rb") as f:
                    content = f.read()
                response = Response(content)
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Content-Type"] = "image/png"
                return response

    raise HTTPException(status_code=404, detail=f"字符图片不存在: {char_id}")