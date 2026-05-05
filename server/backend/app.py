import json
import datetime
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

# ======================================
# 从 config.py 导入配置
# ======================================
from config import (
    DATASET_ID,
    DATASET_DIR,
    RAW_IMAGES_DIR,
    RULE_JSONS_DIR,
    MODEL_JSONS_DIR,
    FUSION_JSONS_DIR,
    ANNOTATIONS_DIR,
    TOP_SAMPLES_PATH,
    CLUSTERS_JSON,
    LABELS_JSON,
    pseudo_label_cache,
    executor
)

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

# 自定义JSON响应，确保UTF-8编码
@app.middleware("http")
async def set_encoding(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

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

        # 根据筛选条件过滤
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
        "is_postponed": False,
        "chars": chars_list,
        "updated_at": datetime.datetime.now().isoformat()
    }

    save_path = ANNOTATIONS_DIR / f"{image_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "保存成功"}


class BatchLabelItem(BaseModel):
    char: str
    charIndex: int


class BatchLabelSave(BaseModel):
    clusterId: int
    labels: List[BatchLabelItem]


@app.post("/api/cluster-labels/batch-save")
def batch_save_cluster_labels(body: BatchLabelSave):
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)

    if LABELS_JSON.exists():
        labels_data = load_json_file(LABELS_JSON) or {}
    else:
        labels_data = {}

    cluster_key = str(body.clusterId)

    if cluster_key not in labels_data:
        labels_data[cluster_key] = {
            "char": None,
            "chars": {},
            "status": "unlabeled",
            "confidence": None,
            "alias": "",
            "char_labels": {}
        }

    if "char_labels" not in labels_data[cluster_key]:
        labels_data[cluster_key]["char_labels"] = {}

    existing_chars = set()
    for cluster_data in labels_data.values():
        if cluster_data.get("status") == "labeled" and cluster_data.get("char_labels"):
            for label_info in cluster_data["char_labels"].values():
                if label_info.get("char"):
                    existing_chars.add(label_info["char"])

    new_chars = set()

    for item in body.labels:
        char_key = str(item.charIndex)
        labels_data[cluster_key]["char_labels"][char_key] = {
            "char": item.char,
            "labeled_at": datetime.datetime.now().isoformat()
        }
        
        if item.char not in existing_chars:
            new_chars.add(item.char)

    all_chars = [v["char"] for v in labels_data[cluster_key]["char_labels"].values() if v.get("char")]
    if all_chars:
        from collections import Counter
        char_counts = Counter(all_chars)
        most_common = char_counts.most_common(1)[0]
        labels_data[cluster_key]["char"] = most_common[0]
        labels_data[cluster_key]["confidence"] = most_common[1] / len(all_chars)
        labels_data[cluster_key]["chars"] = dict(char_counts)
        labels_data[cluster_key]["status"] = "labeled"

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    cluster_id = str(body.clusterId)
    for item in body.labels:
        char = item.char
        if char in pseudo_label_cache:
            if cluster_id in pseudo_label_cache[char]:
                before_count = len(pseudo_label_cache[char])
                del pseudo_label_cache[char][cluster_id]
                after_count = len(pseudo_label_cache[char]) if char in pseudo_label_cache else 0
                if len(pseudo_label_cache[char]) == 0:
                    del pseudo_label_cache[char]
            else:
                pass
        else:
            pass

    new_chars_list = sorted(list(new_chars))
    return {
        "code": 0, 
        "msg": f"保存成功，共 {len(body.labels)} 条",
        "new_chars_count": len(new_chars_list),
        "new_chars": new_chars_list
    }

# ======================================
# 3.1 暂不标注接口
# ======================================
@app.post("/api/images/{image_id:path}/postpone")
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


# ======================================
# OCR标注相关配置
# ======================================
CLUSTERS_DIR = DATASET_DIR / "clusters"
CLUSTERS_JSON = CLUSTERS_DIR / "hog_clusters.json"
LABELS_JSON = CLUSTERS_DIR / "labeling" / "labels.json"


# ======================================
# 5. 获取聚类数据
# ======================================
@app.get("/api/clusters")
def get_clusters():
    if not CLUSTERS_JSON.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")
    
    data = load_json_file(CLUSTERS_JSON)
    return {
        "code": 0,
        "msg": "success",
        "clusters": data.get("clusters", {}),
        "config": data.get("config", {})
    }


# ======================================
# 6. 获取聚类标注数据
# ======================================
@app.get("/api/cluster-labels")
def get_cluster_labels():
    if not LABELS_JSON.exists():
        return {"code": 0, "msg": "success", "data": {}}

    data = load_json_file(LABELS_JSON)

    for cluster_data in data.values():
        char_labels = cluster_data.get("char_labels", {})
        if char_labels and "chars" not in cluster_data:
            from collections import Counter
            all_chars = [v["char"] for v in char_labels.values() if v.get("char")]
            if all_chars:
                cluster_data["chars"] = dict(Counter(all_chars))

    return {"code": 0, "msg": "success", "data": data}


# ======================================
# 7. 保存聚类标注
# ======================================
class ClusterLabelSave(BaseModel):
    clusterId: int
    alias: Optional[str] = None
    char: Optional[str] = None
    charIndex: Optional[int] = None

@app.post("/api/cluster-labels/save")
def save_cluster_label(body: ClusterLabelSave):
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)

    if LABELS_JSON.exists():
        labels_data = load_json_file(LABELS_JSON) or {}
    else:
        labels_data = {}

    cluster_key = str(body.clusterId)

    if cluster_key not in labels_data:
        labels_data[cluster_key] = {
            "char": None,
            "chars": {},
            "status": "unlabeled",
            "confidence": None,
            "alias": "",
            "char_labels": {}
        }

    if body.alias is not None:
        labels_data[cluster_key]["alias"] = body.alias

    if body.char is not None and body.charIndex is not None:
        char_key = str(body.charIndex)
        if "char_labels" not in labels_data[cluster_key]:
            labels_data[cluster_key]["char_labels"] = {}
        labels_data[cluster_key]["char_labels"][char_key] = {
            "char": body.char,
            "labeled_at": datetime.datetime.now().isoformat()
        }

        all_chars = [v["char"] for v in labels_data[cluster_key]["char_labels"].values()]
        if all_chars:
            from collections import Counter
            char_counts = Counter(all_chars)
            most_common = char_counts.most_common(1)[0]
            labels_data[cluster_key]["char"] = most_common[0]
            labels_data[cluster_key]["confidence"] = most_common[1] / len(all_chars)
            labels_data[cluster_key]["chars"] = dict(char_counts)

        labels_data[cluster_key]["status"] = "labeled"
    
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    return {"code": 0, "msg": "保存成功"}


# ======================================
# 8. 标记聚类为暂不标记
# ======================================
@app.post("/api/clusters/{cluster_id}/skip")
def skip_cluster(cluster_id: int):
    if not LABELS_JSON.exists():
        labels_data = {}
    else:
        labels_data = load_json_file(LABELS_JSON) or {}

    cluster_key = str(cluster_id)

    if cluster_key not in labels_data:
        labels_data[cluster_key] = {
            "char": None,
            "chars": {},
            "status": "skipped",
            "confidence": None,
            "alias": "",
            "char_labels": {},
            "skipped_at": datetime.datetime.now().isoformat()
        }
    else:
        labels_data[cluster_key]["status"] = "skipped"
        labels_data[cluster_key]["skipped_at"] = datetime.datetime.now().isoformat()

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "已标记为暂不标记"}

# ======================================
# 9. 获取单个聚类的图片
# ======================================
@app.get("/api/clusters/{cluster_id}/images")
def get_cluster_images(cluster_id: int):
    if not CLUSTERS_JSON.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")
    
    data = load_json_file(CLUSTERS_JSON)
    clusters = data.get("clusters", {})
    
    cluster_key = str(cluster_id)
    if cluster_key not in clusters:
        raise HTTPException(status_code=404, detail="聚类不存在")
    
    chars = clusters[cluster_key]
    
    labels_data = {}
    if LABELS_JSON.exists():
        labels_data = load_json_file(LABELS_JSON) or {}
    
    cluster_labels = labels_data.get(cluster_key, {}).get("char_labels", {})
    
    result = []
    for idx, char_info in enumerate(chars):
        char_key = str(idx)
        result.append({
            "index": idx,
            "char_id": char_info.get("char_id", ""),
            "image_path": char_info.get("image_path", ""),
            "lineage": char_info.get("lineage", {}),
            "label": cluster_labels.get(char_key, {}).get("char", None)
        })
    
    return {
        "code": 0,
        "msg": "success",
        "cluster_id": cluster_id,
        "total": len(result),
        "images": result
    }


# ======================================
# 8.1 获取聚类推荐标签
# ======================================
class RecommendMode(BaseModel):
    mode: str = "global"

@app.get("/api/clusters/{cluster_id}/recommend")
def get_cluster_recommend(cluster_id: int, mode: str = "global"):
    if not CLUSTERS_JSON.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")

    clusters_data = load_json_file(CLUSTERS_JSON)
    clusters = clusters_data.get("clusters", {})

    cluster_key = str(cluster_id)
    if cluster_key not in clusters:
        raise HTTPException(status_code=404, detail="聚类不存在")

    chars = clusters[cluster_key]
    labels_data = load_json_file(LABELS_JSON) or {}
    cluster_labels = labels_data.get(cluster_key, {})

    labeled_info = get_labeled_clusters_info()

    if mode == "local" and cluster_labels.get("char_labels"):
        anchor_char_ids = [
            chars[int(idx)]["char_id"]
            for idx, label_info in cluster_labels["char_labels"].items()
            if label_info.get("char")
        ]
        anchor_char_ids = anchor_char_ids[:10]
    else:
        # 全局模式：优先选择标记数量多的字符作为锚点
        all_anchors = []
        for cid, info in labeled_info.items():
            for idx, label_info in info["char_labels"].items():
                if label_info.get("char"):
                    cluster_chars = clusters.get(str(cid), [])
                    if int(idx) < len(cluster_chars):
                        all_anchors.append({
                            "char_id": cluster_chars[int(idx)]["char_id"],
                            "char": label_info["char"]
                        })
        
        # 按字符分组，统计每个字符的标注数量
        from collections import Counter
        char_counter = Counter([a["char"] for a in all_anchors])
        
        # 按字符频率排序
        sorted_chars = [char for char, count in char_counter.most_common()]
        
        anchor_char_ids = []
        selected_chars = set()
        
        # 优化锚点选择策略：确保所有字符都有机会进入推荐
        # 第一阶段：确保每个字符至少有一个锚点（优先保证字符覆盖）
        for char in sorted_chars:
            if char not in selected_chars:
                char_anchors = [a for a in all_anchors if a["char"] == char]
                if char_anchors:
                    anchor_char_ids.append(char_anchors[0]["char_id"])
                    selected_chars.add(char)
        
        # 第二阶段：根据字符频率分配额外的锚点，直到达到上限
        remaining_quota = 500 - len(anchor_char_ids)
        if remaining_quota > 0:
            for char in sorted_chars:
                if remaining_quota <= 0:
                    break
                count = char_counter[char]
                # 根据标注数量决定还能取多少额外锚点
                if count >= 20:
                    # 高频字：总共取3个（已取1个，还能取2个）
                    extra_anchors = 2
                elif count >= 10:
                    # 中频字：总共取2个（已取1个，还能取1个）
                    extra_anchors = 1
                else:
                    # 低频字：只取1个，不再额外分配
                    extra_anchors = 0
                
                if extra_anchors > 0:
                    char_anchors = [a for a in all_anchors if a["char"] == char][1:]  # 跳过第一个已选的
                    num_to_add = min(extra_anchors, len(char_anchors), remaining_quota)
                    anchor_char_ids.extend([a["char_id"] for a in char_anchors[:num_to_add]])
                    remaining_quota -= num_to_add

    if not anchor_char_ids:
        return {"code": 0, "msg": "无可用锚点", "recommendations": {}}

    anchor_features = {}
    for char_id in anchor_char_ids:
        img_path = str(DATASET_DIR / "pdf_chars" / f"{char_id}.png")
        feat = extract_hog_features(img_path)
        if feat is not None:
            anchor_features[char_id] = feat

    if not anchor_features:
        return {"code": 0, "msg": "无法提取锚点特征", "recommendations": {}}

    char_label_map = {}
    # 本地模式：从当前聚类获取标签映射
    if mode == "local" and cluster_labels.get("char_labels"):
        for idx, label_info in cluster_labels["char_labels"].items():
            if int(idx) < len(chars):
                char_id = chars[int(idx)]["char_id"]
                if char_id in anchor_features:
                    char_label_map[char_id] = label_info["char"]
    # 全局模式：从所有已标记聚类获取标签映射
    else:
        for cid, info in labeled_info.items():
            for idx, label_info in info["char_labels"].items():
                cluster_chars = clusters.get(str(cid), [])
                if int(idx) < len(cluster_chars):
                    char_id = cluster_chars[int(idx)]["char_id"]
                    if char_id in anchor_features:
                        char_label_map[char_id] = label_info["char"]

    recommendations = {}
    for idx, char_info in enumerate(chars):
        char_key = str(idx)
        if cluster_labels.get("char_labels", {}).get(char_key, {}).get("char"):
            continue

        char_id = char_info.get("char_id", "")
        img_path = char_info.get("image_path", "")
        if not img_path:
            img_path = str(DATASET_DIR / "pdf_chars" / f"{char_id}.png")

        target_feat = extract_hog_features(img_path)
        if target_feat is None:
            continue

        best_similarity = 0
        best_char = None

        for anchor_id, anchor_feat in anchor_features.items():
            sim = float(cosine_similarity([target_feat], [anchor_feat])[0][0])
            if sim > best_similarity:
                best_similarity = sim
                best_char = char_label_map.get(anchor_id)

        if best_similarity >= 0.75 and best_char:
            if best_char not in recommendations:
                recommendations[best_char] = {
                    "count": 0,
                    "indices": [],
                    "avg_similarity": 0,
                    "total_similarity": 0
                }
            recommendations[best_char]["count"] += 1
            recommendations[best_char]["indices"].append(idx)
            recommendations[best_char]["total_similarity"] += best_similarity

    for char in recommendations:
        cnt = recommendations[char]["count"]
        recommendations[char]["avg_similarity"] = recommendations[char]["total_similarity"] / cnt
        del recommendations[char]["total_similarity"]
        del recommendations[char]["indices"]

    return {
        "code": 0,
        "msg": "success",
        "mode": mode,
        "anchor_count": len(anchor_features),
        "recommendations": recommendations
    }


# ======================================
# 8.2 获取推荐图片详情
# ======================================
@app.get("/api/clusters/{cluster_id}/recommend/{char}")
def get_recommend_images(cluster_id: int, char: str):
    if not CLUSTERS_JSON.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")

    clusters_data = load_json_file(CLUSTERS_JSON)
    clusters = clusters_data.get("clusters", {})

    cluster_key = str(cluster_id)
    if cluster_key not in clusters:
        raise HTTPException(status_code=404, detail="聚类不存在")

    chars = clusters[cluster_key]
    labels_data = load_json_file(LABELS_JSON) or {}
    cluster_labels = labels_data.get(cluster_key, {})

    labeled_info = get_labeled_clusters_info()

    anchor_char_ids = []
    # 先从当前聚类获取锚点
    if cluster_labels.get("char_labels"):
        for idx, label_info in cluster_labels["char_labels"].items():
            if label_info.get("char") == char:
                if int(idx) < len(chars):
                    anchor_char_ids.append(chars[int(idx)]["char_id"])
    # 如果当前聚类锚点不够，从全局已标记聚类获取
    if len(anchor_char_ids) < 10:
        for cid, info in labeled_info.items():
            for idx, label_info in info["char_labels"].items():
                if label_info.get("char") == char:
                    cluster_chars = clusters.get(str(cid), [])
                    if int(idx) < len(cluster_chars):
                        anchor_char_ids.append(cluster_chars[int(idx)]["char_id"])
            if len(anchor_char_ids) >= 10:
                break

    anchor_features = {}
    for char_id in anchor_char_ids:
        img_path = str(DATASET_DIR / "pdf_chars" / f"{char_id}.png")
        feat = extract_hog_features(img_path)
        if feat is not None:
            anchor_features[char_id] = feat

    if not anchor_features:
        return {"code": 0, "msg": "无可用锚点", "images": []}

    results = []
    for idx, char_info in enumerate(chars):
        char_key = str(idx)
        if cluster_labels.get("char_labels", {}).get(char_key, {}).get("char"):
            continue

        char_id = char_info.get("char_id", "")
        img_path = char_info.get("image_path", "")
        if not img_path:
            img_path = str(DATASET_DIR / "pdf_chars" / f"{char_id}.png")

        target_feat = extract_hog_features(img_path)
        if target_feat is None:
            continue

        best_similarity = 0
        for anchor_id, anchor_feat in anchor_features.items():
            sim = float(cosine_similarity([target_feat], [anchor_feat])[0][0])
            if sim > best_similarity:
                best_similarity = sim

        if best_similarity >= 0.75:
            results.append({
                "index": idx,
                "char_id": char_id,
                "similarity": round(best_similarity, 4)
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = results[:50]  # 限制最多返回50张推荐图片
    return {
        "code": 0,
        "msg": "success",
        "char": char,
        "total": len(results),
        "images": results
    }


# ======================================
# 9. 获取所有已标记标签列表（伪标签传播）
# ======================================
@app.get("/api/pseudo-labels")
def get_pseudo_labels():
    if not LABELS_JSON.exists():
        return {"code": 0, "data": []}

    labels_data = load_json_file(LABELS_JSON) or {}

    char_stats = {}
    for cluster_id, label_info in labels_data.items():
        if label_info.get("status") == "labeled" and label_info.get("char_labels"):
            # 记录该聚类中出现的所有字符
            chars_in_cluster = set()
            
            for idx, info in label_info["char_labels"].items():
                char = info.get("char")
                if char:
                    chars_in_cluster.add(char)
                    if char not in char_stats:
                        char_stats[char] = {
                            "char": char,
                            "total_count": 0,
                            "cluster_count": 0,
                            "clusters": []
                        }
                    char_stats[char]["total_count"] += 1

            # 为该聚类中的每个字符记录聚类ID
            for char in chars_in_cluster:
                if cluster_id not in char_stats[char]["clusters"]:
                    char_stats[char]["clusters"].append(cluster_id)
                    char_stats[char]["cluster_count"] += 1

    result = list(char_stats.values())
    result.sort(key=lambda x: x["total_count"], reverse=True)

    # 计算全局统计
    total_chars = len(char_stats)
    total_clusters = len([cid for cid, info in labels_data.items() if info.get("status") == "labeled"])
    total_images = sum(info.get("total_count", 0) for info in char_stats.values())

    return {"code": 0, "data": result, "stats": {
        "total_chars": total_chars,
        "total_clusters": total_clusters,
        "total_images": total_images
    }}


# ======================================
# 9.1 获取某标签的推荐聚类（用于伪标签传播）
# ======================================
@app.post("/api/pseudo-labels/clusters")
async def get_pseudo_label_clusters(request: Request):
    data = await request.json()
    char = data.get("char", "")

    if not char:
        raise HTTPException(status_code=400, detail="字符不能为空")

    if char in pseudo_label_cache:
        cache_size = len(pseudo_label_cache[char])
        cluster_ids = list(pseudo_label_cache[char].keys())
        clusters = list(pseudo_label_cache[char].values())
        clusters.sort(key=lambda x: x["avg_similarity"], reverse=True)
        return {"code": 0, "char": char, "clusters": clusters, "cached": True}


    def compute_clusters():
        return compute_pseudo_clusters_sync(char)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, compute_clusters)

    if result["error"]:
        return {"code": 1, "msg": result["error"], "clusters": []}

    clusters = result["clusters"]
    pseudo_label_cache[char] = {item["cluster_id"]: item for item in clusters}
    clusters.sort(key=lambda x: x["avg_similarity"], reverse=True)

    return {"code": 0, "char": char, "clusters": clusters, "cached": False}


def compute_pseudo_clusters_sync(char):
    import time
    start_time = time.time()
    result = {"clusters": [], "error": None, "duration": 0}

    try:
        if not CLUSTERS_JSON.exists():
            result["error"] = "聚类数据不存在"
            return result

        clusters_data = load_json_file(CLUSTERS_JSON)
        clusters = clusters_data.get("clusters", {})

        labeled_info = get_labeled_clusters_info()

        anchor_char_ids = []
        for cid, info in labeled_info.items():
            for idx, label_info in info["char_labels"].items():
                if label_info.get("char") == char:
                    cluster_chars = clusters.get(str(cid), [])
                    if int(idx) < len(cluster_chars):
                        anchor_char_ids.append({
                            "char_id": cluster_chars[int(idx)]["char_id"],
                            "cluster_id": cid
                        })
                    if len(anchor_char_ids) >= 30:
                        break
            if len(anchor_char_ids) >= 30:
                break

        if not anchor_char_ids:
            return result

        anchor_features = {}
        for item in anchor_char_ids:
            img_path = str(DATASET_DIR / "pdf_chars" / f'{item["char_id"]}.png')
            feat = extract_hog_features(img_path)
            if feat is not None:
                anchor_features[item["char_id"]] = feat

        if not anchor_features:
            result["error"] = "无法提取锚点特征"
            return result

        labels_data = load_json_file(LABELS_JSON) or {}
        cluster_recommendations = {}

        for cluster_id, cluster_chars in clusters.items():
            if labels_data.get(cluster_id, {}).get("status") == "labeled":
                continue

            unlabeled_indices = []
            for idx, char_info in enumerate(cluster_chars):
                char_key = str(idx)
                if not labels_data.get(cluster_id, {}).get("char_labels", {}).get(char_key, {}).get("char"):
                    unlabeled_indices.append((idx, char_info))

            if not unlabeled_indices:
                continue

            matched_count = 0
            total_similarity = 0
            sample_indices = []

            for idx, char_info in unlabeled_indices[:20]:
                char_id = char_info.get("char_id", "")
                img_path = str(DATASET_DIR / "pdf_chars" / f'{char_id}.png')
                target_feat = extract_hog_features(img_path)
                if target_feat is None:
                    continue

                best_sim = 0
                for anchor_feat in anchor_features.values():
                    sim = float(cosine_similarity([target_feat], [anchor_feat])[0][0])
                    if sim > best_sim:
                        best_sim = sim

                if best_sim >= 0.75:
                    matched_count += 1
                    total_similarity += best_sim
                    sample_indices.append(idx)

            if matched_count >= 3:
                avg_sim = total_similarity / matched_count
                cluster_recommendations[cluster_id] = {
                    "cluster_id": cluster_id,
                    "matched_count": matched_count,
                    "avg_similarity": round(avg_sim, 4),
                    "total_count": len(cluster_chars),
                    "sample_indices": sample_indices[:5]
                }

        result["clusters"] = list(cluster_recommendations.values())
        result["duration"] = round(time.time() - start_time, 2)
        return result

    except Exception as e:
        result["error"] = str(e)
        result["duration"] = round(time.time() - start_time, 2)
        return result


@app.delete("/api/pseudo-labels/cache/{char}")
def clear_pseudo_label_cache(char: str):
    if char in pseudo_label_cache:
        del pseudo_label_cache[char]
    return {"code": 0, "msg": f"已清除缓存: {char}"}


@app.delete("/api/pseudo-labels/cache")
def clear_all_pseudo_label_cache():
    count = len(pseudo_label_cache)
    pseudo_label_cache.clear()
    return {"code": 0, "msg": f"已清除全部缓存，共 {count} 条"}


# ======================================
# 9.2 获取某聚类中某标签的推荐图片
# ======================================
@app.post("/api/pseudo-labels/clusters/images")
async def get_pseudo_label_cluster_images(request: Request):
    data = await request.json()
    char = data.get("char", "")
    cluster_id = data.get("cluster_id", 0)
    
    if not char:
        raise HTTPException(status_code=400, detail="字符不能为空")
    
    if not CLUSTERS_JSON.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")

    clusters_data = load_json_file(CLUSTERS_JSON)
    clusters = clusters_data.get("clusters", {})

    cluster_key = str(cluster_id)
    if cluster_key not in clusters:
        raise HTTPException(status_code=404, detail="聚类不存在")

    cluster_chars = clusters[cluster_key]
    labeled_info = get_labeled_clusters_info()

    anchor_char_ids = []
    for cid, info in labeled_info.items():
        for idx, label_info in info["char_labels"].items():
            if label_info.get("char") == char:
                cluster_chars_list = clusters.get(str(cid), [])
                if int(idx) < len(cluster_chars_list):
                    anchor_char_ids.append(cluster_chars_list[int(idx)]["char_id"])
                if len(anchor_char_ids) >= 20:
                    break
        if len(anchor_char_ids) >= 20:
            break

    anchor_features = {}
    for char_id in anchor_char_ids:
        img_path = str(DATASET_DIR / "pdf_chars" / f'{char_id}.png')
        feat = extract_hog_features(img_path)
        if feat is not None:
            anchor_features[char_id] = feat

    if not anchor_features:
        return {"code": 0, "msg": "无可用锚点", "images": []}

    labels_data = load_json_file(LABELS_JSON) or {}
    cluster_labels = labels_data.get(cluster_key, {})

    results = []
    for idx, char_info in enumerate(cluster_chars):
        char_key = str(idx)
        if cluster_labels.get("char_labels", {}).get(char_key, {}).get("char"):
            continue

        char_id = char_info.get("char_id", "")
        img_path = str(DATASET_DIR / "pdf_chars" / f'{char_id}.png')
        target_feat = extract_hog_features(img_path)
        if target_feat is None:
            continue

        best_sim = 0
        for anchor_feat in anchor_features.values():
            sim = float(cosine_similarity([target_feat], [anchor_feat])[0][0])
            if sim > best_sim:
                best_sim = sim

        if best_sim >= 0.75:
            results.append({
                "index": idx,
                "char_id": char_id,
                "similarity": round(best_sim, 4)
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = results[:50]

    return {"code": 0, "char": char, "cluster_id": cluster_id, "images": results}


# ======================================
# 10. 获取汉字图片列表
# ======================================
@app.get("/api/char-images/search")
async def get_char_images_list(char: str):
    try:
        # 从labels.json中筛选该汉字的图片
        labeled_images = []
        label_file = DATASET_DIR / "clusters" / "labeling" / "labels.json"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
            
            # 从hog_clusters.json中查找图片
            clusters_file = DATASET_DIR / "clusters" / "hog_clusters.json"
            if clusters_file.exists():
                with open(clusters_file, "r", encoding="utf-8") as f:
                    clusters_data = json.load(f)
            
            # 遍历所有已标注的聚类
            for cid, info in labels.items():
                if info.get("status") == "labeled" and info.get("char_labels"):
                    # 检查每个图片的标注是否为目标汉字
                    for char_idx, label_info in info["char_labels"].items():
                        if label_info.get("char") == char:
                            # 获取该图片的信息
                            cluster_items = clusters_data.get("clusters", {}).get(str(cid), [])
                            idx = int(char_idx)
                            if idx < len(cluster_items):
                                char_id = cluster_items[idx].get("char_id")
                                if char_id:
                                    filename = f"{char_id}.png"
                                    labeled_images.append({
                                        "filename": filename,
                                        "cluster_id": int(cid),
                                        "char_index": idx
                                    })
        
        return {"code": 0, "images": labeled_images}
    except Exception as e:
        return {"code": -1, "msg": str(e)}

# ======================================
# 11. 获取聚类图片文件
# ======================================
@app.get("/api/char-images/{image_name:path}")
async def get_char_image(image_name: str):
    char_dir = DATASET_DIR / "pdf_chars"
    
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

# ======================================
# 10. 获取行图片文件
# ======================================
@app.get("/api/line-images/{line_path:path}")
async def get_line_image(line_path: str):
    line_dir = DATASET_DIR / "pdf_lines"
    line_file = line_dir / f"{line_path}.png"

    if not line_file.exists():
        raise HTTPException(status_code=404, detail="行图片不存在")

    with open(line_file, "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Content-Type"] = "image/png"
    return response

# ======================================
# 12. 获取行标注状态
# ======================================
@app.get("/api/line-status")
async def get_line_status():
    try:
        # 读取聚类数据
        clusters_file = DATASET_DIR / "clusters" / "hog_clusters.json"
        labels_file = DATASET_DIR / "clusters" / "labeling" / "labels.json"
        
        if not clusters_file.exists() or not labels_file.exists():
            return {"code": -1, "msg": "文件不存在"}
        
        with open(clusters_file, "r", encoding="utf-8") as f:
            clusters_data = json.load(f)
        
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        # 构建图片到聚类和标注的映射
        image_cluster_map = {}
        image_label_map = {}
        
        for cid, cluster_items in clusters_data.get("clusters", {}).items():
            for idx, item in enumerate(cluster_items):
                char_id = item.get("char_id")
                if char_id:
                    image_cluster_map[char_id] = {"cluster_id": int(cid), "char_index": idx}
                    
                    # 检查标注
                    if cid in labels:
                        label_info = labels[cid]
                        if label_info.get("status") == "labeled" and label_info.get("char_labels"):
                            if str(idx) in label_info["char_labels"]:
                                image_label_map[char_id] = label_info["char_labels"][str(idx)].get("char")
        
        # 按行分组
        line_data = {}

        for cid, cluster_items in clusters_data.get("clusters", {}).items():
            for item in cluster_items:
                lineage = item.get("lineage", {})
                line_name = lineage.get("line_name")
                char_id = item.get("char_id")
                char_index = lineage.get("char_idx", 0)

                if line_name and char_id:
                    if line_name not in line_data:
                        line_data[line_name] = []

                    labeled = char_id in image_label_map
                    line_data[line_name].append({
                        "char_id": char_id,
                        "cluster_id": int(cid),
                        "char": image_label_map.get(char_id),
                        "labeled": labeled,
                        "char_index": char_index
                    })
        
        # 转换为返回格式
        result = []
        for line_name, chars in line_data.items():
            # 按字符位置排序
            chars.sort(key=lambda x: x.get("char_index", 0))

            total = len(chars)
            labeled = sum(1 for c in chars if c["labeled"])
            ratio = labeled / total if total > 0 else 0

            status = "full" if ratio == 1 else "partial" if ratio > 0 else "unlabeled"

            result.append({
                "line_name": line_name,
                "total": total,
                "labeled": labeled,
                "ratio": ratio,
                "status": status,
                "chars": chars
            })
        
        # 按行号排序
        result.sort(key=lambda x: (
            int(x["line_name"].split("_")[1].replace("page", "")),
            int(x["line_name"].split("_")[3])
        ))
        
        return {"code": 0, "data": result}
    
    except Exception as e:
        print(f"获取行状态失败: {e}")
        return {"code": -1, "msg": str(e)}