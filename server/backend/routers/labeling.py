from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import json
from pathlib import Path

router = APIRouter()

from config import config, PROJECT_ROOT
from labeling_task_manager import LabelingTaskManager
from pseudo_label_generator import PseudoLabelGenerator
from simple_char_clustering import SimpleCharClustering
from datastore.stats_manager import StatsManager

task_manager = LabelingTaskManager(config._config)
prelabel_generator = PseudoLabelGenerator(config._config)
simple_clusterer = SimpleCharClustering(config._config)
stats_manager = StatsManager(config.get("dataset.current", "pdf5823"))

class CharStats(BaseModel):
    char: str
    total: int
    labeled: int
    prelabeled: int

class DatasetStats(BaseModel):
    dataset: str
    total_images: int
    labeled_count: int
    prelabeled_count: int
    unlabeled_count: int
    char_stats: List[CharStats]

class PrelabelConfirmRequest(BaseModel):
    char: str
    image_paths: List[str]

class PrelabelModifyRequest(BaseModel):
    image_path: str
    new_char: str

class SimpleConfirmRequest(BaseModel):
    char_id: str
    char: str

class SimpleModifyRequest(BaseModel):
    char_id: str
    char: str

class BatchConfirmRequest(BaseModel):
    items: List[SimpleConfirmRequest]

@router.get("/api/labeling/stats", response_model=DatasetStats)
def get_labeling_stats():
    stats = stats_manager.get_stats()
    
    char_stats = []
    for char, info in stats.get("char_stats", {}).items():
        char_stats.append({
            "char": char,
            "total": info.get("total", 0),
            "labeled": info.get("confirmed", 0),
            "prelabeled": info.get("pending", 0)
        })
    
    char_stats.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "dataset": stats.get("dataset", config.get("dataset.current", "pdf5823")),
        "total_images": stats.get("total_images", 0),
        "labeled_count": stats.get("labeled_count", 0),
        "prelabeled_count": stats.get("total_images", 0),
        "unlabeled_count": stats.get("unlabeled_count", 0),
        "char_stats": char_stats
    }

@router.get("/api/labeling/char-list")
def get_char_list(page: int = 1, page_size: int = 20, search: Optional[str] = None, dataset: str = "pdf5823", sort_by: Optional[str] = None, sort_order: str = "desc"):
    # 使用数据抽象层
    global stats_manager
    stats_manager = StatsManager(dataset)
    return stats_manager.get_char_list(page, page_size, search, sort_by, sort_order)

@router.get("/api/labeling/prelabels/{char}")
def get_char_prelabels(char: str, page: int = 1, page_size: int = 20, confidence_min: Optional[float] = None, dataset: str = "pdf5823"):
    prelabels_path = PROJECT_ROOT / "bussiness" / "datahome" / dataset / "pre_labels.json"
    lineage_path = PROJECT_ROOT / "bussiness" / "datahome" / dataset / "lineage.json"

    if not prelabels_path.exists():
        prelabels_path = PROJECT_ROOT / "bussiness" / "pre_labels.json"
        lineage_path = PROJECT_ROOT / "bussiness" / "datahome" / dataset / "lineage.json"

    if not prelabels_path.exists():
        return {"code": 0, "data": [], "total": 0}

    with open(prelabels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lineage_data = {}
    if lineage_path.exists():
        with open(lineage_path, 'r', encoding='utf-8') as f:
            lineage_data = json.load(f)

    prelabels = [p for p in data.get("prelabels", []) if p.get("predicted_char") == char]

    if confidence_min is not None:
        prelabels = [p for p in prelabels if p.get("confidence", 0) >= confidence_min]

    chars_info = lineage_data.get("chars", {})
    for prelabel in prelabels:
        if "lineage" not in prelabel or prelabel["lineage"].get("col_start") is None:
            char_id = prelabel.get("char_id", "")
            if char_id in chars_info:
                char_info = chars_info[char_id]
                col_start = char_info.get("col_start")
                col_end = char_info.get("col_end")
                line_name = char_info.get("line_name", "")
                page_num = char_info.get("page_num", 0)
                line_num = char_info.get("line_idx", 0)

                prelabel["lineage"] = {
                    "line_name": line_name,
                    "char_idx": char_info.get("char_idx", 0),
                    "page": page_num,
                    "line": line_num,
                    "col_start": col_start,
                    "col_end": col_end,
                    "width": char_info.get("width")
                }

    total = len(prelabels)
    confirmed = len([p for p in prelabels if p.get("status") == "confirmed"])
    pending = len([p for p in prelabels if p.get("status") == "pending"])

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "code": 0,
        "char": char,
        "total": total,
        "confirmed": confirmed,
        "pending": pending,
        "data": prelabels[start:end]
    }

@router.post("/api/labeling/confirm")
def simple_confirm(request: SimpleConfirmRequest):
    global stats_manager
    dataset = config.get("dataset.current", "pdf5823")
    stats_manager = StatsManager(dataset)
    stats_manager.confirm_annotation(request.char_id, request.char)

    return {
        "code": 0,
        "msg": f"成功确认图片 {request.char_id} 为 {request.char}"
    }

@router.post("/api/labeling/modify")
def simple_modify(request: SimpleModifyRequest):
    global stats_manager
    dataset = config.get("dataset.current", "pdf5823")
    stats_manager = StatsManager(dataset)
    stats_manager.confirm_annotation(request.char_id, request.char)

    return {
        "code": 0,
        "msg": f"已将图片 {request.char_id} 修改为: {request.char}"
    }

@router.post("/api/labeling/confirm/batch")
def batch_confirm(request: BatchConfirmRequest):
    global stats_manager
    dataset = config.get("dataset.current", "pdf5823")
    stats_manager = StatsManager(dataset)
    results = stats_manager.batch_confirm_annotations(request.items)

    return {
        "code": 0,
        "msg": f"成功确认 {results['success_count']}/{results['total_count']} 个标注",
        "success_count": results['success_count'],
        "total_count": results['total_count']
    }

@router.post("/api/labeling/prelabels/confirm")
def confirm_prelabels(request: PrelabelConfirmRequest):
    updates = []
    for image_path in request.image_paths:
        updates.append({
            "image_path": image_path,
            "status": "labeled",
            "char": request.char
        })
    
    task_manager.batch_update_labels(updates)
    
    return {
        "code": 0,
        "msg": f"成功确认 {len(request.image_paths)} 张图片",
        "count": len(request.image_paths)
    }

@router.post("/api/labeling/prelabels/modify")
def modify_prelabel(request: PrelabelModifyRequest):
    task_manager.update_label_status(
        request.image_path,
        "labeled",
        request.new_char
    )
    
    return {
        "code": 0,
        "msg": f"已将图片标注为: {request.new_char}"
    }

@router.post("/api/labeling/prelabels/skip")
def skip_prelabels(image_paths: List[str]):
    updates = []
    for image_path in image_paths:
        updates.append({
            "image_path": image_path,
            "status": "unlabeled"
        })
    
    task_manager.batch_update_labels(updates)
    
    return {
        "code": 0,
        "msg": f"已跳过 {len(image_paths)} 张图片"
    }

@router.get("/api/labeling/clusters")
def get_clusters(method: str = "simple_char"):
    cluster_path = PROJECT_ROOT / "bussiness" / "cluster_results" / method / f"{config.get('dataset.current')}_clusters.json"
    
    if not cluster_path.exists():
        return {"code": 0, "data": [], "method": method}
    
    with open(cluster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        "code": 0,
        "method": method,
        "data": data.get("clusters", [])
    }

@router.get("/api/labeling/clusters/{cluster_id}")
def get_cluster_detail(cluster_id: str, method: str = "simple_char"):
    cluster_path = PROJECT_ROOT / "bussiness" / "cluster_results" / method / f"{config.get('dataset.current')}_clusters.json"
    
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")
    
    with open(cluster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data.get("clusters", [])
    cluster = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    
    if not cluster:
        raise HTTPException(status_code=404, detail="聚类不存在")
    
    return {
        "code": 0,
        "data": cluster
    }

@router.post("/api/labeling/clusters/{cluster_id}/label")
def label_cluster(cluster_id: str, char: str, image_indices: Optional[List[int]] = None, method: str = "simple_char"):
    cluster_path = PROJECT_ROOT / "bussiness" / "cluster_results" / method / f"{config.get('dataset.current')}_clusters.json"
    
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="聚类数据不存在")
    
    with open(cluster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data.get("clusters", [])
    cluster = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    
    if not cluster:
        raise HTTPException(status_code=404, detail="聚类不存在")
    
    images = cluster.get("images", [])
    if image_indices is None:
        target_images = images
    else:
        target_images = [images[i] for i in image_indices if i < len(images)]
    
    updates = []
    for img in target_images:
        updates.append({
            "image_path": img.get("image_path"),
            "status": "labeled",
            "char": char
        })
    
    task_manager.batch_update_labels(updates)
    
    for img in target_images:
        img["labeled"] = True
        img["char"] = char
    
    with open(cluster_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {
        "code": 0,
        "msg": f"成功标注 {len(target_images)} 张图片为: {char}"
    }

@router.post("/api/labeling/clusters/run")
def run_clustering(min_samples: int = 3):
    """执行简单字符聚类"""
    try:
        result = simple_clusterer.run_clustering(min_samples=min_samples)
        if result:
            return {
                "code": 0,
                "msg": "聚类完成",
                "cluster_count": len(result.get("clusters", []))
            }
        else:
            return {"code": -1, "msg": "没有未标注图片"}
    except Exception as e:
        return {"code": -1, "msg": f"聚类失败: {str(e)}"}