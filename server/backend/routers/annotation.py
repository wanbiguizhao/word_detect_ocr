"""标注管理路由"""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Optional
from collections import defaultdict

from config import (
    DATASET_DIR,
    ANNOTATIONS_DIR,
    CLUSTERS_JSON,
    LABELS_JSON,
    executor
)

from utils import load_json_file

router = APIRouter()

@router.get("/api/annotation/priority-queue")
def get_priority_queue(limit: int = 50):
    """获取优先级队列（待标注图片）"""
    result = []
    
    # 获取已标注的图片ID
    annotated_ids = set()
    if ANNOTATIONS_DIR.exists():
        for annot_file in ANNOTATIONS_DIR.glob("*.json"):
            annotated_ids.add(annot_file.stem)
    
    # 从融合结果中获取图片列表
    fusion_files = sorted(DATASET_DIR.glob("fusion*.json"))
    if fusion_files:
        latest_fusion = fusion_files[-1]
        try:
            with open(latest_fusion, "r", encoding="utf-8") as f:
                fusion_data = json.load(f)
            
            for item in fusion_data.get("images", []):
                image_id = item.get("image_id")
                if image_id and image_id not in annotated_ids:
                    result.append({
                        "image_id": image_id,
                        "priority": item.get("priority", 0),
                        "conflict": item.get("conflict", False),
                        "rules": item.get("rules", []),
                        "models": item.get("models", [])
                    })
            
            # 按优先级排序
            result.sort(key=lambda x: -x["priority"])
            
        except Exception as e:
            print(f"加载融合文件失败: {e}")
    
    return {"code": 0, "data": result[:limit]}

@router.get("/api/annotation/next")
def get_next_annotation():
    """获取下一张待标注图片"""
    queue = get_priority_queue(limit=100).get("data", [])
    
    if not queue:
        return {"code": 1, "msg": "暂无待标注图片"}
    
    # 优先选择有冲突的图片
    conflict_items = [item for item in queue if item.get("conflict")]
    if conflict_items:
        return {"code": 0, "data": conflict_items[0]}
    
    # 否则选择优先级最高的
    return {"code": 0, "data": queue[0]}

@router.get("/api/annotation/stats")
def get_annotation_stats():
    """获取标注统计信息"""
    total_count = 0
    annotated_count = 0
    
    # 统计总图片数
    fusion_files = sorted(DATASET_DIR.glob("fusion*.json"))
    if fusion_files:
        latest_fusion = fusion_files[-1]
        try:
            with open(latest_fusion, "r", encoding="utf-8") as f:
                fusion_data = json.load(f)
            total_count = len(fusion_data.get("images", []))
        except Exception as e:
            print(f"加载融合文件失败: {e}")
    
    # 统计已标注数
    if ANNOTATIONS_DIR.exists():
        annotated_count = len(list(ANNOTATIONS_DIR.glob("*.json")))
    
    return {
        "code": 0,
        "data": {
            "total": total_count,
            "annotated": annotated_count,
            "unannotated": total_count - annotated_count,
            "progress": round((annotated_count / total_count) * 100) if total_count > 0 else 0
        }
    }

@router.get("/api/annotation/cluster-priority-queue")
def get_cluster_priority_queue(limit: int = 50):
    """获取聚类优先级队列"""
    result = []
    
    # 加载聚类数据
    clusters_data = load_json_file(CLUSTERS_JSON)
    if not clusters_data:
        return {"code": 0, "data": result}
    
    # 加载标签数据
    labels_data = load_json_file(LABELS_JSON)
    labeled_clusters = set()
    if labels_data:
        labeled_clusters = set(labels_data.keys())
    
    for cluster_id, cluster_info in clusters_data.items():
        if cluster_id in labeled_clusters:
            continue
        
        result.append({
            "cluster_id": cluster_id,
            "size": len(cluster_info.get("images", [])),
            "center_distance": cluster_info.get("center_distance", 0),
            "diversity": cluster_info.get("diversity", 0),
            "priority": cluster_info.get("priority", 0)
        })
    
    # 按优先级排序
    result.sort(key=lambda x: -x["priority"])
    
    return {"code": 0, "data": result[:limit]}

@router.get("/api/annotation/cluster-next")
def get_next_cluster():
    """获取下一个待标注聚类"""
    queue = get_cluster_priority_queue(limit=100).get("data", [])
    
    if not queue:
        return {"code": 1, "msg": "暂无待标注聚类"}
    
    return {"code": 0, "data": queue[0]}

@router.get("/api/annotation/cluster-stats")
def get_cluster_stats():
    """获取聚类标注统计信息"""
    clusters_data = load_json_file(CLUSTERS_JSON)
    labels_data = load_json_file(LABELS_JSON)
    
    total_clusters = len(clusters_data) if clusters_data else 0
    labeled_clusters = len(labels_data) if labels_data else 0
    
    return {
        "code": 0,
        "data": {
            "total": total_clusters,
            "labeled": labeled_clusters,
            "unlabeled": total_clusters - labeled_clusters,
            "progress": round((labeled_clusters / total_clusters) * 100) if total_clusters > 0 else 0
        }
    }
