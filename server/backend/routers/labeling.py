from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import json
import logging
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

from config import config, PROJECT_ROOT
from labeling_task_manager import LabelingTaskManager
from simple_char_clustering import SimpleCharClustering
from datastore.stats_manager import StatsManager
from datastore.data_store import DataStore

task_manager = LabelingTaskManager(config._config)
simple_clusterer = SimpleCharClustering(config._config)


class CharStats(BaseModel):
    char: str
    total: int
    labeled: int
    pending: int
    skipped: int = 0
    unlabeled: int = 0


class DatasetStats(BaseModel):
    dataset: str
    total_images: int
    labeled_count: int
    pending_count: int
    skipped_count: int = 0
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


class SimpleRevokeRequest(BaseModel):
    char_id: str


class BatchRevokeRequest(BaseModel):
    char_ids: List[str]


class SimpleSkipRequest(BaseModel):
    char_id: str


class BatchSkipRequest(BaseModel):
    char_ids: List[str]


class SimpleUnskipRequest(BaseModel):
    char_id: str


class BatchUnskipRequest(BaseModel):
    char_ids: List[str]


class ModifyAnnotationRequest(BaseModel):
    char_id: str
    old_char: Optional[str]
    new_char: str


class BatchModifyRequest(BaseModel):
    char_ids: List[str]
    new_char: str


@router.get("/api/labeling/stats", response_model=DatasetStats)
def get_labeling_stats(refresh: bool = False):
    dataset = config.get("dataset.current", "pdf5826")
    stats_manager = StatsManager(dataset)
    
    if refresh:
        stats_manager.refresh_cache()
    
    stats = stats_manager.get_stats()
    
    char_stats = []
    char_stats_dict = stats.get("char_stats", {})
    
    if isinstance(char_stats_dict, dict):
        for char, info in char_stats_dict.items():
            char_stats.append({
                "char": char,
                "total": info.get("total", 0),
                "labeled": info.get("labeled", 0),
                "pending": info.get("pending", 0),
                "skipped": info.get("skipped", 0),
                "unlabeled": info.get("unlabeled", 0)
            })
    elif isinstance(char_stats_dict, list):
        char_stats = char_stats_dict
    
    char_stats.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "dataset": stats.get("dataset", dataset),
        "total_images": stats.get("total_images", 0),
        "labeled_count": stats.get("labeled_count", 0),
        "pending_count": stats.get("pending_count", 0),
        "skipped_count": stats.get("skipped_count", 0),
        "unlabeled_count": stats.get("unlabeled_count", 0),
        "char_stats": char_stats
    }


@router.get("/api/labeling/char-list")
def get_char_list(page: int = 1, page_size: int = 20, search: Optional[str] = None, 
                  dataset: Optional[str] = None, sort_by: Optional[str] = None, sort_order: str = "desc"):
    if dataset is None:
        dataset = config.get("dataset.current", "pdf5826")
    stats_manager = StatsManager(dataset)
    return stats_manager.get_char_list(page, page_size, search, sort_by, sort_order)


@router.get("/api/labeling/prelabels/{char}")
def get_char_prelabels(char: str, page: int = 1, page_size: int = 20, 
                       confidence_min: Optional[float] = None, dataset: Optional[str] = None):
    """使用新的 DataStore 获取预标注数据"""
    if dataset is None:
        dataset = config.get("dataset.current", "pdf5826")
    
    store = DataStore(dataset)
    prelabels = store.get_prelabels_by_char(char)
    
    if confidence_min is not None:
        prelabels = [p for p in prelabels if p.get("confidence", 0) >= confidence_min]
    
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
    dataset = config.get("dataset.current", "pdf5826")
    stats_manager = StatsManager(dataset)
    stats_manager.confirm_annotation(request.char_id, request.char)

    return {
        "code": 0,
        "msg": f"成功确认图片 {request.char_id} 为 {request.char}"
    }





@router.post("/api/labeling/confirm/batch")
def batch_confirm(request: BatchConfirmRequest):
    logger.info(f"[Labeling Router] batch_confirm called")
    
    try:
        logger.debug(f"[Labeling Router] Request items count: {len(request.items)}")
        if request.items:
            logger.debug(f"[Labeling Router] First item: {request.items[0]}")
    except Exception as e:
        logger.warning(f"[Labeling Router] Failed to log request items: {e}")
    
    dataset = config.get("dataset.current", "pdf5826")
    logger.debug(f"[Labeling Router] Current dataset: {dataset}")
    
    stats_manager = StatsManager(dataset)
    logger.debug(f"[Labeling Router] StatsManager initialized with dataset: {stats_manager.dataset_id}")
    
    results = stats_manager.batch_confirm_annotations(request.items)
    logger.info(f"[Labeling Router] Batch confirm completed - success: {results['success_count']}/{results['total_count']}")

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


# ==================== 标注历史记录接口 ====================

@router.get("/api/labeling/history")
def get_label_history(char_id: Optional[str] = None, limit: int = 100, dataset: Optional[str] = None):
    """获取标注历史记录"""
    if dataset is None:
        dataset = config.get("dataset.current", "pdf5826")
    
    stats_manager = StatsManager(dataset)
    history = stats_manager.get_label_history(char_id, limit)
    
    return {
        "code": 0,
        "msg": "success",
        "data": history,
        "total": len(history)
    }


# ==================== 同步日志查询接口 ====================

@router.get("/api/labeling/sync/logs")
def get_sync_logs(limit: int = 100, date: Optional[str] = None):
    """获取同步日志"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    if date:
        logs = store.get_sync_logs_by_date(date)
    else:
        logs = store.get_sync_logs(limit)
    
    return {
        "code": 0,
        "msg": "success",
        "data": logs,
        "total": len(logs)
    }


@router.get("/api/labeling/sync/logs/today")
def get_today_sync_logs():
    """获取今日同步日志"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    logs = store.get_today_sync_logs()
    
    return {
        "code": 0,
        "msg": "success",
        "data": logs,
        "total": len(logs)
    }


@router.get("/api/labeling/sync/stats")
def get_sync_statistics():
    """获取同步统计信息"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    stats = store.get_sync_statistics()
    
    return {
        "code": 0,
        "msg": "success",
        "data": stats
    }


@router.post("/api/labeling/repair-prelabel-status")
def repair_prelabel_status():
    """修复 prelabel_status.json 中缺失的记录"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    result = store.repair_prelabel_status()
    return {
        "code": 0,
        "msg": f"修复了 {result['repaired_count']} 条缺失记录",
        "data": result
    }


# ==================== 修改/撤回/跳过操作接口 ====================

@router.post("/api/labeling/modify")
def modify_annotation(request: ModifyAnnotationRequest):
    """修改预标注（直接修改prelabel_status）"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        # 对于预标注页面的修改，直接修改prelabel_status
        store.modify_prelabel(
            char_id=request.char_id,
            new_char=request.new_char,
            changed_by="user"
        )
        
        logger.info(f"修改预标注: {request.char_id} - {request.old_char} -> {request.new_char}")
        
        return {
            "code": 0,
            "msg": f"成功修改预标注: {request.char_id} - {request.new_char}"
        }
    except Exception as e:
        logger.error(f"修改预标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"修改预标注失败: {str(e)}")


@router.post("/api/labeling/revoke")
def revoke_annotation(request: SimpleRevokeRequest):
    """撤回已确认的标注"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        store.revoke_annotation(
            char_id=request.char_id,
            changed_by="user",
            comment="API撤回确认"
        )
        
        logger.info(f"撤回确认: {request.char_id}")
        
        return {
            "code": 0,
            "msg": f"成功撤回确认: {request.char_id}"
        }
    except Exception as e:
        logger.error(f"撤回确认失败: {e}")
        raise HTTPException(status_code=500, detail=f"撤回确认失败: {str(e)}")


@router.post("/api/labeling/skip")
def skip_prelabel(request: SimpleSkipRequest):
    """跳过单个预标注"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        store.skip_prelabel(
            char_id=request.char_id,
            changed_by="user"
        )
        
        logger.info(f"跳过预标注: {request.char_id}")
        
        return {
            "code": 0,
            "msg": f"成功跳过预标注: {request.char_id}"
        }
    except Exception as e:
        logger.error(f"跳过预标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"跳过预标注失败: {str(e)}")


@router.post("/api/labeling/skip/batch")
def batch_skip_prelabels(request: BatchSkipRequest):
    """批量跳过预标注"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        store.batch_skip_prelabels(
            char_ids=request.char_ids,
            changed_by="user"
        )
        
        logger.info(f"批量跳过预标注: {len(request.char_ids)} 条")
        
        return {
            "code": 0,
            "msg": f"成功批量跳过 {len(request.char_ids)} 条预标注"
        }
    except Exception as e:
        logger.error(f"批量跳过预标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量跳过预标注失败: {str(e)}")


@router.post("/api/labeling/unskip")
def unskip_prelabel(request: SimpleUnskipRequest):
    """取消跳过预标注"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        store.unskip_prelabel(
            char_id=request.char_id,
            changed_by="user"
        )
        
        logger.info(f"取消跳过预标注: {request.char_id}")
        
        return {
            "code": 0,
            "msg": f"成功取消跳过预标注: {request.char_id}"
        }
    except Exception as e:
        logger.error(f"取消跳过预标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消跳过预标注失败: {str(e)}")


@router.post("/api/labeling/unskip/batch")
def batch_unskip_prelabels(request: BatchUnskipRequest):
    """批量取消跳过预标注"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        # 单独处理，因为没有 batch_unskip 方法，所以逐个处理
        for char_id in request.char_ids:
            store.unskip_prelabel(char_id, changed_by="user")
        
        logger.info(f"批量取消跳过预标注: {len(request.char_ids)} 条")
        
        return {
            "code": 0,
            "msg": f"成功批量取消跳过 {len(request.char_ids)} 条预标注"
        }
    except Exception as e:
        logger.error(f"批量取消跳过预标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量取消跳过预标注失败: {str(e)}")


@router.post("/api/labeling/modify/batch")
def batch_modify_annotations(request: BatchModifyRequest):
    """批量修改标注（优化版，减少文件IO次数）"""
    dataset = config.get("dataset.current", "pdf5826")
    store = DataStore(dataset)
    
    try:
        # 构造 {char_id: new_char} 的字典
        char_updates = {char_id: request.new_char for char_id in request.char_ids}
        
        # 使用批量方法，一次性完成
        success_count = store.batch_modify_prelabels(char_updates, changed_by="user")
        
        return {
            "code": 0,
            "msg": f"成功批量修改 {success_count}/{len(request.char_ids)} 个标注",
            "success_count": success_count,
            "total_count": len(request.char_ids)
        }
    except Exception as e:
        logger.error(f"批量修改标注失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量修改标注失败: {str(e)}")