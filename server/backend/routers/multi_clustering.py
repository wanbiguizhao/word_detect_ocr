from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from services.multi_clustering_manager import MultiClusteringManager
from services.char_pool_manager import CharPoolManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mc", tags=["多轮聚类"])

# 模型定义
class LabelItem(BaseModel):
    charIndex: int
    char: str

class BatchLabelSave(BaseModel):
    labels: List[LabelItem]

class BatchSkipRequest(BaseModel):
    char_ids: List[str]

class NewRoundRequest(BaseModel):
    data_source: Optional[str] = "unlabeled"
    method: Optional[str] = "hdbscan"
    n_clusters: Optional[int] = None
    description: Optional[str] = ""
    min_cluster_size: Optional[int] = 5
    min_samples: Optional[int] = 2
    max_cluster_size: Optional[int] = 100
    confidence_threshold: Optional[float] = 0.7


@router.get("/rounds")
def get_rounds():
    """获取所有轮次"""
    try:
        manager = MultiClusteringManager()
        history = manager.get_round_history()
        return {"code": 0, "msg": "success", "data": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds")
def start_new_round(request: NewRoundRequest):
    """启动新一轮聚类"""
    try:
        print(f"[DEBUG] 接收到的聚类参数:")
        print(f"  data_source: {request.data_source}")
        print(f"  method: {request.method}")
        print(f"  n_clusters: {request.n_clusters}")
        print(f"  description: {request.description}")
        print(f"  min_cluster_size: {request.min_cluster_size}")
        print(f"  min_samples: {request.min_samples}")
        print(f"  max_cluster_size: {request.max_cluster_size}")
        print(f"  confidence_threshold: {request.confidence_threshold}")

        manager = MultiClusteringManager()
        round_num = manager.start_new_round(
            data_source=request.data_source,
            n_clusters=request.n_clusters,
            description=request.description,
            method=request.method,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            max_cluster_size=request.max_cluster_size,
            confidence_threshold=request.confidence_threshold
        )
        return {
            "code": 0,
            "msg": "success",
            "round": round_num,
            "message": f"第{round_num}轮聚类已启动（方法: {request.method}）"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_num}")
def get_round_info(round_num: int):
    """获取指定轮次信息"""
    try:
        manager = MultiClusteringManager()
        clusters = manager.get_round_clusters(round_num)
        progress = manager.get_round_progress(round_num)

        if not clusters:
            raise HTTPException(status_code=404, detail=f"轮次 {round_num} 不存在")

        return {
            "code": 0,
            "msg": "success",
            "round": round_num,
            "clusters": clusters.get("clusters", {}),
            "total_clusters": len(clusters.get("clusters", {})),
            "total_chars": clusters.get("total_chars", 0),
            "progress": progress
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_num}/clusters")
def get_round_clusters(round_num: int):
    """获取指定轮次的聚类列表"""
    try:
        manager = MultiClusteringManager()
        clusters = manager.get_round_clusters(round_num)

        if not clusters:
            raise HTTPException(status_code=404, detail=f"轮次 {round_num} 不存在")

        cluster_list = []
        for cluster_id, chars in clusters.get("clusters", {}).items():
            labels = manager.get_round_labels(round_num)
            cluster_labels = labels.get("labels", {}).get(cluster_id, {}) if labels else {}

            # 统计已标注字符数量和汉字分布
            char_labels = cluster_labels.get("char_labels", {})
            labeled_count = sum(1 for v in char_labels.values() if v.get("char"))

            # 统计每个汉字的数量
            char_counts = {}
            for v in char_labels.values():
                char = v.get("char")
                if char:
                    char_counts[char] = char_counts.get(char, 0) + 1

            cluster_list.append({
                "cluster_id": cluster_id,
                "char_count": len(chars),
                "labeled_count": labeled_count,
                "status": cluster_labels.get("status", "unlabeled"),
                "char": cluster_labels.get("char"),
                "char_counts": char_counts,
                "confidence": cluster_labels.get("confidence")
            })

        return {
            "code": 0,
            "msg": "success",
            "round": round_num,
            "clusters": cluster_list,
            "total": len(cluster_list)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_num}/clusters/{cluster_id}")
def get_cluster_detail(round_num: int, cluster_id: str):
    """获取聚类详情"""
    try:
        manager = MultiClusteringManager()
        clusters = manager.get_round_clusters(round_num)
        labels = manager.get_round_labels(round_num)

        if not clusters:
            raise HTTPException(status_code=404, detail=f"轮次 {round_num} 不存在")

        cluster_chars = clusters.get("clusters", {}).get(cluster_id)
        if not cluster_chars:
            raise HTTPException(status_code=404, detail=f"聚类 {cluster_id} 不存在")

        cluster_labels = labels.get("labels", {}).get(cluster_id, {}) if labels else {}
        logger.info(f"[mc_detail] round={round_num}, cluster={cluster_id}, cluster_labels={cluster_labels}")

        all_chars = manager.char_pool.load_all_chars()

        char_details = []
        for idx, char_info in enumerate(cluster_chars):
            char_key = str(idx)
            char_label = cluster_labels.get("char_labels", {}).get(char_key, {})
            char_id = char_info.get("char_id", "")
            char_status = all_chars.get(char_id, {}).get("status", "unlabeled")

            prelabel = manager.data_store.get_prelabel_by_char_id(char_id)
            predicted_char = None
            confidence = None
            confidence_level = None
            if prelabel:
                predicted_char = prelabel.get("predicted_char")
                confidence = prelabel.get("confidence")
                confidence_level = prelabel.get("confidence_level")

            char_details.append({
                "index": idx,
                "char_id": char_id,
                "line_name": char_info.get("line_name", ""),
                "col_start": char_info.get("col_start", 0),
                "col_end": char_info.get("col_end", 0),
                "label": char_label.get("char"),
                "labeled_at": char_label.get("labeled_at"),
                "char_status": char_status,
                "predicted_char": predicted_char,
                "confidence": confidence,
                "confidence_level": confidence_level
            })

        return {
            "code": 0,
            "msg": "success",
            "round": round_num,
            "cluster_id": cluster_id,
            "status": cluster_labels.get("status", "unlabeled"),
            "char": cluster_labels.get("char"),
            "confidence": cluster_labels.get("confidence"),
            "chars": char_details,
            "total": len(char_details)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/clusters/{cluster_id}/labels")
def save_cluster_label(round_num: int, cluster_id: str, body: LabelItem):
    """保存单个标注"""
    try:
        print(f"[API] save_cluster_label: round={round_num}, cluster={cluster_id}, charIndex={body.charIndex}, char='{body.char}'")
        manager = MultiClusteringManager()
        success = manager.save_label(round_num, cluster_id, body.charIndex, body.char)

        if success:
            print(f"[API] save_cluster_label 成功")
            return {"code": 0, "msg": "保存成功"}
        else:
            print(f"[API] save_cluster_label 返回False")
            raise HTTPException(status_code=500, detail="保存失败")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] save_cluster_label 异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/clusters/{cluster_id}/labels/batch")
def batch_save_cluster_labels(round_num: int, cluster_id: str, body: BatchLabelSave):
    """批量保存标注"""
    try:
        manager = MultiClusteringManager()
        saved_count = manager.save_batch_labels(round_num, cluster_id, body.labels)

        return {
            "code": 0,
            "msg": f"批量保存成功，共 {saved_count} 条",
            "saved_count": saved_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/clusters/{cluster_id}/skip")
def skip_cluster(round_num: int, cluster_id: str):
    """跳过聚类"""
    try:
        manager = MultiClusteringManager()
        success = manager.skip_cluster(round_num, cluster_id)

        if success:
            return {"code": 0, "msg": "已跳过"}
        else:
            raise HTTPException(status_code=500, detail="操作失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/chars/{char_id}/skip")
def skip_char(round_num: int, char_id: str):
    """跳过单个字符"""
    try:
        manager = MultiClusteringManager()
        success = manager.skip_char(char_id, round_num)

        if success:
            return {"code": 0, "msg": "已跳过"}
        else:
            raise HTTPException(status_code=500, detail="操作失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/chars/batch-skip")
def batch_skip_chars(round_num: int, request: BatchSkipRequest):
    """批量跳过字符"""
    try:
        manager = MultiClusteringManager()
        result = manager.batch_skip_chars(request.char_ids, round_num)
        return {"code": 0, "msg": f"已跳过 {result['skipped']} 个字符", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/chars/{char_id}/unskip")
def unskip_char(round_num: int, char_id: str):
    """撤回跳过单个字符"""
    try:
        manager = MultiClusteringManager()
        success = manager.unskip_char(char_id, round_num)
        if success:
            return {"code": 0, "msg": "已撤回跳过"}
        else:
            raise HTTPException(status_code=500, detail="操作失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_num}/chars/batch-unskip")
def batch_unskip_chars(round_num: int, request: BatchSkipRequest):
    """批量撤回跳过字符"""
    try:
        manager = MultiClusteringManager()
        result = manager.batch_unskip_chars(request.char_ids, round_num)
        return {"code": 0, "msg": f"已撤回跳过 {result['unskipped']} 个字符", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/char-pool")
def get_char_pool_stats():
    """获取字符池统计"""
    try:
        char_pool = CharPoolManager()
        stats = char_pool.get_stats()

        return {
            "code": 0,
            "msg": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/char-pool/init")
def init_char_pool():
    """初始化字符池"""
    try:
        char_pool = CharPoolManager()
        count = char_pool.init_from_lineage()

        return {
            "code": 0,
            "msg": f"字符池初始化成功",
            "total_chars": count
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/char-pool/unlabeled")
def get_unlabeled_chars(page: int = Query(1, ge=1), page_size: int = Query(100, ge=1)):
    """获取未标注字符列表（分页）"""
    try:
        char_pool = CharPoolManager()
        unlabeled_ids = char_pool.get_unlabeled_char_ids()

        total = len(unlabeled_ids)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = unlabeled_ids[start:end]

        return {
            "code": 0,
            "msg": "success",
            "data": paginated,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_num}/progress")
def get_round_progress(round_num: int):
    """获取轮次进度"""
    try:
        manager = MultiClusteringManager()
        progress = manager.get_round_progress(round_num)

        if not progress:
            raise HTTPException(status_code=404, detail=f"轮次 {round_num} 不存在")

        return {
            "code": 0,
            "msg": "success",
            "data": progress
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unified-labels/count")
def get_unified_labels_count():
    """获取统一标注数量"""
    try:
        manager = MultiClusteringManager()
        unified_data = manager._load_json(manager.unified_labels_path)
        count = len(unified_data.get("annotations", [])) if unified_data else 0

        return {
            "code": 0,
            "msg": "success",
            "count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))