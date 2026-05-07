"""
迁移学习API路由
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

router = APIRouter()


class ShareRequest(BaseModel):
    source_dataset_id: str
    target_dataset_ids: List[str]
    merge_strategy: str = "merge"


@router.get("/api/migration/priority-list")
async def get_priority_list(top_n: int = 50):
    """获取推荐标注的汉字优先级列表"""
    try:
        from config import DATASET_ID
        target_dataset_id = DATASET_ID
        
        data_dir = Path(__file__).parent.parent.parent / "bussiness" / "migration" / "data"
        priority_path = data_dir / "char_priority_list.json"
        
        if priority_path.exists():
            with open(priority_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if data.get("dataset_id") == target_dataset_id:
                priorities = data.get("priorities", [])[:top_n]
                return {"code": 0, "data": priorities}
        
        return {"code": -1, "msg": "优先级列表不存在或需要重新生成"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/api/migration/image-matches/{char}")
async def get_image_matches(char: str, top_n: int = 20):
    """获取指定汉字在目标数据集中的匹配图片"""
    try:
        from config import DATASET_ID
        target_dataset_id = DATASET_ID
        
        # 直接从文件读取，不使用缓存
        data_dir = Path(__file__).parent.parent.parent / "bussiness" / "migration" / "data"
        matches_path = data_dir / "image_matches.json"
        
        if not matches_path.exists():
            return {"code": -1, "msg": "匹配结果文件不存在"}
        
        with open(matches_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        matches = data.get("matches", {})
        if char in matches:
            result = matches[char][:top_n]
            print(f"DEBUG: 返回 {len(result)} 个匹配结果")
            if result:
                print(f"DEBUG: 第一个匹配相似度: {result[0].get('similarity', 'N/A')}")
            return {"code": 0, "matches": result}
        
        return {"code": 0, "matches": []}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"code": -1, "msg": str(e)}


@router.post("/api/migration/precompute")
async def precompute_matches():
    """预计算所有汉字的图片匹配（离线批处理）"""
    try:
        from config import DATASET_ID
        target_dataset_id = DATASET_ID
        
        from bussiness.migration.manager import MigrationManager
        migration_manager = MigrationManager()
        result = migration_manager.precompute_all_image_matches(target_dataset_id)
        return {"code": 0, "msg": "预计算完成", "matched_chars": len(result)}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


class BatchLabelRequest(BaseModel):
    char: str
    charIds: List[str]


@router.post("/api/migration/batch-label")
async def batch_label(request: BatchLabelRequest):
    """批量标注图片为指定汉字（与聚类标注页面数据格式一致）"""
    try:
        from config import LABELS_JSON, CLUSTERS_JSON, pseudo_label_cache
        from utils import load_json_file, save_json_file
        from collections import Counter
        import datetime
        
        char = request.char
        char_ids = request.charIds
        
        if not char or not char_ids:
            return {"code": -1, "msg": "参数错误"}
        
        labels_data = load_json_file(LABELS_JSON) or {}
        clusters_data = load_json_file(CLUSTERS_JSON) or {}
        clusters = clusters_data.get("clusters", {})
        
        labeled_count = 0
        existing_chars = set()
        new_chars = set()
        
        # 收集已存在的汉字
        for cluster_data in labels_data.values():
            if cluster_data.get("status") == "labeled" and cluster_data.get("char_labels"):
                for label_info in cluster_data["char_labels"].values():
                    if label_info.get("char"):
                        existing_chars.add(label_info["char"])
        
        for char_id in char_ids:
            parts = char_id.split("_")
            if len(parts) >= 3:
                cluster_id = parts[0]
                char_index = parts[-1]
                
                if cluster_id in clusters:
                    cluster_chars = clusters[cluster_id]
                    if int(char_index) < len(cluster_chars):
                        if cluster_id not in labels_data:
                            labels_data[cluster_id] = {
                                "char": None,
                                "chars": {},
                                "status": "unlabeled",
                                "confidence": None,
                                "alias": "",
                                "char_labels": {}
                            }
                        
                        if "char_labels" not in labels_data[cluster_id]:
                            labels_data[cluster_id]["char_labels"] = {}
                        
                        labels_data[cluster_id]["char_labels"][char_index] = {
                            "char": char,
                            "labeled_at": datetime.datetime.now().isoformat()
                        }
                        
                        if char not in existing_chars:
                            new_chars.add(char)
                        
                        labeled_count += 1
        
        # 更新每个聚类的汇总信息
        for cluster_id, cluster_data in labels_data.items():
            if cluster_data.get("char_labels"):
                all_chars = [v["char"] for v in cluster_data["char_labels"].values() if v.get("char")]
                if all_chars:
                    char_counts = Counter(all_chars)
                    most_common = char_counts.most_common(1)[0]
                    cluster_data["char"] = most_common[0]
                    cluster_data["confidence"] = most_common[1] / len(all_chars)
                    cluster_data["chars"] = dict(char_counts)
                    cluster_data["status"] = "labeled"
        
        save_json_file(LABELS_JSON, labels_data)
        
        # 更新伪标签缓存
        for char_id in char_ids:
            parts = char_id.split("_")
            if len(parts) >= 3:
                cluster_id = parts[0]
                if char in pseudo_label_cache:
                    if cluster_id in pseudo_label_cache[char]:
                        del pseudo_label_cache[char][cluster_id]
                        if len(pseudo_label_cache[char]) == 0:
                            del pseudo_label_cache[char]
        
        new_chars_list = sorted(list(new_chars))
        return {
            "code": 0, 
            "msg": f"成功标注 {labeled_count} 张图片", 
            "labeled_count": labeled_count,
            "new_chars_count": len(new_chars_list),
            "new_chars": new_chars_list
        }
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/api/migration/export-labels")
async def export_labels(dataset_id: str, description: str = ""):
    """导出标注结果（整合版）"""
    try:
        from bussiness.migration.manager import MigrationManager
        
        manager = MigrationManager()
        export_path = manager.export_labels(dataset_id, description)
        
        if export_path:
            return {"code": 0, "msg": "导出成功", "export_path": export_path}
        else:
            return {"code": -1, "msg": "标注文件不存在"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/api/migration/list-exports")
async def list_exports():
    """列出所有已导出的标注文件"""
    try:
        from bussiness.migration.sharing import LabelSharingManager
        
        data_dir = Path(__file__).parent.parent.parent / "bussiness" / "migration" / "data"
        sharing_manager = LabelSharingManager(data_dir)
        
        exports = sharing_manager.list_exported_files()
        
        return {"code": 0, "data": exports}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/api/migration/share-labels")
async def share_labels(request: ShareRequest):
    """跨数据集共享标注（整合版）"""
    try:
        from bussiness.migration.manager import MigrationManager
        
        manager = MigrationManager()
        result = manager.share_labels(
            request.source_dataset_id,
            request.target_dataset_ids,
            request.merge_strategy
        )
        
        return result
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/api/migration/import-labels")
async def import_labels(file_path: str, target_dataset_id: str, merge_strategy: str = "merge"):
    """导入标注结果"""
    try:
        from bussiness.migration.sharing import LabelSharingManager
        
        data_dir = Path(__file__).parent.parent.parent / "bussiness" / "migration" / "data"
        sharing_manager = LabelSharingManager(data_dir)
        
        result = sharing_manager.import_labels(file_path, target_dataset_id, merge_strategy)
        
        return result
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.delete("/api/migration/delete-export/{filename}")
async def delete_export(filename: str):
    """删除已导出的标注文件"""
    try:
        from bussiness.migration.sharing import LabelSharingManager
        
        data_dir = Path(__file__).parent.parent.parent / "bussiness" / "migration" / "data"
        sharing_manager = LabelSharingManager(data_dir)
        
        success = sharing_manager.delete_exported_file(filename)
        
        if success:
            return {"code": 0, "msg": "删除成功"}
        else:
            return {"code": -1, "msg": "文件不存在"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/api/migration/share-statistics")
async def get_share_statistics():
    """获取标注共享统计信息（整合版）"""
    try:
        from bussiness.migration.manager import MigrationManager
        
        manager = MigrationManager()
        
        # 获取共享统计
        stats = manager.sharing_manager.get_share_statistics()
        
        # 获取所有数据集及其标注统计
        datasets = manager.get_all_datasets()
        dataset_stats = []
        for dataset_id in datasets:
            stats_info = manager.get_dataset_label_stats(dataset_id)
            dataset_stats.append(stats_info)
        
        return {
            "code": 0,
            "data": {
                "share_stats": stats,
                "datasets": dataset_stats
            }
        }
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/api/migration/datasets")
async def get_datasets():
    """获取所有数据集列表"""
    try:
        from bussiness.migration.manager import MigrationManager
        
        manager = MigrationManager()
        datasets = manager.get_all_datasets()
        
        # 获取每个数据集的标注统计
        dataset_info = []
        for dataset_id in datasets:
            stats = manager.get_dataset_label_stats(dataset_id)
            dataset_info.append(stats)
        
        return {"code": 0, "data": dataset_info}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/api/migration/dataset-stats/{dataset_id}")
async def get_dataset_stats(dataset_id: str):
    """获取指定数据集的标注统计"""
    try:
        from bussiness.migration.manager import MigrationManager
        
        manager = MigrationManager()
        stats = manager.get_dataset_label_stats(dataset_id)
        
        return {"code": 0, "data": stats}
    except Exception as e:
        return {"code": -1, "msg": str(e)}