import asyncio
import time
from collections import Counter
from fastapi import APIRouter, HTTPException, Request
from sklearn.metrics.pairwise import cosine_similarity
from config import CLUSTERS_JSON, LABELS_JSON, PROJECT_ROOT, pseudo_label_cache, executor
from utils import load_json_file, extract_hog_features, get_labeled_clusters_info

router = APIRouter()


@router.get("/api/pseudo-labels")
def get_pseudo_labels():
    if not LABELS_JSON.exists():
        return {"code": 0, "data": []}

    labels_data = load_json_file(LABELS_JSON) or {}

    char_stats = {}
    for cluster_id, label_info in labels_data.items():
        if label_info.get("status") == "labeled" and label_info.get("char_labels"):
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

            for char in chars_in_cluster:
                if cluster_id not in char_stats[char]["clusters"]:
                    char_stats[char]["clusters"].append(cluster_id)
                    char_stats[char]["cluster_count"] += 1

    result = list(char_stats.values())
    result.sort(key=lambda x: x["total_count"], reverse=True)

    total_chars = len(char_stats)
    total_clusters = len([cid for cid, info in labels_data.items() if info.get("status") == "labeled"])
    total_images = sum(info.get("total_count", 0) for info in char_stats.values())

    return {"code": 0, "data": result, "stats": {
        "total_chars": total_chars,
        "total_clusters": total_clusters,
        "total_images": total_images
    }}


@router.post("/api/pseudo-labels/clusters")
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
            img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f'{item["char_id"]}.png')
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
                img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f'{char_id}.png')
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


@router.delete("/api/pseudo-labels/cache/{char}")
def clear_pseudo_label_cache(char: str):
    if char in pseudo_label_cache:
        del pseudo_label_cache[char]
    return {"code": 0, "msg": f"已清除缓存: {char}"}


@router.delete("/api/pseudo-labels/cache")
def clear_all_pseudo_label_cache():
    count = len(pseudo_label_cache)
    pseudo_label_cache.clear()
    return {"code": 0, "msg": f"已清除全部缓存，共 {count} 条"}


@router.post("/api/pseudo-labels/clusters/images")
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
        img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f'{char_id}.png')
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
        img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f'{char_id}.png')
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