import datetime
import json
from collections import Counter
from fastapi import APIRouter, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from config import CLUSTERS_JSON, LABELS_JSON, PROJECT_ROOT, pseudo_label_cache
from models import BatchLabelSave, ClusterLabelSave
from utils import load_json_file, extract_hog_features, get_labeled_clusters_info

router = APIRouter()


@router.get("/api/clusters")
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


@router.get("/api/cluster-labels")
def get_cluster_labels():
    if not LABELS_JSON.exists():
        return {"code": 0, "msg": "success", "data": {}}

    data = load_json_file(LABELS_JSON)

    for cluster_data in data.values():
        char_labels = cluster_data.get("char_labels", {})
        if char_labels and "chars" not in cluster_data:
            all_chars = [v["char"] for v in char_labels.values() if v.get("char")]
            if all_chars:
                cluster_data["chars"] = dict(Counter(all_chars))

    return {"code": 0, "msg": "success", "data": data}


@router.post("/api/cluster-labels/save")
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
            char_counts = Counter(all_chars)
            most_common = char_counts.most_common(1)[0]
            labels_data[cluster_key]["char"] = most_common[0]
            labels_data[cluster_key]["confidence"] = most_common[1] / len(all_chars)
            labels_data[cluster_key]["chars"] = dict(char_counts)

        labels_data[cluster_key]["status"] = "labeled"

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    return {"code": 0, "msg": "保存成功"}


@router.post("/api/cluster-labels/batch-save")
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
                del pseudo_label_cache[char][cluster_id]
                if len(pseudo_label_cache[char]) == 0:
                    del pseudo_label_cache[char]

    new_chars_list = sorted(list(new_chars))
    return {
        "code": 0,
        "msg": f"保存成功，共 {len(body.labels)} 条",
        "new_chars_count": len(new_chars_list),
        "new_chars": new_chars_list
    }


@router.post("/api/clusters/{cluster_id}/skip")
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


@router.get("/api/clusters/{cluster_id}/images")
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


@router.get("/api/clusters/{cluster_id}/recommend")
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

        char_counter = Counter([a["char"] for a in all_anchors])
        sorted_chars = [char for char, count in char_counter.most_common()]

        anchor_char_ids = []
        selected_chars = set()

        for char in sorted_chars:
            if char not in selected_chars:
                char_anchors = [a for a in all_anchors if a["char"] == char]
                if char_anchors:
                    anchor_char_ids.append(char_anchors[0]["char_id"])
                    selected_chars.add(char)

        remaining_quota = 500 - len(anchor_char_ids)
        if remaining_quota > 0:
            for char in sorted_chars:
                if remaining_quota <= 0:
                    break
                count = char_counter[char]
                if count >= 20:
                    extra_anchors = 2
                elif count >= 10:
                    extra_anchors = 1
                else:
                    extra_anchors = 0

                if extra_anchors > 0:
                    char_anchors = [a for a in all_anchors if a["char"] == char][1:]
                    num_to_add = min(extra_anchors, len(char_anchors), remaining_quota)
                    anchor_char_ids.extend([a["char_id"] for a in char_anchors[:num_to_add]])
                    remaining_quota -= num_to_add

    if not anchor_char_ids:
        return {"code": 0, "msg": "无可用锚点", "recommendations": {}}

    anchor_features = {}
    for char_id in anchor_char_ids:
        img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f"{char_id}.png")
        feat = extract_hog_features(img_path)
        if feat is not None:
            anchor_features[char_id] = feat

    if not anchor_features:
        return {"code": 0, "msg": "无法提取锚点特征", "recommendations": {}}

    char_label_map = {}
    if mode == "local" and cluster_labels.get("char_labels"):
        for idx, label_info in cluster_labels["char_labels"].items():
            if int(idx) < len(chars):
                char_id = chars[int(idx)]["char_id"]
                if char_id in anchor_features:
                    char_label_map[char_id] = label_info["char"]
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
            img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f"{char_id}.png")

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


@router.get("/api/clusters/{cluster_id}/recommend/{char}")
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
    if cluster_labels.get("char_labels"):
        for idx, label_info in cluster_labels["char_labels"].items():
            if label_info.get("char") == char:
                if int(idx) < len(chars):
                    anchor_char_ids.append(chars[int(idx)]["char_id"])
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
        img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f"{char_id}.png")
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
            img_path = str(PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "pdf_chars" / f"{char_id}.png")

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
    results = results[:50]
    return {
        "code": 0,
        "msg": "success",
        "char": char,
        "total": len(results),
        "images": results
    }