import json
from fastapi import APIRouter
from config import PROJECT_ROOT

router = APIRouter()


@router.get("/api/char-images/search")
async def get_char_images_list(char: str):
    try:
        labeled_images = []
        label_file = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "labeling" / "labels.json"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                labels = json.load(f)

            clusters_file = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "hog_clusters.json"
            if clusters_file.exists():
                with open(clusters_file, "r", encoding="utf-8") as f:
                    clusters_data = json.load(f)

            for cid, info in labels.items():
                if info.get("status") == "labeled" and info.get("char_labels"):
                    for char_idx, label_info in info["char_labels"].items():
                        if label_info.get("char") == char:
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


@router.get("/api/line-status")
async def get_line_status():
    try:
        clusters_file = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "hog_clusters.json"
        labels_file = PROJECT_ROOT / "bussiness" / "datahome" / "pdf01" / "clusters" / "labeling" / "labels.json"

        if not clusters_file.exists() or not labels_file.exists():
            return {"code": -1, "msg": "文件不存在"}

        with open(clusters_file, "r", encoding="utf-8") as f:
            clusters_data = json.load(f)

        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)

        image_cluster_map = {}
        image_label_map = {}

        for cid, cluster_items in clusters_data.get("clusters", {}).items():
            for idx, item in enumerate(cluster_items):
                char_id = item.get("char_id")
                if char_id:
                    image_cluster_map[char_id] = {"cluster_id": int(cid), "char_index": idx}

                    if cid in labels:
                        label_info = labels[cid]
                        if label_info.get("status") == "labeled" and label_info.get("char_labels"):
                            if str(idx) in label_info["char_labels"]:
                                image_label_map[char_id] = label_info["char_labels"][str(idx)].get("char")

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

        result = []
        for line_name, chars in line_data.items():
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

        result.sort(key=lambda x: (
            int(x["line_name"].split("_")[1].replace("page", "")),
            int(x["line_name"].split("_")[3])
        ))

        return {"code": 0, "data": result}

    except Exception as e:
        print(f"获取行状态失败: {e}")
        return {"code": -1, "msg": str(e)}