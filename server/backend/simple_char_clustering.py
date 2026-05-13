import json
from pathlib import Path


class SimpleCharClustering:
    def __init__(self, config):
        self.config = config
        self.dataset_id = config.get("dataset.current", "pdf5826")
        
        self.project_root = Path(__file__).parent.parent.parent
        self.datahome_dir = self.project_root / "bussiness" / "datahome"
        self.dataset_dir = self.datahome_dir / self.dataset_id
        self.cluster_results_dir = self.project_root / "bussiness" / "cluster_results"
        self.simple_clusters_dir = self.cluster_results_dir / "simple_char"
        
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        if not self.unified_labels_path.exists():
            self.unified_labels_path = self.project_root / "bussiness" / "unified_labels.json"

        self.simple_clusters_dir.mkdir(parents=True, exist_ok=True)

    def get_unlabeled_images(self):
        unlabeled_images = []
        existing_paths = set()

        # 1. 从 unified_labels.json 中查找未标注图片
        if self.unified_labels_path.exists():
            with open(self.unified_labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ann in data.get("annotations", []):
                if ann.get("dataset") == self.dataset_id:
                    if ann.get("status") != "labeled":
                        img_path = self.project_root / "bussiness" / ann["image_path"]
                        if img_path.exists():
                            unlabeled_images.append({
                                "path": img_path,
                                "predicted_char": ann.get("char", "")
                            })
                            existing_paths.add(str(img_path))

        # 2. 从 pre_labels.json 中查找所有预标注图片（未在 unified_labels 中的）
        prelabels_path = self.dataset_dir / "pre_labels.json"
        if not prelabels_path.exists():
            prelabels_path = self.project_root / "bussiness" / "pre_labels.json"

        if prelabels_path.exists():
            with open(prelabels_path, 'r', encoding='utf-8') as f:
                prelabel_data = json.load(f)

            for prelabel in prelabel_data.get("prelabels", []):
                # 处理可能的反斜杠路径
                img_rel_path = prelabel["image_path"].replace("\\", "/")
                img_path = self.project_root / "bussiness" / img_rel_path
                if str(img_path) not in existing_paths and img_path.exists():
                    unlabeled_images.append({
                        "path": img_path,
                        "predicted_char": prelabel.get("predicted_char", ""),
                        "char_id": prelabel.get("char_id", "")
                    })

        return unlabeled_images

    def run_clustering(self, min_samples=3):
        print(f"📊 开始简单字符聚类 - 数据集: {self.dataset_id}")
        
        image_items = self.get_unlabeled_images()
        print(f"🔍 找到 {len(image_items)} 张未标注图片")
        
        if len(image_items) == 0:
            print("⚠️  没有未标注图片，跳过聚类")
            return
        
        # 按预测的字符分组
        char_groups = {}
        for item in image_items:
            char = item["predicted_char"]
            if char:
                if char not in char_groups:
                    char_groups[char] = []
                char_groups[char].append(item)
        
        # 过滤掉样本数少于 min_samples 的组
        filtered_clusters = {char: items for char, items in char_groups.items() 
                            if len(items) >= min_samples}
        
        print(f"✅ 生成 {len(filtered_clusters)} 个簇")
        
        # 加载已有的标注信息
        existing_path = self.simple_clusters_dir / f"{self.dataset_id}_clusters.json"
        labeled_images = {}
        
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            for cluster in existing_data.get("clusters", []):
                for img in cluster.get("images", []):
                    if img.get("labeled"):
                        labeled_images[img["image_path"]] = img["char"]
        
        # 构建聚类结果
        clusters = []
        for char, items in sorted(filtered_clusters.items(), key=lambda x: len(x[1]), reverse=True):
            cluster_images = []
            for item in items:
                img_path_str = str(item["path"].relative_to(self.project_root / "bussiness"))
                cluster_img = {
                    "char_id": item.get("char_id", item["path"].stem),
                    "image_path": img_path_str,
                    "labeled": img_path_str in labeled_images
                }
                if cluster_img["labeled"]:
                    cluster_img["char"] = labeled_images[img_path_str]
                cluster_images.append(cluster_img)
            
            clusters.append({
                "cluster_id": char,
                "representative": cluster_images[0]["image_path"],
                "predicted_char": char,
                "images": cluster_images
            })
        
        result = {
            "version": "1.0",
            "method": "simple_char",
            "dataset": self.dataset_id,
            "config": {
                "min_samples": min_samples
            },
            "clusters": clusters
        }
        
        with open(existing_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 聚类完成！共生成 {len(clusters)} 个簇")
        return result
