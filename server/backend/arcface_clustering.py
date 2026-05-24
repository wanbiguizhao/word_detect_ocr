import json
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2

from config import DATASET_ID, SOURCE_DATASET_ID

class ArcFaceClustering:
    def __init__(self, config):
        self.config = config
        self.dataset_id = DATASET_ID
        self.source_dataset_id = SOURCE_DATASET_ID
        
        self.project_root = Path(__file__).parent.parent.parent
        self.datahome_dir = self.project_root / "bussiness" / "datahome"
        self.dataset_dir = self.datahome_dir / self.dataset_id
        self.cluster_results_dir = self.project_root / "bussiness" / "cluster_results"
        self.arcface_clusters_dir = self.cluster_results_dir / "arcface"
        self.embeddings_dir = self.project_root / "bussiness" / "embeddings"

        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        if not self.unified_labels_path.exists():
            self.unified_labels_path = self.project_root / "bussiness" / "unified_labels.json"

        self.arcface_clusters_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.arcface_model = None

    def load_arcface_model(self):
        try:
            from insightface.app import FaceAnalysis
            self.arcface_model = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.arcface_model.prepare(ctx_id=0, det_size=(64, 64))
            print("✅ ArcFace 模型加载成功")
        except Exception as e:
            print(f"❌ ArcFace 模型加载失败: {e}")
            raise

    def extract_features(self, image_paths):
        if self.arcface_model is None:
            self.load_arcface_model()
            
        features = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                faces = self.arcface_model.get(img)
                if faces:
                    features.append(faces[0].embedding)
                    valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️  特征提取失败 {img_path}: {e}")
        
        return np.array(features), valid_paths

    def cluster(self, features, eps=0.5, min_samples=3):
        if len(features) == 0:
            return []
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_scaled)
        
        return labels

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
                            unlabeled_images.append(img_path)
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
                    unlabeled_images.append(img_path)

        return unlabeled_images

    def run_clustering(self, eps=0.5, min_samples=3):
        print(f"📊 开始 ArcFace 聚类 - 数据集: {self.dataset_id}")
        
        image_paths = self.get_unlabeled_images()
        print(f"🔍 找到 {len(image_paths)} 张未标注图片")
        
        if len(image_paths) == 0:
            print("⚠️  没有未标注图片，跳过聚类")
            return
        
        print("🔄 提取 ArcFace 特征...")
        features, valid_paths = self.extract_features(image_paths)
        print(f"✅ 成功提取 {len(features)} 个特征")
        
        features_path = self.embeddings_dir / f"{self.dataset_id}_arcface_embeddings.npy"
        paths_path = self.embeddings_dir / f"{self.dataset_id}_arcface_paths.json"
        
        np.save(features_path, features)
        with open(paths_path, 'w', encoding='utf-8') as f:
            json.dump([str(p) for p in valid_paths], f, ensure_ascii=False)
        
        print(f"💾 特征已保存到 {features_path}")
        
        print("🔄 执行聚类...")
        labels = self.cluster(features, eps, min_samples)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "char_id": valid_paths[idx].stem,
                "image_path": str(valid_paths[idx].relative_to(self.project_root / "bussiness")),
                "labeled": False
            })
        
        if -1 in clusters:
            del clusters[-1]
        
        existing_path = self.arcface_clusters_dir / f"{self.dataset_id}_clusters.json"
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            labeled_images = {}
            for cluster in existing_data.get("clusters", []):
                for img in cluster.get("images", []):
                    if img.get("labeled"):
                        labeled_images[img["image_path"]] = img["char"]
            
            for cluster_id, images in clusters.items():
                for img in images:
                    if img["image_path"] in labeled_images:
                        img["labeled"] = True
                        img["char"] = labeled_images[img["image_path"]]
        
        result = {
            "version": "1.0",
            "method": "arcface",
            "dataset": self.dataset_id,
            "config": {
                "eps": eps,
                "min_samples": min_samples
            },
            "clusters": [
                {
                    "cluster_id": str(i),
                    "representative": clusters[list(clusters.keys())[i]][0]["image_path"] if clusters else "",
                    "images": clusters[list(clusters.keys())[i]]
                }
                for i, key in enumerate(sorted(clusters.keys()))
            ]
        }
        
        with open(existing_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 聚类完成！共生成 {len(result['clusters'])} 个簇")
        return result

if __name__ == "__main__":
    from server.backend.config import config
    clusterer = ArcFaceClustering(config._config)
    clusterer.run_clustering()