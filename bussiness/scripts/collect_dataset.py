import os
import json
import shutil
from pathlib import Path
import random

class DatasetCollector:
    def __init__(self, datahome_path, output_path):
        self.datahome = Path(datahome_path)
        self.output = Path(output_path)
        self.output.mkdir(parents=True, exist_ok=True)
        
        self.annotations = []
        self.char_distribution = {}
        self.seen_char_ids = set()
        self.seen_image_paths = set()
        self.seen_chars = set()
        self.dataset_new_chars = {}
    
    def load_unified_labels(self, dataset_name):
        unified_path = self.datahome / dataset_name / "unified_labels.json"
        if not unified_path.exists():
            print(f"跳过: {unified_path} 不存在")
            return
        
        print(f"加载 {dataset_name} 的统一标注...")
        with open(unified_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for ann in data.get("annotations", []):
            self.add_annotation(ann, dataset_name, source="unified_labels")
    
    def load_multi_clustering_labels(self, dataset_name):
        """加载多轮聚类标注数据"""
        mc_unified_path = self.datahome / dataset_name / "multi_clustering" / "unified_labels.json"
        
        if not mc_unified_path.exists():
            print(f"跳过: {mc_unified_path} 不存在")
            return
        
        print(f"加载 {dataset_name} 的多轮聚类标注...")
        try:
            with open(mc_unified_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取失败: {e}")
            return
        
        for ann in data.get("annotations", []):
            char_id = ann.get("char_id")
            if not char_id:
                continue
            
            self.add_annotation({
                "char_id": char_id,
                "char": ann.get("char", ""),
                "image_path": f"datahome/{dataset_name}/pdf_chars/{char_id}.png",
                "dataset": dataset_name,
                "status": ann.get("status", "labeled"),
                "cluster_id": ""
            }, dataset_name, source="multi_clustering")
    
    def load_cluster_labels(self, dataset_name):
        labels_path = self.datahome / dataset_name / "clusters" / "labeling" / "labels.json"
        clusters_path = self.datahome / dataset_name / "clusters" / "hog_clusters.json"
        
        if not labels_path.exists() or not clusters_path.exists():
            print(f"跳过: {labels_path} 或 {clusters_path} 不存在")
            return
        
        print(f"加载 {dataset_name} 的聚类标注...")
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        
        clusters = clusters_data.get("clusters", {})
        
        for cluster_id, label_info in labels_data.items():
            char_labels = label_info.get("char_labels", {})
            cluster_chars = clusters.get(cluster_id, [])
            
            for idx_str, char_label_info in char_labels.items():
                char = char_label_info.get("char")
                if not char:
                    continue
                
                try:
                    idx = int(idx_str)
                    if idx < len(cluster_chars):
                        char_info = cluster_chars[idx]
                        char_id = char_info.get("char_id", f"{dataset_name}_cluster_{cluster_id}_char_{idx}")
                        
                        self.add_annotation({
                            "char_id": char_id,
                            "char": char,
                            "image_path": f"datahome/{dataset_name}/pdf_chars/{char_id}.png",
                            "dataset": dataset_name,
                            "status": "labeled",
                            "cluster_id": cluster_id
                        }, dataset_name, source="cluster_labels")
                except (ValueError, IndexError):
                    continue
    
    def add_annotation(self, ann, dataset_name, source):
        char_id = ann.get("char_id")
        if not char_id:
            return
        
        orig_image_path = ann.get("image_path", "")
        
        if char_id in self.seen_char_ids:
            return
        if orig_image_path in self.seen_image_paths:
            return
        
        record = {
            "char_id": char_id,
            "char": ann.get("char", ""),
            "image_path": f"datahome/{dataset_name}/pdf_chars/{char_id}.png",
            "dataset": dataset_name,
            "status": ann.get("status", "labeled"),
            "cluster_id": ann.get("cluster_id", "")
        }
        
        self.annotations.append(record)
        self.seen_char_ids.add(char_id)
        if orig_image_path:
            self.seen_image_paths.add(orig_image_path)
        
        char = ann.get("char", "")
        if char:
            self.char_distribution[char] = self.char_distribution.get(char, 0) + 1
            if char not in self.seen_chars:
                self.seen_chars.add(char)
                if dataset_name not in self.dataset_new_chars:
                    self.dataset_new_chars[dataset_name] = []
                self.dataset_new_chars[dataset_name].append(char)
    
    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        
        datasets = sorted(set(a["dataset"] for a in self.annotations))
        dataset_stats = {}
        for ds in datasets:
            dataset_stats[ds] = {
                "labeled_count": sum(1 for a in self.annotations if a["dataset"] == ds)
            }
        
        unified = {
            "datasets": datasets,
            "total_labeled": len(self.annotations),
            "dataset_stats": dataset_stats,
            "char_distribution": self.char_distribution,
            "annotations": self.annotations
        }
        
        output_path = self.output / "unified_labels.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified, f, ensure_ascii=False, indent=2)
        
        print(f"数据集保存完成！共 {len(self.annotations)} 条标注，{len(self.char_distribution)} 个不同汉字")
        print(f"输出文件: {output_path}")
    
    def run(self,dataset_name_list):
        print("开始收集数据集...")
        print()
        for dataset_name in dataset_name_list:
            before_chars = len(self.seen_chars)
            self.load_unified_labels(dataset_name)
            self.load_multi_clustering_labels(dataset_name)
            self.load_cluster_labels(dataset_name)
            after_chars = len(self.seen_chars)
            new_count = after_chars - before_chars
            print(f"  [{dataset_name}] 新增 {new_count} 个未识别汉字")
        print()
        self.save()
        print()
        print("各数据集新增汉字统计:")
        for ds in dataset_name_list:
            new_chars = self.dataset_new_chars.get(ds, [])
            print(f"  [{ds}] 新增 {len(new_chars)} 个: {', '.join(sorted(new_chars)[:30])}{'...' if len(new_chars) > 30 else ''}")

if __name__ == "__main__":
    PROJECT_ROOT= Path(__file__).resolve().parent.parent.parent
    collector = DatasetCollector(
        datahome_path=PROJECT_ROOT / "bussiness/datahome",
        output_path=PROJECT_ROOT /"bussiness"
    )
    dataset_name_list = ["pdf01","pdf5823"]
    collector.run(dataset_name_list)