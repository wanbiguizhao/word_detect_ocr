#!/usr/bin/env python3
"""数据同步脚本 - 统一更新所有数据文件（安全模式）"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
from datetime import datetime

class DataSync:
    def __init__(self, dataset_id: str, dry_run=False, backup=True):
        self.dataset_id = dataset_id
        self.dry_run = dry_run
        self.backup = backup
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.dataset_dir = self.project_root / "bussiness" / "datahome" / dataset_id

        # 路径配置
        self.prelabels_path = self.dataset_dir / "pre_labels.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"
        self.clusters_path = self.dataset_dir / "clusters" / "hog_clusters.json"
        self.labels_path = self.dataset_dir / "clusters" / "labeling" / "labels.json"

        # 回退路径
        self.fallback_unified_labels = self.project_root / "bussiness" / "unified_labels.json"
        self.fallback_prelabels = self.project_root / "bussiness" / "pre_labels.json"

        # 统计信息
        self.stats = {
            "prelabels_updated": 0,
            "unified_labels_added": 0,
            "cluster_labels_updated": 0,
            "char_counts_updated": 0,
            "errors": [],
            "warnings": []
        }

        # 备份文件列表
        self.backup_files = []

    def _backup_file(self, path: Path):
        """备份文件"""
        if self.backup and path.exists():
            backup_path = path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
            shutil.copy(path, backup_path)
            self.backup_files.append(str(backup_path))
            print(f"    [BAK] 已备份: {path.name}")

    def _load_file(self, path: Path, fallback_path=None):
        """加载 JSON 文件"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif fallback_path and fallback_path.exists():
            print(f"    [FALLBACK] 使用回退路径: {fallback_path}")
            with open(fallback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_file(self, path: Path, data: dict):
        """保存 JSON 文件"""
        if self.dry_run:
            print(f"    [DRYRUN] 模拟保存文件: {path}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _validate_char_id(self, char_id: str) -> bool:
        """验证 char_id 格式"""
        if not char_id:
            return False
        parts = char_id.split('_')
        if len(parts) != 6:
            return False
        if parts[0] != "page" or parts[2] != "line" or parts[4] != "char":
            return False
        try:
            int(parts[1]), int(parts[3]), int(parts[5])
            return True
        except ValueError:
            return False

    def _validate_char(self, char: str) -> bool:
        """验证字符（非空且不是乱码）"""
        if not char:
            return False
        if len(char) > 10:  # 太长的不是正常字符
            return False
        return True

    def sync_unified_to_prelabels(self):
        """同步统一标注到预标注（统一标注为权威来源）"""
        print("[SYNC] 正在同步统一标注到预标注...")

        self._backup_file(self.prelabels_path)

        prelabels_data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        unified_data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)

        prelabels = prelabels_data.get("prelabels", [])
        annotations = unified_data.get("annotations", [])

        labeled_chars = {}
        for ann in annotations:
            char_id = ann.get("char_id")
            char = ann.get("char")
            status = ann.get("status")

            if not self._validate_char_id(char_id):
                self.stats["warnings"].append(f"无效的 char_id: {char_id}")
                continue
            if not self._validate_char(char):
                self.stats["warnings"].append(f"无效的字符: {repr(char)} (char_id: {char_id})")
                continue
            if status != "labeled":
                continue

            labeled_chars[char_id] = char

        print(f"    找到 {len(labeled_chars)} 个有效的已标注字符")

        for prelabel in prelabels:
            char_id = prelabel.get("char_id")
            original_status = prelabel.get("status")

            if char_id in labeled_chars:
                new_char = labeled_chars[char_id]
                prelabel["status"] = "confirmed"

                if prelabel.get("predicted_char") != new_char:
                    prelabel["predicted_char"] = new_char
                    self.stats["prelabels_updated"] += 1
                elif original_status != "confirmed":
                    self.stats["prelabels_updated"] += 1

        prelabels_data["prelabels"] = prelabels
        self._save_file(self.prelabels_path, prelabels_data)
        print(f"    [OK] 成功更新 {self.stats['prelabels_updated']} 条预标注")

    def sync_prelabels_to_unified(self):
        """同步预标注到统一标注（添加未存在的，更新字符不一致的）"""
        print("[SYNC] 正在同步预标注到统一标注...")

        self._backup_file(self.unified_labels_path)

        prelabels_data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        
        # 对于统一标注，优先使用数据集特定路径，不使用回退路径（避免数据污染）
        if self.unified_labels_path.exists():
            with open(self.unified_labels_path, 'r', encoding='utf-8') as f:
                unified_data = json.load(f)
        else:
            unified_data = {"annotations": []}
        
        # 如果使用了回退路径，只保留当前数据集的数据
        if unified_data and not self.unified_labels_path.exists():
            annotations = unified_data.get("annotations", [])
            unified_data["annotations"] = [a for a in annotations if a.get("dataset") == self.dataset_id]

        prelabels = prelabels_data.get("prelabels", [])
        annotations = unified_data.get("annotations", [])

        # 构建 char_id -> char 映射
        existing_char_map = {ann.get("char_id"): ann.get("char") for ann in annotations if ann.get("char_id")}

        confirmed_prelabels = []
        for p in prelabels:
            if p.get("status") == "confirmed":
                char_id = p.get("char_id")
                char = p.get("predicted_char")

                if not self._validate_char_id(char_id):
                    self.stats["warnings"].append(f"预标注无效 char_id: {char_id}")
                    continue
                if not self._validate_char(char):
                    self.stats["warnings"].append(f"预标注无效字符: {repr(char)}")
                    continue

                confirmed_prelabels.append(p)

        print(f"    找到 {len(confirmed_prelabels)} 个有效的已确认预标注")

        for prelabel in confirmed_prelabels:
            char_id = prelabel.get("char_id")
            new_char = prelabel.get("predicted_char")

            if char_id not in existing_char_map:
                # 添加新标注
                annotation = {
                    "char_id": char_id,
                    "char": new_char,
                    "image_path": prelabel.get("image_path", ""),
                    "dataset": self.dataset_id,
                    "status": "labeled",
                    "cluster_id": prelabel.get("cluster_id", "")
                }
                annotations.append(annotation)
                existing_char_map[char_id] = new_char
                self.stats["unified_labels_added"] += 1
            elif existing_char_map[char_id] != new_char:
                # 更新字符不一致的标注（以预标注为准）
                for ann in annotations:
                    if ann.get("char_id") == char_id:
                        ann["char"] = new_char
                        ann["status"] = "labeled"
                        self.stats["unified_labels_added"] += 1
                        break
            elif existing_char_map[char_id] == new_char:
                # 字符相同但状态可能不对
                for ann in annotations:
                    if ann.get("char_id") == char_id:
                        if ann.get("status") != "labeled":
                            ann["status"] = "labeled"
                            self.stats["unified_labels_added"] += 1
                        break

        char_distribution = defaultdict(int)
        dataset_stats = defaultdict(lambda: {"labeled_count": 0})

        for ann in annotations:
            if ann.get("char"):
                char_distribution[ann["char"]] += 1
            if ann.get("dataset"):
                dataset_stats[ann["dataset"]]["labeled_count"] += 1

        unified_data["annotations"] = annotations
        unified_data["char_distribution"] = dict(char_distribution)
        unified_data["total_labeled"] = len(annotations)
        unified_data["datasets"] = list(dataset_stats.keys())
        unified_data["dataset_stats"] = dict(dataset_stats)

        self._save_file(self.unified_labels_path, unified_data)
        print(f"    [OK] 成功添加 {self.stats['unified_labels_added']} 条标注")

    def sync_cluster_labels(self):
        """同步聚类数据中的确认标记"""
        print("[SYNC] 正在同步聚类标注...")

        self._backup_file(self.clusters_path)

        clusters_data = self._load_file(self.clusters_path)
        unified_data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)

        clusters = clusters_data.get("clusters", {})
        annotations = unified_data.get("annotations", [])

        labeled_chars = {
            ann["char_id"]: ann["char"]
            for ann in annotations
            if ann.get("char_id") and ann.get("char") and ann.get("status") == "labeled"
        }

        for cluster_id, chars in clusters.items():
            for char in chars:
                char_id = char.get("char_id")
                if char_id in labeled_chars:
                    if char.get("confirmed") != True or char.get("confirmed_char") != labeled_chars[char_id]:
                        char["confirmed"] = True
                        char["confirmed_char"] = labeled_chars[char_id]
                        self.stats["cluster_labels_updated"] += 1

        clusters_data["clusters"] = clusters
        self._save_file(self.clusters_path, clusters_data)
        print(f"    [OK] 成功更新 {self.stats['cluster_labels_updated']} 个聚类字符")

    def update_char_counts(self):
        """更新字符统计（基于统一标注）"""
        print("[SYNC] 正在更新字符统计...")

        self._backup_file(self.prelabels_path)

        prelabels_data = self._load_file(self.prelabels_path, self.fallback_prelabels)
        unified_data = self._load_file(self.unified_labels_path, self.fallback_unified_labels)

        prelabels = prelabels_data.get("prelabels", [])
        annotations = unified_data.get("annotations", [])

        char_counts = defaultdict(lambda: {"total": 0, "confirmed": 0, "pending": 0})

        for prelabel in prelabels:
            char = prelabel.get("predicted_char")
            if self._validate_char(char):
                char_counts[char]["total"] += 1

        dataset_annotations = [a for a in annotations if a.get("dataset") == self.dataset_id]
        for ann in dataset_annotations:
            char = ann.get("char")
            if self._validate_char(char) and ann.get("status") == "labeled":
                char_counts[char]["confirmed"] += 1

        for char, counts in char_counts.items():
            counts["pending"] = max(0, counts["total"] - counts["confirmed"])

        prelabels_data["char_counts"] = {k: dict(v) for k, v in char_counts.items()}
        prelabels_data["stats"] = {
            "total": len(prelabels),
            "confirmed": sum(v["confirmed"] for v in char_counts.values()),
            "pending": sum(v["pending"] for v in char_counts.values())
        }

        self.stats["char_counts_updated"] = len(char_counts)
        self._save_file(self.prelabels_path, prelabels_data)
        print(f"    [OK] 成功更新 {self.stats['char_counts_updated']} 个字符的统计")

    def run_all(self):
        """运行所有同步任务"""
        print(f"[START] 开始数据同步 - 数据集: {self.dataset_id}")
        print(f"   模式: {'DRYRUN' if self.dry_run else '实际'}")
        print(f"   备份: {'开启' if self.backup else '关闭'}")
        print("=" * 60)

        try:
            self.sync_unified_to_prelabels()
            self.sync_prelabels_to_unified()
            self.sync_cluster_labels()
            self.update_char_counts()

            print("=" * 60)
            print("[DONE] 同步完成！")
            print(f"    - 预标注更新: {self.stats['prelabels_updated']}")
            print(f"    - 统一标注添加: {self.stats['unified_labels_added']}")
            print(f"    - 聚类标记更新: {self.stats['cluster_labels_updated']}")
            print(f"    - 字符统计更新: {self.stats['char_counts_updated']}")

            if self.stats["warnings"]:
                print("\n[WARN] 警告列表:")
                for warning in self.stats["warnings"][:5]:
                    print(f"    - {warning}")
                if len(self.stats["warnings"]) > 5:
                    print(f"    - ... 还有 {len(self.stats['warnings']) - 5} 个警告")

            if self.backup_files:
                print(f"\n[BAK] 备份文件:")
                for backup in self.backup_files[:3]:
                    print(f"    - {backup}")
                if len(self.backup_files) > 3:
                    print(f"    - ... 还有 {len(self.backup_files) - 3} 个备份")

        except Exception as e:
            print(f"\n[ERROR] 同步失败: {str(e)}")
            self.stats["errors"].append(str(e))
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="数据同步脚本（安全模式）")
    parser.add_argument("--dataset", "-d", default="pdf5823", help="数据集ID")
    parser.add_argument("--all", "-a", action="store_true", help="同步所有数据集")
    parser.add_argument("--dry-run", "-n", action="store_true", help="模拟运行，不实际修改文件")
    parser.add_argument("--no-backup", action="store_true", help="不备份文件")

    args = parser.parse_args()

    if args.all:
        datahome_dir = Path(__file__).parent.parent.parent.parent / "bussiness" / "datahome"
        if datahome_dir.exists():
            dataset_dirs = sorted([d for d in datahome_dir.iterdir() if d.is_dir()])
            print(f"[INFO] 找到 {len(dataset_dirs)} 个数据集")
            print()

            for dataset_dir in dataset_dirs:
                dataset_id = dataset_dir.name
                sync = DataSync(dataset_id, dry_run=args.dry_run, backup=not args.no_backup)
                sync.run_all()
                print()
        else:
            print(f"[ERROR] 数据目录不存在: {datahome_dir}")
    else:
        sync = DataSync(args.dataset, dry_run=args.dry_run, backup=not args.no_backup)
        sync.run_all()

if __name__ == "__main__":
    main()