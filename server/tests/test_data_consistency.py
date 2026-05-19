import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(r"d:\projects\word_detect_ocr")
DATA_HOME = PROJECT_ROOT / "bussiness" / "datahome" / "pdf5826"
MC_DIR = DATA_HOME / "multi_clustering"
CHAR_POOL_DIR = MC_DIR / "char_pool"


def load_json(path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def unified_labels():
    data = load_json(DATA_HOME / "unified_labels.json")
    return data.get("annotations", []) if data else []


@pytest.fixture
def char_pool_all():
    data = load_json(CHAR_POOL_DIR / "all_chars.json")
    return data.get("chars", {}) if data else {}


@pytest.fixture
def char_pool_labeled_ids():
    data = load_json(CHAR_POOL_DIR / "labeled.json")
    return set(data.get("char_ids", [])) if data else set()


@pytest.fixture
def char_pool_unlabeled_ids():
    data = load_json(CHAR_POOL_DIR / "unlabeled.json")
    return set(data.get("char_ids", [])) if data else set()


@pytest.fixture
def prelabel_status():
    data = load_json(DATA_HOME / ".meta" / "prelabel_status.json")
    return data.get("status", {}) if data else {}


@pytest.fixture
def prelabels():
    data = load_json(DATA_HOME / "pre_labels.json")
    return data.get("prelabels", []) if data else []


class TestTotalConsistency:
    """总数字符一致性：Dashboard 和多轮聚类的 total 应一致"""

    def test_total_count_matches(self, char_pool_all, prelabels):
        mc_total = len(char_pool_all)
        dashboard_total = len(prelabels)
        assert mc_total == dashboard_total, (
            f"总数不一致: char_pool total={mc_total}, prelabels total={dashboard_total}"
        )


class TestLabeledConsistency:
    """已标注数据一致性：unified_labels.labeled 与 char_pool.labeled 应一致"""

    def test_labeled_count_matches(self, unified_labels, char_pool_labeled_ids):
        unified_labeled_ids = {
            a["char_id"] for a in unified_labels
            if a.get("status") == "labeled"
            and (a.get("dataset") == "pdf5826" or not a.get("dataset"))
        }
        only_in_unified = unified_labeled_ids - char_pool_labeled_ids
        only_in_pool = char_pool_labeled_ids - unified_labeled_ids
        assert len(only_in_unified) == 0, (
            f"有 {len(only_in_unified)} 个字符在unified_labels中为labeled，"
            f"但不在char_pool/labeled.json中: {list(only_in_unified)[:10]}"
        )
        assert len(only_in_pool) == 0, (
            f"有 {len(only_in_pool)} 个字符在char_pool/labeled.json中，"
            f"但不在unified_labels中: {list(only_in_pool)[:10]}"
        )

    def test_labeled_in_all_chars_status(self, char_pool_labeled_ids, char_pool_all):
        for cid in char_pool_labeled_ids:
            char_info = char_pool_all.get(cid)
            assert char_info is not None, f"labeled字符 {cid} 不在all_chars中"
            assert char_info.get("status") == "labeled", (
                f"labeled字符 {cid} 在all_chars中状态为 '{char_info.get('status')}'，期望 'labeled'"
            )


class TestSkippedConsistency:
    """已跳过数据一致性：prelabel_status.skipped 与 char_pool.all_chars.skipped 应一致"""

    def test_skipped_count_matches(self, prelabel_status, char_pool_all):
        prelabel_skipped_ids = {cid for cid, st in prelabel_status.items() if st == "skipped"}
        pool_skipped_ids = {cid for cid, info in char_pool_all.items() if info.get("status") == "skipped"}
        only_in_prelabel = prelabel_skipped_ids - pool_skipped_ids
        only_in_pool = pool_skipped_ids - prelabel_skipped_ids
        assert len(only_in_prelabel) == 0, (
            f"有 {len(only_in_prelabel)} 个字符在prelabel_status中为skipped，"
            f"但不在char_pool/all_chars的skipped中: {list(only_in_prelabel)[:10]}"
        )
        assert len(only_in_pool) == 0, (
            f"有 {len(only_in_pool)} 个字符在char_pool/all_chars中为skipped，"
            f"但不在prelabel_status的skipped中: {list(only_in_pool)[:10]}"
        )


class TestMutualExclusion:
    """互斥性：labeled / skipped / unlabeled 不应有重叠"""

    def test_labeled_and_unlabeled_no_overlap(self, char_pool_labeled_ids, char_pool_unlabeled_ids):
        overlap = char_pool_labeled_ids & char_pool_unlabeled_ids
        assert len(overlap) == 0, (
            f"labeled和unlabeled有 {len(overlap)} 个重叠: {list(overlap)[:10]}"
        )

    def test_labeled_and_skipped_no_overlap(self, char_pool_labeled_ids, char_pool_all):
        skipped_ids = {cid for cid, info in char_pool_all.items() if info.get("status") == "skipped"}
        overlap = char_pool_labeled_ids & skipped_ids
        assert len(overlap) == 0, (
            f"labeled和skipped有 {len(overlap)} 个重叠: {list(overlap)[:10]}"
        )

    def test_all_chars_partition(self, char_pool_all, char_pool_labeled_ids, char_pool_unlabeled_ids):
        skipped_ids = {cid for cid, info in char_pool_all.items() if info.get("status") == "skipped"}
        all_ids = set(char_pool_all.keys())
        covered = char_pool_labeled_ids | char_pool_unlabeled_ids | skipped_ids
        uncovered = all_ids - covered
        assert len(uncovered) == 0, (
            f"有 {len(uncovered)} 个字符不在labeled/unlabeled/skipped任何集合中"
        )


class TestDashboardVsMultiClustering:
    """Dashboard 与多轮聚类页面统计一致性"""

    def test_labeled_count_same(self, unified_labels, char_pool_labeled_ids):
        unified_labeled = {
            a["char_id"] for a in unified_labels
            if a.get("status") == "labeled"
            and (a.get("dataset") == "pdf5826" or not a.get("dataset"))
        }
        assert len(unified_labeled) == len(char_pool_labeled_ids), (
            f"Dashboard labeled={len(unified_labeled)}, "
            f"多轮聚类 labeled={len(char_pool_labeled_ids)}, "
            f"差异={len(unified_labeled) - len(char_pool_labeled_ids)}"
        )

    def test_skipped_count_same(self, prelabel_status, char_pool_all):
        prelabel_skipped = sum(1 for v in prelabel_status.values() if v == "skipped")
        pool_skipped = sum(1 for info in char_pool_all.values() if info.get("status") == "skipped")
        assert prelabel_skipped == pool_skipped, (
            f"Dashboard skipped={prelabel_skipped}, "
            f"多轮聚类 skipped={pool_skipped}, "
            f"差异={prelabel_skipped - pool_skipped}"
        )

    def test_pending_vs_unlabeled_consistency(self, unified_labels, prelabel_status,
                                               prelabels, char_pool_unlabeled_ids):
        unified_labeled = {
            a["char_id"] for a in unified_labels
            if a.get("status") == "labeled"
            and (a.get("dataset") == "pdf5826" or not a.get("dataset"))
        }
        prelabel_skipped = {cid for cid, st in prelabel_status.items() if st == "skipped"}
        all_prelabel_ids = {p.get("char_id") for p in prelabels if p.get("char_id")}
        dashboard_pending = len(all_prelabel_ids - unified_labeled - prelabel_skipped)
        mc_unlabeled = len(char_pool_unlabeled_ids)
        assert dashboard_pending == mc_unlabeled, (
            f"Dashboard 待确认={dashboard_pending}, "
            f"多轮聚类 待标注={mc_unlabeled}, "
            f"差异={dashboard_pending - mc_unlabeled}"
        )


class TestLowConfidenceFilter:
    """低置信度筛选条件一致性（需求规格说明书 5.3 节）

    筛选条件（必须同时满足）：
    1. confidence < confidence_threshold
    2. 不在 labeled.json 中（未标注）
    3. 不在 all_chars.json 的 skipped 状态中（未跳过）
    4. 不在 prelabel_status.json 的 confirmed 状态中（未确认）
    """

    def test_filter_excludes_labeled(self, char_pool_labeled_ids, prelabels):
        low_conf = [p for p in prelabels if p.get("confidence", 1.0) < 0.7]
        low_conf_labeled = [p for p in low_conf if p.get("char_id") in char_pool_labeled_ids]
        assert len(low_conf_labeled) == 0, (
            f"低置信度筛选应排除已标注，但有 {len(low_conf_labeled)} 个已标注字符"
        )

    def test_filter_excludes_skipped(self, char_pool_all, prelabels):
        skipped_ids = {cid for cid, info in char_pool_all.items() if info.get("status") == "skipped"}
        low_conf = [p for p in prelabels if p.get("confidence", 1.0) < 0.7]
        low_conf_skipped = [p for p in low_conf if p.get("char_id") in skipped_ids]
        assert len(low_conf_skipped) == 0, (
            f"低置信度筛选应排除已跳过，但有 {len(low_conf_skipped)} 个已跳过字符"
        )

    def test_filter_excludes_confirmed(self, prelabel_status, prelabels):
        prelabel_confirmed = {cid for cid, st in prelabel_status.items() if st == "confirmed"}
        low_conf = [p for p in prelabels if p.get("confidence", 1.0) < 0.7]
        low_conf_confirmed = [p for p in low_conf if p.get("char_id") in prelabel_confirmed]
        assert len(low_conf_confirmed) == 0, (
            f"低置信度筛选应排除已确认，但有 {len(low_conf_confirmed)} 个已确认字符"
        )


class TestClusterStatusConsistency:
    """聚类状态一致性：labels.json 静态状态与动态计算状态应一致"""

    def _get_round_dirs(self):
        rounds_dir = MC_DIR / "rounds"
        if not rounds_dir.exists():
            return []
        return [d for d in rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]

    def test_cluster_status_matches_dynamic(self, char_pool_all):
        chars_info = char_pool_all
        for round_dir in self._get_round_dirs():
            labels_path = round_dir / "labels.json"
            clusters_path = round_dir / "hog_clusters.json"
            if not labels_path.exists() or not clusters_path.exists():
                continue

            labels_data = load_json(labels_path)
            clusters_data = load_json(clusters_path)
            labels = labels_data.get("labels", {})
            clusters = clusters_data.get("clusters", {})

            for cluster_id, chars in clusters.items():
                char_ids_in_cluster = []
                for c in chars:
                    if isinstance(c, str):
                        char_ids_in_cluster.append(c)
                    elif isinstance(c, dict):
                        char_ids_in_cluster.append(c.get("char_id", ""))

                labeled_count = 0
                skipped_count = 0
                for char_id in char_ids_in_cluster:
                    char_info = chars_info.get(char_id, {})
                    if char_info.get("status") == "skipped":
                        skipped_count += 1
                    elif char_info.get("status") == "labeled":
                        labeled_count += 1

                remaining = len(char_ids_in_cluster) - labeled_count - skipped_count

                if remaining == 0 and skipped_count > 0 and labeled_count == 0:
                    expected = "skipped"
                elif remaining == 0 and labeled_count > 0:
                    expected = "labeled"
                elif labeled_count > 0 or skipped_count > 0:
                    expected = "partial"
                else:
                    expected = "unlabeled"

                static_status = labels.get(cluster_id, {}).get("status", "unlabeled")
                assert static_status == expected, (
                    f"轮次{round_dir.name} 聚类{cluster_id}: "
                    f"labels.json状态='{static_status}', 动态计算='{expected}' "
                    f"(labeled={labeled_count}, skipped={skipped_count}, total={len(char_ids_in_cluster)})"
                )
