# tests/test_filter_dips.py
from __future__ import annotations

import pytest

from centrodip.bedtable import BedTable, IntervalRecord

# Import the functions under test.
# Adjust the module path if your file name differs (e.g. centrodip/filter_dips.py).
from centrodip.filter_dips import filterDips, clusterFilter


def _bt(*recs: IntervalRecord) -> BedTable:
    return BedTable(list(recs), inferred_kind="bed", inferred_ncols=6)


def _rec(chrom: str, start: int, end: int, score: int | None) -> IntervalRecord:
    return IntervalRecord(chrom=chrom, start=start, end=end, name="CDR", score=score, strand=".")


# ----------------------------
# filterDips tests
# ----------------------------

def test_filterDips_empty_returns_empty():
    dips = _bt()
    out = filterDips(dips, min_size=1000, min_score=100, cluster_distance=500_000)
    assert isinstance(out, BedTable)
    assert len(out._records) == 0


def test_filterDips_size_filter_removes_small():
    dips = _bt(
        _rec("chr1", 0, 500, 200),      # len 500 -> removed
        _rec("chr1", 1000, 2500, 200),  # len 1500 -> kept
    )
    out = filterDips(dips, min_size=1000, min_score=0, cluster_distance=-1)
    assert [(r.start, r.end) for r in out._records] == [(1000, 2500)]


def test_filterDips_score_filter_removes_low_and_none_scores():
    dips = _bt(
        _rec("chr1", 0, 2000, None),   # removed (None)
        _rec("chr1", 3000, 5000, 99),  # removed (<100)
        _rec("chr1", 6000, 9000, 100), # kept
    )
    out = filterDips(dips, min_size=0, min_score=100, cluster_distance=-1)
    assert [(r.start, r.end, r.score) for r in out._records] == [(6000, 9000, 100)]


def test_filterDips_clusterFilter_applied_after_other_filters():
    # Two clusters: (A) has higher mean score; (B) has more members but lower mean
    dips = _bt(
        _rec("chr1", 0, 1000, 800),        # cluster A
        _rec("chr1", 10_000, 11_000, 800), # cluster A
        _rec("chr1", 1_000_000, 1_001_000, 200),  # cluster B
        _rec("chr1", 1_010_000, 1_011_000, 200),  # cluster B
        _rec("chr1", 1_020_000, 1_021_000, 200),  # cluster B
    )
    out = filterDips(dips, min_size=0, min_score=0, cluster_distance=50_000)
    # Expect cluster A chosen (mean 800 > 200)
    assert [(r.start, r.end) for r in out._records] == [(0, 1000), (10_000, 11_000)]


# ----------------------------
# clusterFilter tests
# ----------------------------

def test_clusterFilter_negative_distance_returns_input_identity():
    dips = _bt(_rec("chr1", 0, 1000, 10), _rec("chr1", 2000, 3000, 20))
    out = clusterFilter(dips, -1)
    # Same object is fine but not required; content must match
    assert [(r.start, r.end, r.score) for r in out._records] == [(0, 1000, 10), (2000, 3000, 20)]


def test_clusterFilter_single_or_empty_returns_input():
    d0 = _bt()
    d1 = _bt(_rec("chr1", 0, 1000, 10))
    assert len(clusterFilter(d0, 1000)._records) == 0
    assert len(clusterFilter(d1, 1000)._records) == 1


def test_clusterFilter_selects_best_mean_score_cluster():
    # cluster A: two high scores
    # cluster B: three lower scores
    dips = _bt(
        _rec("chr1", 0, 1000, 900),
        _rec("chr1", 20_000, 21_000, 900),
        _rec("chr1", 1_000_000, 1_001_000, 200),
        _rec("chr1", 1_010_000, 1_011_000, 200),
        _rec("chr1", 1_020_000, 1_021_000, 200),
    )
    out = clusterFilter(dips, cluster_distance=50_000)
    assert [(r.start, r.end, r.score) for r in out._records] == [
        (0, 1000, 900),
        (20_000, 21_000, 900),
    ]


def test_clusterFilter_tie_breaker_prefers_more_members_then_span_then_earlier_start():
    # Make mean_score equal for two candidate clusters:
    # Cluster A: 2 members mean 500
    # Cluster B: 3 members mean 500 -> should win by count
    dips = _bt(
        # Cluster A around 0
        _rec("chr1", 0, 1000, 500),
        _rec("chr1", 10_000, 11_000, 500),
        # Cluster B around 1,000,000
        _rec("chr1", 1_000_000, 1_001_000, 500),
        _rec("chr1", 1_010_000, 1_011_000, 500),
        _rec("chr1", 1_020_000, 1_021_000, 500),
    )
    out = clusterFilter(dips, cluster_distance=50_000)
    assert [(r.start, r.end) for r in out._records] == [
        (1_000_000, 1_001_000),
        (1_010_000, 1_011_000),
        (1_020_000, 1_021_000),
    ]


def test_clusterFilter_require_min_members_filters_out_small_clusters():
    dips = _bt(
        _rec("chr1", 0, 1000, 900),          # isolated if distance small
        _rec("chr1", 1_000_000, 1_001_000, 100),
        _rec("chr1", 1_010_000, 1_011_000, 100),
    )
    # With distance=50k, second and third are a cluster of 2; first is cluster of 1.
    out = clusterFilter(dips, cluster_distance=50_000, require_min_members=3)
    assert len(out._records) == 0


def test_clusterFilter_sorts_input_by_start_before_clustering():
    # Intentionally unsorted input: cluster should still be found correctly.
    dips = _bt(
        _rec("chr1", 1_010_000, 1_011_000, 300),
        _rec("chr1", 0, 1000, 900),
        _rec("chr1", 1_000_000, 1_001_000, 300),
        _rec("chr1", 20_000, 21_000, 900),
    )
    out = clusterFilter(dips, cluster_distance=50_000)
    # Mean score cluster A = 900 (two members) vs cluster B = 300 (two members)
    assert [(r.start, r.end, r.score) for r in out._records] == [
        (0, 1000, 900),
        (20_000, 21_000, 900),
    ]
