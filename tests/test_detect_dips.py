import numpy as np
import pytest

from centrodip.bedtable import BedTable, IntervalRecord
from centrodip.detect_dips import (
    detectDips,
    find_dip_centers,
    find_simple_edges,
    estimate_background_from_masked,
    find_edges,
)


def _bg_rec(chrom: str, start: int, end: int, sm: float, dy: float = 0.0) -> IntervalRecord:
    # LOWESS-output bedGraph-like: extras=(smoothedY, dY)
    return IntervalRecord(
        chrom=chrom,
        start=start,
        end=end,
        name=None,
        score=None,
        strand=None,
        extras=(float(sm), float(dy)),
    )


@pytest.fixture
def bedgraph_single_dip_chr1() -> BedTable:
    # Simple "valley" shape dip around idx=5
    # smoothed:  1.0 1.0 1.0 0.8 0.5 0.2 0.5 0.8 1.0 1.0 1.0
    sm = [1.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0]
    recs = [_bg_rec("chr1", i * 100, i * 100 + 1, v, 0.0) for i, v in enumerate(sm)]
    return BedTable(recs, inferred_kind="bedgraph", inferred_ncols=5)


@pytest.fixture
def bedgraph_two_chroms() -> BedTable:
    # Not ideal for detectDips (expects per chrom), but should not crash.
    recs = []
    for i in range(10):
        recs.append(_bg_rec("chr1", i * 100, i * 100 + 1, 1.0))
    for i in range(10):
        recs.append(_bg_rec("chr2", i * 100, i * 100 + 1, 1.0))
    return BedTable(recs, inferred_kind="bedgraph", inferred_ncols=5)


# -------------------------
# find_dip_centers
# -------------------------

def test_find_dip_centers_empty():
    centers = find_dip_centers(np.array([]), prominence=0.25, height=0.1, enrichment=False)
    assert centers.dtype == int
    assert centers.size == 0


def test_find_dip_centers_detects_single_dip():
    sm = np.array([1, 1, 1, 0.8, 0.5, 0.2, 0.5, 0.8, 1, 1, 1], dtype=float)
    centers = find_dip_centers(sm, prominence=0.1, height=0.1, enrichment=False)

    # For a single symmetric valley, center should be around index 5.
    assert centers.size >= 1
    assert int(centers[np.argmin(np.abs(centers - 5))]) == 5


def test_find_dip_centers_detects_single_peak_when_enrichment_true():
    # In enrichment mode, we detect peaks (high values).
    sm = np.array([0.2, 0.5, 0.8, 1.0, 0.8, 0.5, 0.2], dtype=float)
    centers = find_dip_centers(sm, prominence=0.1, height=0.1, enrichment=True)

    assert centers.size >= 1
    # peak at index 3
    assert int(centers[np.argmin(np.abs(centers - 3))]) == 3


# -------------------------
# find_simple_edges
# -------------------------

def test_find_simple_edges_empty_returns_empty():
    bt, idxs = find_simple_edges(
        chrom_for_output="chr1",
        data=np.array([]),
        positions=np.array([]),
        bounding_threshold=0.5,
        centers=np.array([1, 2], dtype=int),
    )
    assert isinstance(bt, BedTable)
    assert len(list(bt)) == 0
    assert idxs == []


def test_find_simple_edges_returns_expected_interval():
    # One dip center at idx=5, threshold=1.0 means edges are the first >=1.0 on either side.
    data = np.array([1, 1, 1, 0.8, 0.5, 0.2, 0.5, 0.8, 1, 1, 1], dtype=float)
    positions = np.arange(data.size) * 100

    bt, idxs = find_simple_edges(
        chrom_for_output="chr1",
        data=data,
        positions=positions,
        bounding_threshold=1.0,
        centers=np.array([5], dtype=int),
    )

    # left edge should be at idx=2 (data[2]=1) scanning from 4 down; first hit is 2
    # right edge should be at idx=8 (data[8]=1) scanning from 6 up; first hit is 8
    assert idxs == [(2, 8)]

    recs = list(bt)
    assert len(recs) == 1
    r = recs[0]
    assert r.chrom == "chr1"
    assert r.start == int(positions[2])
    assert r.end == int(positions[8])
    assert r.name == "simpleDip_1"


def test_find_simple_edges_deduplicates():
    data = np.array([1, 1, 1, 0.8, 0.5, 0.2, 0.5, 0.8, 1, 1, 1], dtype=float)
    positions = np.arange(data.size) * 100
    centers = np.array([5, 5, 5], dtype=int)

    bt, idxs = find_simple_edges(
        chrom_for_output="chr1",
        data=data,
        positions=positions,
        bounding_threshold=1.0,
        centers=centers,
    )

    assert idxs == [(2, 8)]
    assert len(list(bt)) == 1


# -------------------------
# estimate_background_from_masked
# -------------------------

def test_estimate_background_from_masked_empty():
    out = estimate_background_from_masked(np.array([]), masked_regions=[])
    assert np.isnan(out["median"])
    assert out["values"].size == 0
    assert out["mask"].size == 0


def test_estimate_background_from_masked_masks_and_computes_stats():
    sm = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0], dtype=float)
    # Mask the low region indices [3..5]
    out = estimate_background_from_masked(sm, masked_regions=[(3, 5)])

    assert out["n_total"] == 8
    assert out["n_masked"] == 3
    assert out["n_bg"] == 5
    # background values are the 1.0s only => median 1.0
    assert out["median"] == pytest.approx(1.0)


def test_estimate_background_from_masked_all_masked_falls_back_to_all_finite():
    sm = np.array([1.0, 2.0, 3.0], dtype=float)
    out = estimate_background_from_masked(sm, masked_regions=[(0, 2)])

    # fallback to all finite points, so n_bg should be 3
    assert out["n_bg"] == 3
    assert out["median"] == pytest.approx(2.0)


# -------------------------
# find_edges (half-depth) + merging
# -------------------------

def test_find_edges_validates_shapes_and_background():
    sm = np.array([1.0, 0.5, 1.0], dtype=float)
    pos = np.array([0, 100, 200], dtype=int)

    with pytest.raises(ValueError, match="same length"):
        find_edges(
            chrom="chr1",
            smoothed=sm,
            positions=np.array([0, 100], dtype=int),
            background_median=1.0,
            dip_center_idxs=np.array([1], dtype=int),
            label="CDR",
            color="0,0,0",
        )

    with pytest.raises(ValueError, match="background_median must be finite"):
        find_edges(
            chrom="chr1",
            smoothed=sm,
            positions=pos,
            background_median=np.nan,
            dip_center_idxs=np.array([1], dtype=int),
            label="CDR",
            color="0,0,0",
        )

    with pytest.raises(ValueError, match="k_consecutive must be"):
        find_edges(
            chrom="chr1",
            smoothed=sm,
            positions=pos,
            background_median=1.0,
            dip_center_idxs=np.array([1], dtype=int),
            label="CDR",
            color="0,0,0",
            k_consecutive=0,
        )


def test_find_edges_produces_bed_and_scores_in_range():
    sm = np.array([1.0, 1.0, 0.8, 0.3, 0.8, 1.0, 1.0], dtype=float)
    pos = np.arange(sm.size) * 100
    centers = np.array([3], dtype=int)

    dips_bt, idxs = find_edges(
        chrom="chr1",
        smoothed=sm,
        positions=pos,
        background_median=1.0,
        dip_center_idxs=centers,
        label="CDR",
        color="50,50,255",
        width=0.5,
    )

    recs = list(dips_bt)
    assert len(recs) >= 1
    for r in recs:
        assert r.chrom == "chr1"
        assert r.name == "CDR"
        assert 0 <= int(r.score) <= 1000
        assert r.extras is not None
        assert len(r.extras) == 3  # (start, end, color)


def test_find_edges_merges_overlapping_intervals():
    # Construct a signal with two nearby "centers" whose halfpoint windows overlap
    sm = np.array([1.0, 0.9, 0.4, 0.2, 0.4, 0.2, 0.4, 0.9, 1.0], dtype=float)
    pos = np.arange(sm.size) * 100
    centers = np.array([3, 5], dtype=int)

    dips_bt, idxs = find_edges(
        chrom="chr1",
        smoothed=sm,
        positions=pos,
        background_median=1.0,
        dip_center_idxs=centers,
        label="CDR",
        color="50,50,255",
        width=0.5,
        k_consecutive=1,
    )

    # The key property: after merging, you should have fewer or equal intervals than centers.
    assert len(idxs) <= len(centers)
    # If they overlap, we expect a single merged idx interval:
    assert len(idxs) == 1

    recs = list(dips_bt)
    assert len(recs) == 1
    r = recs[0]
    assert r.start <= r.end


def test_find_edges_end_inclusive_false_extends_end_by_one():
    sm = np.array([1.0, 1.0, 0.2, 1.0, 1.0], dtype=float)
    pos = np.array([0, 100, 200, 300, 400], dtype=int)
    centers = np.array([2], dtype=int)

    dips_bt_true, _ = find_edges(
        chrom="chr1",
        smoothed=sm,
        positions=pos,
        background_median=1.0,
        dip_center_idxs=centers,
        label="CDR",
        color="x",
        end_inclusive=True,
    )
    dips_bt_false, _ = find_edges(
        chrom="chr1",
        smoothed=sm,
        positions=pos,
        background_median=1.0,
        dip_center_idxs=centers,
        label="CDR",
        color="x",
        end_inclusive=False,
    )

    r_true = list(dips_bt_true)[0]
    r_false = list(dips_bt_false)[0]
    assert r_false.end == r_true.end + 1


# -------------------------
# detectDips (integration-ish)
# -------------------------

def test_detectDips_bad_x_mode_raises(bedgraph_single_dip_chr1: BedTable):
    with pytest.raises(ValueError, match="x_mode must be"):
        detectDips(
            bedgraph_single_dip_chr1,
            prominence=0.1,
            height=0.1,
            enrichment=False,
            broadness=0.5,
            x_mode="nope",
        )


def test_detectDips_empty_returns_empty_like_original():
    # Your detectDips currently returns early with dict+[] (legacy behavior).
    out = detectDips(
        BedTable([], inferred_kind="bedgraph", inferred_ncols=5),
        prominence=0.1,
        height=0.1,
        enrichment=False,
        broadness=0.5,
    )
    # Don't over-constrain, just check "emptiness" shape.
    dip_regions, bg_stats = out
    assert isinstance(bg_stats, list) or isinstance(bg_stats, dict)
    assert (dip_regions == {"starts": [], "ends": []}) or (isinstance(dip_regions, BedTable) and len(list(dip_regions)) == 0)


def test_detectDips_single_dip_finds_region_and_bg_stats(bedgraph_single_dip_chr1: BedTable):
    dips_bt, bg = detectDips(
        bedgraph_single_dip_chr1,
        prominence=0.1,
        height=0.1,
        enrichment=False,
        broadness=0.5,
        label="CDR",
        color="50,50,255",
    )

    assert isinstance(dips_bt, BedTable)
    assert isinstance(bg, dict)
    assert "median" in bg

    recs = list(dips_bt)
    # For this synthetic dip, expect at least one dip call.
    assert len(recs) >= 1
    for r in recs:
        assert r.chrom == "chr1"
        assert r.name == "CDR"
        assert 0 <= int(r.score) <= 1000
