import numpy as np
import pytest

from centrodip.bedtable import BedTable, IntervalRecord
from centrodip.detect_dips import (
    detectDips,
    find_dip_centers,
    estimate_bkgrd_median,
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
# estimate_bkgrd_median
# -------------------------

def test_estimate_background_from_masked_empty():
    out = estimate_bkgrd_median(np.array([]), masked_regions=[])
    assert np.isnan(out)


def test_estimate_bkgrd_median_from_masked_masks_and_computes_stats():
    sm = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0], dtype=float)
    # Mask the low region indices [3..5]
    out = estimate_bkgrd_median(sm, masked_regions=[(3, 5)])
    # background values are the 1.0s only => median 1.0
    assert out == pytest.approx(1.0)


def test_estimate_bkgrd_median_from_masked_all_masked_falls_back_to_all_finite():
    sm = np.array([1.0, 2.0, 3.0], dtype=float)
    out = estimate_bkgrd_median(sm, masked_regions=[(0, 2)])

    # fallback to all finite points, so n_bg should be 3
    assert out == pytest.approx(2.0)


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
            score_sensitivity=0.5,
            broadness=0.5,
            enrichment=False,
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
            score_sensitivity=0.5,
            broadness=0.5,
            enrichment=False,
            label="CDR",
            color="0,0,0",
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
        score_sensitivity=0.5,
        broadness=0.5,
        enrichment=False,
        label="CDR",
        color="0,0,0",            
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
        score_sensitivity=0.5,
        broadness=0.5,
        enrichment=False,
        label="CDR",
        color="0,0,0",            
    )

    # The key property: after merging, you should have fewer or equal intervals than centers.
    assert len(idxs) <= len(centers)
    # If they overlap, we expect a single merged idx interval:
    assert len(idxs) == 1

    recs = list(dips_bt)
    assert len(recs) == 1
    r = recs[0]
    assert r.start <= r.end


# -------------------------
# detectDips (integration-ish)
# -------------------------


def test_detectDips_empty_returns_empty_like_original():
    # Your detectDips currently returns early with dict+[] (legacy behavior).
    out = detectDips(
        chrom="chr1",
        bedgraph=BedTable([], inferred_kind="bedgraph", inferred_ncols=5),
        prominence=0.1,
        height=0.1,
        broadness=0.5,
        score_sensitivity=0.5,
        enrichment=False,
    )
    # Don't over-constrain, just check "emptiness" shape.
    dip_regions, bg_stats = out
    assert isinstance(bg_stats, list) or isinstance(bg_stats, dict)
    assert (dip_regions == {"starts": [], "ends": []}) or (isinstance(dip_regions, BedTable) and len(list(dip_regions)) == 0)


def test_detectDips_single_dip_finds_region_and_bg_stats(bedgraph_single_dip_chr1: BedTable):
    dips_bt, bg = detectDips(
        chrom="chr1",
        bedgraph=bedgraph_single_dip_chr1,
        prominence=0.1,
        height=0.1,
        broadness=0.5,
        score_sensitivity=0.5,
        enrichment=False,
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
