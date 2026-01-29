import math
import numpy as np
import pytest

from centrodip.bedtable import BedTable, IntervalRecord
from centrodip.bedmethyl_smooth import bedMethyl_LOWESS


# Helper: build a "bedMethyl-like" record where:
#   cov_col_1based=10 => extras[3]
#   y_col_1based=11   => extras[4]
def _rec(chrom: str, start: int, end: int, cov: float, y: float) -> IntervalRecord:
    extras = (0.0, 0.0, 0.0, cov, y)  # len>=5 so col10/11 exist
    return IntervalRecord(chrom=chrom, start=start, end=end, name="x", score=0, strand=".", extras=extras)


@pytest.fixture
def bt_bedmethyl_linear_start() -> BedTable:
    # Linear in "start": y = 2*start + 1
    recs = [
        _rec("chr1", 0, 10, 10.0, 1.0),
        _rec("chr1", 10, 20, 10.0, 21.0),
        _rec("chr1", 20, 30, 10.0, 41.0),
        _rec("chr1", 30, 40, 10.0, 61.0),
        _rec("chr1", 40, 50, 10.0, 81.0),
    ]
    return BedTable(recs, inferred_kind="bed", inferred_ncols=11)


@pytest.fixture
def bt_bedmethyl_linear_midpoint() -> BedTable:
    # Linear in midpoint: y = 3*mid + 2, with varying intervals
    recs = [
        _rec("chr1", 0, 10, 10.0, 3.0 * 5.0 + 2.0),     # mid=5
        _rec("chr1", 10, 30, 10.0, 3.0 * 20.0 + 2.0),   # mid=20
        _rec("chr1", 30, 40, 10.0, 3.0 * 35.0 + 2.0),   # mid=35
        _rec("chr1", 40, 80, 10.0, 3.0 * 60.0 + 2.0),   # mid=60
        _rec("chr1", 80, 90, 10.0, 3.0 * 85.0 + 2.0),   # mid=85
    ]
    return BedTable(recs, inferred_kind="bed", inferred_ncols=11)


def test_empty_input_returns_empty_bedgraph():
    out = bedMethyl_LOWESS(
        BedTable([], inferred_kind="bed", inferred_ncols=11),
        window_bp=1000,
        cov_conf=1.0,
    )
    assert isinstance(out, BedTable)
    assert len(list(out)) == 0
    assert out.inferred_kind == "bedgraph"
    assert out.inferred_ncols == 5


def test_invalid_x_mode_raises(bt_bedmethyl_linear_start: BedTable):
    with pytest.raises(ValueError, match="x_mode must be"):
        bedMethyl_LOWESS(
            bt_bedmethyl_linear_start,
            window_bp=1000,
            cov_conf=1.0,
            x_mode="banana",
        )


def test_cov_conf_must_be_positive(bt_bedmethyl_linear_start: BedTable):
    with pytest.raises(ValueError, match="cov_conf must be greater than 0"):
        bedMethyl_LOWESS(bt_bedmethyl_linear_start, window_bp=1000, cov_conf=0.0)

    with pytest.raises(ValueError, match="cov_conf must be greater than 0"):
        bedMethyl_LOWESS(bt_bedmethyl_linear_start, window_bp=1000, cov_conf=-5.0)


def test_negative_coverage_raises():
    bt = BedTable([_rec("chr1", 0, 10, -1.0, 0.5)], inferred_kind="bed", inferred_ncols=11)
    with pytest.raises(ValueError, match="coverage must be non-negative"):
        bedMethyl_LOWESS(bt, window_bp=1000, cov_conf=1.0)


def test_output_preserves_order_and_has_two_extras(bt_bedmethyl_linear_start: BedTable):
    out = bedMethyl_LOWESS(
        bt_bedmethyl_linear_start,
        window_bp=1_000_000,  # include all points
        cov_conf=1.0,
        x_mode="start",
        drop_nonfinite=True,
    )

    out_recs = list(out)
    in_recs = list(bt_bedmethyl_linear_start)

    assert len(out_recs) == len(in_recs)

    for in_r, out_r in zip(in_recs, out_recs):
        assert (out_r.chrom, out_r.start, out_r.end) == (in_r.chrom, in_r.start, in_r.end)
        assert out_r.name is None
        assert out_r.score is None
        assert out_r.strand is None
        assert out_r.extras is not None
        assert len(out_r.extras) == 2
        sm, dy = out_r.extras
        assert isinstance(sm, float)
        assert isinstance(dy, float)


def test_linear_start_recovers_slope_and_values(bt_bedmethyl_linear_start: BedTable):
    # y = 2*x + 1; with a huge window, weighted LS should recover slope ~2 everywhere
    out = bedMethyl_LOWESS(
        bt_bedmethyl_linear_start,
        window_bp=1_000_000,
        cov_conf=1.0,
        x_mode="start",
        drop_nonfinite=True,
    )
    out_recs = list(out)
    xs = np.array([r.start for r in bt_bedmethyl_linear_start], dtype=float)
    expected_y = 2.0 * xs + 1.0

    smoothed = np.array([r.extras[0] for r in out_recs], dtype=float)
    slopes = np.array([r.extras[1] for r in out_recs], dtype=float)

    assert np.allclose(smoothed, expected_y, atol=1e-8, rtol=0)
    assert np.allclose(slopes, 2.0, atol=1e-8, rtol=0)


def test_linear_midpoint_recovers_slope(bt_bedmethyl_linear_midpoint: BedTable):
    # y = 3*mid + 2; test x_mode="midpoint" slope ~3
    out = bedMethyl_LOWESS(
        bt_bedmethyl_linear_midpoint,
        window_bp=1_000_000,
        cov_conf=1.0,
        x_mode="midpoint",
        drop_nonfinite=True,
    )
    slopes = np.array([r.extras[1] for r in out], dtype=float)
    assert np.allclose(slopes, 3.0, atol=1e-8, rtol=0)


def test_drop_nonfinite_true_drops_nan_rows():
    recs = [
        _rec("chr1", 0, 10, 10.0, 1.0),
        _rec("chr1", 10, 20, 10.0, np.nan),
        _rec("chr1", 20, 30, 10.0, 3.0),
    ]
    bt = BedTable(recs, inferred_kind="bed", inferred_ncols=11)

    out = bedMethyl_LOWESS(bt, window_bp=1000, cov_conf=1.0, drop_nonfinite=True)
    out_recs = list(out)

    assert len(out_recs) == 2
    assert [r.start for r in out_recs] == [0, 20]
    assert all(np.isfinite(r.extras[0]) and np.isfinite(r.extras[1]) for r in out_recs)


def test_drop_nonfinite_false_keeps_nan_rows_with_nan_extras():
    recs = [
        _rec("chr1", 0, 10, 10.0, 1.0),
        _rec("chr1", 10, 20, 10.0, np.nan),
        _rec("chr1", 20, 30, 10.0, 3.0),
    ]
    bt = BedTable(recs, inferred_kind="bed", inferred_ncols=11)

    out = bedMethyl_LOWESS(bt, window_bp=1000, cov_conf=1.0, drop_nonfinite=False)
    out_recs = list(out)

    assert len(out_recs) == 3
    assert [r.start for r in out_recs] == [0, 10, 20]

    # middle row should be NaNs in extras
    sm, dy = out_recs[1].extras
    assert np.isnan(sm)
    assert np.isnan(dy)


def test_window_bp_tiny_still_returns_outputs(bt_bedmethyl_linear_start: BedTable):
    # window so small many points may fall into the "m.sum() < 2" fallback
    out = bedMethyl_LOWESS(
        bt_bedmethyl_linear_start,
        window_bp=1,     # extremely small
        cov_conf=1.0,
        drop_nonfinite=True,
    )
    out_recs = list(out)
    assert len(out_recs) == len(list(bt_bedmethyl_linear_start))
    # should still have numeric extras
    assert all(np.isfinite(r.extras[0]) and np.isfinite(r.extras[1]) for r in out_recs)
