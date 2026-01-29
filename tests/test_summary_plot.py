# tests/test_summary_plot.py
from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless backend for CI / pytest
import matplotlib.pyplot as plt

from centrodip.bedtable import BedTable, IntervalRecord

# Adjust this import to your actual module path
import centrodip.summary_plot as sp


def _bt(records):
    return BedTable(records, inferred_kind="bed", inferred_ncols=6)


def _rec(
    chrom: str,
    start: int,
    end: int,
    name: str | None = None,
    score: int | None = None,
    strand: str | None = ".",
    extras=None,
):
    return IntervalRecord(chrom=chrom, start=start, end=end, name=name, score=score, strand=strand, extras=extras)


# -----------------------
# Pure helper unit tests
# -----------------------

def test_normalise_interval_orders_and_handles_zero_len():
    left, right = sp._normalise_interval(10, 5)
    assert left == 5.0
    assert right == 10.0

    left, right = sp._normalise_interval(7, 7)
    assert left < 7.0
    assert right > 7.0
    assert pytest.approx(right - left) == 1.0  # widened by 0.5 each side


def test_position_edges_none_and_empty():
    assert sp._position_edges(None).size == 0
    assert sp._position_edges([]).size == 0


def test_position_edges_all_nan_returns_empty():
    out = sp._position_edges([np.nan, np.nan])
    assert out.size == 0


def test_position_edges_single_value():
    out = sp._position_edges([100.0])
    assert np.allclose(out, np.array([99.5, 100.5]))


def test_position_edges_multiple_values_sorted_and_edges():
    # positions are unsorted on purpose
    out = sp._position_edges([20.0, 10.0, 30.0])
    # ordered: 10,20,30 -> edges: [5,15,25,35]? Actually:
    # deltas=[10,10], left_edge=10-5=5, midpoints=[15,25], right_edge=30+5=35
    assert np.allclose(out, np.array([5.0, 15.0, 25.0, 35.0]))


def test_bt_filter_chrom_returns_empty_bedtable_if_no_hits():
    bt = _bt([_rec("chr1", 0, 10), _rec("chr2", 0, 10)])
    sub = sp._bt_filter_chrom(bt, "chr3")
    assert isinstance(sub, BedTable)
    assert len(sub._records) == 0


def test_bt_filter_chrom_filters_correctly():
    bt = _bt([_rec("chr1", 0, 10), _rec("chr2", 0, 10), _rec("chr1", 20, 30)])
    sub = sp._bt_filter_chrom(bt, "chr1")
    assert [r.start for r in sub._records] == [0, 20]


@pytest.mark.parametrize("x_mode,expected", [
    ("start", [0, 20]),
    ("midpoint", [5, 25]),
])
def test_bt_positions(x_mode, expected):
    bt = _bt([_rec("chr1", 0, 10), _rec("chr1", 20, 30)])
    out = sp._bt_positions(bt, x_mode=x_mode)
    assert out.tolist() == expected


def test_bt_positions_bad_mode_raises():
    bt = _bt([_rec("chr1", 0, 10)])
    with pytest.raises(ValueError):
        sp._bt_positions(bt, x_mode="nope")


def test_lowess_smoothed_from_bedgraph_reads_extras0_and_nan_when_missing():
    low = BedTable(
        [
            _rec("chr1", 0, 1, extras=(1.0, 0.1)),
            _rec("chr1", 1, 2, extras=None),
            _rec("chr1", 2, 3, extras=(3.0, 0.3)),
        ],
        inferred_kind="bedgraph",
        inferred_ncols=5,
    )
    xs, ys = sp._lowess_smoothed_from_bedgraph(low)
    assert xs.tolist() == [0, 1, 2]
    assert np.isfinite(ys[0])
    assert np.isnan(ys[1])
    assert ys[2] == 3.0


# -----------------------------------------
# _bt_get_col_1based: both code paths
# -----------------------------------------

def test_bt_get_col_1based_uses_method_if_present(monkeypatch):
    bt = _bt([_rec("chr1", 0, 10, score=5), _rec("chr1", 20, 30, score=7)])

    def fake_get_column_1based(self, col, as_float=True):
        assert col == 5
        return [111, 222]

    monkeypatch.setattr(BedTable, "get_column_1based", fake_get_column_1based, raising=False)

    out = sp._bt_get_col_1based(bt, 5, as_float=True)
    assert out.tolist() == [111.0, 222.0]


def test_bt_get_col_1based_fallback_reads_bed_and_extras(monkeypatch):
    # Remove get_column_1based if it exists so we test fallback
    if hasattr(BedTable, "get_column_1based"):
        monkeypatch.delattr(BedTable, "get_column_1based", raising=False)

    bt = _bt([
        _rec("chr1", 0, 10, name="a", score=5, strand="+", extras=("xtra1", 3.14)),
        _rec("chr1", 20, 30, name="b", score=7, strand="-", extras=None),
    ])

    col1 = sp._bt_get_col_1based(bt, 1, as_float=False)
    assert col1.tolist() == ["chr1", "chr1"]

    col2 = sp._bt_get_col_1based(bt, 2, as_float=True)
    assert col2.tolist() == [0.0, 20.0]

    # extras[0] is column 7
    col7 = sp._bt_get_col_1based(bt, 7, as_float=False)
    assert col7[0] == "xtra1"
    assert col7[1] is None  # missing extras -> None when as_float=False

    # extras[1] is column 8
    col8 = sp._bt_get_col_1based(bt, 8, as_float=True)
    assert col8[0] == pytest.approx(3.14)
    assert np.isnan(col8[1])  # missing -> NaN when as_float=True


# -----------------------
# Plotting smoke tests
# -----------------------

def test_add_track_legends_adds_legend_and_does_not_crash():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim(0.0, 4.2)

    norm = sp.Normalize(vmin=0, vmax=10, clip=True)
    sp._add_track_legends(ax, norm)

    # legend is created
    leg = ax.get_legend()
    assert leg is not None

    plt.close(fig)


def test_centrodipSummaryPlot_bedtable_writes_file(tmp_path: Path):
    # Minimal, single-chrom input that exercises coverage band + lines + regions
    bedMethyl = BedTable(
        [
            # extras need to be long enough for cov_col_1based=10 and frac_mod_col_1based=11
            _rec("chr1", 0, 1, extras=(None, None, None, None,  # 4
                                      None, None, None, None,  # 8
                                      5.0, 25.0)),             # extras[8]=cov, extras[9]=frac_mod
            _rec("chr1", 500_000, 500_001, extras=(None, None, None, None,
                                                  None, None, None, None,
                                                  10.0, 50.0)),
            _rec("chr1", 1_000_000, 1_000_001, extras=(None, None, None, None,
                                                      None, None, None, None,
                                                      3.0, 75.0)),
        ],
        inferred_kind="bed",
        inferred_ncols=11,
    )

    regions = BedTable([_rec("chr1", 100_000, 200_000)], inferred_kind="bed", inferred_ncols=6)

    lowess_bg = BedTable(
        [
            _rec("chr1", 0, 1, extras=(30.0, 0.0)),
            _rec("chr1", 500_000, 500_001, extras=(55.0, 0.0)),
            _rec("chr1", 1_000_000, 1_000_001, extras=(70.0, 0.0)),
        ],
        inferred_kind="bedgraph",
        inferred_ncols=5,
    )

    dips_final = BedTable([_rec("chr1", 600_000, 800_000, score=500)], inferred_kind="bed", inferred_ncols=6)
    dips_unfiltered = BedTable([_rec("chr1", 550_000, 850_000, score=100)], inferred_kind="bed", inferred_ncols=6)

    out = tmp_path / "plots" / "summary.png"
    written = sp.centrodipSummaryPlot_bedtable(
        bedMethyl=bedMethyl,
        regions=regions,
        lowess_bg=lowess_bg,
        dips_final=dips_final,
        dips_unfiltered=dips_unfiltered,
        output_path=out,
    )

    assert Path(written).exists()
    assert Path(written).stat().st_size > 0
