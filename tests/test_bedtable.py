# tests/test_bedtable.py
#
# Run:
#   pytest -q
#
# Notes:
# - Adjust the import line to match your package layout:
#     from centrodip.bedtable import BedTable, IntervalRecord
#   or:
#     from bedtable import BedTable, IntervalRecord
#
# - These tests assume your BedTable supports:
#     - BedTable([...], inferred_kind=..., inferred_ncols=...)
#     - BedTable.from_path(path)
#     - bt.to_path(path)
#     - iteration over records (or bt._records)
#     - len(bt)
#     - groupby_chrom() -> iterable of (chrom, BedTable_for_chrom) OR dict-like (we handle both)
#     - IntervalRecord fields: chrom,start,end,name,score,strand,extras + .length + .overlaps(...)
#     - optional: bt.get_column_1based(col, as_float=...) (tests skip if missing)
#
# If any of these differ, tell me your actual API and I’ll tailor the tests exactly.

from __future__ import annotations

from pathlib import Path
import math
import pytest

# --- EDIT THIS IMPORT ---
from centrodip.bedtable import BedTable, IntervalRecord  # noqa: F401


def _records(bt: BedTable):
    """Best-effort to get records list from a BedTable."""
    if hasattr(bt, "_records"):
        return list(bt._records)
    return list(bt)


def _as_groupby_items(grouped):
    """
    groupby_chrom() varies by implementation; normalize to iterable of (chrom, BedTable).
    Supports:
      - dict-like .items()
      - list/iterable of tuples
    """
    if hasattr(grouped, "items"):
        return list(grouped.items())
    return list(grouped)


@pytest.fixture
def bt_bed() -> BedTable:
    recs = [
        IntervalRecord("chr1", 10, 20, "a", 100, ".", extras=("x",)),
        IntervalRecord("chr1", 30, 40, "b", 200, ".", extras=None),
        IntervalRecord("chr2", 5, 15, "c", 0, ".", extras=(1, 2, 3)),
    ]
    return BedTable(recs, inferred_kind="bed", inferred_ncols=6)


@pytest.fixture
def bt_bedgraph() -> BedTable:
    # Typical bedGraph: chrom, start, end, value (often stored as score or extras)
    recs = [
        IntervalRecord("chr1", 0, 1, ".", 0, ".", extras=(1.0,)),
        IntervalRecord("chr1", 1, 2, ".", 0, ".", extras=(2.5,)),
        IntervalRecord("chr2", 0, 1, ".", 0, ".", extras=(0.0,)),
    ]
    return BedTable(recs, inferred_kind="bedgraph", inferred_ncols=4)


def test_intervalrecord_length(bt_bed: BedTable):
    recs = _records(bt_bed)
    assert recs[0].length == 10
    assert recs[1].length == 10
    assert recs[2].length == 10


def test_intervalrecord_overlap_semantics():
    r = IntervalRecord("chr1", 10, 20, "x", 0, ".")
    # Overlap (same chrom, overlapping coords)
    assert r.overlaps("chr1", 15, 25) is True
    # No overlap
    assert r.overlaps("chr1", 21, 30) is False
    # Different chrom
    assert r.overlaps("chr2", 15, 25) is False

    # Edge-case: end-touching behavior depends on your convention.
    # If you're using half-open BED [start,end), then [10,20) and [20,30) DO NOT overlap.
    # If using closed intervals, they do.
    # We accept either, but ensure it returns a bool.
    v = r.overlaps("chr1", 20, 30)
    assert isinstance(v, bool)


def test_bedtable_len_and_iter(bt_bed: BedTable):
    assert len(bt_bed) == 3
    assert len(list(bt_bed)) == 3


def test_bedtable_preserves_inferred_kind(bt_bed: BedTable, bt_bedgraph: BedTable):
    assert getattr(bt_bed, "inferred_kind", None) in ("bed", None) or bt_bed.inferred_kind == "bed"
    assert getattr(bt_bedgraph, "inferred_kind", None) in ("bedgraph", None) or bt_bedgraph.inferred_kind == "bedgraph"


def test_groupby_chrom_splits(bt_bed: BedTable):
    grouped = bt_bed.groupby_chrom()
    items = _as_groupby_items(grouped)
    chroms = {c for c, _ in items}
    assert chroms == {"chr1", "chr2"}

    chr1_bt = next(bt for c, bt in items if c == "chr1")
    chr2_bt = next(bt for c, bt in items if c == "chr2")
    assert len(chr1_bt) == 2
    assert len(chr2_bt) == 1
    assert all(r.chrom == "chr1" for r in _records(chr1_bt))
    assert all(r.chrom == "chr2" for r in _records(chr2_bt))


def test_groupby_chrom_sorted_optional(bt_bed: BedTable):
    # Not all implementations sort; this is a "soft" test: if it is sorted, great.
    # If not, we still ensure correctness.
    grouped = _as_groupby_items(bt_bed.groupby_chrom())
    # Ensure each group preserves chrom membership
    for chrom, sub in grouped:
        assert all(r.chrom == chrom for r in _records(sub))


def test_from_path_to_path_roundtrip_bed(tmp_path: Path):
    p = tmp_path / "x.bed"
    text = "\n".join(
        [
            "chr1\t10\t20\ta\t100\t.",
            "chr1\t30\t40\tb\t200\t.",
            "chr2\t5\t15\tc\t0\t.",
        ]
    ) + "\n"
    p.write_text(text)

    bt = BedTable.from_path(str(p))
    assert len(bt) == 3

    outp = tmp_path / "y.bed"
    bt.to_path(str(outp))
    assert outp.exists()
    out_txt = outp.read_text().strip().splitlines()
    assert len(out_txt) == 3
    # We don't enforce exact formatting beyond tab-separated, but sanity-check first fields.
    first = out_txt[0].split("\t")
    assert first[0] == "chr1"
    assert int(first[1]) == 10
    assert int(first[2]) == 20


def test_from_path_handles_comments_and_blank_lines(tmp_path: Path):
    p = tmp_path / "x.bed"
    p.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "chr1\t0\t10\t.\t0\t.",
                "track name=foo",
                "chr1\t10\t20\t.\t0\t.",
                "",
            ]
        )
        + "\n"
    )
    bt = BedTable.from_path(p)
    assert len(bt) == 2
    recs = _records(bt)
    assert recs[0].chrom == "chr1"
    assert recs[0].start == 0
    assert recs[0].end == 10


def test_to_path_creates_parent_dirs(tmp_path: Path, bt_bed: BedTable):
    outp = tmp_path / "nested" / "dir" / "out.bed"
    bt_bed.to_path(outp)
    assert outp.exists()


def test_get_column_1based_basic(bt_bed: BedTable):
    if not hasattr(bt_bed, "get_column_1based"):
        pytest.skip("BedTable.get_column_1based not implemented; skipping.")
    chroms = bt_bed.get_column_1based(1, as_float=False)
    starts = bt_bed.get_column_1based(2, as_float=True)
    ends = bt_bed.get_column_1based(3, as_float=True)
    names = bt_bed.get_column_1based(4, as_float=False)
    scores = bt_bed.get_column_1based(5, as_float=True)
    strands = bt_bed.get_column_1based(6, as_float=False)

    assert list(chroms) == ["chr1", "chr1", "chr2"]
    assert list(map(int, starts)) == [10, 30, 5]
    assert list(map(int, ends)) == [20, 40, 15]
    assert list(names) == ["a", "b", "c"]
    assert list(map(int, scores)) == [100, 200, 0]
    assert list(strands) == [".", ".", "."]


def test_get_column_1based_extras(bt_bed: BedTable):
    if not hasattr(bt_bed, "get_column_1based"):
        pytest.skip("BedTable.get_column_1based not implemented; skipping.")
    # Column 7 should be extras[0] if you support it
    col7 = bt_bed.get_column_1based(7, as_float=False)
    # first record has extras=("x",), second None, third extras=(1,2,3)
    assert col7[0] == "x"
    assert col7[1] is None or (isinstance(col7[1], float) and math.isnan(col7[1]))
    assert str(col7[2]) in ("1", "1.0")  # depending on casting


def test_sorting_stability_optional(bt_bed: BedTable):
    # If you have a sort method, test it. Otherwise skip.
    if not hasattr(bt_bed, "sort"):
        pytest.skip("BedTable.sort not implemented; skipping.")
    s = bt_bed.sort()
    recs = _records(s)
    assert (recs[0].chrom, recs[0].start) <= (recs[1].chrom, recs[1].start) <= (recs[2].chrom, recs[2].start)


def test_empty_bedtable_roundtrip(tmp_path: Path):
    p = tmp_path / "empty.bed"
    p.write_text("")
    bt = BedTable.from_path(p)
    assert len(bt) == 0
    outp = tmp_path / "empty_out.bed"
    bt.to_path(outp)
    assert outp.exists()
    assert outp.read_text().strip() == ""


def test_intervalrecord_repr_or_str(bt_bed: BedTable):
    # Just ensure no crashes when printing records (helps debugging)
    for r in _records(bt_bed):
        _ = str(r)
        _ = repr(r)
