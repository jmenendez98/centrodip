# tests/test_cli_flags.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# -----------------------------
# Inline (non-parallel) executor
# -----------------------------
class _InlineFuture:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._ran = False
        self._result = None
        self._exc = None

    def result(self, timeout=None):
        if not self._ran:
            self._ran = True
            try:
                self._result = self._fn(*self._args, **self._kwargs)
            except Exception as e:
                self._exc = e
        if self._exc is not None:
            raise self._exc
        return self._result


class _InlineExecutor:
    def __init__(self, *args, **kwargs):
        # accept max_workers=..., mp_context=..., etc.
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _InlineFuture(fn, *args, **kwargs)


@pytest.fixture
def mpl_agg(monkeypatch):
    # ensure --plot works in headless test runs
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    yield


@pytest.fixture
def run_main_no_parallel(monkeypatch):
    """
    Run centrodip.main.main() with:
      - sys.argv overridden
      - ProcessPoolExecutor replaced by an inline executor
      - as_completed replaced by identity iteration
    """
    import centrodip.main as cmain

    monkeypatch.setattr(cmain.concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(cmain.concurrent.futures, "as_completed", lambda futs: futs)

    def _run(argv):
        monkeypatch.setattr(sys, "argv", ["centrodip"] + list(map(str, argv)))
        cmain.main()

    return _run


# -----------------------------
# Tiny synthetic inputs (fast)
# -----------------------------
def _write_synthetic_bedmethyl(path: Path, chrom: str = "chr1", mod_code: str = "m") -> Path:
    """
    Write a minimal bedMethyl-like TSV with >= 11 columns so that:
      cov_col_1based=10 and y_col_1based=11 exist.

    Column 4 is mod-code in YOUR CLI's intended meaning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 21 points, 1kb step
    starts = [i * 1000 for i in range(21)]
    ends = [s + 1 for s in starts]

    # strong dip in middle (raw frac mod)
    frac = []
    for i in range(21):
        frac.append(10.0 if 8 <= i <= 12 else 80.0)

    cov = [10] * 21

    lines = []
    for s, e, c, y in zip(starts, ends, cov, frac):
        fields = [
            chrom,          # 1
            str(s),         # 2
            str(e),         # 3
            mod_code,       # 4 (this is what --mod-code should match)
            "0",            # 5 score
            ".",            # 6 strand
            ".",            # 7
            ".",            # 8
            ".",            # 9
            str(c),         # 10 coverage
            str(y),         # 11 frac_mod percent
        ]
        lines.append("\t".join(fields))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_synthetic_bedgraph4(path: Path, chrom: str = "chr1") -> Path:
    """
    bedGraph4 format:
      chrom start end value
    Here value is fraction modified (0..100) in your pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    starts = [i * 1000 for i in range(21)]
    ends = [s + 1 for s in starts]

    # dip
    vals = []
    for i in range(21):
        vals.append(10.0 if 8 <= i <= 12 else 80.0)

    lines = [f"{chrom}\t{s}\t{e}\t{v}" for s, e, v in zip(starts, ends, vals)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_regions(path: Path, chrom: str = "chr1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{chrom}\t0\t25000\tREGION\t0\t.\n", encoding="utf-8")
    return path


def _run_cli_bedmethyl(run_main_no_parallel, tmp_path: Path, extra_args: tuple[str, ...] = ()) -> Path:
    bedmethyl = _write_synthetic_bedmethyl(tmp_path / "toy.bedmethyl.bed", chrom="chr1", mod_code="m")
    regions = _write_regions(tmp_path / "toy.regions.bed", chrom="chr1")
    out = tmp_path / "out.bed"
    argv = (str(bedmethyl), str(regions), str(out)) + tuple(extra_args)
    run_main_no_parallel(argv)
    return out


def _run_cli_bedgraph(run_main_no_parallel, tmp_path: Path, extra_args: tuple[str, ...] = ()) -> Path:
    bedgraph = _write_synthetic_bedgraph4(tmp_path / "toy.bedgraph", chrom="chr1")
    regions = _write_regions(tmp_path / "toy.regions.bed", chrom="chr1")
    out = tmp_path / "out.bed"
    argv = (str(bedgraph), str(regions), str(out), "--bedgraph") + tuple(extra_args)
    run_main_no_parallel(argv)
    return out


# -----------------------------
# Tests: bedMethyl mode
# -----------------------------
def test_cli_runs_with_defaults_bedmethyl(run_main_no_parallel, tmp_path):
    out = _run_cli_bedmethyl(run_main_no_parallel, tmp_path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_cli_mod_code_filters_bedmethyl(run_main_no_parallel, tmp_path):
    # Synthetic uses mod_code="m", so matching should keep rows
    out = _run_cli_bedmethyl(run_main_no_parallel, tmp_path, extra_args=("--mod-code", "m"))
    assert out.exists()
    assert out.stat().st_size > 0

    # Non-matching should produce empty output (or a file with only headers/track lines)
    out2 = _run_cli_bedmethyl(run_main_no_parallel, tmp_path / "nomatch", extra_args=("--mod-code", "Z"))
    assert out2.exists()
    # allow empty file here; behavior depends on your writer (track lines, etc.)
    # but it must not crash.


def test_cli_debug_writes_extra_outputs_bedmethyl(run_main_no_parallel, tmp_path):
    out = _run_cli_bedmethyl(run_main_no_parallel, tmp_path, extra_args=("--debug",))
    assert out.exists()

    lowess_path = out.with_suffix(".LOWESS.bedgraph")
    detected_path = out.with_suffix(".detected_dips.bed")

    assert lowess_path.exists()
    assert lowess_path.stat().st_size > 0

    assert detected_path.exists()
    assert detected_path.stat().st_size > 0


def test_cli_plot_writes_pngs_bedmethyl(run_main_no_parallel, tmp_path, mpl_agg):
    out = _run_cli_bedmethyl(run_main_no_parallel, tmp_path, extra_args=("--plot",))
    assert out.exists()

    plot_dir = out.parent / f"{out.stem}_plots"
    assert plot_dir.exists()

    pngs = list(plot_dir.glob(f"{out.stem}.*.summary.png"))
    assert len(pngs) >= 1
    assert all(p.stat().st_size > 0 for p in pngs)


@pytest.mark.parametrize(
    "flag_args",
    [
        ("--enrichment",),
        ("--prominence", "0.4"),
        ("--height", "0.15"),
        ("--broadness", "0.8"),
        ("--window-size", "20000"),
        ("--cov-conf", "8"),
        ("--min-size", "1500"),
        ("--min-score", "400"),
        ("--cluster-distance", "-1"),
        ("--label", "TESTCDR"),
        ("--color", "255,0,0"),
        ("--threads", "1"),
    ],
)
def test_cli_flag_variants_do_not_crash_bedmethyl(run_main_no_parallel, tmp_path, mpl_agg, flag_args):
    out = _run_cli_bedmethyl(run_main_no_parallel, tmp_path, extra_args=tuple(flag_args))
    assert out.exists()
    # may be empty depending on thresholds; key is: must not crash


def test_cli_combined_flags_do_not_crash_bedmethyl(run_main_no_parallel, tmp_path, mpl_agg):
    out = _run_cli_bedmethyl(
        run_main_no_parallel,
        tmp_path,
        extra_args=(
            "--debug",
            "--plot",
            "--window-size", "20000",
            "--cov-conf", "8",
            "--prominence", "0.4",
            "--height", "0.15",
            "--broadness", "0.8",
            "--min-size", "1500",
            "--min-score", "400",
            "--cluster-distance", "-1",
            "--label", "TESTCDR",
            "--color", "255,0,0",
            "--threads", "1",
        ),
    )
    assert out.exists()

    # debug artifacts
    assert out.with_suffix(".LOWESS.bedgraph").exists()
    assert out.with_suffix(".detected_dips.bed").exists()

    # plot artifacts
    plot_dir = out.parent / f"{out.stem}_plots"
    assert plot_dir.exists()
    pngs = list(plot_dir.glob(f"{out.stem}.*.summary.png"))
    assert len(pngs) >= 1


# -----------------------------
# Tests: --bedgraph mode
# -----------------------------
def test_cli_runs_bedgraph_mode(run_main_no_parallel, tmp_path):
    out = _run_cli_bedgraph(run_main_no_parallel, tmp_path)
    assert out.exists()
    # may be empty depending on detection thresholds; must not crash


def test_cli_bedgraph_plot_and_debug(run_main_no_parallel, tmp_path, mpl_agg):
    out = _run_cli_bedgraph(run_main_no_parallel, tmp_path, extra_args=("--debug", "--plot"))
    assert out.exists()

    # debug artifacts should still exist if your main writes them in debug mode
    assert out.with_suffix(".LOWESS.bedgraph").exists()
    assert out.with_suffix(".detected_dips.bed").exists()

    plot_dir = out.parent / f"{out.stem}_plots"
    assert plot_dir.exists()
    pngs = list(plot_dir.glob(f"{out.stem}.*.summary.png"))
    assert len(pngs) >= 1
