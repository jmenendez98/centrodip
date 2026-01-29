from __future__ import annotations

from pathlib import Path
from typing import Dict
import pytest

from centrodip.bedtable import BedTable


def _dataset_type(remote_dataset_paths: Dict[str, Dict[str, Path]], key: str) -> str:
    # you store "type" as a downloaded file too; it's not a real url
    # so it ends up as a Path in remote_dataset_paths. Instead read REMOTE_DATASETS in conftest,
    # OR simplest: infer from extension
    bedmethyl_name = remote_dataset_paths[key]["bedmethyl"].name
    return "bedgraph" if bedmethyl_name.endswith(".bedgraph") else "bedmethyl"


def _count_overlaps(bm: BedTable, regions: BedTable) -> int:
    regions_by_chrom = regions.groupby_chrom()
    n = 0
    for r in bm:
        chrom_regs = regions_by_chrom.get(r.chrom)
        if chrom_regs is None:
            continue
        for reg in chrom_regs:
            if r.overlaps(reg.chrom, reg.start, reg.end):
                n += 1
                break
    return n


@pytest.mark.parametrize("dataset_key", ["chm13_chr1", "hg002_chrXY"])
def test_remote_datasets_do_not_crash(
    run_main_no_parallel,
    remote_dataset_paths: Dict[str, Dict[str, Path]],
    tmp_path: Path,
    mpl_agg,
    dataset_key: str,
):
    ds = remote_dataset_paths[dataset_key]
    bedmethyl_path = ds["bedmethyl"]
    regions_path = ds["regions"]

    out = tmp_path / f"{dataset_key}.out.bed"

    dtype = _dataset_type(remote_dataset_paths, dataset_key)

    base_args = [
        str(bedmethyl_path),
        str(regions_path),
        str(out),
        "--threads",
        "1",
        "--debug",
        "--plot",
    ]

    if dtype == "bedgraph":
        base_args += ["--bedgraph"]
    else:
        base_args += ["--mod-code", "m"]

    # ---- run ----
    run_main_no_parallel(base_args)

    # ---- assertions: outputs exist ----
    assert out.exists()

    lowess_path = out.with_suffix(".LOWESS.bedgraph")
    detected_path = out.with_suffix(".detected_dips.bed")

    assert lowess_path.exists()
    assert detected_path.exists()

    # ---- plot expectations ----
    plot_dir = out.parent / f"{out.stem}_plots"
    assert plot_dir.exists()

    pngs = list(plot_dir.glob(f"{out.stem}.*.summary.png"))

    # Only require PNGs if there is at least one overlapping record.
    bm = BedTable.from_path(bedmethyl_path)
    rg = BedTable.from_path(regions_path)
    n_overlap = _count_overlaps(bm, rg)

    if n_overlap > 0:
        assert len(pngs) >= 1, f"Expected >=1 PNG because overlap_count={n_overlap}"
    else:
        # Helpful diagnostic if you *expected* overlap
        pytest.xfail(
            f"No overlaps between bedmethyl and regions for {dataset_key}; "
            "no per-chrom plots will be produced. Likely chrom-name mismatch or coords."
        )
