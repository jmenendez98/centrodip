from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Generator, Optional


RegionDict = Dict[str, Dict[str, List[int]]]
MethylationDict = Dict[str, Dict[str, List[float]]]


class DataHandler:
    """Parser to read in region and methylation bed files for centrodip."""
    def __init__(
        self,
        regions_path: Path | str,
        methylation_path: Path | str,
        mod_code: str,
        debug: bool = False,
    ) -> None:
        self.mod_code = mod_code
        self.debug = debug

        self.regions_path = Path(regions_path)
        self.methylation_path = Path(methylation_path)

        if debug:
            print(f"[DEBUG] Regions path: {self.regions_path}...")
            print(f"[DEBUG] bedMethyl path: {self.methylation_path}...")

        self._assert_bed_sorted(self.regions_path, chrom_col=0, start_col=1, min_cols=3, label="Regions")
        self._assert_bed_sorted(self.methylation_path, chrom_col=0, start_col=1, min_cols=3, label="Methylation")


    def _assert_bed_sorted(
        self,
        bed_path: Path,
        chrom_col: int = 0,
        start_col: int = 1,
        min_cols: int = 3,
        label: str = "BED",
    ) -> None:
        """
        Ensure the BED-like file is grouped by chromosome and sorted by start
        within each chromosome block.

        Raises:
            TypeError: if columns are insufficient.
            ValueError: if rows are not grouped/sorted.
        """
        last_chrom: str | None = None
        last_start: int = -1
        seen_chroms: set[str] = set()

        with bed_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < min_cols:
                    raise TypeError(
                        f"[{label}] Insufficient columns at line {line_no} in {bed_path}. "
                        f"Expected ≥{min_cols}."
                    )

                chrom = cols[chrom_col]
                try:
                    start = int(cols[start_col])
                except ValueError:
                    raise ValueError(
                        f"[{label}] Non-integer start at line {line_no} in {bed_path!s}: {cols[start_col]!r}"
                    ) from None

                if last_chrom is None:
                    last_chrom, last_start = chrom, start
                    continue

                if chrom == last_chrom:
                    if start < last_start:
                        raise ValueError(
                            f"[{label}] Not sorted by start within {chrom} near line {line_no} in {bed_path!s}: "
                            f"{start} < previous {last_start}. "
                            "Sort with: sort -k1,1 -k2,2n"
                        )
                    last_start = start
                else:
                    # starting a new chromosome block
                    if chrom in seen_chroms:
                        raise ValueError(
                            f"[{label}] Chromosomes are not grouped in {bed_path!s}: "
                            f"encountered {chrom!r} again after seeing other chromosomes (line {line_no}). "
                            "Ensure rows for each chromosome are contiguous. "
                            "Sort with: sort -k1,1 -k2,2n"
                        )
                    seen_chroms.add(last_chrom)
                    last_chrom, last_start = chrom, start


    def read_regions_bed(self, regions_path: Path | str) -> RegionDict:
        """Read regions from a BED file (assumes sortedness has been checked)."""
        regions_path = Path(regions_path)
        if not regions_path.exists():
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict: RegionDict = defaultdict(lambda: {"starts": [], "ends": []})

        with regions_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 3:
                    raise TypeError(
                        f"[Regions] Less than 3 columns at line {line_no} in {regions_path}."
                    )
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])
                if end < start:
                    raise ValueError(
                        f"[Regions] End < start for {chrom}:{start}-{end} at line {line_no} in {regions_path}"
                    )
                region_dict[chrom]["starts"].append(start)
                region_dict[chrom]["ends"].append(end)

        return region_dict

    def load_data(
        self,
    ) -> Generator[MethylationDict, None, None]:
        """
        Stream the methylation BED and yield a per-chromosome MethylationDict
        as soon as each chromosome block is finished.

        Yields:
            {chrom: {"cpg_pos": [...], "fraction_modified": [...], "valid_coverage": [...]}}
        """
        rpath = self.regions_path
        mpath = self.methylation_path

        if not rpath.exists():
            raise FileNotFoundError(f"File not found: {rpath}")
        if not mpath.exists():
            raise FileNotFoundError(f"File not found: {mpath}")

        region_dict = self.read_regions_bed(rpath)

        # Prepare quick lookup per chrom
        region_lookup: Dict[str, tuple[List[int], List[int]]] = {
            chrom: (coords["starts"], coords["ends"])
            for chrom, coords in region_dict.items()
        }

        current_chrom: Optional[str] = None
        payload: Dict[str, List[float | int]] = {
            "cpg_pos": [],
            "fraction_modified": [],
            "valid_coverage": [],
        }

        def _flush():
            """Yield current chrom if we have collected any data."""
            nonlocal payload, current_chrom
            if current_chrom and payload["cpg_pos"]:
                yield {current_chrom: {
                    "cpg_pos": payload["cpg_pos"],
                    "fraction_modified": payload["fraction_modified"],
                    "valid_coverage": payload["valid_coverage"],
                }}

        with mpath.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) != 18:
                    raise TypeError(
                        f"[Methylation] Insufficient columns at line {line_no} in {mpath}. Expected 18."
                    )

                if cols[3] != self.mod_code:
                    continue

                chrom = cols[0]
                pos = int(cols[1])

                # If chrom changed, flush previous (if any) and reset payload
                if current_chrom is None:
                    current_chrom = chrom
                elif chrom != current_chrom:
                    # flush previous chrom
                    for out in _flush():
                        yield current_chrom, out
                    # reset for new chrom
                    current_chrom = chrom
                    payload = {
                        "cpg_pos": [],
                        "fraction_modified": [],
                        "valid_coverage": [],
                    }

                # Skip if this chrom has no regions of interest
                if chrom not in region_lookup:
                    continue

                starts, ends = region_lookup[chrom]
                # Identify containing region (if any)
                idx = bisect_right(starts, pos) - 1
                if idx >= 0:
                    r_start = starts[idx]
                    r_end = ends[idx]
                    if r_start < pos < r_end:
                        payload["cpg_pos"].append(pos)
                        payload["fraction_modified"].append(float(cols[10]))
                        payload["valid_coverage"].append(float(cols[4]))

        # Flush the final chromosome after file ends
        for out in _flush():
            yield current_chrom, out