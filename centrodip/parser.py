from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import warnings


RegionDict = Dict[str, Dict[str, List[int]]]
MethylationDict = Dict[str, Dict[str, List[float]]]


class Parser:
    """Parser to read in region and methylation bed files for centrodip."""
    def __init__(
        self,
        mod_code: str,
        bedgraph: bool,
    ) -> None:
        self.mod_code = mod_code
        self.bedgraph = bedgraph

    def read_regions_bed(self, regions_path: Path | str) -> RegionDict:
        """ Read and filter regions from a BED file. """

        regions_path = Path(regions_path)
        if not regions_path.exists():
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict: RegionDict = defaultdict(lambda: {"starts": [], "ends": []})

        with regions_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 3:
                    raise TypeError(
                        f"Less than 3 columns in {regions_path}. Likely incorrectly formatted bed file."
                    )
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])
                region_dict[chrom]["starts"].append(start)
                region_dict[chrom]["ends"].append(end)

        return region_dict
    
    def read_and_filter_methylation(
        self,
        methylation_path: Path | str,
        region_dict: RegionDict,
    ) -> MethylationDict:
        """ Read and filter methylation data from a BED file. """

        methylation_path = Path(methylation_path)
        if not methylation_path.exists():
            raise FileNotFoundError(f"File not found: {methylation_path}")

        methylation_dict: Dict[str, Dict[str, List[float | int]]] = defaultdict(
            lambda: {
                "position": [],
                "fraction_modified": [],
                "valid_coverage": [],
            }
        )

        region_lookup = {
            chrom: (coords["starts"], coords["ends"])
            for chrom, coords in region_dict.items()
        }

        with methylation_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                min_columns = 4 if self.bedgraph else 11
                if len(columns) < min_columns:
                    raise TypeError(
                        f"Insufficient columns in {methylation_path}. "
                        "Likely incorrectly formatted."
                    )
                if self.bedgraph and len(columns) > 4:
                    warnings.warn(
                        f"Warning: {methylation_path} has more than 4 columns, and was "
                        "passed in as bedgraph. Potentially incorrectly formatted bedgraph file.",
                        stacklevel=2,
                    )
                if not self.bedgraph and columns[3] != self.mod_code:
                    continue

                chrom = columns[0]
                methylation_position = int(columns[1])
                if chrom not in region_lookup:
                    continue

                starts, ends = region_lookup[chrom]
                idx = bisect_right(starts, methylation_position) - 1
                if idx < 0:
                    continue
                region_start = starts[idx]
                region_end = ends[idx]
                if not (region_start < methylation_position < region_end):
                    continue

                region_key = f"{chrom}:{region_start}-{region_end}"
                entry = methylation_dict[region_key]
                entry["position"].append(methylation_position)
                entry["fraction_modified"].append(
                    float(columns[3]) if self.bedgraph else float(columns[10])
                )
                entry["valid_coverage"].append(
                    1.0 if self.bedgraph else float(columns[4])
                )

        sorted_dict: MethylationDict = {}
        for region, values in methylation_dict.items():
            sorted_entries = sorted(
                zip(
                    values["position"],
                    values["fraction_modified"],
                    values["valid_coverage"],
                ),
                key=lambda entry: entry[0],
            )
            if sorted_entries:
                positions, frac_mod, coverage = zip(*sorted_entries)
            else:
                starts = ends = frac_mod = coverage = ()
            sorted_dict[region] = {
                "position": list(positions),
                "fraction_modified": list(frac_mod),
                "valid_coverage": list(coverage),
            }

        return sorted_dict

    def process_files(
        self,
        methylation_path: Path | str,
        regions_path: Path | str,
    ) -> tuple[MethylationDict, RegionDict]:
        """Read and intersect methylation and region BED files."""

        regions = self.read_regions_bed(regions_path)
        methylation = self.read_and_filter_methylation(methylation_path, regions)
        return methylation, regions


__all__ = ["Parser"]
