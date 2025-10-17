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
        region_merge_distance: int,
        region_edge_filter: int,
    ) -> None:
        self.mod_code = mod_code
        self.bedgraph = bedgraph
        self.region_merge_distance = region_merge_distance
        self.region_edge_filter = region_edge_filter

    def read_and_filter_regions(self, regions_path: Path | str) -> RegionDict:
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

        merged_regions: RegionDict = {}
        for chrom, coords in region_dict.items():
            sorted_regions = sorted(zip(coords["starts"], coords["ends"]))

            merged_starts: List[int] = []
            merged_ends: List[int] = []
            for start, end in sorted_regions:
                if not merged_starts or start - merged_ends[-1] > self.region_merge_distance:
                    trimmed_start = start + self.region_edge_filter
                    trimmed_end = end - self.region_edge_filter
                    if trimmed_end <= trimmed_start:
                        continue
                    merged_starts.append(trimmed_start)
                    merged_ends.append(trimmed_end)
                else:
                    merged_ends[-1] = max(merged_ends[-1], end - self.region_edge_filter)

            merged_regions[chrom] = {"starts": merged_starts, "ends": merged_ends}

        return merged_regions
    
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
                # handle potential issues with reading in methylation file
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                min_columns = 4 if self.bedgraph else 11
                if len(columns) < min_columns:
                    raise TypeError(
                        f"Insufficient columns in {methylation_path}. Likely incorrectly formatted."
                    )
                if self.bedgraph and len(columns) > 4:
                    warnings.warn(
                        f"Warning: {methylation_path} has more than 4 columns, and was passed in as bedgraph. "
                        "Potentially incorrectly formatted bedgraph file.",
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

                entry = methylation_dict[chrom]
                entry["position"].append(methylation_position)
                entry["fraction_modified"].append(
                    float(columns[3]) if self.bedgraph else float(columns[10])
                )
                entry["valid_coverage"].append(None if self.bedgraph else float(columns[4]))

        return methylation_dict