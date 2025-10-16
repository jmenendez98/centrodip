from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import warnings

class Parser:
    """Parser to read in region and methylation bed files for centrodip."""

    def __init__(
        self,
        mod_code,
        bedgraph,
        region_merge_distance,
        region_edge_filter,
    ):
        """
        Initialize the parser with optional filtering parameters.

        Args:
            mod_code: Modification code to filter
            bedgraph: True if methylation file is a bedgraph
            edge_filter: Amount to remove from edges of active_hor regions
            regions_prefiltered: Whether the regions bed is already subset
        """
        self.mod_code = mod_code
        self.bedgraph = bedgraph
        self.region_merge_distance = region_merge_distance
        self.region_edge_filter = region_edge_filter

    def read_and_filter_regions(self, regions_path):
        """
        Read and filter regions from a BED file.
        Args:
            regions_path: Path to the regions BED file
        Returns:
            Dictionary mapping chromosomes to their start/end positions
        Raises:
            FileNotFoundError: If regions_path doesn't exist
            TypeError: If BED file is incorrectly formatted
        """

        regions_path = Path(regions_path)
        if not regions_path.exists():
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"starts": [], "ends": []})

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

        merged_regions: dict[str, dict[str, list[int]]] = {}
        for chrom, coords in region_dict.items():
            sorted_regions = sorted(zip(coords["starts"], coords["ends"]))

            merged_starts: list[int] = []
            merged_ends: list[int] = []
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
    
    def read_and_filter_methylation(self, methylation_path, region_dict):
        """
        Read and filter methylation data from a BED file.
        Args:
            methylation_path: Path to the methylation BED file
        Returns:
            Dictionary mapping chromosomes to their methylation data
        Raises:
            FileNotFoundError: If methylation_path doesn't exist
            TypeError: If BED file is incorrectly formatted
            ValueError: If trying to filter bedgraph by coverage
        """
        methylation_path = Path(methylation_path)
        if not methylation_path.exists():
            raise FileNotFoundError(f"File not found: {methylation_path}")

        methylation_dict: defaultdict[str, dict[str, list[float | int]]] = defaultdict(
            lambda: {"starts": [], "ends": [], "fraction_modified": [], "valid_coverage": []}
        )

        region_lookup = {
            chrom: (coords["starts"], coords["ends"])
            for chrom, coords in region_dict.items()
        }

        with methylation_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split('\t')
                if len(columns) < (4 if self.bedgraph else 11):
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
                region_key = f"{chrom}:{region_start}-{region_end}"
                entry = methylation_dict[region_key]
                entry["starts"].append(methylation_position)
                entry["ends"].append(methylation_position + 1)
                entry["fraction_modified"].append(
                    float(columns[3]) if self.bedgraph else float(columns[10])
                )
                entry["valid_coverage"].append(1 if self.bedgraph else float(columns[4]))

        return {region: dict(values) for region, values in methylation_dict.items()}

    def process_files(self, methylation_path, regions_path):
        """
        Process and intersect methylation and regions files.
        Args:
            methylation_path: Path to methylation BED file
            regions_path: Path to regions BED file
        Returns:
            Tuple of (region_dict, filtered_methylation_dict)
        """
        region_dict = self.read_and_filter_regions(regions_path)
        methylation_dict = self.read_and_filter_methylation(methylation_path, region_dict)

        return region_dict, methylation_dict