import os
import numpy as np
from typing import Dict, Optional, List, Union
import tempfile

class bed_parser:
    """hmmCDR parser to read in region and methylation bed files."""

    def __init__(
        self,
        mod_code: Optional[str] = None,
        min_valid_cov: int = 1,
        methyl_bedgraph: bool = False,
        region_edge_filter: int = 10000,
    ):
        """
        Initialize the parser with optional filtering parameters.

        Args:
            mod_code: Modification code to filter
            min_valid_cov: Minimum coverage threshold
            methyl_bedgraph: Whether the file is a bedgraph
            sat_type: Satellite type(s) to filter
            edge_filter: Amount to remove from edges of active_hor regions
            regions_prefiltered: Whether the regions bed is already subset
        """
        self.mod_code = mod_code
        self.min_valid_cov = min_valid_cov
        self.methyl_bedgraph = methyl_bedgraph

        self.region_edge_filter = region_edge_filter
        
        self.temp_dir = tempfile.gettempdir()

    def read_and_filter_regions(self, regions_path: str) -> Dict[str, Dict[str, list]]:
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
        if not os.path.exists(regions_path):
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict: Dict[str, Dict[str, list]] = {}

        with open(regions_path, 'r') as file:
            lines = file.readlines()
            
            if any(len(cols) < 3 for cols in lines):
                raise TypeError(f"Less than 3 columns in {regions_path}. Likely incorrectly formatted bed file.")

            for line in lines:
                columns = line.strip().split("\t")
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])

                if chrom not in region_dict:
                    region_dict[chrom] = {"starts": [], "ends": []}

                if (end - self.region_edge_filter) < start:
                    continue
                region_dict[chrom]["starts"].append(start+self.region_edge_filter)
                region_dict[chrom]["ends"].append(end-self.region_edge_filter)

        return region_dict
    
    def read_and_filter_methylation(self, methylation_path: str) -> Dict[str, Dict[str, list]]:
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
        if not os.path.exists(methylation_path):
            raise FileNotFoundError(f"File not found: {methylation_path}")
        
        if self.methyl_bedgraph and self.min_valid_cov > 1:
            raise ValueError(f"{methylation_path} {self.min_valid_cov} bedgraph file cannot be filtered by coverage.")

        methylation_dict: Dict[str, Dict[str, list]] = {}

        with open(methylation_path, 'r') as file:
            lines = file.readlines()
            
            if any(len(cols) < (4 if self.methyl_bedgraph else 11) for cols in lines):
                raise TypeError(f"Insufficient columns in {methylation_path}. Likely incorrectly formatted.")

            for line in lines:
                columns = line.strip().split('\t')
                chrom = columns[0]
                start = int(columns[1])
                if chrom not in methylation_dict.keys():
                    methylation_dict[chrom] = {"starts": [], "ends": [], "fraction_modified": []}
                    
                if self.methyl_bedgraph:
                    frac_mod = float(columns[3])
                    methylation_dict[chrom]["starts"].append(start)
                    methylation_dict[chrom]["ends"].append(start+1)
                    methylation_dict[chrom]["fraction_modified"].append(frac_mod)
                elif columns[3] == self.mod_code and int(columns[4]) >= self.min_valid_cov:
                    frac_mod = float(columns[10])
                    methylation_dict[chrom]["starts"].append(start)
                    methylation_dict[chrom]["ends"].append(start+1)
                    methylation_dict[chrom]["fraction_modified"].append(frac_mod)
                    
        return methylation_dict

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
        methylation_dict = self.read_and_filter_methylation(methylation_path)

        filtered_methylation_dict = {}

        for chrom, regions in region_dict.items():
            if chrom not in methylation_dict:
                continue

            methylation_data = methylation_dict[chrom]

            # Vectorized overlap check
            region_starts = np.array(regions["starts"], dtype=int)
            region_ends = np.array(regions["ends"], dtype=int)
            methyl_starts = np.array(methylation_data["starts"], dtype=int)

            overlaps = np.zeros(len(methyl_starts), dtype=bool)
            for region_start, region_end in zip(region_starts, region_ends):
                for i, methyl_start in enumerate(methyl_starts):
                    if (methyl_start < region_end) and (methyl_start >= region_start):
                        overlaps[i] = True

            if np.any(overlaps):
                filtered_methylation_dict[chrom] = {
                    "starts": [start for start, overlap in zip(methylation_data["starts"], overlaps) if overlap],
                    "ends": [end for end, overlap in zip(methylation_data["ends"], overlaps) if overlap],
                    "fraction_modified": [frac_mod for frac_mod, overlap in zip(methylation_data["fraction_modified"], overlaps) if overlap]
                }
            else: 
                ValueError(f"No methylation data from {methylation_path} overlapping with {regions_path}.")

        return region_dict, filtered_methylation_dict