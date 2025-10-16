import argparse
import concurrent.futures
import warnings
import os
import time

import numpy as np


class Parser:
    """hmmCDR parser to read in region and methylation bed files."""

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
        if not os.path.exists(regions_path):
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict = {}

        with open(regions_path, 'r') as file:
            lines = file.readlines()
            
            if any(len(line.strip().split("\t")) < 3 for line in lines):
                raise TypeError(f"Less than 3 columns in {regions_path}. Likely incorrectly formatted bed file.")

            for line in lines:
                columns = line.strip().split("\t")
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])

                if chrom not in region_dict:
                    region_dict[chrom] = {"starts": [], "ends": []}

                region_dict[chrom]["starts"].append(start)
                region_dict[chrom]["ends"].append(end)

        # merge regions that are closer than self.region_merge_distance
        for chrom in region_dict:
            starts = region_dict[chrom]["starts"]
            ends = region_dict[chrom]["ends"]
            
            # sort regions by start position
            sorted_regions = sorted(zip(starts, ends))
            
            merged_starts, merged_ends = [], []
            for start, end in sorted_regions:
                if not merged_starts or start - merged_ends[-1] > self.region_merge_distance:
                    if (end - self.region_edge_filter) < (start + self.region_edge_filter):
                        continue
                    merged_starts.append(start + self.region_edge_filter)
                    merged_ends.append(end - self.region_edge_filter)
                else:
                    merged_ends[-1] = max(merged_ends[-1], end)
            
            region_dict[chrom]["starts"] = merged_starts
            region_dict[chrom]["ends"] = merged_ends

        return region_dict
    
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
        if not os.path.exists(methylation_path):
            raise FileNotFoundError(f"File not found: {methylation_path}")

        methylation_dict = {}

        # helper function to add methylation data entries
        def add_methylation_entry(methyl_dict, region_key, start, end, frac_mod, cov):
            methyl_dict[region_key]["starts"].append(start)
            methyl_dict[region_key]["ends"].append(start+1)
            methyl_dict[region_key]["fraction_modified"].append(frac_mod)
            methyl_dict[region_key]["valid_coverage"].append(cov)
            return methyl_dict

        with open(methylation_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                columns = line.strip().split('\t') # split line of bed into columns

                # raise errors if: 1. not enough columns, 2. too many columns in bedgraph
                if (len(columns) < (4 if self.bedgraph else 11)):
                    raise TypeError(f"Insufficient columns in {methylation_path}. Likely incorrectly formatted.")
                elif (self.bedgraph) and (len(columns) > 4):
                    warnings.warn(f"Warning: {methylation_path} has more than 4 columns, and was passed in as bedgraph. Potentially incorrectly formatted bedgraph file.")

                # do not process entry if it has the wrong mod code (default: 'm')
                if (not self.bedgraph) and (columns[3] != self.mod_code):
                    continue

                chrom = columns[0] # get chromosome
                methylation_position = int(columns[1]) # get methlation position

                if chrom not in region_dict.keys():
                    continue

                # check if the methylation position is within one of the regions on the matching chromosome
                for r_s, r_e in zip(region_dict[chrom]['starts'], region_dict[chrom]['ends']): 
                    if (methylation_position > r_s) and (methylation_position < r_e):
                        region_key = f'{chrom}:{r_s}-{r_e}' # make a dictionary key that is the area of that region

                        if region_key not in methylation_dict.keys(): # if that key is not in methylation dictionary create it
                            methylation_dict[region_key] = {"starts": [], "ends": [], "fraction_modified": [], "valid_coverage": []}

                        # use helper to add entry into methylation dictionary
                        methylation_dict = add_methylation_entry( 
                            methylation_dict, 
                            region_key, 
                            methylation_position, 
                            methylation_position+1, 
                            float(columns[3]) if self.bedgraph else float(columns[10]),
                            1 if self.bedgraph else float(columns[4])
                        )
                        break

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
        methylation_dict = self.read_and_filter_methylation(methylation_path, region_dict)

        return region_dict, methylation_dict
