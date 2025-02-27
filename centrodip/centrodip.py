import argparse
import concurrent.futures
import warnings
import os

import numpy as np
import scipy

from typing import Dict, Optional, List, Union
import tempfile

class BedParse:
    """hmmCDR parser to read in region and methylation bed files."""

    def __init__(
        self,
        mod_code: Optional[str] = None,
        methyl_bedgraph: bool = False,
        region_edge_filter: int = 0,
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
        self.methyl_bedgraph = methyl_bedgraph

        self.region_edge_filter = region_edge_filter
        
        self.temp_dir = tempfile.gettempdir()

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
            
            if any(len(cols) < 3 for cols in lines):
                raise TypeError(f"Less than 3 columns in {regions_path}. Likely incorrectly formatted bed file.")

            for line in lines:
                columns = line.strip().split("\t")
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])

                if chrom not in region_dict:
                    region_dict[chrom] = {"starts": [], "ends": []}

                if (end - self.region_edge_filter) < (start + self.region_edge_filter):
                    continue
                region_dict[chrom]["starts"].append(start+self.region_edge_filter)
                region_dict[chrom]["ends"].append(end-self.region_edge_filter)

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
            methyl_dict[region_key]["valid_coverage"].append(1)
            return methyl_dict

        with open(methylation_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                columns = line.strip().split('\t') # split line of bed into columns

                # raise errors if: 1. not enough columns, 2. too many columns in bedgraph
                if (len(columns) < (4 if self.methyl_bedgraph else 11)):
                    raise TypeError(f"Insufficient columns in {methylation_path}. Likely incorrectly formatted.")
                elif (self.methyl_bedgraph) and (len(columns) > 4):
                    warnings.warn(f"Warning: {methylation_path} has more than 4 columns, and was passed in as bedgraph. Potentially incorrectly formatted bedgraph file.")

                # do not process entry if it has the wrong mod code (default: 'm')
                if (not self.methyl_bedgraph) and (columns[3] != self.mod_code):
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
                            float(columns[3]) if self.methyl_bedgraph else float(columns[10]),
                            1 if self.methyl_bedgraph else float(columns[4])
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


class CentroDip:
    def __init__(
        self,
        window_size,
        step_size,
        min_valid_cov,
        min_sig_cpgs,
        merge_distance,
        enrichment,
        cdr_color,
        transition_color,
        low_coverage_color,
        threads,
        output_label,
    ):
        self.window_size = window_size
        self.step_size = step_size

        self.min_valid_cov = min_valid_cov

        self.min_sig_cpgs = min_sig_cpgs
        self.merge_distance = merge_distance

        self.enrichment = enrichment

        self.cdr_color = cdr_color
        self.transition_color = transition_color
        self.low_coverage_color = low_coverage_color

        self.threads = threads
        self.output_label = output_label

    def smooth_methylation(self, methylation):
        methyl_frac_mod = np.array(methylation["fraction_modified"], dtype=float)

        # calculate the rolling mean of the fraction modified data
        half_window = self.window_size // 2

        frac_mod_cum_sum = np.cumsum(methyl_frac_mod)
        rolling_avg = np.full(len(methyl_frac_mod), 100.0)
        
        for i in range(half_window, len(methyl_frac_mod) - half_window):
            start = i - half_window
            end = i + half_window + 1
            rolling_avg[i] = np.mean(methyl_frac_mod[start:end])
        
        methylation["rolling_avg_frac_mod"] = rolling_avg

        return methylation

    def optimize_threshold(self, methylation):
        starts = np.array(methylation["starts"], dtype=int)
        ends = np.array(methylation["ends"], dtype=int)

        data = np.array(methylation["rolling_avg_frac_mod"], dtype=float)
        fracmod = np.array(methylation["fraction_modified"], dtype=float)
        rolling_slope = np.diff(data)
        methylation["derivative"] = rolling_slope

        median = np.median(data)
        hist, bin_edges = np.histogram(data[data < median], bins=1000)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        weighted_hist = hist * np.exp(-bin_centers * 0.1)
        weight1 = np.cumsum(weighted_hist)
        weight2 = np.cumsum(weighted_hist[::-1])[::-1]

        mean1 = np.cumsum(weighted_hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((weighted_hist * bin_centers)[::-1]) / (weight2[::-1])[::-1] + 1e-10)

        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = np.argmax(variance)
        threshold = bin_centers[idx]

        return threshold

    def call_cdrs(self, methylation_smoothed, threshold):
        cdrs = {
            "starts": [],
            "ends": [],
            "names": [],
            "scores": [],
            "strands": [],
            "itemRgbs": []
        }

        raw_data = np.array(methylation_smoothed["fraction_modified"], dtype=float)
        smoothed_data = np.array(methylation_smoothed["rolling_avg_frac_mod"], dtype=float)

        sig_cpgs = np.where(smoothed_data < threshold)[0]

        if len(sig_cpgs) == 0:
            return cdrs

        sig_regions = np.split(sig_cpgs, np.where(np.diff(sig_cpgs) != 1)[0] + 1)

        for region in sig_regions:
            start = methylation_smoothed["starts"][region[0]]
            end = methylation_smoothed["ends"][region[-1]]
            
            cdrs["starts"].append(start)
            cdrs["ends"].append(end)
            cdrs["names"].append(f"{self.output_label}")
            cdrs["scores"].append(np.mean(raw_data[region]))
            cdrs["strands"].append(".")
            cdrs["itemRgbs"].append(f"{self.cdr_color}") 

        return cdrs

    def find_low_coverage(self, methylation):
        methyl_starts = np.array(methylation["starts"], dtype=int)
        methyl_coverage = np.array(methylation["valid_coverage"], dtype=int)

        # find all low coverage CpGs
        low_cov_idxs = np.where(methyl_coverage < self.min_valid_cov)[0]
        if len(low_cov_idxs) == 0:  
            # check if there are any low coverage indices
            return {"starts": [], "ends": []}

        low_cov_diff = np.diff(low_cov_idxs)
        low_cov_breaks = np.where(low_cov_diff > self.merge_distance)[0] + 1 
        low_cov_regions = np.split(low_cov_idxs, low_cov_breaks)

        low_covs = {"starts": [], "ends": []}

        for region in low_cov_regions:
            # make region only if it has more cpgs than self.min_sig_cpgs
            lcs = [cpg for cpg in region if cpg in low_cov_idxs]
            if len(lcs) > self.min_sig_cpgs:
                start_idx, end_idx = region[0], region[-1]
                low_covs["starts"].append(methyl_starts[start_idx])
                low_covs["ends"].append(methyl_starts[end_idx] + 1)

        return low_covs

    def adjust_for_low_coverage(self, cdrs, low_cov_regions):
        adjusted_cdrs = {
            "starts": [],
            "ends": [],
            "names": [],
            "scores": [],
            "strands": [],
            "itemRgbs": []
        }

        i = 0
        while i < len(cdrs["starts"]):
            cdr_start, cdr_end = cdrs["starts"][i], cdrs["ends"][i]
            adjusted = False

            for j in range(len(low_cov_regions["starts"])):
                low_cov_start, low_cov_end = low_cov_regions["starts"][j], low_cov_regions["ends"][j]

                if (low_cov_start < cdr_end) and (low_cov_end > cdr_start):
                    # CDR overlaps with low coverage region
                    if cdr_start < low_cov_start:
                        # add adjusted CDR before low coverage region
                        adjusted_cdrs["starts"].append(cdr_start)
                        adjusted_cdrs["ends"].append(low_cov_start)
                        adjusted_cdrs["names"].append(cdrs["names"][i])
                        adjusted_cdrs["scores"].append(cdrs["scores"][i])
                        adjusted_cdrs["strands"].append(cdrs["strands"][i])
                        adjusted_cdrs["itemRgbs"].append(cdrs["itemRgbs"][i])

                    # add low coverage entry
                    adjusted_cdrs["starts"].append(low_cov_start)
                    adjusted_cdrs["ends"].append(low_cov_end)
                    adjusted_cdrs["names"].append(f"low_coverage_region(>{self.min_valid_cov})")
                    adjusted_cdrs["scores"].append("0")
                    adjusted_cdrs["strands"].append(".")
                    adjusted_cdrs["itemRgbs"].append(self.low_coverage_color)

                    if cdr_end > low_cov_end:
                        # add adjusted CDR
                        adjusted_cdrs["starts"].append(low_cov_end)
                        adjusted_cdrs["ends"].append(cdr_end)
                        adjusted_cdrs["names"].append(cdrs["names"][i])
                        adjusted_cdrs["scores"].append(cdrs["scores"][i])
                        adjusted_cdrs["strands"].append(cdrs["strands"][i])
                        adjusted_cdrs["itemRgbs"].append(cdrs["itemRgbs"][i])

                    adjusted = True
                    break

            if not adjusted:
                # If CDR doesn't overlap with any low coverage region, add it as is
                for key in adjusted_cdrs:
                    adjusted_cdrs[key].append(cdrs[key][i])

            i += 1

        return adjusted_cdrs

    def centrodip_single_chromosome(self, region, methylation):
        methylation_smoothed = self.smooth_methylation(methylation)

        threshold = self.optimize_threshold(methylation_smoothed)
        cdrs = self.call_cdrs(methylation_smoothed, threshold)

        if self.min_valid_cov > 1:
            low_cov_regions = self.find_low_coverage(methylation)
            filtered_cdrs = self.adjust_for_low_coverage(cdrs, low_cov_regions)
            return ( region, filtered_cdrs, low_cov_regions, methylation_smoothed)

        return ( region, cdrs, {}, methylation_smoothed)

    def centrodip_all_chromosomes(self, methylation_per_region, regions_per_chrom):
        cdrs_all_chroms, low_cov_all_chroms, methylation_pvalues_all_chroms = {}, {}, {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            # launch parallized processing of regions
            futures = {
                executor.submit(
                    self.centrodip_single_chromosome,
                    region, methylation_per_region[region],
                ): region
                for region in list(methylation_per_region.keys())
            }

            for future in concurrent.futures.as_completed(futures):
                (
                    region, cdrs, low_cov_regions, methylation_pvalues,
                ) = future.result()

                cdrs_all_chroms[region] = cdrs
                low_cov_all_chroms[region] = low_cov_regions
                methylation_pvalues_all_chroms[region] = methylation_pvalues

        return cdrs_all_chroms, low_cov_all_chroms, methylation_pvalues_all_chroms


def main():
    argparser = argparse.ArgumentParser(
        description="Process bedMethyl and CenSat BED file to produce CDR predictions."
    )

    # required inputs
    argparser.add_argument("bedmethyl", type=str, help="Path to the bedmethyl file")
    argparser.add_argument("regions", type=str, help="Path to BED file of regions")
    argparser.add_argument("output", type=str, help="Path to the output BED file")

    # bed parser arguments
    argparser.add_argument(
        "-m",
        "--mod_code",
        type=str,
        default="m",
        help='Modification code to filter bedMethyl file (default: "m")',
    )
    argparser.add_argument(
        "--methyl_bedgraph",
        action="store_true",
        default=False,
        help="Flag indicating the input is a bedgraph. If passed --mod_code and --min_valid_cov are ignored. (default: False)",
    )
    argparser.add_argument(
        "--min_valid_cov",
        type=int,
        default=1,
        help="Minimum valid coverage required to call CDR site. Ignored if bedmethyl input is a bedgraph. (default: 1)",
    )
    argparser.add_argument(
        "--region_edge_filter",
        type=int,
        default=0,
        help="Filter out this many base pairs from the edges of each region in regions input. (default: 0)",
    )

    # CentroDip arguments
    argparser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=101,
        help="Number of CpGs to include in rolling window for CDR calculation. (default: 101)",
    )
    argparser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for rolling window calculations of CDRs. (default: 1)",
    )
    argparser.add_argument(
        "--min_sig_cpgs",
        type=int,
        default=50,
        help="Minimum size for a region to be labelled as an MDR. (default: 50)",
    )
    argparser.add_argument(
        "--merge_distance",
        type=int,
        default=50,
        help="Distance in bp to merge low confidence MDR annotations. (default: 50)",
    )
    argparser.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Pass this flag in if you are looking for hypermethylation within the region. Recommended to asjust --output_label if used. (default: False)",
    )
    argparser.add_argument(
        "--threads",
        type=int,
        default=4,
        help='Number of threads to use for multithreading. (default: 4)',
    )

    # output arguments
    argparser.add_argument(
        "--cdr_color",
        type=str,
        default="50,50,255",
        help='Pass flag in if you want to output all data generated throughout mwCDR process. (default: False)',
    )
    argparser.add_argument(
        "--transition_color",
        type=str,
        default="150,150,255",
        help='Pass flag in if you want to output all data generated throughout mwCDR process. (default: False)',
    )
    argparser.add_argument(
        "--low_coverage_color",
        type=str,
        default="150,150,150",
        help='Pass flag in if you want to output all data generated throughout mwCDR process. (default: False)',
    )
    argparser.add_argument(
        "--output_all",
        action='store_true',
        default=False,
        help='Pass flag in if you want to output all data generated throughout mwCDR process. (default: False)',
    )
    argparser.add_argument(
        "--output_label",
        type=str,
        default="subCDR",
        help='Label to use for name column of output file. (default: "subCDR")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

    if args.methyl_bedgraph and args.min_valid_cov > 1:
        raise ValueError("Cannot pass --min_valid_cov > 1 with --methyl_bedgraph")


    def generate_output_bed(bed_dict, output_file, key_is_regions, columns=["starts", "ends"]):
        if not bed_dict:
            return

        all_lines = []
        all_keys = list(bed_dict.keys())

        if key_is_regions:
            all_chroms = [region.split(':')[0] for region in list(bed_dict.keys())]
        else:
            all_chroms = all_keys
            
        for key, chrom in zip(all_keys, all_chroms):
            chrom_data = bed_dict[key]

            if chrom_data:
                for i in range( len(chrom_data.get("starts", []))-1 ):
                    line = [chrom]
                    for col in columns:
                        if col in chrom_data:
                            line.append(str(chrom_data[col][i])) 
                    all_lines.append(line)

        # if nothing is in all_lines, return nothing and don't write to file
        if all_lines:        
            all_lines = sorted(all_lines, key=lambda x: (x[0], int(x[1])))
            with open(output_file, 'w') as file:
                for line in all_lines: 
                    file.write("\t".join(line) + "\n")
        else:
            return


    bed_parser = BedParse(
        mod_code=args.mod_code,
        methyl_bedgraph=args.methyl_bedgraph,
        region_edge_filter=args.region_edge_filter
    )

    regions_per_chrom_dict, methylation_per_region_dict = bed_parser.process_files(
        methylation_path=args.bedmethyl,
        regions_path=args.regions
    )

    centro_dip = CentroDip(
        window_size=args.window_size,
        step_size=args.step_size,
        min_valid_cov=args.min_valid_cov,
        min_sig_cpgs=args.min_sig_cpgs,
        merge_distance=args.merge_distance,
        enrichment=args.enrichment,
        threads=args.threads,
        cdr_color=args.cdr_color,
        transition_color=args.transition_color,
        low_coverage_color=args.low_coverage_color,
        output_label=args.output_label,
    )

    (
        cdrs_per_region,
        low_coverage_per_region,
        methylation_sig_per_region,
    ) = centro_dip.centrodip_all_chromosomes(methylation_per_region=methylation_per_region_dict, regions_per_chrom=regions_per_chrom_dict)

    if args.output_all:
        generate_output_bed(low_coverage_per_region, f"{output_prefix}_low_cov.bed", key_is_regions=True, columns=["starts", "ends"])
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_rolling_avg_frac_mod.bedgraph", key_is_regions=True, columns=["starts", "ends", "rolling_avg_frac_mod"])
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_derivative.bedgraph", key_is_regions=True, columns=["starts", "ends", "derivative"])

    generate_output_bed(cdrs_per_region, f"{args.output}", key_is_regions=True, columns=["starts", "ends", "names", "scores", "strands", "starts", "ends", "itemRgbs"])


if __name__ == "__main__":
    import cProfile
    with cProfile.Profile() as profile:
        main()
    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    
    # main()