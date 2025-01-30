import argparse
import concurrent.futures
import os

import numpy as np
import scipy.stats as stats

from typing import Dict, Optional, List, Union
import tempfile

class BedParser:
    """hmmCDR parser to read in region and methylation bed files."""

    def __init__(
        self,
        mod_code: Optional[str] = None,
        min_valid_cov: int = 1,
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

                if (end - self.region_edge_filter) < (start + self.region_edge_filter):
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

class CentroDip:
    def __init__(
        self,
        window_size,
        step_size,
        stat,
        cdr_p,
        transition_p,
        min_sig_cpgs,
        merge_distance,
        enrichment,
        cdr_color,
        transition_color,
        threads,
        output_label,
    ):

        self.window_size = window_size
        self.step_size = step_size

        self.stat = stat
        self.cdr_p = cdr_p
        self.transition_p = transition_p

        self.min_sig_cpgs = min_sig_cpgs
        self.merge_distance = merge_distance

        self.enrichment = enrichment

        self.cdr_color = cdr_color
        self.transition_color = transition_color

        self.threads = threads
        self.output_label = output_label

    def calculate_regional_stats(self, methylation):
        methyl_starts = np.array(methylation["starts"], dtype=int)
        methyl_frac_mod = np.array(methylation["fraction_modified"], dtype=float)

        p_values = np.array([np.ones(self.window_size) for _ in range(len(methyl_starts))])

        for i in range(0, len(methyl_starts), self.step_size):
            start = max(0, i - (self.window_size//2))
            end = min(len(methyl_starts), i + (self.window_size//2) + 1)
            current_region_frac_mods = methyl_frac_mod[start:end]

            alt = "less" if not self.enrichment else "greater"
            if self.stat == "ks":
                _, p_value, _, _ = stats.ks_2samp(current_region_frac_mods, methyl_frac_mod, alternative=alt, nan_policy="omit")
            elif self.stat == "t":
                _, p_value, _ = stats.ttest_ind(current_region_frac_mods, methyl_frac_mod, alternative=alt, nan_policy="omit")
            elif "mannwhitneyu": 
                _, p_value = stats.mannwhitneyu(current_region_frac_mods, methyl_frac_mod, alternative=alt, nan_policy="omit")
            else:
                raise ValueError(f"Invalid statistical test: {self.stat}")

            for j in range(self.window_size):
                p_values[i][j] = p_value
        methylation["p-values"] = [np.median(ps) for ps in p_values]

        return methylation

    def find_priors(self, methylation):
        methyl_starts = np.array(methylation["starts"], dtype=int)
        methyl_p_values = np.array(methylation["p-values"], dtype=float)

        # create an array to store confidence levels (0: no confidence, 1: low confidence, 2: high confidence)
        mdr_conf = np.zeros(len(methyl_p_values), dtype=int)
        mdr_conf = np.where(np.array(methyl_p_values) <= self.transition_p, 1, mdr_conf)
        mdr_conf = np.where(np.array(methyl_p_values) <= self.cdr_p, 2, mdr_conf)
        
        mdrs = {"starts": [], "ends": [], "names": [], "scores": [], "strands": [], "itemRgbs": []}
        
        # helper function to add entries to the mdrs dictionary
        def create_entry_helper(start_idx, end_idx, name, scores, color):
            mdrs["starts"].append(methyl_starts[start_idx])
            mdrs["ends"].append(methyl_starts[end_idx] + 1)
            mdrs["names"].append(name)
            mdrs["scores"].append(np.median(scores))
            mdrs["strands"].append(".")
            mdrs["itemRgbs"].append(color)

        # find potential MDR regions
        potential_mdr_idxs = np.where(mdr_conf > 0)[0]
        potential_mdr_diff = np.diff(potential_mdr_idxs)
        potential_mdr_breaks = np.where(potential_mdr_diff > self.merge_distance)[0] + 1 
        potential_mdrs = np.split(potential_mdr_idxs, potential_mdr_breaks)

        temp_entries = []
        
        for mdr in potential_mdrs:
            if any(mdr_conf[m]==2 for m in mdr):
                # process high confidence regions
                hc_idxs = np.where(mdr_conf[mdr] == 2)[0]
                hc_diff = np.diff(hc_idxs)
                hc_breaks = np.where(hc_diff > self.merge_distance)[0] + 1 
                hc_mdrs = np.split(hc_idxs, hc_breaks)

                lc_start_idx = mdr[0]
                lc_scores = []

                for hc in hc_mdrs: 
                    hc_start_idx, hc_end_idx = mdr[hc[0]], mdr[hc[-1]]
                    hc_scores = methyl_p_values[hc_start_idx:hc_end_idx+1]

                    # the number of hc_idxs that are in hc is >= self.min_sig_cpgs
                    if len([idx for idx in hc_idxs if idx in hc]) >= self.min_sig_cpgs:
                        # Add low confidence region before high confidence region
                        if lc_scores:
                            create_entry_helper(lc_start_idx, hc_start_idx, f"transition_{self.output_label}", lc_scores, self.transition_color)
                            lc_scores = []
                        # Add high confidence region
                        create_entry_helper(hc_start_idx, hc_end_idx, f"{self.output_label}", hc_scores, self.cdr_color)
                        lc_start_idx = hc_end_idx + 1
                    else:
                        # If high confidence region is too small, add it to low confidence scores
                        lc_scores.extend(hc_scores)

                # Add remaining low confidence region
                if lc_scores or lc_start_idx <= mdr[-1]:
                    lc_scores.extend(methyl_p_values[lc_start_idx:mdr[-1]+1])
                    create_entry_helper(lc_start_idx, mdr[-1], f"transition_{self.output_label}", lc_scores, self.transition_color)
                    
        return mdrs


    def centrodip_single_chromosome(self, chrom, methylation, regions):
        methylation_sig = self.calculate_regional_stats(methylation)
        priors = self.find_priors(methylation_sig)
        return ( chrom, priors, methylation_sig)

    def centrodip_all_chromosomes(self, methylation_all_chroms, regions_all_chroms):
        priors_all_chroms, methylation_emissions_priors_all_chroms = {}, {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(
                    self.centrodip_single_chromosome,
                    chrom, methylation_all_chroms[chrom], regions_all_chroms[chrom],
                ): chrom
                for chrom in methylation_all_chroms
            }

            for future in concurrent.futures.as_completed(futures):
                (
                    chrom, priors, methylation_emissions_priors,
                ) = future.result()

                priors_all_chroms[chrom] = priors
                methylation_emissions_priors_all_chroms[chrom] = methylation_emissions_priors

        return priors_all_chroms, methylation_emissions_priors_all_chroms


def main():
    argparser = argparse.ArgumentParser(
        description="Process bedMethyl and CenSat BED file to produce hmmCDR priors"
    )

    # required inputs
    argparser.add_argument("bedmethyl", type=str, help="Path to the bedmethyl file")
    argparser.add_argument("regions", type=str, help="Path to BED file of regions")
    argparser.add_argument("output", type=str, help="Path to the output BED file")

    # BedParser arguments
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

    # mwCDR arguments
    argparser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=101,
        help="Number of CpGs to include in rolling window for CDR calculation. (default: 31)",
    )
    argparser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for rolling window calculations of CDRs. (default: 1)",
    )
    argparser.add_argument(
        "--stat",
        type=str,
        default='mannwhitneyu',
        help="Statistical test to perform when determining p-value of CpG site. Options are 'mannwhitneyu', 'ks', or 't' (default: 'mannwhitneyu')",
    )
    argparser.add_argument(
        "--cdr_p",
        type=float,
        default=0.000001,
        help="Cutoff for high confidence MDR p-value. (default: 0.000001)",
    )
    argparser.add_argument(
        "--transition_p",
        type=float,
        default=0.0,
        help="Cutoff for low confidence MDR p-value. (default: 0.0)",
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
        help='Label to use for name column of priorCDR BED file. (default: "subCDR")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

    def generate_output_bed(all_chroms_dict, output_file, columns=["starts", "ends"]):
        all_lines = []
        for chrom in all_chroms_dict:
            chrom_data = all_chroms_dict[chrom]
            for i in range(len(chrom_data["starts"])):
                line = [chrom]
                for col in columns:
                    if col in chrom_data:
                        line.append(str(chrom_data[col][i])) 
                all_lines.append(line)
                
        all_lines = sorted(all_lines, key=lambda x: (x[0], int(x[1])))
        with open(output_file, 'w') as file:
            for line in all_lines: 
                file.write("\t".join(line) + "\n")

    parse_beds = BedParser(
        mod_code=args.mod_code,
        methyl_bedgraph=args.methyl_bedgraph,
        min_valid_cov=args.min_valid_cov,
        region_edge_filter=args.region_edge_filter,
    )

    regions_dict, methylation_dict = parse_beds.process_files(
        methylation_path=args.bedmethyl,
        regions_path=args.regions,
    )

    if args.output_all:
        generate_output_bed(regions_dict, f"{output_prefix}_regions.bed", columns=["starts", "ends"])
        generate_output_bed(methylation_dict, f"{output_prefix}_region_frac_mod.bedgraph", columns=["starts", "ends", "fraction_modified"])

    centro_dip = CentroDip(
        window_size=args.window_size,
        step_size=args.step_size,
        stat=args.stat,
        cdr_p=args.cdr_p,
        transition_p=args.transition_p,
        min_sig_cpgs=args.min_sig_cpgs,
        merge_distance=args.merge_distance,
        enrichment=args.enrichment,
        threads=args.threads,
        cdr_color=args.cdr_color,
        transition_color=args.transition_color,
        output_label=args.output_label,
    )

    (
        cdrs_all_chroms,
        methylation_sig_all_chroms,
    ) = centro_dip.centrodip_all_chromosomes(methylation_all_chroms=methylation_dict, regions_all_chroms=regions_dict)

    generate_output_bed(cdrs_all_chroms, f"{args.output}", columns=["starts", "ends", "names", "scores", "strands", "starts", "ends", "itemRgbs"])
    generate_output_bed(methylation_sig_all_chroms, f"{output_prefix}_pvalues.bedgraph", columns=["starts", "ends", "p-values"])

if __name__ == "__main__":
    main()