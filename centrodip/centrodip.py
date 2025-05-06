import argparse
import concurrent.futures
import warnings
import os

import numpy as np
import scipy


class BedParse:
    """parser to read in region and methylation bed files."""

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

                if (end - self.region_edge_filter) < (start + self.region_edge_filter):
                    continue

                region_dict[chrom]["starts"].append(start + self.region_edge_filter)
                region_dict[chrom]["ends"].append(end - self.region_edge_filter)

        # merge regions that are closer than self.region_merge_distance
        for chrom in region_dict:
            starts = region_dict[chrom]["starts"]
            ends = region_dict[chrom]["ends"]
            
            # sort regions by start position
            sorted_regions = sorted(zip(starts, ends))
            
            merged_starts, merged_ends = [], []
            for start, end in sorted_regions:
                if not merged_starts or start - merged_ends[-1] > self.region_merge_distance:
                    merged_starts.append(start)
                    merged_ends.append(end)
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
                            methyl_dict=methylation_dict, 
                            region_key=region_key, 
                            start=methylation_position, 
                            end=methylation_position+1, 
                            frac_mod=float(columns[3]) if self.bedgraph else float(columns[10]),
                            cov=1 if self.bedgraph else float(columns[4])
                        )
                        
                        break # break if the methylation line is placed

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
        mdr_threshold,
        transition_threshold,
        prominence_constant,
        significance,
        min_size,
        min_cov,
        enrichment,
        mdr_color,
        transition_color,
        low_cov_color,
        threads,
        label,
    ):
        self.window_size = window_size
        self.mdr_threshold = mdr_threshold
        self.transition_threshold = transition_threshold
        self.prominence_constant = prominence_constant

        self.significance = significance
        self.min_size = min_size

        self.min_cov = min_cov

        self.enrichment = enrichment

        self.mdr_color = mdr_color
        self.transition_color = transition_color
        self.low_cov_color = low_cov_color

        self.threads = threads
        self.label = label

    def smooth_methylation(self, methylation):
        data = np.array(methylation["fraction_modified"], dtype=float)

        # run data through savgol filtering
        savgol_frac_mod = scipy.signal.savgol_filter(
            x=data,
            window_length=self.window_size, 
            polyorder=3, 
            mode='mirror',
            deriv=0
        )
        dy = scipy.signal.savgol_filter(
            x=data,
            window_length=self.window_size, 
            polyorder=3, 
            mode='mirror',
            deriv=1
        )
        ddy = scipy.signal.savgol_filter(
            x=data,
            window_length=self.window_size, 
            polyorder=3, 
            mode='mirror',
            deriv=2
        )
       
        # store values in methylation dictionary
        methylation["savgol_frac_mod"] = savgol_frac_mod
        methylation["first_deriv"] = dy
        methylation["second_deriv"] = ddy

        return methylation


    def detect_dips(self, methylation):
        savgol = np.array(methylation["savgol_frac_mod"], dtype=float)

        height_threshold = np.mean(savgol)-(np.std(savgol)*self.mdr_threshold) # calculate the height threshold
        prominence_threshold = self.prominence_constant * (np.percentile(savgol, 99) - np.percentile(savgol, 1)) # calculate the prominence threshold

        if not self.enrichment:
            dips, _ = scipy.signal.find_peaks(
                -savgol,
                height=-height_threshold, 
                prominence=prominence_threshold
            )
        else:
            dips, _ = scipy.signal.find_peaks(
                -savgol,
                height=height_threshold, 
                prominence=prominence_threshold
            )

        return dips

    def detect_edges(self, methylation, dips):
        data = np.array(methylation["fraction_modified"], dtype=float)
        savgol = np.array(methylation["savgol_frac_mod"], dtype=float)
        dy = np.array(methylation["first_deriv"], dtype=float)
        ddy = np.array(methylation["second_deriv"], dtype=float)

        mdr_threshold = np.mean(savgol)-(np.std(savgol)*self.mdr_threshold)
        transition_threshold = np.mean(savgol)-(np.std(savgol)*self.transition_threshold)

        print(mdr_threshold, transition_threshold)

        def find_nearest_inflection(index, direction, condition):
            step = -1 if direction == 'left' else 1
            while 0 <= index < len(ddy) - 1:
                if np.sign(ddy[index]) != np.sign(ddy[index + step]) and condition(dy[index]):
                    return index
                index += step
            return index

        # find the edge indices for each of the dips
        edges = []
        for dip in dips:
            left = right = dip
            while left > 0 and savgol[left] < mdr_threshold:
                left -= 1
            left = find_nearest_inflection(left, 'left', lambda x: x < 0)
            while right < len(savgol) - 1 and savgol[right] < mdr_threshold:
                right += 1
            right = find_nearest_inflection(right, 'right', lambda x: x > 0)

            # find the edge indices for each of the transitions near dips
            transition_edges = []
            t_left = left       
            t_right = right
            if transition_threshold > mdr_threshold:
                while t_left > 0 and savgol[t_left] < transition_threshold:
                    t_left -= 1
                while t_right < len(savgol) - 1 and savgol[t_right] < transition_threshold:
                    t_right += 1
                print(savgol[t_right])

            if left != right:
                edges.append((t_left, left, right, t_right))
                
        return edges

    def filter_dips_and_make_entries(self, methylation, edges):
        mdrs = {
            "starts": [],
            "ends": [],
            "names": [],
            "scores": [],
            "strands": [],
            "itemRgbs": []
        }

        def add_region(start_i, end_i, name, score, color):
            mdrs["starts"].append(int(starts[start_i]))
            mdrs["ends"].append(int(starts[end_i]+1))
            mdrs["names"].append(f"{name}")
            mdrs["scores"].append(score)
            mdrs["strands"].append(".")
            mdrs["itemRgbs"].append(f"{color}")

        data = np.array(methylation["fraction_modified"], dtype=float)
        starts = np.array(methylation["starts"], dtype=float)

        for edge in edges:
            # pull out relavant indices for ease
            mdr_start_i, mdr_end_i = edge[1], edge[2]
            mdr_size = starts[mdr_end_i] - starts[mdr_start_i]

            left_transition_start_i, left_transition_end_i = edge[0], edge[1]
            right_transition_start_i, right_transition_end_i = edge[2], edge[3]
            
            # remove stretches of dips that are too small
            for i in range(len(mdr_idxs)):
                # if mdr is too small skip it and its transitions
                if starts[mdr_idxs[i][1]]-starts[mdr_idxs[i][0]] > self.min_size:
                    add_region(mdr_idxs[i][0], mdr_idxs[i][1], self.label, '.', self.mdr_color)
                    if transition_idxs:
                        for transition in transition_idxs[i]:
                            start_i, end_i = min(transition), max(transition)
                            if starts[end_i]-starts[start_i] >= self.min_size:
                                add_region(start_i, end_i, f'transition_{self.label}', '.', self.transition_color)

        return mdrs

    def find_low_coverage(self, methylation):
        starts = np.array(methylation["starts"], dtype=int)
        coverage = np.array(methylation["valid_coverage"], dtype=int)

        # find all low coverage CpGs
        low_cov_idxs = np.where(coverage < self.min_cov)[0]
        if len(low_cov_idxs) == 0:  
            # check if there are any low coverage indices
            return {"starts": [], "ends": []}

        low_cov_diff = np.diff(low_cov_idxs)
        low_cov_breaks = np.where(low_cov_diff > 1)[0] + 1 
        low_cov_regions = np.split(low_cov_idxs, low_cov_breaks)

        low_covs = {"starts": [], "ends": []}

        for region in low_cov_regions:
            # make region only if is > than self.min_size
            start_idx, end_idx = region[0], region[-1]
            start, end = starts[start_idx], starts[end_idx]+1
            region_size = end - start
            if region_size >= self.min_size:
                low_covs["starts"].append(start)
                low_covs["ends"].append(end)

        return low_covs

    def adjust_for_low_coverage(self, mdrs, low_cov_regions):
        adjusted_mdrs = {
            "starts": [],
            "ends": [],
            "names": [],
            "scores": [],
            "strands": [],
            "itemRgbs": []
        }

        i = 0
        while i < len(mdrs["starts"]):
            mdr_start, mdr_end = mdrs["starts"][i], mdrs["ends"][i]
            adjusted = False

            for j in range(len(low_cov_regions["starts"])):
                low_cov_start, low_cov_end = low_cov_regions["starts"][j], low_cov_regions["ends"][j]

                if (low_cov_start < mdr_end) and (low_cov_end > mdr_start):
                    # CDR overlaps with low coverage region
                    if mdr_start < low_cov_start:
                        # add adjusted CDR before low coverage region
                        adjusted_mdrs["starts"].append(mdr_start)
                        adjusted_mdrs["ends"].append(low_cov_start)
                        adjusted_mdrs["names"].append(mdrs["names"][i])
                        adjusted_mdrs["scores"].append(mdrs["scores"][i])
                        adjusted_mdrs["strands"].append(mdrs["strands"][i])
                        adjusted_mdrs["itemRgbs"].append(mdrs["itemRgbs"][i])

                    # add low coverage entry
                    adjusted_mdrs["starts"].append(low_cov_start)
                    adjusted_mdrs["ends"].append(low_cov_end)
                    adjusted_mdrs["names"].append(f"low_coverage_region(>{self.min_cov})")
                    adjusted_mdrs["scores"].append("0")
                    adjusted_mdrs["strands"].append(".")
                    adjusted_mdrs["itemRgbs"].append(self.low_cov_color)

                    if mdr_end > low_cov_end:
                        # add adjusted CDR
                        adjusted_mdrs["starts"].append(low_cov_end)
                        adjusted_mdrs["ends"].append(mdr_end)
                        adjusted_mdrs["names"].append(mdrs["names"][i])
                        adjusted_mdrs["scores"].append(mdrs["scores"][i])
                        adjusted_mdrs["strands"].append(mdrs["strands"][i])
                        adjusted_mdrs["itemRgbs"].append(mdrs["itemRgbs"][i])

                    adjusted = True
                    break

            if not adjusted:
                # If CDR doesn't overlap with any low coverage region, add it as is
                for key in adjusted_mdrs:
                    adjusted_mdrs[key].append(mdrs[key][i])

            i += 1

        return adjusted_mdrs

    def centrodip_single_chromosome(self, region, methylation):
        if len(methylation['starts']) < self.window_size:
            # if the region has less CpG's than the window size do not process
            return ( region, {}, {}, {} )
        
        methylation = self.smooth_methylation(methylation) 
        dips = self.detect_dips(methylation)
        edges = self.detect_edges(methylation, dips)
        mdrs = self.filter_dips_and_make_entries(methylation, edges)

        if self.min_cov > 1:
            low_cov_regions = self.find_low_coverage(methylation)
            mdrs = self.adjust_for_low_coverage(mdrs, low_cov_regions)
            return ( region, mdrs, low_cov_regions, methylation)

        return ( region, mdrs, {}, methylation)

    def centrodip_all_chromosomes(self, methylation_per_region, regions_per_chrom):
        mdrs_all_chroms, low_cov_all_chroms, methylation_all_chroms = {}, {}, {}

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
                    region, mdrs, low_cov_regions, methylation_pvalues,
                ) = future.result()

                mdrs_all_chroms[region] = mdrs
                low_cov_all_chroms[region] = low_cov_regions
                methylation_all_chroms[region] = methylation_pvalues

        return mdrs_all_chroms, low_cov_all_chroms, methylation_all_chroms


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
        "--mod-code",
        type=str,
        default="m",
        help='Modification code to filter bedMethyl file (default: "m")',
    )
    argparser.add_argument(
        "--bedgraph",
        action="store_true",
        default=False,
        help="Flag indicating the input is a bedgraph. If passed --mod-code and --min-cov are ignored. (default: False)",
    )
    argparser.add_argument(
        "--region-merge-distance",
        type=int,
        default=10000,
        help="Merge gaps in nearby centrodip regions up to this many base pairs. (default: 10000)",
    )
    argparser.add_argument(
        "--region-edge-filter",
        type=int,
        default=0,
        help="Remove edges of merged regions in base pairs. (default: 0)",
    )

    # CentroDip arguments
    argparser.add_argument(
        "--window-size",
        type=int,
        default=101,
        help="Number of CpGs to include in Savitzky-Golay filtering of Fraction Modified. (default: 101)",
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        default=1,
        help="Number of standard deviations to be subtracted from the mean smoothed data to consider as minimum MDR height. Lower values increase leniency of MDR calls. (default: 1)",
    )
    argparser.add_argument(
        "--prominence",
        type=float,
        default=0.5,
        help="Scalar factor to decide the prominence required for an MDR peak. Scalar is multiplied by smoothed data's difference in the 99th and 1st percentiles. Lower values increase leniency of MDR calls. (default: 0.5)",
    )
    argparser.add_argument(
        "--minor-threshold",
        type=int,
        default=0,
        help="Number of standard deviations from the mean smoothed data to consider the transition threshold. (default: 0)",
    )
    argparser.add_argument(
        "--min-size",
        type=int,
        default=1000,
        help="Minimum MDR or Transition size in base pairs. Smaller MDR/Transitions are filtered out. (default: 2500)",
    )
    argparser.add_argument(
        "--min-cov",
        type=int,
        default=1,
        help="Minimum valid coverage (read from bedmethyl) required to call MDR site. Ignored if bedmethyl input is a bedgraph. (default: 1)",
    )
    argparser.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Use centrodip to find areas of enriched methylation. (default: False)",
    )
    argparser.add_argument(
        "--threads",
        type=int,
        default=4,
        help='Number of workers to use for parallelization. (default: 4)',
    )

    # output arguments
    argparser.add_argument(
        "--mdr-color",
        type=str,
        default="50,50,255",
        help='Color of predicted MDRs. (default: 50,50,255)',
    )
    argparser.add_argument(
        "--transition-color",
        type=str,
        default="150,150,255",
        help='Color of predicted MDR Transitions. (default: 150,150,255)',
    )
    argparser.add_argument(
        "--low-cov-color",
        type=str,
        default="211,211,211",
        help='Color of low coverage regions. (default: 211,211,211)',
    )
    argparser.add_argument(
        "--output-all",
        action='store_true',
        default=False,
        help='Output all intermediate files. (default: False)',
    )
    argparser.add_argument(
        "--label",
        type=str,
        default="MDR",
        help='Label to use for regions in BED output. (default: "MDR")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

    if args.bedgraph and args.min_cov > 1:
        raise ValueError("Cannot pass --min-cov > 1 with --bedgraph")

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
                for i in range(len(chrom_data.get("starts", []))):
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
        bedgraph=args.bedgraph,
        region_merge_distance=args.region_merge_distance,
        region_edge_filter=args.region_edge_filter
    )

    regions_per_chrom_dict, methylation_per_region_dict = bed_parser.process_files(
        methylation_path=args.bedmethyl,
        regions_path=args.regions
    )

    centro_dip = CentroDip(
        window_size=args.window_size,
        mdr_threshold=args.mdr_threshold,
        transition_threshold=args.transition_threshold,
        prominence_constant=args.prominence_constant,
        significance=args.significance,
        min_size=args.min_size,
        min_cov=args.min_cov,
        enrichment=args.enrichment,
        threads=args.threads,
        mdr_color=args.mdr_color,
        transition_color=args.transition_color,
        low_cov_color=args.low_cov_color,
        label=args.label
    )

    (
        mdrs_per_region,
        low_coverage_per_region,
        methylation_sig_per_region,
    ) = centro_dip.centrodip_all_chromosomes(methylation_per_region=methylation_per_region_dict, regions_per_chrom=regions_per_chrom_dict)

    if args.output_all:
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_coverages.bedgraph", key_is_regions=True, columns=["starts", "ends", "valid_coverage"])
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_savgol_frac_mod.bedgraph", key_is_regions=True, columns=["starts", "ends", "savgol_frac_mod"])
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_first_deriv.bedgraph", key_is_regions=True, columns=["starts", "ends", "first_deriv"])
        generate_output_bed(methylation_sig_per_region, f"{output_prefix}_second_deriv.bedgraph", key_is_regions=True, columns=["starts", "ends", "second_deriv"])

    print(mdrs_per_region)
    generate_output_bed(mdrs_per_region, f"{args.output}", key_is_regions=True, columns=["starts", "ends", "names", "scores", "strands", "starts", "ends", "itemRgbs"])


if __name__ == "__main__":
    '''
    import cProfile
    with cProfile.Profile() as profile:
        main()
    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    '''
    
    main()