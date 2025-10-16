import argparse
import concurrent.futures
import os

import numpy as np
import scipy

class Dip_Detector:
    def __init__(
        self,
        window_size,
        threshold,
        prominence,
        min_size,
        enrichment,
        color,
        threads,
        label,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.prominence = prominence

        self.min_size = min_size

        self.enrichment = enrichment

        self.color = color

        self.threads = threads
        self.label = label

    def smooth_methylation(self, methylation):
        # run data through savgol filtering
        methyl_frac_mod = np.array(methylation["fraction_modified"], dtype=float)
        methylation["savgol_frac_mod"] = scipy.signal.savgol_filter(
            x=methyl_frac_mod, 
            window_length=self.window_size, 
            polyorder=2, 
            mode='mirror'
        )
        methylation["savgol_frac_mod_dy"] = scipy.signal.savgol_filter(
            x=methyl_frac_mod, 
            window_length=self.window_size, 
            polyorder=2, 
            deriv=1,
            mode='mirror'
        )

        return methylation

    def detect_dips(self, methylation):
        data = np.array(methylation["savgol_frac_mod"], dtype=float)
        data_range = np.max(data) - np.min(data)

        height_threshold = np.mean(data)-(np.std(data)*self.threshold)          # calculate the height threshold
        prominence_threshold = self.prominence * data_range                     # calculate the prominence threshold

        if not self.enrichment:
            dips, _ = scipy.signal.find_peaks(
                -data,
                height=-height_threshold, 
                prominence=prominence_threshold,
                wlen=(len(data))
            )
        else:
            dips, _ = scipy.signal.find_peaks(
                data,
                height=height_threshold, 
                prominence=prominence_threshold,
                wlen=(len(data))
            )

        return dips

    def extend_dips(self, methylation, dips):
        data = np.array(methylation["savgol_frac_mod"], dtype=float)
        dy = np.array(methylation["savgol_frac_mod_dy"], dtype=float)
        median = np.median(data)

        mask = data > median if self.enrichment else data < median              # mask out invalid values
        prev = np.r_[False, mask[:-1]]
        next_ = np.r_[mask[1:], False]
        starts = np.flatnonzero(mask & ~prev)
        ends = np.flatnonzero(mask & ~next_)
        lefts = np.maximum(starts - 1, 0)                                       # safe finding of right/left index
        rights = np.minimum(ends + 1, data.size - 1)
        dips_arr = np.asarray(list(dips), dtype=int)                            # np array of dips
        idx = np.searchsorted(starts, dips_arr, side="right") - 1
        valid = (idx >= 0)
        valid &= mask[dips_arr]
        valid &= dips_arr <= ends[idx.clip(min=0)]                              # safe gather
        idx = idx[valid]
        dip_bounds = [(int(lefts[i]), int(rights[i])) for i in idx]

        # trim dip calls - set the edge to be where the slope is the most prominent, while within the bounds
        dip_bounds_adj = []
        for d, (l, r) in zip(dips, dip_bounds):
            l_adj = int( np.argmin(dy[range(l, d+1)]) + l )
            r_adj = int( np.argmax(dy[range(d, r+1)]) + d )
            dip_bounds_adj.append((l_adj, r_adj))

        return dip_bounds_adj

    def filter_dips(self, methylation, idxs):
        dip_calls = {
            "starts": [],
            "ends": [],
            "names": [],
            "scores": [],
            "strands": [],
            "itemRgbs": []
        }

        starts = np.array(methylation["starts"], dtype=int)

        # remove dip calls that are too small
        for i in range(len(idxs)):
            li, ri = idxs[i][0], idxs[i][1]
            lp, rp = starts[li], starts[ri]+1
            if abs(starts[li]-starts[ri]+1) >= self.min_size:
                dip_calls["starts"].append(lp)
                dip_calls["ends"].append(rp)
                dip_calls["names"].append(f"{self.label}")
                dip_calls["scores"].append(1)
                dip_calls["strands"].append(".")
                dip_calls["itemRgbs"].append(f"{self.color}")

        return dip_calls

    def centrodip_single_chromosome(self, region, methylation):
        # if the region has less CpG's than the window size do not process
        if len(methylation['starts']) < self.window_size:
            return ( region, {}, {}, {} )
        methylation_smoothed = self.smooth_methylation(methylation)

        dip_sites = self.detect_dips(methylation_smoothed)
        dip_idxs = self.extend_dips(methylation_smoothed, dip_sites)
        dips = self.filter_dips(methylation_smoothed, dip_idxs)

        return ( region, dips, methylation_smoothed)

    def centrodip_all_chromosomes(self, methylation_per_region, regions_per_chrom):
        dips_all_chroms, methylation_all_chroms = {}, {}

        regions = list(methylation_per_region.keys())
        if not regions:
            return dips_all_chroms, methylation_all_chroms

        if self.threads <= 1 or len(regions) == 1:
            for region in regions:
                region, mdrs, methylation_pvalues = self.centrodip_single_chromosome(
                    region, methylation_per_region[region]
                )
                dips_all_chroms[region] = mdrs
                methylation_all_chroms[region] = methylation_pvalues
            return dips_all_chroms, methylation_all_chroms

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(
                    self.centrodip_single_chromosome,
                    region,
                    methylation_per_region[region],
                ): region for region in regions
            }    

            for future in concurrent.futures.as_completed(futures):
                (    
                    region,
                    mdrs,
                    methylation_pvalues,
                ) = future.result()

                dips_all_chroms[region] = mdrs
                methylation_all_chroms[region] = methylation_pvalues

        return dips_all_chroms, methylation_all_chroms


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
        help="Number of standard deviations from the smoothed mean to be the minimum dip. Lower values increase leniency of dip calls. (default: 1)",
    )
    argparser.add_argument(
        "--prominence",
        type=float,
        default=0.66,
        help="Scalar factor to decide the prominence required for an dip. Scalar is multiplied by smoothed data's difference in the minimum and maxiumum values. Lower values increase leniency of MDR calls. (default: 0.66)",
    )
    argparser.add_argument(
        "--min-size",
        type=int,
        default=5000,
        help="Minimum dip size in base pairs. Small dips are removed. (default: 5000)",
    )
    argparser.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Use centrodip to find areas enriched in aggregated methylation calls. (default: False)",
    )
    argparser.add_argument(
        "--threads",
        type=int,
        default=4,
        help='Number of workers. (default: 4)',
    )

    # output arguments
    argparser.add_argument(
        "--color",
        type=str,
        default="50,50,255",
        help='Color of predicted dips. (default: 50,50,255)',
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
        default="CDR",
        help='Label to use for regions in BED output. (default: "CDR")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

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
        threshold=args.threshold,
        prominence=args.prominence,
        min_size=args.min_size,
        enrichment=args.enrichment,
        threads=args.threads,
        color=args.color,
        label=args.label
    )

    (
        mdrs_per_region,
        methylation_per_region
    ) = centro_dip.centrodip_all_chromosomes(methylation_per_region=methylation_per_region_dict, regions_per_chrom=regions_per_chrom_dict)

    def generate_output_bed(bed_dict, output_file, columns=["starts", "ends"]):
        if not bed_dict:
            return

        lines = []
        keys = list(bed_dict.keys())
        chroms = [region.split(':')[0] for region in list(bed_dict.keys())]
            
        for key, chrom in zip(keys, chroms):
            chrom_data = bed_dict[key]

            if chrom_data:
                for i in range( len(chrom_data.get("starts", [])) ):
                    line = [chrom]
                    for col in columns:
                        if col in chrom_data:
                            line.append(str(chrom_data[col][i])) 
                    lines.append(line)

        # if nothing is in all_lines, return nothing and don't write to file
        if lines:        
            lines = sorted(lines, key=lambda x: (x[0], int(x[1])))
            with open(output_file, 'w') as file:
                for line in lines: 
                    file.write("\t".join(line) + "\n")
        else:
            return

    if args.output_all:
        generate_output_bed(methylation_per_region, f"{output_prefix}_savgol_frac_mod.bedgraph", columns=["starts", "ends", "savgol_frac_mod"])

    generate_output_bed(mdrs_per_region, f"{args.output}", columns=["starts", "ends", "names", "scores", "strands", "starts", "ends", "itemRgbs"])