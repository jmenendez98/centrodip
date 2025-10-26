from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List

from .load_data import DataHandler
from .detect_dips import detectDips
from .filter_dips import filterDips
from .summary_plot import centrodip_summary_plot


def main() -> None:
    def zero_to_one_float(x):
        """Type function for argparse to ensure 0 < x < 1."""
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x!r} is not a valid float")
        if x <= 0.0 or x >= 1.0:
            raise argparse.ArgumentTypeError(f"{x} not in range (0, 1)")
        return x

    argparser = argparse.ArgumentParser(
        description="Process bedMethyl and region BED files to produce CDR predictions.",
    )

    argparser.add_argument("bedmethyl", type=str, help="Path to the bedMethyl file")
    argparser.add_argument("regions", type=str, help="Path to BED file of regions to search for dips")
    argparser.add_argument("output", type=str, help="Path to the output BED file")

    parsing_group = argparser.add_argument_group('Input Options')
    parsing_group.add_argument(
        "--mod-code",
        type=str,
        default="m",
        help='Modification code to filter bedMethyl file. Selects rows with this as fourth column value. (default: "m")',
    )
    parsing_group.add_argument(
        "--bedgraph",
        action="store_true",
        default=False,
        help="Treat methylation input as a bedGraph. Takes Fraction Modified from the fourth column. (default: False)",
    )
    parsing_group.add_argument(
        "--window-size",
        type=int,
        default=10000,
        help="Window size (bp) to use in LOWESS smoothing of fraction modified. (default: 10000)",
    )

    dip_detect_group = argparser.add_argument_group('Detection Options')
    dip_detect_group.add_argument(
        "--prominence",
        type=zero_to_one_float,
        default=0.10,
        help="Sensitivity of dip detection. Must be a float between 0 and 1. Higher values require more pronounced dips. (default: 0.5)",
    )
    dip_detect_group.add_argument(
        "--height",
        type=float,
        default=10,
        help="Height for dip detection. Must be a float between 0 and 100. Lower values filter more dips. (default: 10)",
    )
    dip_detect_group.add_argument(
        "--broadness",
        type=float,
        default=75,
        help="Broadness of dips called. Higher values make broader entries. (default: 75)",
    )
    dip_detect_group.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Find regions enriched (rather than depleted) for methylation.",
    )

    dip_filter_group = argparser.add_argument_group('Filtering Options')
    dip_filter_group.add_argument(
        "--min-size",
        type=int,
        default=5000,
        help="Minimum dip size in base pairs. (default: 5000)",
    )
    dip_filter_group.add_argument(
        "--min-z-score",
        type=int,
        default=1.5,
        help="Minimum difference in Z-score that an entry must be from the rest of the data to be kept. (default: 1)",
    )
    dip_filter_group.add_argument(
        "--cluster-distance",
        type=int,
        default=250000,
        help="Cluster distance in base pairs. Attempts to keep the single largest cluster of annotationed dips. Negative Values turn it off. (default: 250000)",
    )

    other_arguments_group = argparser.add_argument_group('Other Options', 'Miscellaneous arguments affecting outputs and runtime.')
    other_arguments_group.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker processes. (default: 4)",
    )
    other_arguments_group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Dumps smoothed methylation values, their derivatives, methylation peaks, and derivative peaks. Each to separate BED/BEDGraph files. (default: False)",
    )
    other_arguments_group.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Create summary plot of the results. Written to <output_prefix>.summary.png (default: False)",
    )
    other_arguments_group.add_argument(
        "--label",
        type=str,
        default="CDR",
        help='Label to use for regions in BED output. (default: "CDR")',
    )
    other_arguments_group.add_argument(
      "--color",
        type=str,
        default="50,50,255",
        help='Color of predicted dips. (default: "50,50,255")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

    # -- load data --
    input_data = DataHandler(
        methylation_path=args.bedmethyl,
        regions_path=args.regions,
        mod_code=args.mod_code,
        bedgraph=args.bedgraph,
        smooth_window_bp=args.window_size,
        threads=args.threads,
        debug=args.debug,
    )

    # if debug is on, write out the smoothed values
    if args.debug:
        with open(f"{output_prefix}.debug.smooth_frac_mod.bedgraph", "w", encoding="utf-8") as handle:
            lines = []
            for region, values in input_data.methylation_dict.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]

                assert len(values.get("position", [])) == len(values.get("lowess_fraction_modified", [])), "Error mismatch size in output data."
                for i in range(len(values.get("position", []))):
                    start = values["position"][i]
                    frac_mod = values["lowess_fraction_modified"][i]
                    lines.append(f"{chrom}\t{start}\t{start + 1}\t{frac_mod}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines)

        with open(f"{output_prefix}.debug.smooth_frac_mod_dy.bedgraph", "w", encoding="utf-8") as handle:
            lines = []
            for region, values in input_data.methylation_dict.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]

                assert len(values.get("position", [])) == len(values.get("lowess_fraction_modified_dy", [])), "Error mismatch size in output data."
                for i in range(len(values.get("position", []))):
                    start = values["position"][i]
                    frac_mod = values["lowess_fraction_modified_dy"][i]
                    lines.append(f"{chrom}\t{start}\t{start + 1}\t{frac_mod}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines)

    # -- detect dips --
    raw_dips = detectDips(
        methylation_data=input_data.methylation_dict,
        regions_data=input_data.region_dict,
        prominence=args.prominence,
        height=args.height,
        broadness=args.broadness,
        enrichment=args.enrichment,
        threads=args.threads,
        debug=args.debug,
    )

    # if debug is on, write out the dip centers and edges
    if args.debug:
        with open(f"{output_prefix}.debug.dip_centers.bed", "w", encoding="utf-8") as handle:
            lines = []
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]
                for pos in values.get("dip_centers", []) or []:
                    start = int(pos)
                    lines.append(f"{chrom}\t{start}\t{start + 1}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines)

        with open(f"{output_prefix}.debug.dip_edges.bed", "w", encoding="utf-8") as handle:
            lines = []
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]
                for pos in values.get("dip_edges", []) or []:
                    start = int(pos)
                    lines.append(f"{chrom}\t{start}\t{start + 1}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines) 

        with open(f"{output_prefix}.debug.no_filt_dips.bed", "w", encoding="utf-8") as handle:
            lines = []
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]
                assert len(values.get("starts", [])) == len(values.get("ends", [])), "Mismatched dip starts and ends lengths in debug output."
                for i in range(len(values.get("starts", []))):
                    start = values["starts"][i]
                    end = values["ends"][i]
                    lines.append(f"{chrom}\t{start}\t{end}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines)

    # -- filter dips --
    final_dips = filterDips(
        dip_dict=raw_dips,
        min_size=args.min_size,
        min_zscore=args.min_z_score,
        cluster_distance=args.cluster_distance,
    )

    # write final dips to output BED file
    with open(f"{output_prefix}.bed", "w", encoding="utf-8") as handle:
        lines = []
        for region, values in final_dips.items():
            if not values:
                continue
            chrom = region.split(":", 1)[0]
            assert len(values.get("starts", [])) == len(values.get("ends", [])), "Mismatched dip starts and ends lengths in debug output."
            for i in range(len(values.get("starts", []))):
                start = values["starts"][i]
                end = values["ends"][i]
                lines.append(f"{chrom}\t{start}\t{end}\t{args.label}\t0\t.\t{start}\t{end}\t{args.color}\n")
        lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
        handle.writelines(lines)

    # -- make summary plot --
    if args.plot:
        summary_path = f"{output_prefix}.centrodip_summary.png"
        centrodip_summary_plot(
            regions_per_chrom=input_data.region_dict,
            methylation_per_region=input_data.methylation_dict,
            final_dips=final_dips,
            unfiltered_dips=raw_dips,
            output_path=summary_path
        )


if __name__ == "__main__":
    main()
