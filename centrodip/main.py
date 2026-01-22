#!/usr/bin/env python3

import argparse

from pathlib import Path
from bedtable import BedTable

import bedmethyl_smooth as bms
import detect_dips as dd
import filter_dips as fd
import summary_plot as pd


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect BED / bedGraph files using BedTable"
    )

    # take in positional - file paths
    parser.add_argument("bedMethyl", type=str, help="Path to the bedMethyl file")
    parser.add_argument("regions", type=str, help="Path to BED file of regions to search for dips")
    parser.add_argument("output", type=str, help="Path to the output BED file")

    parsing_group = parser.add_argument_group('Input Options')
    parsing_group.add_argument(
        "--mod-code",
        type=str,
        default="m",
        help='Modification code to filter bedMethyl file. Selects rows with this as fourth column value. (default: "m")',
    )

    smoothing_group = parser.add_argument_group('Smoothing Options')
    smoothing_group.add_argument(
        "--window-size",
        type=int,
        default=10000,
        help="Window size (bp) to use in LOWESS smoothing of fraction modified. (default: 10000)",
    )
    smoothing_group.add_argument(
        "--cov-conf",
        type=int,
        default=1,
        help="Minimum coverage required to be a confident CpG site. (default: 10)",
    )

    dip_detect_group = parser.add_argument_group('Detection Options')
    dip_detect_group.add_argument(
        "--prominence",
        type=float,
        default=0.25,
        help="Sensitivity of dip detection. Must be a float between 0 and 1. Higher values require more pronounced dips. (default: 0.25)",
    )
    dip_detect_group.add_argument(
        "--height",
        type=float,
        default=0.1,
        help="Height for dip detection. Lower values filter more dips. Must be a float between 0 and 1. (default: 0.1)",
    )
    dip_detect_group.add_argument(
        "--broadness",
        type=float,
        default=0.5,
        help="Broadness of dips called. Higher values make broader entries. Must be a float between 0 and 1. (default: 0.5)",
    )
    dip_detect_group.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Find regions enriched (rather than depleted) for methylation.",
    )

    dip_filter_group = parser.add_argument_group('Filtering Options')
    dip_filter_group.add_argument(
        "--min-size",
        type=int,
        default=1000,
        help="Minimum dip size in base pairs. (default: 1000)",
    )
    dip_filter_group.add_argument(
        "--min-score",
        type=float,
        default=100,
        help="Minimum score that a dip must have to be kept. Must be an int between 0 and 1000.  (default: 100)",
    )
    dip_filter_group.add_argument(
        "--cluster-distance",
        type=int,
        default=500000,
        help="Cluster distance in base pairs. Attempts to keep the single largest cluster of annotationed dips. Negative Values turn it off. (default: 500000)",
    )

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--label",
        type=str,
        default="CDR",
        help='Label to use for regions in BED output. (default: "CDR")',
    )
    output_group.add_argument(
      "--color",
        type=str,
        default="50,50,255",
        help='Color of predicted dips. (default: "50,50,255")',
    )

    other_arguments_group = parser.add_argument_group('Other Options')
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

    args = parser.parse_args()

    # -------------------------
    # Load files
    # -------------------------
    bedMethyl = BedTable.from_path(args.bedMethyl)
    regions = BedTable.from_path(args.regions)

    # -------------------------
    # Subset bedMethyl to overlap w/ regions
    # -------------------------
    overlapping_records = []
    regions_by_chrom = regions.groupby_chrom()

    for r in bedMethyl:
        chrom_regions = regions_by_chrom.get(r.chrom)
        if chrom_regions is None:
            continue

        # If this bedMethyl record overlaps ANY region, keep it
        for reg in chrom_regions:
            if r.overlaps(reg.chrom, reg.start, reg.end):
                overlapping_records.append(r)
                break

    bedMethyl_in_region = BedTable(
        overlapping_records,
        inferred_kind=bedMethyl.inferred_kind,
    )

    lowess_records = []
    dip_records = []

    for r in bedMethyl_in_region.groupby_chrom():
        # -------------------------
        # Smooth bedMethyl
        # -------------------------
        bedGraph_LOWESS = bms.bedMethyl_LOWESS(
            bedMethyl_in_region,
            window_bp = args.window_size,
            cov_conf = args.cov_conf,
            y_col_1based = 11,
            cov_col_1based = 10,
        )

        if True:
            print("Smoothed bedMethyl out: {}".format("bedMethyl_LOWESS.bedgraph"))
            bedGraph_LOWESS.to_path("bedMethyl_LOWESS.bedgraph")

        # -------------------------
        # Detect dips
        # -------------------------
        dips = dd.detectDips(
            bedgraph=bedGraph_LOWESS,
            prominence=args.prominence,
            height=args.height,
            enrichment=args.enrichment,
            broadness=args.broadness,
            label=args.label,
            color=args.color,
            debug=True,
        )

        # -------------------------
        # Filter dips
        # -------------------------
        filtered_dips = fd.filterDips(
            dips=dips,
            min_size=args.min_size,
            min_score=args.min_score,
            cluster_distance=args.cluster_distance
        )

    # -------------------------
    # Write output
    # -------------------------
    print(f"Writing output to: {args.output}")
    dips.to_path(args.output)

    if args.plot:


        plot_path = str(Path(args.output).with_suffix(".summary.png"))
        print(f"Writing summary plot to: {plot_path}")
        pd.centrodipSummaryPlot_bedtable(
            bedMethyl=bedMethyl,
            lowess_bg=bedGraph_LOWESS,
            dips_unfiltered=dips,
            dips_final=filtered_dips,
            output_path=plot_path
        )


if __name__ == "__main__":
    main()