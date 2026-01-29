#!/usr/bin/env python3

import os
import sys
import argparse

import concurrent.futures

from pathlib import Path

from centrodip.bedtable import BedTable
import centrodip.bedmethyl_smooth as bms
import centrodip.detect_dips as dd
import centrodip.filter_dips as fd
import centrodip.summary_plot as pd


def _process_chrom(item):
    chrom, bm_chr, argsd = item

    bedGraph_LOWESS = bms.bedMethyl_LOWESS(
        bm_chr,
        window_bp=argsd["window_size"],
        cov_conf=argsd["cov_conf"],
        y_col_1based=11 if not argsd["bedgraph"] else 7,
        cov_col_1based=10 if not argsd["bedgraph"] else None,
    )

    dips, lowess_bg_stats = dd.detectDips(
        bedgraph=bedGraph_LOWESS,
        prominence=argsd["prominence"],
        height=argsd["height"],
        enrichment=argsd["enrichment"],
        broadness=argsd["broadness"],
        label=argsd["label"],
        color=argsd["color"],
    )

    filtered_dips = fd.filterDips(
        dips=dips,
        min_size=argsd["min_size"],
        min_score=argsd["min_score"],
        cluster_distance=argsd["cluster_distance"],
    )

    debug_msg = None
    if argsd["debug"]:
        debug_msg = (
            f"[DEBUG] {chrom}:\n"
            f" - Smoothed {len(bedGraph_LOWESS)} CpG sites.\n"
            f" - Background identified: median={lowess_bg_stats['median']:.3f}; "
            f"IQR=({lowess_bg_stats['p25']:.3f}, {lowess_bg_stats['p75']:.3f}); "
            f"n={len(lowess_bg_stats['values'])}.\n"
            f" - Detected {len(dips)} dips.\n"
            f" - Filtered to {len(filtered_dips)} dips.\n"
        )

    plot_path = None
    if argsd["plot"]:
        out_path = Path(argsd["output"])
        plot_dir = out_path.parent / f"{out_path.stem}_plots"
        plot_path = str(plot_dir / f"{out_path.stem}.{chrom}.summary.png")

    return chrom, bm_chr, bedGraph_LOWESS, dips, filtered_dips, lowess_bg_stats, debug_msg, plot_path


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
        help='Modification code to filter bedMethyl file. Selects rows with this value in the fourth column. (default: "m")',
    )
    parsing_group.add_argument(
        "--bedgraph",
        action="store_true",
        default=False,
        help='Input file in a bedGraph format rather than bedMethyl. Requires bedGraph4 with the fourth column being fraction modified (default: False)',
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
        default=10,
        help="Minimum coverage required to be a confident CpG site. (default: 10)",
    )

    dip_detect_group = parser.add_argument_group('Detection Options')
    dip_detect_group.add_argument(
        "--prominence",
        type=float,
        default=0.5,
        help="Sensitivity of dip detection for scipy.signal.find_peaks. Higher values require more pronounced dips. Must be a float between 0 and 1. (default: 0.5)",
    )
    dip_detect_group.add_argument(
        "--height",
        type=float,
        default=0.1,
        help="Minimum depth for dip detection, lower values require deeper dips. Must be a float between 0 and 1. (default: 0.1)",
    )
    dip_detect_group.add_argument(
        "--broadness",
        type=float,
        default=0.75,
        help="Broadness of dips called, higher values make broader entries. Must be a float between 0 and 1. (default: 0.75)",
    )
    dip_detect_group.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Find regions that are enriched (rather than depleted) for methylation. (default: False)",
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
        default=500,
        help="Minimum score that a dip must have to be kept. Must be an int between 0 and 1000.  (default: 500)",
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
        "--plot",
        action="store_true",
        default=False,
        help="Create summary plot of the results. Written to <output_prefix>.summary.png (default: False)",
    )
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
    if not args.bedgraph:
        bedMethyl_in_region = bedMethyl_in_region.filter(lambda r: r.name == args.mod_code)

    out_path = Path(args.output)
    plot_dir = out_path.parent / f"{out_path.stem}_plots"
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    argsd = {
        "mod_code": args.mod_code,
        "bedgraph": args.bedgraph,
        "window_size": args.window_size,
        "cov_conf": args.cov_conf,
        "prominence": args.prominence,
        "height": args.height,
        "enrichment": args.enrichment,
        "broadness": args.broadness,
        "label": args.label,
        "color": args.color,
        "min_size": args.min_size,
        "min_score": args.min_score,
        "cluster_distance": args.cluster_distance,
        "debug": args.debug,
        "plot": args.plot,
        "output": out_path,
    }

    chrom_map = bedMethyl_in_region.groupby_chrom()  # should be dict-like: chrom -> BedTable or list[IntervalRecord]

    work_items = []
    for chrom, chrom_records in chrom_map.items():
        # If groupby_chrom returns lists of records, wrap into BedTable
        if isinstance(chrom_records, BedTable):
            bm_chr = chrom_records
        else:
            bm_chr = BedTable(list(chrom_records), inferred_kind=bedMethyl_in_region.inferred_kind)

        work_items.append((chrom, bm_chr, argsd))

    all_lowess = []
    all_dips = []
    all_filtered = []
    all_bg_stats = {}  

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(_process_chrom, item) for item in work_items]

        for fut in concurrent.futures.as_completed(futures):
            chrom, bm_chr, bedGraph_LOWESS, dips, filtered_dips, lowess_bg_stats, debug_msg, plot_path = fut.result()

            # debug printing (main process so logs aren't interleaved as badly)
            if debug_msg:
                print(debug_msg, end="")

            # plotting (do this in main process to avoid matplotlib multiprocessing issues)
            if args.plot and plot_path is not None:
                Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
                if args.debug:
                    print(f"Writing summary plot to: {plot_path}")
                pd.centrodipSummaryPlot_bedtable(
                    bedMethyl=bm_chr,              # plot only this chrom
                    regions=regions,
                    lowess_bg=bedGraph_LOWESS,
                    dips_unfiltered=dips,
                    dips_final=filtered_dips,
                    output_path=plot_path,
                )

            # Collect outputs
            all_lowess.extend(list(bedGraph_LOWESS._records))
            all_dips.extend(list(dips._records))
            all_filtered.extend(list(filtered_dips._records))
            all_bg_stats[chrom] = lowess_bg_stats

    # ---- Concatenate into single BedTables ----
    bedGraph_LOWESS_all = BedTable(all_lowess, inferred_kind="bedgraph", inferred_ncols=4)
    dips_all = BedTable(all_dips, inferred_kind="bed", inferred_ncols=6)
    filtered_dips_all = BedTable(all_filtered, inferred_kind="bed", inferred_ncols=6)

    if args.debug:
        # save smoothed bedMethyl
        lowess_path = str(Path(args.output).with_suffix(".LOWESS.bedgraph"))
        print(f"Smoothed bedMethyl out: {lowess_path}")
        bedGraph_LOWESS_all.to_path(lowess_path)

        # save unfiltered/detected dips
        unfiltered_path = str(Path(args.output).with_suffix(".detected_dips.bed"))
        print(f"Detected dips out: {unfiltered_path}")
        dips_all.to_path(unfiltered_path)

    # -------------------------
    # Write output (FINAL dips)
    # -------------------------
    if args.debug:
        print(f"Writing output to: {args.output}")
    filtered_dips_all.to_path(out_path)


if __name__ == "__main__":
    main()