#!/usr/bin/env python3

import os
import sys
import argparse
from importlib.metadata import version, PackageNotFoundError

import concurrent.futures

from pathlib import Path

from centrodip.bedtable import BedTable
import centrodip.bedmethyl_smooth as bms
import centrodip.detect_dips as dd
import centrodip.filter_dips as fd
import centrodip.summary_plot as sp


def _process_chrom(item):
    chrom, bm_chr, r_chr, argsd = item

    bedGraph_LOWESS = bms.bedMethyl_LOWESS(
        chrom=chrom,
        bedMethyl=bm_chr,
        window_bp=argsd["window_size"],
        cov_conf=argsd["cov_conf"],
        y_col_1based=11 if not argsd["bedgraph"] else 7,
        cov_col_1based=10 if not argsd["bedgraph"] else None,
        debug=argsd["debug"],
    )

    dips, bkgrd_stats = dd.detectDips(
        chrom=chrom,
        bedgraph=bedGraph_LOWESS,
        prominence=argsd["prominence"],
        height=argsd["height"],
        enrichment=argsd["enrichment"],
        broadness=argsd["broadness"],
        score_sensitivity=argsd["score_sensitivity"],
        label=argsd["label"],
        color=argsd["color"],
        debug=argsd["debug"],
    )

    filtered_dips = fd.filterDips(
        chrom=chrom,
        dips=dips,
        regions=r_chr,
        min_size=argsd["min_size"],
        min_score=argsd["min_score"],
        cluster_distance=argsd["cluster_distance"],
        debug=argsd["debug"],
    )

    plot_path = None
    if argsd["plot"]:
        out_path = Path(argsd["output"])
        plot_dir = out_path.parent / f"{out_path.stem}_plots"
        plot_path = str(plot_dir / f"{out_path.stem}.{chrom}.summary.png")

    return chrom, bm_chr, bedGraph_LOWESS, dips, filtered_dips, bkgrd_stats, None, plot_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect Centromeric Dip Regions (CDRs) from bedMethyl file.")

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
        "-w", "--window-size",
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
        "-p", "--prominence",
        type=float,
        default=0.333,
        help="Sensitivity of dip detection for scipy.signal.find_peaks. Higher values require more pronounced dips. Must be a float between 0 and 1. (default: 0.334)",
    )
    dip_detect_group.add_argument(
        "--height",
        type=float,
        default=0.1,
        help="Minimum depth for dip detection, lower values require deeper dips. Must be a float between 0 and 1. (default: 0.1)",
    )
    dip_detect_group.add_argument(
        "-b", "--broadness",
        type=float,
        default=0.9,
        help="Broadness of dips called, higher values make broader entries. Recommended to use float between 0 and 1. (default: 0.9)",
    )
    dip_detect_group.add_argument(
        "-s", "--score-sensitivity",
        type=float,
        default=0.333,
        help="Sensitivity of score calculation for dip detection. Must be a float between 0 and 1. (default: 0.334)",
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
        default=100,
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
        default=-1,
        help="Cluster distance in base pairs. Attempts to keep the single largest cluster of annotationed dips. Negative Values turn it off. (default: -1)",
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
    other_arguments_group.add_argument(
        "--version",
        action="version",
        version=f"centrodip {version('centrodip')}"
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


    argsd = {
        "mod_code": args.mod_code,
        "bedgraph": args.bedgraph,
        "window_size": args.window_size,
        "cov_conf": args.cov_conf,
        "prominence": args.prominence,
        "height": args.height,
        "broadness": args.broadness,
        "score_sensitivity": args.score_sensitivity,
        "enrichment": args.enrichment,
        "label": args.label,
        "color": args.color,
        "min_size": args.min_size,
        "min_score": args.min_score,
        "cluster_distance": args.cluster_distance,
        "debug": args.debug,
        "plot": args.plot,
        "output": args.output,
    }

    out_path = Path(argsd["output"])
    plot_dir = out_path.parent / f"{out_path.stem}_plots"
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    chrom_map = bedMethyl_in_region.groupby_chrom()  # should be dict-like: chrom -> BedTable or list[IntervalRecord]

    work_items = []
    for chrom in chrom_map.keys():
        chrom_records = chrom_map[chrom]
        region_records = regions_by_chrom[chrom]
        # If groupby_chrom returns lists of records, wrap into BedTable
        if isinstance(chrom_records, BedTable):
            bm_chr = chrom_records
            r_chr = region_records
        else:
            bm_chr = BedTable(list(chrom_records), inferred_kind=bedMethyl_in_region.inferred_kind)
            r_chr = BedTable(list(region_records), inferred_kind=regions.inferred_kind)

        work_items.append((chrom, bm_chr, r_chr, argsd))

    all_lowess = []
    all_dips = []
    all_filtered = []
    all_plots = {}
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
                plot_path = str(plot_dir / f"{out_path.stem}.{chrom}.summary.png")
                Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
                if args.debug:
                    print(f"[DEBUG] {chrom}: Writing summary plot: {plot_path}")
                fig, output_path = sp.centrodipChromSummaryPlot(
                    bedMethyl=bm_chr,              # plot only this chrom
                    regions=regions,
                    lowess_bg=bedGraph_LOWESS,
                    dips_unfiltered=dips,
                    bkgrd_median=lowess_bg_stats["median"],
                    dips_final=filtered_dips,
                    output_path=plot_path,
                    args=argsd,
                )
                all_plots[chrom] = fig

            # Collect outputs
            all_lowess.extend(list(bedGraph_LOWESS._records))
            all_dips.extend(list(dips._records))
            all_filtered.extend(list(filtered_dips._records))
            all_bg_stats[chrom] = lowess_bg_stats

    # ---- Concatenate into single BedTables ----
    bedGraph_LOWESS_all = BedTable(all_lowess, inferred_kind="bedgraph", inferred_ncols=4).sort()
    dips_all = BedTable(all_dips, inferred_kind="bed", inferred_ncols=6).sort()
    filtered_dips_all = BedTable(all_filtered, inferred_kind="bed", inferred_ncols=6).sort()

    if args.debug:
        # save smoothed bedMethyl
        lowess_path = str(Path(args.output).with_suffix(".LOWESS.bedgraph"))
        print(f"[DEBUG] Smoothed bedMethyl: {lowess_path}")
        bedGraph_LOWESS_all.to_path(lowess_path)

        # save unfiltered/detected dips
        unfiltered_path = str(Path(args.output).with_suffix(".detected_dips.bed"))
        print(f"[DEBUG] All detected dips: {unfiltered_path}")
        dips_all.to_path(unfiltered_path)
    
    # --- Concatentate into single summary plot (if requested) ---
    if args.plot:
        plot_path = str(plot_dir / f"{out_path.stem}.all.summary.png")
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        if args.debug:
            print(f"[DEBUG] Combined summary plot: {plot_path}")
        sp.centrodipCombinedSummaryPlot(fig_dict=all_plots, output_path=plot_path)

    # -------------------------
    # Write output (FINAL dips)
    # -------------------------
    if args.debug:
        print(f"[DEBUG] Final output: {args.output}")
    filtered_dips_all.to_path(out_path)


if __name__ == "__main__":
    main()