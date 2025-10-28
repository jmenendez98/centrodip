import argparse
import os
3
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, Iterable, List, Union

import numpy as np

from .load_data import DataHandler
from .lowess_smooth import lowess_smooth
from .filter_dips import filterDips
from .summary_plot import centrodip_summary_plot


def single_chromosome_task(
    chrom: str,
    data: Dict[str, List[Union[float, int]]],
    window_size: int, cov_conf: float,
):
    if len(positions) != len(fractions):
        raise ValueError(
            f"Chromosome {chrom}: mismatch between positions ({len(positions)}) and fraction_modified ({len(fractions)})."
        )

    positions = data.get("position", [])
    fractions = data.get("fraction_modified", [])
    coverage = data.get("valid_coverage", [])

    # calculate smoothing + dy, add them to data dict
    smoothed, derivative = lowess_smooth(
        y=fractions, x=positions, c=coverage,
        window_bp=window_size, cov_conf=cov_conf,
    )
    data["lowess_fraction_modified"] = smoothed.tolist()
    data["lowess_fraction_modified_dy"] = derivative.tolist()

    # 


    return chrom, result


def main() -> None:
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

    smoothing_group = argparser.add_argument_group('Smoothing Options')
    smoothing_group.add_argument(
        "--cov-conf",
        type=int,
        default=10,
        help="Minimum coverage required to be a confident CpG site. (default: 10)",
    )
    smoothing_group.add_argument(
        "--window-size",
        type=int,
        default=10000,
        help="Window size (bp) to use in LOWESS smoothing of fraction modified. (default: 10000)",
    )


    dip_detect_group = argparser.add_argument_group('Detection Options')
    dip_detect_group.add_argument(
        "--prominence",
        type=float,
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
        default=50,
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

    other_arguments_group = argparser.add_argument_group('Other Options')
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
    data_handler = DataHandler(
        methylation_path=args.bedmethyl,
        regions_path=args.regions,
        mod_code=args.mod_code,
        debug=args.debug,
    )

    results_by_chrom: Dict[str, Dict[str, List[Union[float, int]]]] = {}

    # add a futures parallelized functionized thing that can take th yielded input from running
    # data_handler.load_data()...
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futures = []

        # stream the methylation BED; each yield is a single-chrom dict: {"chrX": methylation_data}
        for one_chrom_dict in data_handler.load_data():
            (chrom, methylation_data) = one_chrom_dict   # unpack single item

            # print(chrom, methylation_data[chrom].keys())

            fut = ex.submit(
                single_chromosome_task,             # parallelized task name
                chrom, methylation_data[chrom],     # data
                args.window_size, args.cov_conf,    # smoothing arguments

            )
            futures.append(fut)

        # collect as tasks finish (no need to maintain input order)
        for fut in as_completed(futures):
            chrom, result = fut.result()  # handle exceptions here if desired
            results_by_chrom[chrom] = result
            if args.debug:
                print(f"[DEBUG] Completed {chrom}: {result}")

    raw_dips: Dict[str, Dict[str, List[Union[float, int]]]] = {}

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

    # if debug is on, write out the dip centers and edges
    if args.debug:
        with open(f"{output_prefix}.debug.dip_centers.bed", "w", encoding="utf-8") as handle:
            lines = []
            for chrom, values in raw_dips.items():
                if not values:
                    continue
                for pos in values.get("dip_centers", []) or []:
                    start = int(pos)
                    lines.append(f"{chrom}\t{start}\t{start + 1}\n")
            lines.sort(key=lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
            handle.writelines(lines)

        with open(f"{output_prefix}.debug.dip_edges.bed", "w", encoding="utf-8") as handle:
            lines = []
            for chrom, values in raw_dips.items():
                if not values:
                    continue
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
