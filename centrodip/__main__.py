from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List

from .load_data import DataHandler
from .detect_dips import detectDips

from .detect_dips import DipDetector

from .dip_filter import DipFilter
from .plot import create_summary_plot


def _write_bed(output_file: str, rows: Iterable[List[str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(output_file, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")

def _value_at(values: Dict[str, List], key: str, idx: int, default: str) -> str:
    items = values.get(key, [])
    if idx < len(items):
        return str(items[idx])
    return default

def _generate_output_rows(bed_dict: Dict[str, Dict[str, List]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for region, values in bed_dict.items():
        if not values:
            continue
        chrom = region.split(":", 1)[0]
        starts = values.get("starts", [])
        ends = values.get("ends", [])
        length = min(len(starts), len(ends))
        for idx in range(length):
            row = [chrom]
            row.append(str(starts[idx]))
            row.append(str(ends[idx]))
            row.append(_value_at(values, "names", idx, "."))
            row.append(_value_at(values, "scores", idx, "0"))
            row.append(_value_at(values, "strands", idx, "."))
            row.append(_value_at(values, "thick_starts", idx, str(starts[idx])))
            row.append(_value_at(values, "thick_ends", idx, str(ends[idx])))
            row.append(_value_at(values, "item_rgbs", idx, "0,0,0"))
            rows.append(row)
    rows.sort(key=lambda entry: (entry[0], int(entry[1])))
    return rows

def _generate_bedgraph_rows(
    methylation_dict: Dict[str, Dict[str, List]],
    value_key: str,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for region, values in methylation_dict.items():
        if not values:
            continue
        chrom = region.split(":", 1)[0]
        positions = values.get("position", [])
        metrics = values.get(value_key, [])
        length = min(len(positions), len(metrics))
        for idx in range(length):
            start = int(positions[idx])
            end = start + 1
            metric = metrics[idx]
            rows.append([chrom, str(start), str(end), str(metric)])
    rows.sort(key=lambda entry: (entry[0], int(entry[1])))
    return rows



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

    dip_detect_group = argparser.add_argument_group('Dip Detection Options')
    dip_detect_group.add_argument(
        "--sensitivity",
        type=zero_to_one_float,
        default=0.5,
        help="Sensitivity of dip detection. Higher values require more pronounced dips. (default: 0.5)",
    )
    dip_detect_group.add_argument(
        "--edge-sensitivity",
        type=zero_to_one_float,
        default=0.25,
        help="Sensitivity of edge detection. Higher values require more pronounced edges, causing larger dip calls. (default: 0.1)",
    )
    dip_detect_group.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Find regions enriched (rather than depleted) for methylation.",
    )

    dip_filter_group = argparser.add_argument_group('Dip Filtering Options')
    dip_filter_group.add_argument(
        "--min-size",
        type=int,
        default=-1,
        help="Minimum dip size in base pairs. (default: 5000)",
    )
    dip_filter_group.add_argument(
        "--cluster-distance",
        type=int,
        default=-1,
        help="Cluster distance in base pairs. Attempts to keep the single largest cluster of annotationed dips. Negative Values turn it off. (default: -1)",
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
            for region, values in input_data.methylation_dict.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]

                assert len(values.get("position", [])) == len(values.get("lowess_fraction_modified", [])), "Error mismatch size in output data."
                for i in range(len(values.get("position", []))):
                    start = values["position"][i]
                    frac_mod = values["lowess_fraction_modified"][i]
                    handle.write(f"{chrom}\t{start}\t{start + 1}\t{frac_mod}\n")

        with open(f"{output_prefix}.debug.smooth_frac_mod_dy.bedgraph", "w", encoding="utf-8") as handle:
            for region, values in input_data.methylation_dict.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]

                assert len(values.get("position", [])) == len(values.get("lowess_fraction_modified_dy", [])), "Error mismatch size in output data."
                for i in range(len(values.get("position", []))):
                    start = values["position"][i]
                    frac_mod = values["lowess_fraction_modified_dy"][i]
                    handle.write(f"{chrom}\t{start}\t{start + 1}\t{frac_mod}\n")

    # -- detect dips --
    raw_dips = detectDips(
        methylation_data=input_data.methylation_dict,
        regions_data=input_data.region_dict,
        sensitivity=args.sensitivity,
        edge_sensitivity=args.edge_sensitivity,
        enrichment=args.enrichment,
        threads=args.threads,
        debug=args.debug,
    )

    # if debug is on, write out the dip centers and edges
    if args.debug:
        with open(f"{output_prefix}.debug.dip_centers.bed", "w", encoding="utf-8") as handle:
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]
                for pos in values.get("dip_centers", []) or []:
                    start = int(pos)
                    handle.write(f"{chrom}\t{start}\t{start + 1}\n")

        with open(f"{output_prefix}.debug.dip_edges.bed", "w", encoding="utf-8") as handle:
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]
                for pos in values.get("dip_edges", []) or []:
                    start = int(pos)
                    handle.write(f"{chrom}\t{start}\t{start + 1}\n")  

        with open(f"{output_prefix}.debug.no_filt_dips.bed", "w", encoding="utf-8") as handle:
            for region, values in raw_dips.items():
                if not values:
                    continue
                chrom = region.split(":", 1)[0]

                assert len(values.get("starts", [])) == len(values.get("ends", [])), "Mismatched dip starts and ends lengths in debug output."
                for i in range(len(values.get("starts", []))):
                    start = values["starts"][i]
                    end = values["ends"][i]
                    handle.write(f"{chrom}\t{start}\t{end}\n")

    # Create DipFilter class instance
    dip_filter = DipFilter(
        min_size=args.min_size,
        cluster_distance=args.cluster_distance,
    )
    # filter the dips
    dips_filtered = dip_filter.filter(dips)

    '''
    # write filtered dips to output BED file
    dip_rows = _generate_output_rows(dips_filtered)
    _write_bed(args.output, dip_rows)

    if args.plot:
        summary_path = f"{output_prefix}.summary.png"
        create_summary_plot(
            regions_per_chrom=regions,
            methylation_per_region=methylation,
            dip_results=dips_filtered,
            unfiltered_dip_results=dips,
            output_path=summary_path,
        )
    '''


if __name__ == "__main__":
    main()
