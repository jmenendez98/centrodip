from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List

from .parser import Parser
from .dip_detect import DipDetector
from .dip_filter import DipFilter


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

def _generate_track_rows(methylation_dict: Dict[str, Dict[str, List]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for region, values in methylation_dict.items():
        if not values:
            continue
        chrom = region.split(":", 1)[0]
        starts = values.get("starts", [])
        ends = values.get("ends", [])
        smooth = values.get("savgol_frac_mod", [])
        length = min(len(starts), len(ends), len(smooth))
        for idx in range(length):
            row = [chrom]
            row.append(str(starts[idx]))
            row.append(str(ends[idx]))
            row.append(str(smooth[idx]))
            rows.append(row)
    rows.sort(key=lambda entry: (entry[0], int(entry[1])))
    return rows


def main() -> None:
    argparser = argparse.ArgumentParser(
        description="Process bedMethyl and region BED files to produce CDR predictions.",
    )

    argparser.add_argument("bedmethyl", type=str, help="Path to the bedMethyl file")
    argparser.add_argument("regions", type=str, help="Path to BED file of regions to search for dips")
    argparser.add_argument("output", type=str, help="Path to the output BED file")

    parsing_group = argparser.add_argument_group('Parsing Options', 'Arguments related to how the BED files are parsed.')
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

    dip_detect_group = argparser.add_argument_group('Dip Detection Options', 'Arguments related to how the dips are detected/extended.')
    dip_detect_group.add_argument(
        "--window-size",
        type=int,
        default=101,
        help="Number of CpGs to include in Savitzky-Golay filtering of Fraction Modified. (default: 101)",
    )
    dip_detect_group.add_argument(
        "--threshold",
        type=float,
        default=1,
        help="Number of standard deviations from the smoothed mean to be the minimum dip. (default: 1)",
    )
    dip_detect_group.add_argument(
        "--prominence",
        type=float,
        default=0.66,
        help="Prominence required for a dip. Scalar is multiplied by the smoothed data range. (default: 0.66)",
    )
    dip_detect_group.add_argument(
        "--enrichment",
        action="store_true",
        default=False,
        help="Find regions enriched (rather than depleted) for methylation.",
    )

    dip_filter_group = argparser.add_argument_group('Dip Filtering Options', 'Arguments related to how the dips filtered/removed.')
    dip_filter_group.add_argument(
        "--min-size",
        type=int,
        default=5000,
        help="Minimum dip size in base pairs. (default: 5000)",
    )

    other_arguments_group = argparser.add_argument_group('Other Options', 'Miscellaneous arguments affecting outputs and runtime.')
    other_arguments_group.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker processes. (default: 4)",
    )
    other_arguments_group.add_argument(
        "--color",
        type=str,
        default="50,50,255",
        help='Color of predicted dips. (default: "50,50,255")',
    )
    other_arguments_group.add_argument(
        "--output-all",
        action="store_true",
        default=False,
        help="Output smoothed methylation values as a bedGraph. (default: False)",
    )
    other_arguments_group.add_argument(
        "--label",
        type=str,
        default="CDR",
        help='Label to use for regions in BED output. (default: "CDR")',
    )

    args = argparser.parse_args()
    output_prefix = os.path.splitext(args.output)[0]

    # Create Parser instance
    parse = Parser(
        mod_code=args.mod_code,
        bedgraph=args.bedgraph,
    )
    # Read in regions BED file and BEDMethyl File
    methylation, regions = parse.process_files(
        methylation_path=args.bedmethyl,
        regions_path=args.regions
    )

    # Create DipDetector class instance
    detector = DipDetector(
        window_size=args.window_size,
        threshold=args.threshold,
        prominence=args.prominence,
        enrichment=args.enrichment,
        threads=args.threads
    )

    # call the dips, here you have them all before filtering
    dips, methylation = detector.dip_detect_all_chromosome(
        methylation_per_region=methylation, 
        regions_per_chrom=regions
    )

    dip_rows = _generate_output_rows(dips)
    _write_bed(args.output, dip_rows)

    # Create DipFilter class instance


if __name__ == "__main__":
    main()