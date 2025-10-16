from .dip_detect import DipDetector
from .parser import Parser


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


if __name__ == "__main__":
    main()