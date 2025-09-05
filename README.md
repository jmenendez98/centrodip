# `centrodip`

### Installation
Conda Install:   
```
conda install jmmenend::centrodip
```

Docker Run:     
```
docker run -it jmmenend/centrodip:latest
```

Pip Install (requires having bedtools already installed):                   
```
pip install centrodip
```

### How to Run
Preprocessing:
```
(1) Align BAM with MM/ML tags to matched reference genome.
(2) Modkit pileup aligned bam and matched reference. 
(3) Region annotation file.
```
Running `centrodip`:
```
centrodip ${bedmethyl} ${regions} ${output}
```

### Inputs: 
1. `bedmethyl` - `modkit pileup` file (Refer to [modkit](https://github.com/nanoporetech/modkit) github).
2. `regions` - bed file of regions you want to search for CDRs.
3. `output` - name of output file.

### Output: 
Output file is a `BED` file with 9 columns. Some columns can be adjusted with flags (`--label`, `--color`, etc.)

## Help Documentation
```
usage: centrodip [-h] [--mod-code MOD_CODE] [--bedgraph] [--region-merge-distance REGION_MERGE_DISTANCE] [--region-edge-filter REGION_EDGE_FILTER] [--window-size WINDOW_SIZE]
                 [--threshold THRESHOLD] [--prominence PROMINENCE] [--min-size MIN_SIZE] [--enrichment] [--threads THREADS] [--color COLOR] [--output-all] [--label LABEL]
                 bedmethyl regions output

Process bedMethyl and CenSat BED file to produce CDR predictions.

positional arguments:
  bedmethyl             Path to the bedmethyl file
  regions               Path to BED file of regions
  output                Path to the output BED file

options:
  -h, --help            show this help message and exit
  --mod-code MOD_CODE   Modification code to filter bedMethyl file (default: "m")
  --bedgraph            Flag indicating the input is a bedgraph. If passed --mod-code and --min-cov are ignored. (default: False)
  --region-merge-distance REGION_MERGE_DISTANCE
                        Merge gaps in nearby centrodip regions up to this many base pairs. (default: 100000)
  --region-edge-filter REGION_EDGE_FILTER
                        Remove edges of merged regions in base pairs. (default: 0)
  --window-size WINDOW_SIZE
                        Number of CpGs to include in Savitzky-Golay filtering of Fraction Modified. (default: 101)
  --threshold THRESHOLD
                        Number of standard deviations from the smoothed mean to be the minimum dip. Lower values increase leniency of dip calls. (default: 1)
  --prominence PROMINENCE
                        Scalar factor to decide the prominence required for an dip. Scalar is multiplied by smoothed data's difference in the minimum and maxiumum values. Lower values increase
                        leniency of MDR calls. (default: 0.66)
  --min-size MIN_SIZE   Minimum dip size in base pairs. Small dips are removed. (default: 5000)
  --enrichment          Use centrodip to find areas enriched in aggregated methylation calls. (default: False)
  --threads THREADS     Number of workers. (default: 4)
  --color COLOR         Color of predicted dips. (default: 50,50,255)
  --output-all          Output all intermediate files. (default: False)
  --label LABEL         Label to use for regions in BED output. (default: "CDR")
```