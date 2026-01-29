# `centrodip`

## Installation
```
# conda Install:   
conda install jmmenend::centrodip

# docker Run:     
docker run -it jmmenend/centrodip:latest

# pip install:                   
pip install centrodip
```

## Preprocessing:
`centrodip` requires two inputs: (1) a `bedMethyl` file from modkit and (2) active-alpha annotations  

### (1) can be created by aligning a BAM and calling modkit pileup   
Example:
```
UBAM="HG002.unaligned.bam"
FA="HG002.fa"

# convert to FQ, then align
samtools fastq -T '*' $UBAM > $FQ
minimap2 ... $FQ > $SAM

# convert to BAM and index
samtools view -bh $SAM > $BAM
samtools index $BAM

# aggregate methylation with modkit
modkit pileup --cpg --ref $FA $BAM $bedMethyl
```

### (2) can be created by subsetting the output from the cenSat Annotation workflow
Documentation for running this workflow can be found [here](https://github.com/kmiga/alphaAnnotation/tree/main)    
Example:
```
CENSAT="HG002.censat.bed"

# filter for only active-alpha censat annotations
grep "active_hor" $CENSAT > $ACTIVE_ALPHA

# it is recommended to perform a bedtools merge on these subset annotations
bedtools merge -d 100000 $ACTIVE_ALPHA > $regions
```

## Running `centrodip`:
```
centrodip $bedMethyl $regions $output
```

### Inputs: 
1. `bedMethyl` - `modkit pileup` file (Refer to [modkit](https://github.com/nanoporetech/modkit) github).
2. `regions` - bed file of regions you want to search for CDRs.
3. `output` - name of output file.

### Output: 
Default output file is a `BED` file with 9 columns
 - Column 4 can be adjusted with the `--label` flag
 - Column 9 can be adjusted with the `--color` flag
 - The `--debug` flag adds chromosomal summary printouts and additional outputs like smoothed methylation, and unfiltered dip calls
 - The `--plot` flag creates a folder that contains summary png files for each chromosome


## Help Documentation
```
usage: centrodip [-h] [--mod-code MOD_CODE] [--bedgraph] [--window-size WINDOW_SIZE] [--cov-conf COV_CONF] [--prominence PROMINENCE] [--height HEIGHT] [--broadness BROADNESS] [--enrichment] [--min-size MIN_SIZE]
                 [--min-score MIN_SCORE] [--cluster-distance CLUSTER_DISTANCE] [--label LABEL] [--color COLOR] [--plot] [--threads THREADS] [--debug]
                 bedMethyl regions output

Inspect BED / bedGraph files using BedTable

positional arguments:
  bedMethyl             Path to the bedMethyl file
  regions               Path to BED file of regions to search for dips
  output                Path to the output BED file

options:
  -h, --help            show this help message and exit

Input Options:
  --mod-code MOD_CODE   Modification code to filter bedMethyl file. Selects rows with this value in the fourth column. (default: "m")
  --bedgraph            Input file in a bedGraph format rather than bedMethyl. Requires bedGraph4 with the fourth column being fraction modified (default: False)

Smoothing Options:
  --window-size WINDOW_SIZE
                        Window size (bp) to use in LOWESS smoothing of fraction modified. (default: 10000)
  --cov-conf COV_CONF   Minimum coverage required to be a confident CpG site. (default: 10)

Detection Options:
  --prominence PROMINENCE
                        Sensitivity of dip detection for scipy.signal.find_peaks. Higher values require more pronounced dips. Must be a float between 0 and 1. (default: 0.5)
  --height HEIGHT       Minimum depth for dip detection, lower values require deeper dips. Must be a float between 0 and 1. (default: 0.1)
  --broadness BROADNESS
                        Broadness of dips called, higher values make broader entries. Must be a float between 0 and 1. (default: 0.75)
  --enrichment          Find regions that are enriched (rather than depleted) for methylation. (default: False)

Filtering Options:
  --min-size MIN_SIZE   Minimum dip size in base pairs. (default: 1000)
  --min-score MIN_SCORE
                        Minimum score that a dip must have to be kept. Must be an int between 0 and 1000. (default: 500)
  --cluster-distance CLUSTER_DISTANCE
                        Cluster distance in base pairs. Attempts to keep the single largest cluster of annotationed dips. Negative Values turn it off. (default: 500000)

Output Options:
  --label LABEL         Label to use for regions in BED output. (default: "CDR")
  --color COLOR         Color of predicted dips. (default: "50,50,255")

Other Options:
  --plot                Create summary plot of the results. Written to <output_prefix>.summary.png (default: False)
  --threads THREADS     Number of worker processes. (default: 4)
  --debug               Dumps smoothed methylation values, their derivatives, methylation peaks, and derivative peaks. Each to separate BED/BEDGraph files. (default: False)
```