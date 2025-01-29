# `centrodip`

WIP tool to call CDRs (and potentially other areas of hypo/hyper-methylation that exist within a semi-homogeneiously methylated region) with high scalability as well as single CpG resolution.

Inputs: 
1. `bedmethyl` - from `modkit pileup` (Refer to [modkit](https://github.com/nanoporetech/modkit) github) or bedgraph equivalent of fraction modified column when using flag `--methyl_bedgraph`
2. `regions` - bed file of regions you want to search for CDRs
3. `output` - name of primary output file (prefix of this used for other outputs)

Workflow Overview:
```
Preprocessing steps are 
    (1) alignment of bam file containing MM/ML tags to matched reference genome 
    (2) modkit pileup called using aligned bam and matched reference. 
```

1. Parse regions and methylation into python dictionary where key is `chromosome` and value is dictionary of with keys for `starts`, `ends`, and `fraction_modified` in the methylation file's case.
2. For each chromosome in parallel take slices of size `--window_size` of the `fraction_modified` list and perform a statistical test (determined by the `--stat` flag that can be passed in) comparing the distribution of the slice to that of the entire region. Repeat for every slice. 
3. Merge regions based on p-value cutoff set using the `--cdr_p` flag (can optionally include transitions flanking subCDRs by setting `--transition_p`), `--merge_distance` is the number of CpGs to merge over to extend CDR calls. `--min_sig_cpgs` determines how many CpGs are required in this merged region to make it `CDR` call. 

## License
This project is licensed under the MIT License.