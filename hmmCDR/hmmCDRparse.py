import pandas as pd
import numpy as np 
import pybedtools
import argparse
import os
import concurrent.futures


class hmmCDRparse:
    '''
    This class processes bedMethyl and CenSat BED files to produce HMM CDR priors. 
    It filters and intersects the input files based on specified modification codes 
    and satellite types. The resulting filtered data is used to create intersected 
    regions, which can be saved as output files.
    
    Attributes:
    -----------
    mod_code : str
        Modification code to filter the bedMethyl file.
    sat_type : str
        Satellite type/name to filter the CenSat BED file.
    bedgraph : bool
        Flag indicating whether the input file is in bedgraph format.
    bedgraphMethyl : DataFrame
        The filtered bedMethyl DataFrame.
    subset_cenSat : DataFrame
        The filtered CenSat DataFrame.
    subset_bedgraphMethyl : DataFrame
        The DataFrame containing intersected regions between the filtered 
        bedMethyl and CenSat files.
    '''
    def __init__(self, min_valid_cov, rolling_window,
                 mod_code, sat_type, bedgraph=False):
        '''
        Initializes the hmmCDR_parser class by reading and processing the bedMethyl 
        and CenSat BED files, filtering them according to the provided modification 
        code and satellite type, and intersecting the filtered regions.
        
        Parameters:
        -----------
        bedMethyl_path : str
            Path to the bedMethyl file.
        cenSat_path : str
            Path to the CenSat BED file.
        mod_code : str
            Modification code to filter the bedMethyl file.
        sat_type : str
            Satellite type/name to filter the CenSat BED file.
        bedgraph : bool, optional
            Flag indicating if the input is a bedgraph file. (default is False)
        
        Raises:
        -------
        ValueError
            If the bedMethyl file does not have the expected number of columns 
            based on the bedgraph flag.
        '''

        self.mod_code = mod_code
        self.sat_type = sat_type
        self.bedgraph = bedgraph
        self.min_valid_cov = min_valid_cov
        self.rolling_window = rolling_window


    def read_bedMethyl(self, path):
        # Check if bedMethyl file exists and is not empty
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            raise ValueError(f"bedMethyl file {path} does not exist or is empty.")
        bedMethyl = pd.read_csv(path, sep='\t', header=None, index_col=None)
        return bedMethyl
    
    def read_cenSat(self, path):
        # Check if cenSat file exists and is not empty
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            raise ValueError(f"cenSat file {path} does not exist or is empty.")
        cenSat = pd.read_csv(path, sep='\t', header=None, index_col=None)
        return cenSat

    def filter_bedMethyl(self, bedMethyl, mod_code, min_valid_cov):
        '''
        Filters the bedMethyl DataFrame by the specified modification code 
        and subsets it to include only relevant columns.

        Parameters:
        -----------
        bedMethyl : DataFrame
            The DataFrame containing the bedMethyl data.
        mod_code : str
            The modification code to filter the bedMethyl data.

        Returns:
        --------
        DataFrame
            A DataFrame containing the filtered bedMethyl data with only 
            the relevant columns (chromosome, start, end, and methylation level).

        Raises:
        -------
        ValueError
            If the filtering results in an empty DataFrame.
        '''
        if self.bedgraph:
            if len(bedMethyl.columns) != 4:
                raise ValueError("Valid bedgraph should have 4 columns(all tab separated) when the bedgraph flag is passed.")
            bed4Methyl = bedMethyl
            return bed4Methyl
        
        if len(bedMethyl.columns) != 18:
            
            raise ValueError("Valid bedMethyl should have 18 columns(all tab separated) when the bedgraph flag is not passed.")

        filtered_bedMethyl = bedMethyl[bedMethyl[3] == mod_code]
        filtered_bedMethyl = bedMethyl[bedMethyl[4] >= min_valid_cov]
        bed4Methyl = filtered_bedMethyl.iloc[:, [0, 1, 2, 10]]
        if bed4Methyl.empty:
            raise ValueError("Filtering bedMethyl by the specified modification code resulted in an empty DataFrame.")
        return bed4Methyl

    def filter_cenSat(self, cenSat, sat_types):
        '''
        Filters the CenSat DataFrame by the specified satellite type.

        Parameters:
        -----------
        cenSat : DataFrame
            The DataFrame containing the CenSat BED data.
        sat_type : str
            The satellite type to filter the CenSat data.

        Returns:
        --------
        DataFrame
            A DataFrame containing the filtered CenSat data.

        Raises:
        -------
        ValueError
            If the filtering results in an empty DataFrame.
        '''
        if len(cenSat.columns) > 3:
            filtered_cenSat = cenSat[cenSat[3].str.contains('|'.join(sat_types))]
            if filtered_cenSat.empty:
                raise ValueError("Filtering CenSat by the specified satellite type resulted in an empty DataFrame.")
        else:
            print('Regions File has no name column. Not filtering by sat_type.')
            filtered_cenSat = cenSat

        return filtered_cenSat

    def intersect_files(self, bed4Methyl, filtered_cenSat):
        '''
        Intersects the filtered bedMethyl and CenSat DataFrames using pybedtools.

        Parameters:
        -----------
        bedgraphMethyl : DataFrame
            The DataFrame containing the filtered bedMethyl data.
        subset_cenSat : DataFrame
            The DataFrame containing the filtered CenSat data.

        Returns:
        --------
        DataFrame
            A DataFrame containing the intersected regions between the filtered 
            bedMethyl and CenSat files.

        Raises:
        -------
        ValueError
            If the intersection results in an empty DataFrame.
        '''
        bedMethyl_bedtool = pybedtools.BedTool.from_dataframe(bed4Methyl)
        cenSat_bedtool = pybedtools.BedTool.from_dataframe(filtered_cenSat)
        intersected_bedtool = bedMethyl_bedtool.intersect(cenSat_bedtool, wa=True, u=True)
        intersected_bed4Methyl = intersected_bedtool.to_dataframe()
        
        if intersected_bed4Methyl.empty:
            raise ValueError("The intersection resulted in an empty DataFrame.")
        
        return intersected_bed4Methyl
    
    def calculate_rolling_average(self, bed4Methyl, rolling_size):
        if rolling_size <= 0:
            return bed4Methyl.iloc[:, -1]  # Return the original last column if rolling_size is 0 or less

        # Ensure the DataFrame is sorted by start positions
        bed4Methyl = bed4Methyl.sort_values(by=bed4Methyl.columns[1])

        # Calculate the rolling sum based on the start and end positions
        # Calculate the rolling sum based on the start and end positions
        rolling_avg = bed4Methyl.apply(lambda row: round(
            bed4Methyl[(bed4Methyl.iloc[:, 1] >= row[1] - rolling_size) & 
                       (bed4Methyl.iloc[:, 1] <= row[2])].iloc[:, -1].sum() / rolling_size, 5), axis=1)
        
        return rolling_avg

    def parse_single_chromosome(self, chrom, bedMethyl_path, cenSat_path):
        '''
        Processes the bedMethyl and CenSat data for each chromosome separately.

        Returns:
        --------
        tuple
            A tuple containing the chromosome name, intersected DataFrame, and filtered CenSat DataFrame.
        '''
        # Read and filter data within the worker function to avoid passing large DataFrames
        bedMethyl = self.read_bedMethyl(bedMethyl_path)
        cenSat = self.read_cenSat(cenSat_path)

        bedMethyl_filtered = self.filter_bedMethyl(bedMethyl, self.mod_code, self.min_valid_cov)
        cenSat_filtered = self.filter_cenSat(cenSat, self.sat_type)
        intersected = self.intersect_files(bedMethyl_filtered, cenSat_filtered)
        intersected.iloc[:, -1] = self.calculate_rolling_average(intersected, self.rolling_window)

        return chrom, intersected, cenSat_filtered

    def parse_all_chromosomes(self, bedMethyl_path, cenSat_path):
        '''
        Processes all chromosomes in parallel using concurrent futures.

        Returns:
        --------
        dict
            A dictionary with chromosome names as keys and intersected DataFrames as values.
        '''
        bed4Methyl_chrom_dict = {}
        cenSat_chrom_dict = {}

        # Read chromosome list without loading large DataFrames into memory
        cenSat = pd.read_csv(cenSat_path, sep='\t', header=None)
        chromosomes = cenSat[0].unique()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.parse_single_chromosome, chrom, bedMethyl_path, cenSat_path
                ): chrom for chrom in chromosomes
            }

            for future in concurrent.futures.as_completed(futures):
                chrom, intersected_df, filtered_cenSat = future.result()

                bed4Methyl_chrom_dict[chrom] = intersected_df
                cenSat_chrom_dict[chrom] = filtered_cenSat

        self.chromosomes = chromosomes
        self.bed4Methyl_chrom_dict = bed4Methyl_chrom_dict
        self.cenSat_chrom_dict = cenSat_chrom_dict
        return bed4Methyl_chrom_dict, cenSat_chrom_dict

def main():
    argparser = argparse.ArgumentParser(description='Process bedMethyl and CenSat BED file to produce hmmCDR priors')
    # Required arguments
    argparser.add_argument('bedMethyl_path', type=str, help='Path to the bedMethyl file')
    argparser.add_argument('cenSat_path', type=str, help='Path to the CenSat BED file')
    argparser.add_argument('output_prefix', type=str, help='Path to the output priorCDRs BED file')
    
    # Optional arguments with default values
    argparser.add_argument('--bedgraph', action='store_true', help='Flag indicating if the input is a bedgraph. (default: False)')
    argparser.add_argument('--rolling_window', type=int, default=0, help='Flag indicating whether or not to use a rolling average and the rolling avg window size. If set to 0 no rolling averages are used. (defualt: 0)')
    argparser.add_argument('--min_valid_cov', type=int, default=10, help='Minimum Valid Coverage to consider a methylation site. (default: 10)')
    argparser.add_argument('-m', '--mod_code', type=str, default='m', help='Modification code to filter bedMethyl file (default: "m")')
    argparser.add_argument('-s', '--sat_type', type=str, default='H1L', help='Comma-separated list of satellite types/names to filter CenSat bed file. (default: "H1L")')
    args = argparser.parse_args()

    sat_types = [st.strip() for st in args.sat_type.split(',')]

    hmmCDRparser = hmmCDRparse(bedMethyl_path=args.bedMethyl_path,
                               cenSat_path=args.cenSat_path,
                               mod_code=args.mod_code,
                               sat_type=sat_types,
                               bedgraph=args.bedgraph,
                               min_valid_cov=args.min_valid_cov,
                               rolling_window=args.rolling_window)

    bed4Methyl_chrom_dict, cenSat_chrom_dict = hmmCDRparser.parse_all_chromosomes(bedMethyl_path=args.bedMethyl_path, cenSat_path=args.cenSat_path)
    
    concatenated_bed4Methyl = pd.concat(bed4Methyl_chrom_dict.values())
    concatenated_cenSat = pd.concat(cenSat_chrom_dict.values())

    concatenated_bed4Methyl.to_csv(f'{args.output_prefix}_filtered_bedMethyl.bedgraph', sep='\t', index=False, header=False)
    concatenated_cenSat.to_csv(f'{args.output_prefix}_filtered_cenSat.bed', sep='\t', index=False, header=False)
    print(f"Filtered bedgraphMethyl saved to: {args.output_prefix}_filtered_bedgraphMethyl.bedgraph")
    print(f"Filtered CenSat saved to: {args.output_prefix}_filtered_cenSat.bed")

    [print(i) for i in bed4Methyl_chrom_dict.values()]

if __name__ == "__main__":
    main()