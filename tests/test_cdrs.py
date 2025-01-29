import os

import pytest

from mwCDR.mwCDR import bed_parser
from mwCDR.mwCDR import mwCDR


class TestMatrix:
    @pytest.fixture
    def test_data(self):
        """Fixture to set up test data and parser"""
        test_data_dir = os.path.join("tests", "data")

        parser = bed_parser(
            mod_code="m",
            methyl_bedgraph=False,
            min_valid_cov=1,
            region_edge_filter=0,
        )

        bedmethyl_test = os.path.join(test_data_dir, "bedmethyl_test.bed")
        censat_test = os.path.join(test_data_dir, "censat_test.bed")

        return parser.process_files(
            methylation_path=bedmethyl_test,
            regions_path=censat_test,
        )

    @pytest.fixture
    def mwcdr(self):
        """Fixture for matrix calculator"""
        return mwCDR(
        window_size=101,
        step_size=1,
        stat="mannwhitneyu",
        cdr_p=0.0000001,
        transition_p=0.01,
        min_sig_cpgs=50,
        merge_distance=50,
        enrichment=False,
        threads=4,
        cdr_color="50,50,255",
        transition_color="150,150,150",
        output_label="subCDR",
        )


    def test_making_matrices(self, test_data, mwcdr):
        """Test making matrices"""
        (
            priors_chrom_dict,
            windowmean_chrom_dict,
            labelled_methylation_chrom_dict,
            emission_matrix_chrom_dict,
            transition_matrix_chrom_dict,
        ) = mwcdr.mwcdr_all_chromosomes(
            regions_all_chroms=test_data[0],
            methylation_all_chroms=test_data[1]
        )

        # Changed from .values to proper dictionary access
        assert isinstance(priors_chrom_dict, dict)
        assert isinstance(windowmean_chrom_dict, dict)
        assert isinstance(labelled_methylation_chrom_dict, dict)
        assert isinstance(emission_matrix_chrom_dict, dict)
        assert isinstance(transition_matrix_chrom_dict, dict)
        assert len(priors_chrom_dict) == 1
        assert len(windowmean_chrom_dict) == 1
        assert len(labelled_methylation_chrom_dict) == 1
        assert len(emission_matrix_chrom_dict) == 1
        assert len(transition_matrix_chrom_dict) == 1

        # Add more specific assertions about the matrices
        for chrom in emission_matrix_chrom_dict:
            assert emission_matrix_chrom_dict[chrom].shape == (2, 4)
            # Add assertions about matrix shape or content if known

        for chrom in transition_matrix_chrom_dict:
            assert transition_matrix_chrom_dict[chrom].shape == (2, 2)
            # Add assertions about matrix shape or content if known
