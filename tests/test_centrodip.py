import os
import pytest
from centrodip.centrodip import BedParser, CentroDip


class TestMatrix:
    @pytest.fixture
    def test_data(self):
        """Fixture to set up test data and parser"""
        test_data_dir = os.path.join("tests", "data")

        bed_parser = BedParser(
            mod_code="m",
            methyl_bedgraph=False,
            min_valid_cov=1,
            region_edge_filter=0,
        )

        bedmethyl_test = os.path.join(test_data_dir, "bedmethyl_test.bed")
        censat_test = os.path.join(test_data_dir, "censat_test.bed")

        return bed_parser.process_files(
            methylation_path=bedmethyl_test,
            regions_path=censat_test,
        )

    @pytest.fixture
    def centro_dip(self):
        """Fixture for matrix calculator"""
        return CentroDip(
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

    def test_centro_dip(self, test_data, centro_dip):
        """Test making matrices"""
        (
            cdrs_all_chroms,
            methylation_sig_all_chroms,
        ) = centro_dip.centrodip_all_chromosomes(
            regions_all_chroms=test_data[0],
            methylation_all_chroms=test_data[1]
        )

        assert isinstance(cdrs_all_chroms, dict)
        assert isinstance(methylation_sig_all_chroms, dict)
        assert len(cdrs_all_chroms) == 1
        assert len(methylation_sig_all_chroms) == 1
        assert len(methylation_sig_all_chroms["chrX_MATERNAL"]["p-values"]) == 62064