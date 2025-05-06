import os
import pytest
from centrodip.centrodip import BedParse, CentroDip


class TestMatrix:
    @pytest.fixture
    def test_data(self):
        """Fixture to set up test data and parser"""
        test_data_dir = os.path.join("tests", "data")

        bed_parser = BedParse(
            mod_code="m",
            bedgraph=False,
            region_edge_filter=0,
            region_merge_distance=0,
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
            mdr_threshold=1,
            transition_threshold=0,
            prominence_constant=0.5,
            significance=1e-10,
            min_size=1000,
            min_cov=1,
            enrichment=False,
            threads=4,
            mdr_color='50,50,255',
            transition_color='150,150,255',
            low_cov_color='211,211,211',
            label='subCDR'
        )

    def test_centrodip(self, test_data, centro_dip):
        """Test making matrices"""
        (
            cdrs_per_region,
            low_cov_per_region,
            methylation_per_region,
        ) = centro_dip.centrodip_all_chromosomes(
            methylation_per_region=test_data[1],
            regions_per_chrom=test_data[0]
        )

        assert isinstance(cdrs_per_region, dict)
        assert isinstance(methylation_per_region, dict)
        assert len(cdrs_per_region) == 1
        assert len(methylation_per_region) == 1
        assert len(methylation_per_region["chrX_MATERNAL:57866525-60979767"]["fraction_modified"]) == 62064