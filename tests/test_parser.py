import os

import pytest

from mwCDR.mwCDR import bed_parser


class TestParser:
    @pytest.fixture
    def test_data_dir(self):
        return os.path.join("tests", "data")

    @pytest.fixture
    def parser(self):
        return bed_parser(
            mod_code="m",
            methyl_bedgraph=False,
            min_valid_cov=1,
            sat_type=["active_hor"],
            regions_prefiltered=False,
        )

    def test_fake_bedfile(self, test_data_dir, parser):
        """Test handling of nonexistent file"""
        nonexistent_file = os.path.join(test_data_dir, "nonexistent.bed")
        with pytest.raises(FileNotFoundError):
            parser.read_and_filter_regions(nonexistent_file)
        with pytest.raises(FileNotFoundError):
            parser.read_and_filter_regions(nonexistent_file)

    def test_empty_bedfile(self, test_data_dir, parser):
        """Test handling of empty file"""
        empty_file = os.path.join(test_data_dir, "empty.bed")

        result1 = parser.read_and_filter_regions(str(empty_file))
        result2 = parser.read_and_filter_regions(str(empty_file))

        assert len(result1) == 0
        assert len(result2) == 0

    def test_censat_bedfile(self, test_data_dir, parser):
        """Test basic censat reading functionality"""
        sample_censat_bed = os.path.join(test_data_dir, "censat_test.bed")
        results = parser.read_and_filter_regions(sample_censat_bed)

        assert list(results.keys())[0] == "chrX_MATERNAL"
        assert len(list(results.keys())) == 1
        
        assert results["chrX_MATERNAL"]["starts"] == [57876525]
        assert results["chrX_MATERNAL"]["ends"] == [60969767]

    def test_bedmethyl_bedfile(self, test_data_dir, parser):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = os.path.join(test_data_dir, "bedmethyl_test.bed")
        results = parser.read_and_filter_methylation(
            sample_bedmethyl_bed
        )

        assert list(results.keys())[0] == "chrX_MATERNAL"
        assert len(list(results.keys())) == 1

        assert len(results["chrX_MATERNAL"]["starts"]) == 62064
        assert len(results["chrX_MATERNAL"]["ends"]) == 62064
        assert len(results["chrX_MATERNAL"]["fraction_modified"]) == 62064

    def test_chrom_dict_len(self, test_data_dir, parser):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = os.path.join(test_data_dir, "bedmethyl_test.bed")
        sample_censat_bed = os.path.join(test_data_dir, "censat_test.bed")

        methylation_chrom_dict, regions_chrom_dict = parser.process_files(
            methylation_path=sample_bedmethyl_bed,
            regions_path=sample_censat_bed,
        )

        # lengths should be 1 because test file is only one chromosome
        assert len(methylation_chrom_dict) == 1
        assert len(regions_chrom_dict) == 1
