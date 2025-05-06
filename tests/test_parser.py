import os
import pytest
from centrodip.centrodip import BedParse


class TestParser:
    @pytest.fixture
    def test_data_dir(self):
        return os.path.join("tests", "data")

    @pytest.fixture
    def bed_parser(self):
        return BedParse(
            mod_code="m",
            bedgraph=False,
            region_edge_filter=0,
            region_merge_distance=0,
        )

    def test_fake_bedfile(self, test_data_dir, bed_parser):
        """Test handling of nonexistent file"""
        nonexistent_file = os.path.join(test_data_dir, "nonexistent.bed")
        with pytest.raises(FileNotFoundError):
            bed_parser.read_and_filter_regions(nonexistent_file)
        with pytest.raises(FileNotFoundError):
            bed_parser.read_and_filter_regions(nonexistent_file)

    def test_empty_bedfile(self, test_data_dir, bed_parser):
        """Test handling of empty file"""
        empty_file = os.path.join(test_data_dir, "empty.bed")

        result1 = bed_parser.read_and_filter_regions(str(empty_file))
        result2 = bed_parser.read_and_filter_methylation(str(empty_file), result1)

        assert len(result1) == 0
        assert len(result2) == 0

    def test_censat_bedfile(self, test_data_dir, bed_parser):
        """Test basic censat reading functionality"""
        sample_censat_bed = os.path.join(test_data_dir, "censat_test.bed")
        results = bed_parser.read_and_filter_regions(sample_censat_bed)

        assert list(results.keys())[0] == "chrX_MATERNAL"
        assert len(list(results.keys())) == 1
        
        assert results["chrX_MATERNAL"]["starts"] == [57866525]
        assert results["chrX_MATERNAL"]["ends"] == [60979767]

    def test_bedmethyl_bedfile(self, test_data_dir, bed_parser):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = os.path.join(test_data_dir, "bedmethyl_test.bed")
        sample_censat_bed = os.path.join(test_data_dir, "censat_test.bed")

        regions_dict = bed_parser.read_and_filter_regions(sample_censat_bed)
        results = bed_parser.read_and_filter_methylation(
            sample_bedmethyl_bed,
            regions_dict
        )

        assert list(results.keys())[0] == "chrX_MATERNAL:57866525-60979767"
        assert len(list(results.keys())) == 1

        assert len(results["chrX_MATERNAL:57866525-60979767"]["starts"]) == 62064
        assert len(results["chrX_MATERNAL:57866525-60979767"]["ends"]) == 62064
        assert len(results["chrX_MATERNAL:57866525-60979767"]["fraction_modified"]) == 62064

    def test_chrom_dict_len(self, test_data_dir, bed_parser):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = os.path.join(test_data_dir, "bedmethyl_test.bed")
        sample_censat_bed = os.path.join(test_data_dir, "censat_test.bed")

        methylation_chrom_dict, regions_chrom_dict = bed_parser.process_files(
            methylation_path=sample_bedmethyl_bed,
            regions_path=sample_censat_bed,
        )

        # lengths should be 1 because test file is only one chromosome
        assert len(methylation_chrom_dict) == 1
        assert len(regions_chrom_dict) == 1