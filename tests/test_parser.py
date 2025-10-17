from pathlib import Path

import pytest

from tests.conftest import EXPECTED_CHROMS, REMOTE_DATASETS

class TestParser:
    def test_fake_bedfile(self, test_data_dir: Path, bed_parser):
        """Test handling of nonexistent file"""
        nonexistent_file = test_data_dir / "nonexistent.bed"
        with pytest.raises(FileNotFoundError):
            bed_parser.read_and_filter_regions(nonexistent_file)
        with pytest.raises(FileNotFoundError):
            bed_parser.read_and_filter_methylation(nonexistent_file, {})

    def test_empty_bedfile(self, test_data_dir: Path, bed_parser):
        """Test handling of empty file"""
        empty_file = test_data_dir / "empty.bed"
        result1 = bed_parser.read_and_filter_regions(empty_file)
        result2 = bed_parser.read_and_filter_methylation(empty_file, result1)
        assert len(result1) == 0
        assert len(result2) == 0

    def test_censat_bedfile(self, test_data_dir: Path, bed_parser):
        sample_censat_bed = test_data_dir / "censat_test.bed"
        results = bed_parser.read_and_filter_regions(sample_censat_bed)
        assert list(results.keys())[0] == "chrX_MATERNAL"
        assert len(list(results.keys())) == 1
        assert results["chrX_MATERNAL"]["starts"] == [57866525]
        assert results["chrX_MATERNAL"]["ends"] == [60979767]

    def test_bedmethyl_bedfile(self, test_data_dir: Path, bed_parser):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = test_data_dir / "bedmethyl_test.bed"
        sample_censat_bed = test_data_dir / "censat_test.bed"
        regions_dict = bed_parser.read_and_filter_regions(sample_censat_bed)
        results = bed_parser.read_and_filter_methylation(
            sample_bedmethyl_bed,
            regions_dict,
        )
        assert list(results.keys())[0] == "chrX_MATERNAL:57866525-60979767"
        assert len(list(results.keys())) == 1
        region_values = results["chrX_MATERNAL:57866525-60979767"]
        assert len(region_values["position"]) == 62064
        assert len(region_values["fraction_modified"]) == 62064
        assert len(region_values["valid_coverage"]) == 62064
    
    def test_chrom_dict_len(self, test_data_dir: Path, bed_parser):
        """Test that process_files returns populated dictionaries"""
        sample_bedmethyl_bed = test_data_dir / "bedmethyl_test.bed"
        sample_censat_bed = test_data_dir / "censat_test.bed"

        methylation_chrom_dict, regions_chrom_dict = bed_parser.process_files(
            methylation_path=sample_bedmethyl_bed,
            regions_path=sample_censat_bed,
        )

        # lengths should be 1 because test file is only one chromosome
        assert len(methylation_chrom_dict) == 1
        assert len(regions_chrom_dict) == 1

    @pytest.mark.parametrize("dataset_key", list(REMOTE_DATASETS))
    def test_remote_dataset_processing(
        self,
        dataset_key: str,
        bed_parser,
        remote_dataset_paths,
    ):
        """Ensure we can process publicly hosted integration datasets."""
        dataset_paths = remote_dataset_paths.get(dataset_key)
        if not dataset_paths:
            pytest.skip(f"Dataset {dataset_key} could not be downloaded.")

        regions_dict = bed_parser.read_and_filter_regions(dataset_paths["regions"])
        methylation_dict = bed_parser.read_and_filter_methylation(
            dataset_paths["bedmethyl"],
            regions_dict,
        )

        expected_chrom = EXPECTED_CHROMS[dataset_key]
        assert expected_chrom in regions_dict
        assert methylation_dict
        assert all(key.startswith(f"{expected_chrom}:") for key in methylation_dict)

        for values in methylation_dict.values():
            counts = {len(values["position"]), len(values["fraction_modified"]), len(values["valid_coverage"])}
            assert counts == {len(values["position"])}