from pathlib import Path

import pytest

from centrodip.load_data import DataHandler

from tests.conftest import EXPECTED_CHROMS, REMOTE_DATASETS

class TestParser:
    def test_fake_bedfile(self, test_data_dir: Path):
        """Test handling of nonexistent file"""
        nonexistent_file = test_data_dir / "nonexistent.bed"
        with pytest.raises(FileNotFoundError):
            DataHandler(
                regions_path=nonexistent_file,
                methylation_path=nonexistent_file,
                mod_code="m",
            )

    def test_empty_bedfile(self, test_data_dir: Path):
        """Test handling of empty file"""
        empty_file = test_data_dir / "empty.bed"
        handler = DataHandler(
            regions_path=empty_file,
            methylation_path=empty_file,
            mod_code="m",
        )
        result1 = handler.read_regions_bed(empty_file)
        assert len(result1) == 0
        assert list(handler.load_data()) == []

    def test_censat_bedfile(self, test_data_dir: Path, bed_parser):
        sample_censat_bed = test_data_dir / "censat_test.bed"
        results = bed_parser.read_regions_bed(sample_censat_bed)
        assert list(results.keys())[0] == "chrX_MATERNAL"
        assert len(list(results.keys())) == 1
        assert results["chrX_MATERNAL"]["starts"] == [57866525]
        assert results["chrX_MATERNAL"]["ends"] == [60979767]

    def test_bedmethyl_bedfile(self, test_data_dir: Path):
        """Test basic bedmethyl reading functionality"""
        sample_bedmethyl_bed = test_data_dir / "bedmethyl_test.bed"
        sample_censat_bed = test_data_dir / "censat_test.bed"
        handler = DataHandler(
            regions_path=sample_censat_bed,
            methylation_path=sample_bedmethyl_bed,
            mod_code="m",
        )
        regions_dict = handler.read_regions_bed(sample_censat_bed)
        chrom_data = list(handler.load_data())

        assert list(regions_dict.keys()) == ["chrX_MATERNAL"]
        assert len(chrom_data) == 1

        chrom, payload = chrom_data[0]
        assert chrom == "chrX_MATERNAL"
        region_values = payload[chrom]
        assert len(region_values["cpg_pos"]) == 62064
        assert len(region_values["fraction_modified"]) == 62064
        assert len(region_values["valid_coverage"]) == 62064
    
    def test_chrom_dict_len(self, test_data_dir: Path):
        """Test that load_data yields data for each chromosome"""
        sample_bedmethyl_bed = test_data_dir / "bedmethyl_test.bed"
        sample_censat_bed = test_data_dir / "censat_test.bed"

        handler = DataHandler(
            regions_path=sample_censat_bed,
            methylation_path=sample_bedmethyl_bed,
            mod_code="m",
        )

        regions = handler.read_regions_bed(sample_censat_bed)
        chrom_payloads = list(handler.load_data())

        assert len(regions) == 1
        assert len(chrom_payloads) == 1

    @pytest.mark.parametrize("dataset_key", list(REMOTE_DATASETS))
    def test_remote_dataset_processing(
        self,
        dataset_key: str,
        remote_dataset_paths,
    ):
        """Ensure we can process publicly hosted integration datasets."""
        dataset_paths = remote_dataset_paths.get(dataset_key)
        if not dataset_paths:
            pytest.skip(f"Dataset {dataset_key} could not be downloaded.")

        handler = DataHandler(
            regions_path=dataset_paths["regions"],
            methylation_path=dataset_paths["bedmethyl"],
            mod_code="m",
        )

        regions_dict = handler.read_regions_bed(dataset_paths["regions"])
        chrom_payloads = list(handler.load_data())

        expected_chrom = EXPECTED_CHROMS[dataset_key]
        assert expected_chrom in regions_dict
        assert chrom_payloads

        for chrom, payload in chrom_payloads:
            assert chrom.split(":", 1)[0] == expected_chrom
            values = payload[chrom]
            counts = {
                len(values["cpg_pos"]),
                len(values["fraction_modified"]),
                len(values["valid_coverage"]),
            }
            assert counts == {len(values["cpg_pos"])}