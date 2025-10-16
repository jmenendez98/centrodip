import pytest

from centrodip.parser import Parser
from centrodip.cdr_detect import Dip_Detector

from tests.conftest import EXPECTED_CHROMS

class TestMatrix:
    @pytest.fixture
    def centro_dip(self):
        """Fixture for matrix calculator"""
        return Dip_Detector(
            window_size=51,
            threshold=1,
            prominence=0.5,
            min_size=1000,
            enrichment=False,
            threads=1,
            color="50,50,255",
            label="slarf",
        )

    def test_centrodip_remote_dataset(
        self,
        bed_parser,
        remote_dataset_paths,
        centro_dip,
    ):
        """Test making matrices against a real dataset."""
        selected_dataset = None

        for dataset_key, dataset_paths in remote_dataset_paths.items():
            regions_dict = bed_parser.read_and_filter_regions(dataset_paths["regions"])
            methylation_dict = bed_parser.read_and_filter_methylation(
                dataset_paths["bedmethyl"],
                regions_dict,
            )

            for region_name, region_values in methylation_dict.items():
                if len(region_values["starts"]) >= centro_dip.window_size:
                    selected_dataset = (
                        dataset_key,
                        region_name,
                        region_values,
                        methylation_dict,
                        regions_dict,
                    )
                    break

            if selected_dataset:
                break

        if not selected_dataset:
            pytest.skip("No downloaded region contains enough CpGs for smoothing.")

        dataset_key, region_name, region_values, methylation_dict, regions_dict = selected_dataset
        expected_chrom = EXPECTED_CHROMS[dataset_key]
        assert region_name.startswith(f"{expected_chrom}:")

        methylation_subset = {region_name: region_values}
        cdrs_per_region, methylation_per_region = centro_dip.centrodip_all_chromosomes(
            methylation_per_region=methylation_subset,
            regions_per_chrom=regions_dict,
        )

        assert region_name in cdrs_per_region
        assert region_name in methylation_per_region
        assert "savgol_frac_mod" in methylation_per_region[region_name]
        assert "savgol_frac_mod_dy" in methylation_per_region[region_name]
        assert len(methylation_per_region[region_name]["savgol_frac_mod"]) == len(
            methylation_subset[region_name]["fraction_modified"]
        )