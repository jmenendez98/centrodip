import pytest

from centrodip.detect_dips import DipDetector

from tests.conftest import EXPECTED_CHROMS

class TestMatrix:
    def test_centrodip_remote_dataset(
        self,
        data_handler_factory,
        remote_dataset_paths,
    ):
        """Test making matrices against a real dataset."""
        selected_dataset = None

        for dataset_key, dataset_paths in remote_dataset_paths.items():
            handler = data_handler_factory(
                methylation_path=dataset_paths["bedmethyl"],
                regions_path=dataset_paths["regions"],
                threads=1,
            )

            for region_name, region_values in handler.methylation_dict.items():
                if len(region_values["position"]) >= 51:
                    selected_dataset = (
                        dataset_key,
                        region_name,
                        region_values,
                        handler,
                    )
                    break

            if selected_dataset:
                break

        if not selected_dataset:
            pytest.skip("No downloaded region contains enough CpGs for smoothing.")

        dataset_key, region_name, region_values, handler = selected_dataset
        expected_chrom = EXPECTED_CHROMS[dataset_key]
        assert region_name.split(":", 1)[0] == expected_chrom

        detector = DipDetector(
            methylation_data=handler.methylation_dict,
            regions_data=handler.region_dict,
            prominence=0.667,
            height=10,
            broadness=10,
            enrichment=False,
            threads=1,
        )
        dip_results = detector.detect_all()

        assert region_name in dip_results
        methylation_per_region = handler.methylation_dict
        assert "lowess_fraction_modified" in methylation_per_region[region_name]
        assert "lowess_fraction_modified_dy" in methylation_per_region[region_name]
        assert len(methylation_per_region[region_name]["lowess_fraction_modified"]) == len(
            region_values["fraction_modified"]
        )