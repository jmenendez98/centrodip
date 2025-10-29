from bisect import bisect_left, bisect_right

import pytest

from centrodip.detect_dips import detectDips
from centrodip.lowess_smooth import lowessSmooth

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
            )

            regions = handler.read_regions_bed(handler.regions_path)
            region_records = {}

            for chrom, payload in handler.load_data():
                chrom_payload = payload.get(chrom, {})
                positions = chrom_payload.get("cpg_pos", [])
                fractions = chrom_payload.get("fraction_modified", [])
                coverage = chrom_payload.get("valid_coverage", [])

                chrom_regions = regions.get(chrom)
                if not chrom_regions:
                    continue

                starts = chrom_regions.get("starts", [])
                ends = chrom_regions.get("ends", [])

                for start, end in zip(starts, ends):
                    left = bisect_right(positions, start)
                    right = bisect_left(positions, end)
                    region_name = f"{chrom}:{start}-{end}"
                    region_records[region_name] = {
                        "position": positions[left:right],
                        "fraction_modified": fractions[left:right],
                        "valid_coverage": coverage[left:right],
                    }

            for region_name, region_values in region_records.items():
                if len(region_values["position"]) >= 51:
                    selected_dataset = (
                        dataset_key,
                        region_name,
                        region_values,
                    )
                    break

            if selected_dataset:
                break

        if not selected_dataset:
            pytest.skip("No downloaded region contains enough CpGs for smoothing.")

        dataset_key, region_name, region_values = selected_dataset
        expected_chrom = EXPECTED_CHROMS[dataset_key]
        assert region_name.split(":", 1)[0] == expected_chrom

        smoothed, derivative = lowessSmooth(
            y=region_values["fraction_modified"],
            x=region_values["position"],
            c=region_values["valid_coverage"],
            window_bp=10000,
            cov_conf=1.0,
        )

        assert len(smoothed) == len(region_values["position"])
        assert len(derivative) == len(region_values["position"])

        dips, dip_edge_indices = detectDips(
            {
                "cpg_pos": region_values["position"],
                "lowess_fraction_modified": smoothed,
                "lowess_fraction_modified_dy": derivative,
            },
            prominence=0.667,
            height=10,
            enrichment=False,
            broadness=10,
            debug=False,
        )

        assert set(dips.keys()) == {"starts", "ends"}
        assert isinstance(dip_edge_indices, list)