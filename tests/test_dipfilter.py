import pytest

from centrodip.dip_filter import DipFilter


class TestDipFilter:
    def test_min_size_filtering(self):
        dip_filter = DipFilter(min_size=50, cluster_distance=1_000)
        dip_results = {
            "chr1:0-500": {
                "starts": [0, 100, 200],
                "ends": [25, 200, 400],
            }
        }

        filtered = dip_filter.filter(dip_results)
        region = filtered["chr1:0-500"]

        assert region["starts"] == [100, 200]
        assert region["ends"] == [200, 400]

    def test_largest_cluster_kept(self):
        dip_filter = DipFilter(min_size=0, cluster_distance=150)
        dip_results = {
            "chr2:0-2000": {
                "starts": [0, 50, 100, 1000, 1100],
                "ends": [10, 70, 120, 1050, 1150],
            }
        }

        filtered = dip_filter.filter(dip_results)
        region = filtered["chr2:0-2000"]

        assert region["starts"] == [0, 50, 100]
        assert region["ends"] == [10, 70, 120]

    @pytest.mark.parametrize("cluster_distance", [0, 1])
    def test_zero_or_small_gap_clusters(self, cluster_distance):
        dip_filter = DipFilter(min_size=0, cluster_distance=cluster_distance)
        dip_results = {
            "chr3:0-300": {
                "starts": [0, 100, 200],
                "ends": [10, 110, 210],
            }
        }

        filtered = dip_filter.filter(dip_results)
        region = filtered["chr3:0-300"]

        # Each interval forms its own cluster when gap is 0, so keep the first one.
        assert region["starts"] == [0]
        assert region["ends"] == [10]