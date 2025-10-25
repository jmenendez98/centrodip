from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple


DipRecord = Dict[str, List[int]]
DipResults = Dict[str, DipRecord]


@dataclass
class DipFilterSettings:
    min_size: int = 50
    cluster_distance: int = 200_000


class DipFilter:
    """Apply post-processing filters to detected dips."""

    def __init__(
        self,
        min_size: int = 50,
        cluster_distance: int = 200_000,
    ) -> None:
        self.settings = DipFilterSettings(
            min_size=min_size,
            cluster_distance=cluster_distance,
        )

    @staticmethod
    def _largest_cluster_indices(
        midpoints: Sequence[int],
        max_gap: int,
    ) -> Sequence[int]:
        if len(midpoints) <= 1:
            return tuple(range(len(midpoints)))
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")

        sorted_midpoints = sorted(enumerate(midpoints), key=lambda item: item[1])
        clusters: List[List[int]] = []
        current_cluster: List[int] = []
        previous_midpoint: int | None = None

        for index, midpoint in sorted_midpoints:
            if previous_midpoint is None or midpoint - previous_midpoint <= max_gap:
                current_cluster.append(index)
            else:
                clusters.append(current_cluster)
                current_cluster = [index]
            previous_midpoint = midpoint

        clusters.append(current_cluster)

        def cluster_key(cluster: Sequence[int]) -> Tuple[int, int]:
            min_midpoint = min(midpoints[idx] for idx in cluster)
            return (len(cluster), -min_midpoint)

        best_cluster = max(clusters, key=cluster_key, default=[])
        return tuple(sorted(best_cluster))

    def _apply_size_filter(
        self,
        starts: Iterable[int],
        ends: Iterable[int],
    ) -> List[Tuple[int, int]]:
        filtered: List[Tuple[int, int]] = []
        for start_raw, end_raw in zip(starts, ends):
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                start, end = end, start
            if (end - start) >= self.settings.min_size:
                filtered.append((start, end))
        return filtered

    def _filter_record(self, record: MutableMapping[str, List[int]]) -> DipRecord:
        starts = record.get("starts", []) or []
        ends = record.get("ends", []) or []

        size_filtered = self._apply_size_filter(starts, ends)
        if not size_filtered:
            filtered_record: DipRecord = {}
            for key, value in record.items():
                if key in {"starts", "ends"}:
                    filtered_record[key] = []
                elif isinstance(value, list):
                    filtered_record[key] = list(value)
                else:
                    filtered_record[key] = value  # type: ignore[assignment]
            if "starts" not in filtered_record:
                filtered_record["starts"] = []
            if "ends" not in filtered_record:
                filtered_record["ends"] = []
            return filtered_record

        midpoints = [start + ((end - start) // 2) for start, end in size_filtered]
        if self.settings.cluster_distance < 0:
            filtered_pairs = list(size_filtered)
        else:
            cluster_indices = self._largest_cluster_indices(
                midpoints,
                self.settings.cluster_distance,
            )
            cluster_set = set(cluster_indices)

            filtered_pairs = [
                size_filtered[idx]
                for idx in range(len(size_filtered))
                if idx in cluster_set
            ]

        starts_out = [start for start, _ in filtered_pairs]
        ends_out = [end for _, end in filtered_pairs]

        filtered_record: DipRecord = {}
        for key, value in record.items():
            if key == "starts":
                filtered_record[key] = starts_out
            elif key == "ends":
                filtered_record[key] = ends_out
            elif isinstance(value, list):
                filtered_record[key] = list(value)
            else:
                filtered_record[key] = value  # type: ignore[assignment]

        if "starts" not in filtered_record:
            filtered_record["starts"] = starts_out
        if "ends" not in filtered_record:
            filtered_record["ends"] = ends_out

        return filtered_record

    def filter(self, dip_results: DipResults) -> DipResults:
        filtered: DipResults = {}
        for region, record in dip_results.items():
            if not isinstance(record, MutableMapping):
                filtered[region] = {"starts": [], "ends": []}
                continue
            filtered[region] = self._filter_record(record)
        return filtered