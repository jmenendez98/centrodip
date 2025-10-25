from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple, Union

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float | int]]
RegionRecord = Dict[str, List[float | int]]

def _empty_dip_record() -> Dict[str, List]:
    return {
        "starts": [],
        "ends": [],
    }

def detectDips():
    return 

class DipDetector:
    """Detect dips in methylation profiles."""
    def __init__(
        self,
        methylation_data: Dict[str, MethylationRecord],
        regions_data: Dict[str, RegionRecord],
        sensitivity: float = 0.667,
        edge_sensitivity: float = 0.5,
        enrichment: bool = False,
        threads: int = 1,
        debug: bool = False,
    ) -> None:
        self.methylation_data = methylation_data
        self.regions_data = regions_data

        self.sensitivity = float(sensitivity)
        self.edge_sensitivity = float(edge_sensitivity)
        self.enrichment = bool(enrichment)

        self.threads = int(threads)
        self.debug = debug

        self.dips: Dict[str, Dict[str, List[int]]] = {}

        cpg_sites = np.array( self.methylation_data["position"], dtype=int )

        smoothed_methylation = np.array( self.methylation_data["lowess_fraction_modified"], dtype=float )
        smoothed_methylation_dy = np.array( self.methylation_data["lowess_fraction_modified_dy"], dtype=float )

        dip_centers = self.find_dip_centers(smoothed_methylation)
        dip_edges = self.find_dip_edges(smoothed_methylation_dy)
        dips = self.extend_dips(
            cpg_sites=cpg_sites, 
            centers=dip_centers, 
            edges=dip_edges
        )

        print(dips)

    @staticmethod
    def _copy_methylation_record(record: Dict[str, Iterable]) -> MethylationRecord:
        return {
            "position": list(record.get("position", [])),
            "fraction_modified": list(record.get("fraction_modified", [])),
            "valid_coverage": list(record.get("valid_coverage", [])),
            "lowess_fraction_modified": list(record.get("lowess_fraction_modified", [])),
            "lowess_fraction_modified_dy": list(record.get("lowess_fraction_modified_dy", [])),
        }

    @staticmethod
    def _sort_region_record(record: MethylationRecord) -> MethylationRecord:
        positions = record.get("position", [])
        if not positions:
            return record

        order = sorted(range(len(positions)), key=positions.__getitem__)
        record["position"] = [positions[idx] for idx in order]

        for key in (
            "fraction_modified",
            "valid_coverage",
            "lowess_fraction_modified",
            "lowess_fraction_modified_dy",
        ):
            values = record.get(key, [])
            if len(values) == len(order):
                record[key] = [values[idx] for idx in order]

        return record

    def find_dip_centers(self, smoothed_methylation: np.ndarray) -> np.ndarray:
        """Return dip center indices in the smoothed methylation data."""
        # get info on data range for prominence calculation
        if smoothed_methylation.size == 0:
            return np.array([], dtype=int)
        data_range = float(np.max(smoothed_methylation) - np.min(smoothed_methylation))
        data_prominence_threshold = max(self.sensitivity * data_range, 0.0)

        # call the dip centers
        peak_kwargs = {}
        if data_prominence_threshold > 0:
            peak_kwargs["prominence"] = data_prominence_threshold
        if self.enrichment:
            centers, _ = signal.find_peaks(smoothed_methylation, **peak_kwargs)
        else:
            centers, _ = signal.find_peaks(-smoothed_methylation, **peak_kwargs)

        return centers.astype(int)

    def find_dip_edges(self, smoothed_methylation_dy: np.ndarray) -> np.ndarray:
        """Return the dip edges for a single region."""
        # get info on data range for prominence calculation
        if smoothed_methylation_dy.size == 0:
            return np.array([], dtype=int)
        derivative_range = float(np.max(smoothed_methylation_dy) - np.min(smoothed_methylation_dy))
        edge_prominence = max(self.edge_sensitivity * derivative_range, 0.0)

        # call the dip edges
        peak_kwargs = {}
        if edge_prominence > 0:
            peak_kwargs["prominence"] = edge_prominence
        edges, _ = signal.find_peaks(np.abs(smoothed_methylation_dy), **peak_kwargs)
        edges = np.asarray(np.unique(edges), dtype=int)

        return edges.astype(int)

    def extend_dips(
        self,
        cpg_sites: np.ndarray,
        centers: Iterable[int],
        edges: Iterable[int],
    ) -> Dict[str, List[int]]:
        """Given dip centers and edges, return dip start and end positions."""
        if cpg_sites.size == 0:
            return _empty_dip_record()

        centers_arr = np.asarray(list(centers), dtype=int)
        edges_arr = np.asarray(list(edges), dtype=int)

        if centers_arr.size == 0:
            return _empty_dip_record()

        dip_bounds: List[Tuple[int, int]] = []

        for center in centers_arr:
            if center < 0 or center >= cpg_sites.size:
                continue

            left_candidates = edges_arr[edges_arr <= center]
            right_candidates = edges_arr[edges_arr >= center]

            if left_candidates.size == 0 and right_candidates.size == 0:
                left_idx = right_idx = int(center)
            else:
                left_idx = int(left_candidates[-1]) if left_candidates.size else int(center)
                right_idx = int(right_candidates[0]) if right_candidates.size else int(center)

            left_idx = max(0, min(left_idx, cpg_sites.size - 1))
            right_idx = max(0, min(right_idx, cpg_sites.size - 1))

            if right_idx < left_idx:
                left_idx, right_idx = right_idx, left_idx

            dip_bounds.append((left_idx, right_idx))

        dips = _empty_dip_record()
        for left_idx, right_idx in dip_bounds:
            dips["starts"].append(int(cpg_sites[left_idx]))
            dips["ends"].append(int(cpg_sites[right_idx]))

        return dips