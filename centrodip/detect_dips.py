from __future__ import annotations

import concurrent.futures
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float | int]]
RegionRecord = Dict[str, List[float | int]]
DipResults = Dict[str, Dict[str, List[int]]]

def _empty_dip_record() -> Dict[str, List[int]]:
    return {
        "starts": [],
        "ends": [],
    }

def detectDips(
    methylation_data: Dict[str, MethylationRecord],
    regions_data: Dict[str, RegionRecord],
    sensitivity: float = 0.667,
    dip_width: float = 11,
    enrichment: bool = False,
    threads: int = 1,
    debug: bool = False,
) -> DipResults:
    """Detect dips across all chromosomes/regions in the provided data."""

    detector = DipDetector(
        methylation_data=methylation_data,
        regions_data=regions_data,
        sensitivity=sensitivity,
        dip_width=dip_width,
        enrichment=enrichment,
        threads=threads,
        debug=debug,
    )
    return detector.detect_all()

class DipDetector:
    """Detect dips in methylation profiles."""

    def __init__(
        self,
        methylation_data: Dict[str, MethylationRecord],
        regions_data: Dict[str, RegionRecord],
        sensitivity: float = 0.667,
        dip_width: float = 11,
        enrichment: bool = False,
        threads: int = 1,
        debug: bool = False,
    ) -> None:
        self.methylation_data = methylation_data
        self.regions_data = regions_data

        self.sensitivity = float(sensitivity)
        self.dip_width = float(dip_width)
        self.enrichment = bool(enrichment)

        self.threads = max(int(threads), 1)
        self.debug = debug

        self.dips: DipResults = {}


    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][DipDetector] {message}", flush=True)

    # ------------------------------------------------------------------
    # public API
    def detect_all(self) -> DipResults:
        """Calculate dips for every chromosome/region in parallel."""

        if not self.methylation_data:
            self.dips = {}
            return {}

        results: DipResults = {}

        if self.threads > 1 and len(self.methylation_data) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = {
                    executor.submit(self._detect_single_region, key, record): key
                    for key, record in self.methylation_data.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    results[key] = future.result()
        else:
            for key, record in self.methylation_data.items():
                results[key] = self._detect_single_region(key, record)

        self.dips = results
        return results

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

    @staticmethod
    def _region_label(key: str) -> str:
        if ":" in key:
            return key.split(":", 1)[0]
        return key

    def _detect_single_region(
        self,
        region_key: str,
        record: MethylationRecord,
    ) -> Dict[str, List[int]]:
        region_label = self._region_label(region_key)

        methylation_record = self._sort_region_record(self._copy_methylation_record(record))

        cpg_sites = np.asarray(methylation_record.get("position", []), dtype=int)
        smoothed = np.asarray(
            methylation_record.get("lowess_fraction_modified", []),
            dtype=float,
        )
        derivative = np.asarray(
            methylation_record.get("lowess_fraction_modified_dy", []),
            dtype=float,
        )

        if cpg_sites.size == 0:
            dip_record = _empty_dip_record()
            dip_record["dip_centers"] = []
            dip_record["dip_edges"] = []
            return dip_record

        centers = self.find_dip_centers(smoothed)
        edges = self.find_potential_dip_edges(derivative, centers)

        dip_record = self.extend_dips(
            cpg_sites=cpg_sites,
            centers=centers,
            edges=edges,
        )

        dip_record["dip_centers"] = [
            int(cpg_sites[idx])
            for idx in centers
            if 0 <= int(idx) < cpg_sites.size
        ]
        dip_record["dip_edges"] = [
            int(cpg_sites[idx])
            for idx in edges
            if 0 <= int(idx) < cpg_sites.size
        ]

        return dip_record

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
            peak_kwargs["width"] = self.dip_width
        if self.enrichment:
            centers, _ = signal.find_peaks(smoothed_methylation, **peak_kwargs)
        else:
            centers, _ = signal.find_peaks(-smoothed_methylation, **peak_kwargs)

        return centers.astype(int)

    def find_potential_dip_edges(
        self,
        smoothed_methylation_dy: np.ndarray,
        centers: np.ndarray,
        include_zero: bool = True,  # treat 0 as part of the run
    ) -> np.ndarray:
        """Return dip edge indices in the smoothed methylation dy."""
        smoothed_methylation_dy = np.abs(smoothed_methylation_dy)

        if smoothed_methylation_dy.size == 0:
            return np.array([], dtype=int)

        data_range = float(np.max(smoothed_methylation_dy) - np.min(smoothed_methylation_dy))
        data_prominence_threshold = max(0.1 * data_range, 0.0)

        # call potential dip edges
        peak_kwargs = {}
        if data_prominence_threshold > 0:
            peak_kwargs["prominence"] = data_prominence_threshold

        peak_kwargs["width"] = self.dip_width

        edges, _ = signal.find_peaks(smoothed_methylation_dy, **peak_kwargs)

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

        dips = _empty_dip_record()
        for center_i in centers_arr:
            # subset edges to those flanking the center
            cur_lefts = edges_arr[edges_arr < center_i]
            left_i = np.argmax(cur_lefts) if cur_lefts.size > 0 else 0
            cur_rights = edges_arr[edges_arr > center_i]
            right_i = np.argmin(cur_rights) if cur_rights.size > 0 else len(cpg_sites) - 1

            dips["starts"].append(int(cpg_sites[left_i]))
            dips["ends"].append(int(cpg_sites[right_i]))

        return dips