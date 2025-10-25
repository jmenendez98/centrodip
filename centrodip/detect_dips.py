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
    edge_sensitivity: float = 0.5,
    enrichment: bool = False,
    threads: int = 1,
    debug: bool = False,
) -> DipResults:
    """Detect dips across all chromosomes/regions in the provided data."""

    detector = DipDetector(
        methylation_data=methylation_data,
        regions_data=regions_data,
        sensitivity=sensitivity,
        edge_sensitivity=edge_sensitivity,
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

        self._log("Starting dip detection across all regions.")

        if not self.methylation_data:
            self._log("No methylation data provided; returning empty results.")
            self.dips = {}
            return {}

        results: DipResults = {}

        if self.threads > 1 and len(self.methylation_data) > 1:
            self._log(f"Using ThreadPoolExecutor with {self.threads} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = {
                    executor.submit(self._detect_single_region, key, record): key
                    for key, record in self.methylation_data.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    results[key] = future.result()
                    self._log(f"Completed detection for {key}.")
        else:
            for key, record in self.methylation_data.items():
                results[key] = self._detect_single_region(key, record)
                self._log(f"Completed detection for {key}.")

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
        self._log(f"Processing region {region_key} (chromosome {region_label}).")

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
            self._log(f"Region {region_key} contains no CpG sites.")
            dip_record = _empty_dip_record()
            dip_record["dip_centers"] = []
            dip_record["dip_edges"] = []
            return dip_record

        self._log(
            f"Region {region_key}: {cpg_sites.size} CpG sites, starting dip computation."
        )

        centers = self.find_dip_centers(smoothed)
        self._log(f"Region {region_key}: identified {centers.size} dip centers.")

        edges = self.find_dip_edges(derivative)
        self._log(f"Region {region_key}: identified {edges.size} dip edges.")

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

        self._log(
            f"Region {region_key}: produced {len(dip_record['starts'])} dip intervals."
        )

        return dip_record

    def find_dip_centers(self, smoothed_methylation: np.ndarray) -> np.ndarray:
        """Return dip center indices in the smoothed methylation data."""
        # get info on data range for prominence calculation
        if smoothed_methylation.size == 0:
            return np.array([], dtype=int)
        data_range = float(np.max(smoothed_methylation) - np.min(smoothed_methylation))
        data_prominence_threshold = max(self.sensitivity * data_range, 0.0)

        self._log(
            "Computing dip centers: "
            f"n={smoothed_methylation.size}, prominence>={data_prominence_threshold:.6f}."
        )

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

        self._log(
            "Computing dip edges: "
            f"n={smoothed_methylation_dy.size}, prominence>={edge_prominence:.6f}."
        )

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

        self._log(
            f"Extended {len(dip_bounds)} dip bounds into genomic coordinates."
        )

        return dips