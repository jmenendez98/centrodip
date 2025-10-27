from __future__ import annotations

import concurrent.futures
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float]]
RegionRecord = Dict[str, List[int]]
DipResults = Dict[str, Dict[str, List[int]]]


def _empty_dip_record() -> Dict[str, List[int]]:
    return {
        "starts": [],
        "ends": [],
    }

def detectDips(
    methylation_data: Dict[str, MethylationRecord],
    regions_data: Dict[str, RegionRecord],
    prominence: float = 0.25,
    height: float = 10,
    broadness: float = 10,
    enrichment: bool = False,
    threads: int = 1,
    debug: bool = False,
) -> DipResults:
    """Detect dips across all chromosomes/regions in the provided data."""

    detector = DipDetector(
        methylation_data=methylation_data,
        regions_data=regions_data,
        prominence=prominence,
        height=height,
        broadness=broadness,
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
        prominence: float = 0.25,
        height: float = 10,
        broadness: float = 10,
        enrichment: bool = False,
        threads: int = 1,
        debug: bool = False,
    ) -> None:
        self.methylation_data = methylation_data
        self.regions_data = regions_data

        self.prominence = float(prominence)
        self.height = float(height)
        self.broadness = float(broadness)
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
            dip_record["positions"] = []
            dip_record["lowess_fraction_modified"] = []
            return dip_record

        # find the dip centers
        centers = self.find_dip_centers(smoothed)
        #print(centers)

        # determine the dip edge bounds
        threshold = np.percentile(smoothed, self.broadness)
        edge_bounds = self.find_edge_bounds(smoothed, threshold, centers)
        #print(threshold)
        #print(edge_bounds)

        # pinpoint the actual edge using dy
        # print(edge_bounds)
        edges = self.find_edges(derivative, centers, edge_bounds)

        dip_record = self.extend_dips(
            cpg_sites=cpg_sites,
            centers=centers,
            edges=edges,
        )

        dip_record["positions"] = [int(pos) for pos in cpg_sites.tolist()]
        dip_record["lowess_fraction_modified"] = [float(val) for val in smoothed.tolist()]

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
        data_prominence_threshold = self.prominence * data_range

        if self.enrichment:
            centers, _ = signal.find_peaks(
                smoothed_methylation, 
                prominence=data_prominence_threshold,
                height=np.percentile(smoothed_methylation, 100-self.height),
                wlen=len(smoothed_methylation)
            )
        else:
            centers, _ = signal.find_peaks(
                -smoothed_methylation, 
                prominence=data_prominence_threshold,
                height=-np.percentile(smoothed_methylation, self.height),
                wlen=len(smoothed_methylation)
            )

        return centers.astype(int)

    def find_edge_bounds(
        self,
        data: np.ndarray,
        bounding_threshold: float,
        centers: np.ndarray,
    ) -> np.ndarray:
        n = data.size
        if n == 0:
            return np.array([], dtype=int)

        edge_bounds = []
        for c in centers:
            if c < 0:
                continue

            # ----- search left -----
            li = c - 1
            left_found = None
            while li >= 0:
                val = data[li]
                if np.isfinite(val) and val >= bounding_threshold:
                    left_found = li
                    break
                li -= 1

            # ----- search right -----
            ri = c + 1
            right_found = None
            while ri < n:
                val = data[ri]
                if np.isfinite(val) and val >= bounding_threshold:
                    right_found = ri
                    break
                ri += 1

            if (left_found is not None) and (right_found is not None):
                edge_bounds.append((int(left_found), int(right_found)))
            else:
                edge_bounds.append((None, None))

        return np.array(edge_bounds)

    def find_edges(
        self,
        dy: np.ndarray,
        centers: Iterable[int],
        edge_bounds: Tuple(int, int),
    ) -> np.ndarray:
        if len(centers) != len(edge_bounds):
            print( "differing number of centers and edges!" )
            return

        edges = []
        for i, (c, (l, r)) in enumerate(zip(centers, edge_bounds)):
            if l <= c <= r:
                continue # if center is not in between edges
            elif l or r is None:
                continue # if edge is not reached through bounds

            
            print(i, (c, e))
        
        return

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
            if cur_lefts.size > 0:
                left_i = int(cur_lefts.max())
            else:
                left_i = max(int(center_i), 0)

            cur_rights = edges_arr[edges_arr > center_i]
            if cur_rights.size > 0:
                right_i = int(cur_rights.min())
            else:
                right_i = min(int(center_i), len(cpg_sites) - 1)

            dips["starts"].append(int(cpg_sites[left_i]))
            dips["ends"].append(int(cpg_sites[right_i]))

        return dips