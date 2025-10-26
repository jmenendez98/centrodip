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
    prominence: float = 0.25,
    height: float = 10,
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
        enrichment: bool = False,
        threads: int = 1,
        debug: bool = False,
    ) -> None:
        self.methylation_data = methylation_data
        self.regions_data = regions_data

        self.prominence = float(prominence)
        self.height = float(height)
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
        edges = self.find_potential_edges(derivative, centers)
        edges = self.correct_edges(
            dy=derivative, 
            centers=centers, 
            edges=edges,
            eps=1e-6,
        )

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

    def find_potential_edges(
        self,
        smoothed_methylation_dy: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """Return dip edge indices in the smoothed methylation dy."""

        if smoothed_methylation_dy.size == 0:
            return np.array([], dtype=int)

        neg_dy=smoothed_methylation_dy[smoothed_methylation_dy<0]
        l_edges, _ = signal.find_peaks(
            -smoothed_methylation_dy, 
            height=-np.percentile(neg_dy, 10)
        )

        pos_dy=smoothed_methylation_dy[smoothed_methylation_dy>0]
        r_edges, _ = signal.find_peaks(
            smoothed_methylation_dy, 
            height=np.percentile(pos_dy, 90)
        )

        edges = np.concatenate([l_edges, r_edges])
        return np.sort( edges.astype(int) )

    @staticmethod
    def correct_edges(
        dy: np.ndarray,
        centers: Iterable[int],
        edges: Iterable[int],
        eps: float = 1e-6,          # hysteresis: treat |dy| <= eps as zero
    ) -> np.ndarray:
        dy = np.asarray(dy, dtype=float)
        centers = np.asarray(list(centers), dtype=int)
        edges = np.asarray(list(edges), dtype=int)

        n = dy.size
        if n == 0 or centers.size == 0:
            return np.empty(0, dtype=int)

        edges = edges[(0 <= edges) & (edges < n)]
        edges.sort()

        corrected: List[int] = []

        for center in centers:
            if not (0 <= center < n):
                continue
            left_idx = np.searchsorted(edges, center, side="left")
            right_idx = np.searchsorted(edges, center, side="right")

            has_left_edge = left_idx > 0
            has_right_edge = right_idx < edges.size

            left = edges[left_idx - 1] if has_left_edge else max(center - 1, 0)
            right = edges[right_idx] if has_right_edge else min(center + 1, n - 1)

            if has_left_edge and left < center:
                lo = hi = int(left)
                while lo - 1 >= 0 and dy[lo - 1] <= eps:
                    lo -= 1
                while hi + 1 < center and dy[hi + 1] <= eps:
                    hi += 1
                hi = min(hi, center - 1)
                if hi >= lo:
                    left = lo + int(np.argmin(dy[lo : hi + 1]))
                else:
                    left = int(left)
            else:
                left = max(center - 1, 0)

            if has_right_edge and right > center:
                lo = hi = int(right)
                while hi + 1 < n and dy[hi + 1] >= -eps:
                    hi += 1
                while lo - 1 > center and dy[lo - 1] >= -eps:
                    lo -= 1
                lo = max(lo, center + 1)
                if hi >= lo:
                    right = lo + int(np.argmax(dy[lo : hi + 1]))
                else:
                    right = int(right)
            else:
                right = min(center + 1, n - 1)
            corrected.extend((int(left), int(right)))

        return np.asarray(corrected, dtype=int)

        def _expand_patch_left(start_i: int, stop_before: int) -> Tuple[int, int]:
            """Expand around start_i while dy <= +eps, capped at stop_before (exclusive)."""
            # left bound
            L = start_i
            while L - 1 >= 0 and dy[L - 1] <= eps:
                if (L - 1) >= 0:
                    L -= 1
                else:
                    break
            # right bound
            R = start_i
            while R + 1 < n and dy[R + 1] <= eps and (R + 1) < stop_before:
                R += 1
            # cap to not cross center: caller sets stop_before = center
            R = min(R, stop_before - 1)
            return L, R

        def _expand_patch_right(start_i: int, start_after: int) -> Tuple[int, int]:
            """Expand around start_i while dy >= -eps, capped at start_after (inclusive lower bound)."""
            # left bound
            L = start_i
            while L - 1 >= 0 and dy[L - 1] >= -eps and (L - 1) > start_after:
                L -= 1
            # right bound
            R = start_i
            while R + 1 < n and dy[R + 1] >= -eps:
                R += 1
            # cap to not cross center: caller sets start_after = center
            L = max(L, start_after + 1)
            return L, R

        for c in centers:
            if c < 0 or c >= n:
                continue

            # ----- find nearest left/right candidate edges relative to c -----
            # left: max edge < c
            left_mask = edges_sorted < c
            if np.any(left_mask):
                left_i0 = edges_sorted[left_mask][-1]
            else:
                left_i0 = max(c - 1, 0)

            # right: min edge > c
            right_mask = edges_sorted > c
            if np.any(right_mask):
                right_i0 = edges_sorted[right_mask][0]
            else:
                right_i0 = min(c + 1, n - 1)

            # ----- refine within slope patches -----
            # LEFT patch: dy <= +eps, capped to < c
            if left_i0 < c:
                L, R = _expand_patch_left(left_i0, stop_before=c)
                if L <= R:
                    # argmin within [L, R]
                    left_i = L + int(np.argmin(dy[L:R + 1]))
                else:
                    left_i = left_i0
            else:
                left_i = max(c - 1, 0)

            # RIGHT patch: dy >= -eps, capped to > c
            if right_i0 > c:
                Lr, Rr = _expand_patch_right(right_i0, start_after=c)
                if Lr <= Rr:
                    # argmax within [Lr, Rr]
                    right_i = Lr + int(np.argmax(dy[Lr:Rr + 1]))
                else:
                    right_i = right_i0
            else:
                right_i = min(c + 1, n - 1)

            corrected.append((int(left_i), int(right_i)))

        return np.asarray(corrected, dtype=int)

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