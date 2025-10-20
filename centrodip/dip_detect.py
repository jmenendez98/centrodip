from __future__ import annotations

import concurrent.futures
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float]]
RegionSlices = Dict[str, Tuple[int, int]]


def _empty_methylation_record() -> MethylationRecord:
    return {
        "position": [],
        "fraction_modified": [],
        "valid_coverage": [],
    }


def _empty_dip_record() -> Dict[str, List]:
    return {
        "starts": [],
        "ends": [],
    }


class DipDetector:
    def __init__(
        self,
        window_size,
        sensitivity,
        enrichment,
        threads,
        debug: bool = False,
    ) -> None:
        self.window_size = window_size
        self.sensitivity = sensitivity

        self.enrichment = enrichment

        self.threads = threads
        self.debug = debug

    @staticmethod
    def _effective_window_length(length: int, requested: int) -> int | None:
        """Return an odd Savitzky-Golay window length <= length."""

        if length < 3:
            return None
        window = min(length if length % 2 == 1 else length - 1, requested)
        if window < 3:
            window = 3 if length >= 3 else None
        if window is None:
            return None
        if window % 2 == 0:
            window -= 1
        return max(window, 3)

    @staticmethod
    def _copy_methylation_record(record: Dict[str, Iterable]) -> MethylationRecord:
        return {
            "position": list(record.get("position", [])),
            "fraction_modified": list(record.get("fraction_modified", [])),
            "valid_coverage": list(record.get("valid_coverage", [])),
        }

    def _smooth_chromosome(self, values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        window = self._effective_window_length(data.size, self.window_size)
        if window is None or window > data.size:
            return data.copy(), np.zeros_like(data, dtype=float)

        smoothed = signal.savgol_filter(
            x=data,
            window_length=window,
            polyorder=2,
            mode="mirror",
        )
        derivative = signal.savgol_filter(
            x=data,
            window_length=window,
            polyorder=2,
            deriv=1,
            mode="mirror",
        )
        return smoothed, derivative

    def _build_chromosome_arrays(
        self,
        chrom: str,
        region_keys: List[str],
        region_records: Dict[str, MethylationRecord],
    ) -> Tuple[MethylationRecord, RegionSlices]:
        combined = _empty_methylation_record()
        slices: RegionSlices = {}

        for region_key in region_keys:

            record = self._copy_methylation_record(region_records.get(region_key, {}))
            positions = record["position"]

            if positions:
                order = sorted(range(len(positions)), key=lambda idx: positions[idx])
                positions = [positions[idx] for idx in order]
                fraction_modified = [record["fraction_modified"][idx] for idx in order]
                valid_coverage = [record["valid_coverage"][idx] for idx in order]
            else:
                fraction_modified = []
                valid_coverage = []

            start_idx = len(combined["position"])
            combined["position"].extend(positions)
            combined["fraction_modified"].extend(fraction_modified)
            combined["valid_coverage"].extend(valid_coverage)
            end_idx = len(combined["position"])
            slices[region_key] = (start_idx, end_idx)


        return combined, slices

    def _slice_region(
        self,
        combined: MethylationRecord,
        smoothed: np.ndarray,
        derivative: np.ndarray,
        slc: slice,
    ) -> MethylationRecord:
        region_data = {
            "position": combined["position"][slc],
            "fraction_modified": combined["fraction_modified"][slc],
            "valid_coverage": combined["valid_coverage"][slc],
            "savgol_frac_mod": smoothed[slc].tolist(),
            "savgol_frac_mod_dy": derivative[slc].tolist(),
        }
        return region_data


    def detect_dips(self, methylation: MethylationRecord) -> np.ndarray:
        data = np.asarray(methylation.get("savgol_frac_mod", []), dtype=float)
        dy = np.asarray(methylation.get("savgol_frac_mod_dy", []), dtype=float)

        if data.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        data_range = float(np.max(data) - np.min(data)) if data.size else 0.0
        dy_range = float(np.max(dy) - np.min(dy)) if dy.size else 0.0

        data_prominence_threshold = self.sensitivity * data_range
        dy_prominence_threshold = self.sensitivity * dy_range

        mean = float(np.mean(data)) if data.size else 0.0
        std = float(np.std(data)) if data.size else 0.0
        if self.enrichment:
            centers, _ = signal.find_peaks(
                data,
                prominence=data_prominence_threshold,
                wlen=data.size if data.size else None,
            )
            lefts, _ = signal.find_peaks(
                dy,
                prominence=dy_prominence_threshold,
                wlen=data.size if data.size else None,
            )
            rights, _ = signal.find_peaks(
                -dy,
                prominence=dy_prominence_threshold,
                wlen=data.size if data.size else None,
            )            
        else:
            centers, _ = signal.find_peaks(
                -data,
                prominence=data_prominence_threshold,
                wlen=data.size if data.size else None,
            )
            lefts, _ = signal.find_peaks(
                -dy,
                prominence=dy_prominence_threshold,
                wlen=data.size if data.size else None,
            )
            rights, _ = signal.find_peaks(
                dy,
                prominence=dy_prominence_threshold,
                wlen=data.size if data.size else None,
            )

        return centers, lefts, rights

    '''
    def extend_dips(
        self,
        methylation: MethylationRecord,
        dips: Iterable[int],
    ) -> List[Tuple[int, int]]:

        # fetch the relevant data
        starts_m = np.asarray(methylation.get("position", []), dtype=int)
        data = np.asarray(methylation.get("savgol_frac_mod", []), dtype=float)
        dy = np.asarray(methylation.get("savgol_frac_mod_dy", []), dtype=float)
        if data.size == 0 or dy.size == 0:
            return []

        # detect spans on below median values that are candidate dips
        median = np.median(data)
        mask = data > median if self.enrichment else data < median
        prev = np.r_[False, mask[:-1]]
        next_ = np.r_[mask[1:], False]
        starts = np.flatnonzero(mask & ~prev)
        ends = np.flatnonzero(mask & ~next_)
        lefts = np.maximum(starts - 1, 0)
        rights = np.minimum(ends + 1, data.size - 1)
        dips_arr = np.asarray(list(dips), dtype=int)
        if dips_arr.size == 0:
            return []

        # assign each called dip to the span that contains it
        idx = np.searchsorted(starts, dips_arr, side="right") - 1
        valid = idx >= 0
        valid &= mask[dips_arr]
        valid &= dips_arr <= ends[idx.clip(min=0)]
        idx = idx[valid]
        dip_bounds = [(int(lefts[i]), int(rights[i])) for i in idx]

        # make edges more accurate by using the spots where the slopes are most drastic
        dip_bounds_adj: List[Tuple[int, int]] = []
        for dip, (left, right) in zip(dips_arr[valid], dip_bounds):
            if left > dip:
                left = dip
            if right < dip:
                right = dip
            left_idx = int(np.argmin(dy[left : dip + 1]) + left)
            right_idx = int(np.argmax(dy[dip : right + 1]) + dip)
            dip_bounds_adj.append((left_idx, right_idx))

        # convert the dip indices back into genomic coordinates
        dips = {}
        dips['starts'] = []
        dips['ends'] = []
        for s, e in dip_bounds_adj:
            dips['starts'].append( starts_m[s] )
            dips['ends'].append( starts_m[e] )

        return dips
    '''
    def extend_dips(
        self,
        methylation: MethylationRecord,
        dips: Iterable[int],
    ) -> List[Tuple[int, int]]:

        # fetch the relevant data
        starts_m = np.asarray(methylation.get("position", []), dtype=int)
        data = np.asarray(methylation.get("savgol_frac_mod", []), dtype=float)
        dy = np.asarray(methylation.get("savgol_frac_mod_dy", []), dtype=float)
        if data.size == 0 or dy.size == 0:
            return _empty_dip_record()

        # detect spans on below median values that are candidate dips
        median = np.median(data)
        mask = data > median if self.enrichment else data < median
        prev = np.r_[False, mask[:-1]]
        next_ = np.r_[mask[1:], False]
        starts = np.flatnonzero(mask & ~prev)
        ends = np.flatnonzero(mask & ~next_)
        lefts = np.maximum(starts - 1, 0)
        rights = np.minimum(ends + 1, data.size - 1)
        dips_arr = np.asarray(list(dips), dtype=int)
        if dips_arr.size == 0:
            return _empty_dip_record()

        # assign each called dip to the span that contains it
        idx = np.searchsorted(starts, dips_arr, side="right") - 1
        valid = idx >= 0
        valid &= mask[dips_arr]
        valid &= dips_arr <= ends[idx.clip(min=0)]
        idx = idx[valid]
        dip_bounds = [(int(lefts[i]), int(rights[i])) for i in idx]

        # make edges more accurate by using the spots where the slopes are most drastic
        dip_bounds_adj: List[Tuple[int, int]] = []
        for dip, (left, right) in zip(dips_arr[valid], dip_bounds):
            if left > dip:
                left = dip
            if right < dip:
                right = dip
            left_idx = int(np.argmin(dy[left : dip + 1]) + left)
            right_idx = int(np.argmax(dy[dip : right + 1]) + dip)
            dip_bounds_adj.append((left_idx, right_idx))

        # convert the dip indices back into genomic coordinates
        dips = _empty_dip_record()
        for s, e in dip_bounds_adj:
            dips['starts'].append( starts_m[s] )
            dips['ends'].append( starts_m[e] )

        return dips

    def _process_single_chromosome(
        self,
        chrom: str,
        region_keys: List[str],
        region_records: Dict[str, MethylationRecord],
    ) -> Tuple[Dict[str, Dict[str, List]], Dict[str, MethylationRecord]]:
        combined, slices = self._build_chromosome_arrays(chrom, region_keys, region_records)

        smoothed, derivative = self._smooth_chromosome(combined["fraction_modified"])

        dip_results: Dict[str, Dict[str, List]] = {}
        methylation_results: Dict[str, MethylationRecord] = {}

        for region_key in region_keys:
            start_idx, end_idx = slices.get(region_key, (0, 0))
            slc = slice(start_idx, end_idx)
            region_data = self._slice_region(combined, smoothed, derivative, slc)
            methylation_results[region_key] = region_data

            if (end_idx - start_idx) < self.window_size:
                dip_results[region_key] = _empty_dip_record()
                continue

            centers, lefts, rights = self.detect_dips(region_data)
            dip_results[region_key] = self.extend_dips(region_data, centers)

            if self.debug:
                positions = region_data.get("position", [])
                debug_payload = {
                    "peak_centers": [positions[idx] for idx in centers if idx < len(positions)],
                    "peak_lefts": [positions[idx] for idx in lefts if idx < len(positions)],
                    "peak_rights": [positions[idx] for idx in rights if idx < len(positions)],
                }
                dip_results.update(debug_payload)

        return {chrom: dip_results}, methylation_results

    def dip_detect_all_chromosome(
        self,
        methylation_per_region: Dict[str, MethylationRecord],
        regions_per_chrom: Dict[str, Dict[str, List[int]]],
    ) -> Tuple[Dict[str, Dict[str, List]], Dict[str, MethylationRecord]]:

        chrom_inputs = []
        for chrom, coords in regions_per_chrom.items():
            region_keys = [
                f"{chrom}:{start}-{end}"
                for start, end in zip(coords.get("starts", []), coords.get("ends", []))
            ]
            if not region_keys:
                continue
            chrom_records = {
                key: self._copy_methylation_record(methylation_per_region.get(key, {}))
                for key in region_keys
            }
            chrom_inputs.append((chrom, region_keys, chrom_records))

        dip_results: Dict[str, Dict[str, List]] = {}
        methylation_results: Dict[str, MethylationRecord] = {}

        if not chrom_inputs:
            return dip_results, methylation_results

        if self.threads <= 1 or len(chrom_inputs) == 1:
            for chrom, region_keys, chrom_records in chrom_inputs:
                dips, methylation = self._process_single_chromosome(
                    chrom, region_keys, chrom_records
                )
                dip_results.update(dips)
                methylation_results.update(methylation)
            return dip_results, methylation_results

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(
                    DipDetector._process_single_chromosome,
                    self,
                    chrom,
                    region_keys,
                    chrom_records,
                ): chrom
                for chrom, region_keys, chrom_records in chrom_inputs
            }
            for future in concurrent.futures.as_completed(futures):
                dips, methylation = future.result()
                dip_results.update(dips)
                methylation_results.update(methylation)

        return dip_results, methylation_results