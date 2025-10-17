from __future__ import annotations

import concurrent.futures
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float]]
RegionSlices = Dict[str, Tuple[int, int]]


def _empty_methylation_record() -> MethylationRecord:
    return {
        "starts": [],
        "ends": [],
        "fraction_modified": [],
        "valid_coverage": [],
    }


def _empty_dip_record() -> Dict[str, List]:
    return {
        "starts": [],
        "ends": [],
        "names": [],
        "scores": [],
        "strands": [],
        "thick_starts": [],
        "thick_ends": [],
        "item_rgbs": [],
    }


class DipDetector:
    def __init__(
        self,
        window_size,
        threshold,
        prominence,
        enrichment,
        threads
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.prominence = prominence

        self.enrichment = enrichment

        self.threads = threads

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
            "starts": list(record.get("starts", [])),
            "ends": list(record.get("ends", [])),
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
            starts = record["starts"]
            ends = record["ends"]
            if starts and ends:
                order = sorted(range(len(starts)), key=lambda idx: starts[idx])
                starts = [starts[idx] for idx in order]
                ends = [ends[idx] for idx in order]
                fraction_modified = [record["fraction_modified"][idx] for idx in order]
                valid_coverage = [record["valid_coverage"][idx] for idx in order]
            else:
                fraction_modified = []
                valid_coverage = []

            start_idx = len(combined["starts"])
            combined["starts"].extend(starts)
            combined["ends"].extend(ends)
            combined["fraction_modified"].extend(fraction_modified)
            combined["valid_coverage"].extend(valid_coverage)
            end_idx = len(combined["starts"])
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
            "starts": combined["starts"][slc],
            "ends": combined["ends"][slc],
            "fraction_modified": combined["fraction_modified"][slc],
            "valid_coverage": combined["valid_coverage"][slc],
            "savgol_frac_mod": smoothed[slc].tolist(),
            "savgol_frac_mod_dy": derivative[slc].tolist(),
        }
        return region_data


    def detect_dips(self, methylation: MethylationRecord) -> np.ndarray:
        data = np.asarray(methylation.get("savgol_frac_mod", []), dtype=float)
        if data.size == 0:
            return np.array([], dtype=int)

        data_range = float(np.max(data) - np.min(data)) if data.size else 0.0
        prominence_threshold = self.prominence * data_range

        mean = float(np.mean(data)) if data.size else 0.0
        std = float(np.std(data)) if data.size else 0.0
        if self.enrichment:
            height_threshold = mean + (std * self.threshold)
            peaks, _ = signal.find_peaks(
                data,
                height=height_threshold,
                prominence=prominence_threshold,
                wlen=data.size if data.size else None,
            )
        else:
            height_threshold = mean - (std * self.threshold)
            peaks, _ = signal.find_peaks(
                -data,
                height=-height_threshold,
                prominence=prominence_threshold,
                wlen=data.size if data.size else None,
            )
        return peaks

    def extend_dips(
        self,
        methylation: MethylationRecord,
        dips: Iterable[int],
    ) -> List[Tuple[int, int]]:
        data = np.asarray(methylation.get("savgol_frac_mod", []), dtype=float)
        dy = np.asarray(methylation.get("savgol_frac_mod_dy", []), dtype=float)
        if data.size == 0 or dy.size == 0:
            return []

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

        idx = np.searchsorted(starts, dips_arr, side="right") - 1
        valid = idx >= 0
        valid &= mask[dips_arr]
        valid &= dips_arr <= ends[idx.clip(min=0)]
        idx = idx[valid]
        dip_bounds = [(int(lefts[i]), int(rights[i])) for i in idx]

        dip_bounds_adj: List[Tuple[int, int]] = []
        for dip, (left, right) in zip(dips_arr[valid], dip_bounds):
            if left > dip:
                left = dip
            if right < dip:
                right = dip
            left_idx = int(np.argmin(dy[left : dip + 1]) + left)
            right_idx = int(np.argmax(dy[dip : right + 1]) + dip)
            dip_bounds_adj.append((left_idx, right_idx))

        dips = {}
        dips['starts'] = []
        dips['ends'] = []
        for s, e in dip_bounds_adj:
            dips['starts'].append(s)
            dips['ends'].append(e)

        return dips

    def dip_detect_single_chromosome(self, chrom, methylation):
        # if the region has less CpG's than the window size do not process
        if len(methylation['starts']) < self.window_size:
            return ( region, {}, {}, {} )
        methylation_smoothed = self.smooth_methylation(methylation)

        dip_sites = self.detect_dips(methylation_smoothed)
        dips = self.extend_dips(methylation_smoothed, dip_sites)

        return ( chrom, dips, methylation_smoothed )

    def dip_detect_all_chromosome(self, methyl_by_chrom, regions_by_chrom):
        dips_all_chroms, methylation_all_chroms = {}, {}

        chroms = list(methyl_by_chrom.keys())
        if not chroms:
            return dips_all_chroms, methylation_all_chroms

        if self.threads <= 1 or len(chroms) == 1:
            for chrom in chroms:
                chrom, dips, methylation_smoothed = self.dip_detect_single_chromosome(
                    chrom, methyl_by_chrom[chrom]
                )
                dips_all_chroms[chrom] = dips
                methylation_all_chroms[chrom] = methylation_smoothed
            return dips_all_chroms, methylation_all_chroms

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(
                    self.centrodip_single_chromosome,
                    chrom,
                    methyl_by_chrom[chrom],
                ): chrom for chrom in chroms
            }    

            for future in concurrent.futures.as_completed(futures):
                (    
                    chrom,
                    dips,
                    methylation_smoothed,
                ) = future.result()

                dips_all_chroms[chrom] = dips
                methylation_all_chroms[chrom] = methylation_smoothed

        return dips_all_chroms, methylation_all_chroms