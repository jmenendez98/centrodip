from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Mapping[str, Sequence[float]]
MethylationData = Mapping[str, MethylationRecord]
RegionRecord = Mapping[str, Sequence[int]]
DipRecord = Dict[str, List[int]]
DipResults = Dict[str, DipRecord]
DipIndices = List[Tuple[int, int]]


def detectDips(
    data: Dict[str, MethylationRecord],
    prominence: float = 0.25,
    height: float = 10,
    enrichment: bool = False,
    broadness: float = 50,
    threads: int = 1,
    debug: bool = False,
) -> DipResults:
    """Detect dips across all chromosomes/regions in the provided data."""

    positions = np.array(data["cpg_pos"], dtype=int)
    smoothed = np.array(data["lowess_fraction_modified"], dtype=float)
    smoothed_dy = np.array(data["lowess_fraction_modified_dy"], dtype=float)

    # call the dip centers using scipy.find_peaks
    dip_centers = find_dip_centers(
        smoothed, 
        prominence, height, enrichment
    )

    # find the dip edge indices
    bounding_threshold = np.percentile(smoothed, broadness)
    dip_edge_idxs = find_edge_idxs(
        smoothed, smoothed_dy,
        bounding_threshold, dip_centers
    )

    # populate the dips dict data
    dips = {"starts": [], "ends": []}
    for dli, dri in dip_edge_idxs:
        dips["starts"].append(positions[dli])
        dips["ends"].append(positions[dri])

    return dips, dip_edge_idxs


def find_dip_centers(
    smoothed_methylation: np.ndarray,
    prominence: float,
    height: float,
    enrichment: bool,
) -> np.ndarray:
    """Return dip center indices in the smoothed methylation data."""

    # get info on data range for prominence calculation
    smoothed_methylation = np.array(smoothed_methylation, dtype=float)
    if smoothed_methylation.size == 0:
        return np.array([], dtype=int)

    data_range = float(np.max(smoothed_methylation) - np.min(smoothed_methylation))
    data_prominence_threshold = prominence * data_range

    if enrichment:
        centers, _ = signal.find_peaks(
            smoothed_methylation,
            prominence=data_prominence_threshold,
            height=np.percentile(smoothed_methylation, 100 - height),
            wlen=len(smoothed_methylation),
        )
    else:
        centers, _ = signal.find_peaks(
            -smoothed_methylation,
            prominence=data_prominence_threshold,
            height=-np.percentile(smoothed_methylation, height),
            wlen=len(smoothed_methylation),
        )

    return centers.astype(int)


def find_edge_idxs(
    data: np.ndarray,
    dy: np.ndarray,
    bounding_threshold: float,
    centers: np.ndarray,
) -> List[Tuple[int, int]]:
    data = np.array(data, dtype=float)
    dy = np.array(dy, dtype=float)

    n = data.size
    if n == 0:
        return []

    edges: List[Tuple[int, int]] = []
    for c in centers:
        if c < 0:
            continue

        # left search
        li = c - 1
        left_found: int | None = None
        while li >= 0:
            val = data[li]
            if np.isfinite(val) and val >= bounding_threshold:
                left_found = li
                break
            li -= 1
        if li == 0:
            left_found=li
        
        # right search
        ri = c + 1
        right_found: int | None = None
        while ri < n:
            val = data[ri]
            if np.isfinite(val) and val >= bounding_threshold:
                right_found = ri
                break
            ri += 1
        if ri == n:
            right_found=ri

        if left_found is None or right_found is None:
            continue

        # keep left value with the lowest slope
        l_search_space = dy[min(c, left_found) : max(c, left_found) + 1]
        left_idx = min(c, left_found) + int(np.argmin(l_search_space))

        # keep right value with the highest slope
        r_search_space = dy[min(c, right_found) : max(c, right_found) + 1]
        right_idx = min(c, right_found) + int(np.argmax(r_search_space))

        edges.append((left_idx, right_idx))

    unique_edges = list(dict.fromkeys(tuple(edge) for edge in edges))
    return unique_edges