from __future__ import annotations

import concurrent.futures
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


MethylationRecord = Dict[str, List[float]]
RegionRecord = Dict[str, List[int]]
DipResults = Dict[str, Dict[str, List[int]]]


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

    positions = np.array(data["position"], dtype=int)
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

    return [(positions[dli], positions[dri]) for dli, dri in dip_edge_idxs]


def find_dip_centers(
    smoothed_methylation: np.ndarray,
    prominence, height, enrichment    
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
            height=np.percentile(smoothed_methylation, 100-height),
            wlen=len(smoothed_methylation)
        )
    else:
        centers, _ = signal.find_peaks(
            -smoothed_methylation, 
            prominence=data_prominence_threshold,
            height=-np.percentile(smoothed_methylation, height),
            wlen=len(smoothed_methylation)
        )

    return centers.astype(int)


def find_edge_idxs(
    data: np.ndarray, dy: np.ndarray,
    bounding_threshold: float,
    centers: np.ndarray,
) -> np.ndarray:
    data = np.array(data, dtype=float)
    dy = np.array(dy, dtype=float)

    n = data.size
    if n == 0:
        return np.array([], dtype=int)

    edges = []
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
            # find index between c and li with lowest dy value
            l_search_space = dy[min(c, li):max(c, li)+1]
            idx_rel = np.argmin(l_search_space)
            l_idx = min(c, li) + idx_rel

            # find index between c and li with lowest dy value
            r_search_space = dy[min(c, ri):max(c, ri)+1]
            idx_rel = np.argmin(r_search_space)
            r_idx = min(c, ri) + idx_rel

            edges.append((l_idx, r_idx))

    return np.array(edges)