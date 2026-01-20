from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

from bedtable import BedTable, IntervalRecord


def detectDips(
    bedgraph: BedTable,
    *,
    prominence: float,
    height: float,
    enrichment: bool,
    broadness: float,
    debug: bool = False,
    x_mode: str = "start",  # "start" or "midpoint",
    label: str = "CDR",
    color: str = "50,50,255",
) -> BedTable:
    """
    Detect dips using a LOWESS-output BedTable (BEDGRAPH+2 style):
      chrom start end smoothedY dY

    Returns:
      BedTable where each record is a dip interval:
        chrom  dip_start  dip_end  (extras contain dip_id, left_idx, right_idx)

    Notes:
    - This function assumes a single chromosome input; if multiple chroms are present,
      it will still work but dip intervals may be nonsense across chrom boundaries.
      Prefer calling per chrom (see detectDips_as_bedtable_all_chroms).
    """

    def _safe_extras(r: IntervalRecord, idx: int) -> float:
        """
        Return extras[idx] as float; NaN if missing/bad.
        """
        try:
            v = r.extras[idx]
        except Exception:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    # -------------------------
    # Extract arrays from BedTable
    # -------------------------
    rows = list(bedgraph)
    if not rows:
        return {"starts": [], "ends": []}, []

    # Ensure we're operating per-chromosome (this function assumes one chrom at a time)
    chroms = {r.chrom for r in rows}
    if len(chroms) != 1 and debug:
        print(f"[WARN] detectDips received multiple chroms: {sorted(chroms)}")
        print("       Consider calling detectDips per chromosome via bedgraph.groupby_chrom().")
    chrom_for_output = rows[0].chrom

    # x positions used for reporting dips (in your old code this was cpg_pos)
    if x_mode == "start":
        positions = np.asarray([r.start for r in rows], dtype=int)
    elif x_mode == "midpoint":
        positions = np.asarray([(r.start + r.end) // 2 for r in rows], dtype=int)
    else:
        raise ValueError("x_mode must be 'start' or 'midpoint'")

    # smoothed and slope come from extras
    smoothed = np.asarray([_safe_extras(r, 0) for r in rows], dtype=float)
    smoothed_dy = np.asarray([_safe_extras(r, 1) for r in rows], dtype=float)

    # -------------------------
    # Find dip centers and edges
    # -------------------------

    # call dip centers using scipy.find_peaks
    dip_center_idxs = find_dip_centers(smoothed, prominence, height, enrichment)

    # find initial edges using simple thresholding
    simple_threshold = np.percentile(smoothed, q=50)
    simple_regions, simple_idxs = find_simple_edges(
        chrom_for_output,
        smoothed, 
        positions,
        simple_threshold, 
        dip_center_idxs
    )

    # estimate out of CDR methylation 
    background_stats = estimate_background_from_masked(
        smoothed=smoothed,
        positions=positions,
        masked_regions=simple_idxs,
    )
    print(
        f"[DEBUG] background methylation "
        f"median={background_stats['median']:.3f} "
        f"IQR=({background_stats['p25']:.3f}, {background_stats['p75']:.3f}) "
        f"n={len(background_stats['values'])}"
    )

    # get half-point edges using dip_centers, smoothed, and background median
    dip_regions, halfpoint_idxs = find_edges(
        chrom_for_output,
        smoothed,
        positions,
        background_stats["median"],
        dip_center_idxs
    )

    return dip_regions


def find_dip_centers(
    smoothed_methylation: np.ndarray,
    prominence: float,
    height: float,
    enrichment: bool,
) -> np.ndarray:
    """Return dip center indices in the smoothed methylation data."""
    smoothed_methylation = np.array(smoothed_methylation, dtype=float)
    if smoothed_methylation.size == 0:
        return np.array([], dtype=int)

    data_range = float(np.max(smoothed_methylation) - np.min(smoothed_methylation))
    data_prominence_threshold = prominence * data_range

    if enrichment:
        centers, _ = signal.find_peaks(
            smoothed_methylation,
            prominence=data_prominence_threshold,
            height=np.percentile(smoothed_methylation, q=(1-height)*100),
        )
    else:
        centers, _ = signal.find_peaks(
            -smoothed_methylation,
            prominence=data_prominence_threshold,
            height=-np.percentile(smoothed_methylation, q=(height)*100),
        )

    return centers.astype(int)

def find_simple_edges(
    chrom_for_output: str,
    data: np.ndarray,
    positions: np.ndarray,
    bounding_threshold: float,
    centers: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Simple edge finder:

    For each dip center c, define edges as the first indices on each side
    where data[idx] >= bounding_threshold.

    - Left edge: scan c-1, c-2, ... until threshold hit (or 0)
    - Right edge: scan c+1, c+2, ... until threshold hit (or n-1)

    Returns list of (left_idx, right_idx) pairs.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        return []

    edges: List[Tuple[int, int]] = []

    for c in np.asarray(centers, dtype=int):
        if c < 0 or c >= n:
            continue

        # ---- left scan ----
        left_idx: Optional[int] = None
        li = c - 1
        while li >= 0:
            val = data[li]
            if np.isfinite(val) and val >= bounding_threshold:
                left_idx = li
                break
            li -= 1
        if left_idx is None:
            left_idx = 0

        # ---- right scan ----
        right_idx: Optional[int] = None
        ri = c + 1
        while ri < n:
            val = data[ri]
            if np.isfinite(val) and val >= bounding_threshold:
                right_idx = ri
                break
            ri += 1
        if right_idx is None:
            right_idx = n - 1

        # Ensure proper ordering and non-degenerate interval
        if right_idx <= left_idx:
            continue

        edges.append((left_idx, right_idx))

    # De-duplicate while preserving order
    unique_edges = list(dict.fromkeys(tuple(e) for e in edges))

    out: List[IntervalRecord] = []
    for dip_id, (l_i, r_i) in enumerate(unique_edges, start=1):
        l_i = max(0, min(l_i, len(positions) - 1))
        r_i = max(0, min(r_i, len(positions) - 1))
        if r_i <= l_i:
            continue

        start = int(positions[l_i])
        end = int(positions[r_i])

        out.append(
            IntervalRecord(
                chrom=chrom_for_output,
                start=start,
                end=end,
                name=f"simpleDip_{dip_id}",
                score=0,
                strand='.',
            )
        )

    return BedTable(out, inferred_kind="bed", inferred_ncols=6), unique_edges

def estimate_background_from_masked(
    smoothed: np.ndarray,
    positions: np.ndarray,
    masked_regions: list[tuple[int, int]],
):
    """
    Estimate background methylation statistics after masking dip/CDR regions.

    Parameters
    ----------
    smoothed : array
        Smoothed methylation values (ordered).
    positions : array
        Genomic positions corresponding to smoothed.
    masked_regions : list of (start_pos, end_pos)
        Regions to exclude (CDRs/dips).

    Returns
    -------
    dict with baseline statistics
    """
    n = len(smoothed)
    if n == 0:
        return {"median": np.nan, "mean": np.nan, "p25": np.nan, "p75": np.nan, "values": np.array([]), "mask": np.array([], bool)}

    mask = np.ones(n, dtype=bool)

    for l, r in masked_regions:
        l = max(0, min(int(l), n - 1))
        r = max(0, min(int(r), n - 1))
        if r <= l:
            continue
        mask[l : r + 1] = False

        good = mask & np.isfinite(smoothed)

    bg_vals = smoothed[good]

    out = {
        "median": float(np.median(bg_vals)) if bg_vals.size else np.nan,
        "mean": float(np.mean(bg_vals)) if bg_vals.size else np.nan,
        "p25": float(np.percentile(bg_vals, 25)) if bg_vals.size else np.nan,
        "p75": float(np.percentile(bg_vals, 75)) if bg_vals.size else np.nan,
        "values": bg_vals,
        "mask": mask,
        "n_total": int(n),
        "n_masked": int((~mask).sum()),
        "n_bg": int(bg_vals.size),
    }
    return out

def find_edges(
    chrom: str,
    smoothed: np.ndarray,
    positions: np.ndarray,
    background_median: float,
    dip_center_idxs: np.ndarray,
    *,
    alpha: float = 0.5,
    min_depth: float = 0.0,
    k_consecutive: int = 1,
    end_inclusive: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Half-depth edge caller using a single background level (median outside masked dips/CDRs).

    For each dip center c:
      ymin = smoothed[c]
      level = ymin + alpha * (background_median - ymin)

    Then:
      left edge  = nearest index left of c where smoothed >= level
      right edge = nearest index right of c where smoothed >= level

    Parameters
    ----------
    chrom : str
        Chrom name (unused here except for readability / future expansion).
    smoothed : np.ndarray
        Smoothed methylation values, aligned to `positions`, assumed sorted by position.
    positions : np.ndarray
        Genomic positions (same length as smoothed), assumed sorted.
    background_median : float
        Background methylation estimate (e.g., median of smoothed outside masked regions).
    dip_center_idxs : np.ndarray
        Indices of dip centers into smoothed/positions.
    alpha : float
        Fraction of recovery from ymin to background used to define halfpoint.
        alpha=0.5 is true half-depth.
    min_depth : float
        Minimum required dip depth (background_median - ymin) to accept a dip.
    k_consecutive : int
        Require k consecutive points meeting the threshold to call an edge (noise robustness).
    end_inclusive : bool
        If True, returned regions are (start_pos, end_pos) inclusive.
        If False, regions are (start_pos, end_pos_exclusive) (BED-style; adds +1 to end).

    Returns
    -------
    halfpoint_regions : list of (start_pos, end_pos)
        Genomic coordinate spans for each dip.
    halfpoint_idxs : list of (left_idx, right_idx)
        Index spans for each dip, into the (filtered) arrays provided.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    positions = np.asarray(positions, dtype=int)
    centers = np.asarray(dip_center_idxs, dtype=int)

    n = len(smoothed)
    if n == 0:
        return [], []
    if len(positions) != n:
        raise ValueError("smoothed and positions must have the same length")
    if not np.isfinite(background_median):
        raise ValueError("background_median must be finite")

    if k_consecutive < 1:
        raise ValueError("k_consecutive must be >= 1")

    def _scan_left(c: int, level: float) -> int:
        # find leftmost index of a run of k_consecutive points >= level
        i = c
        while i >= 0:
            j0 = max(0, i - (k_consecutive - 1))
            window = smoothed[j0 : i + 1]
            if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window >= level):
                return j0
            i -= 1
        return 0

    def _scan_right(c: int, level: float) -> int:
        # find rightmost index of a run of k_consecutive points >= level
        i = c
        while i < n:
            j1 = min(n, i + k_consecutive)
            window = smoothed[i:j1]
            if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window >= level):
                return j1 - 1
            i += 1
        return n - 1

    halfpoint_idxs: List[Tuple[int, int]] = []

    for c in centers:
        if c < 0 or c >= n:
            continue
        y0 = smoothed[c]
        if not np.isfinite(y0):
            continue

        depth = float(background_median - y0)
        if depth < min_depth:
            continue

        level = float(y0 + alpha * depth)

        li = _scan_left(c, level)
        ri = _scan_right(c, level)

        if ri <= li:
            continue

        start_pos = int(positions[li])
        end_pos = int(positions[ri])

        if not end_inclusive:
            end_pos = end_pos + 1

        halfpoint_idxs.append((li, ri))

    # de-duplicate (preserve order)
    halfpoint_idxs = list(dict.fromkeys(tuple(x) for x in halfpoint_idxs))

    out: List[IntervalRecord] = []
    for dip_id, (l_i, r_i) in enumerate(halfpoint_idxs, start=1):
        l_i = max(0, min(l_i, len(positions) - 1))
        r_i = max(0, min(r_i, len(positions) - 1))
        if r_i <= l_i:
            continue

        start = int(positions[l_i])
        end = int(positions[r_i])

        out.append(
            IntervalRecord(
                chrom=chrom,
                start=start,
                end=end,
                name=f"simpleDip_{dip_id}",
                score=0,
                strand='.',
            )
        )

    return BedTable(out, inferred_kind="bed", inferred_ncols=6), halfpoint_idxs