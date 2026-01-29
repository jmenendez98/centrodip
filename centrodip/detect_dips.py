from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

from centrodip.bedtable import BedTable, IntervalRecord


def detectDips(
    bedgraph: BedTable,
    *,
    prominence: float,
    height: float,
    enrichment: bool,
    broadness: float,
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
        return BedTable([], inferred_kind="bed", inferred_ncols=6), {}

    # Ensure we're operating per-chromosome (this function assumes one chrom at a time)
    chroms = {r.chrom for r in rows}
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
        masked_regions=simple_idxs,
    )

    # get half-point edges using dip_centers, smoothed, and background median
    dip_regions, halfpoint_idxs = find_edges(
        chrom = chrom_for_output,
        smoothed = smoothed,
        positions = positions,
        background_median = background_stats["median"],
        dip_center_idxs = dip_center_idxs,
        width = broadness,
        label = label,
        color = color
    )

    return dip_regions, background_stats


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
        return BedTable([], inferred_kind="bed", inferred_ncols=6), []

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
    masked_regions: list[tuple[int, int]],
):
    """
    Estimate background methylation statistics after masking dip/CDR regions.

    Parameters
    ----------
    smoothed : array
        Smoothed methylation values (ordered).
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

    # optional safety: if everything got masked, fall back to any finite points
    if not np.any(good):
        good = np.isfinite(smoothed)

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
    label: str,
    color: str, 
    *,
    width: float = 0.5,
    min_depth: float = 0.0,
    k_consecutive: int = 1,
    end_inclusive: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Half-depth edge caller using a single background level (median outside masked dips/CDRs).

    Returns:
      dips_bed (BedTable): dip intervals with BED score 0-1000
      halfpoint_idxs (list): [(left_idx, right_idx), ...]
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

    # --- 1) call halfpoint edges as indices ---
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
        level = float(y0 + width * depth)
        li = _scan_left(c, level)
        ri = _scan_right(c, level)
        if ri <= li:
            continue
        halfpoint_idxs.append((li, ri))

    halfpoint_idxs = list(dict.fromkeys(tuple(x) for x in halfpoint_idxs)) # de-duplicate 

    # --- merge overlapping / touching index intervals ---
    if halfpoint_idxs:
        # sort by left index, then right
        halfpoint_idxs.sort(key=lambda x: (x[0], x[1]))

        merged: List[Tuple[int, int]] = []
        cur_l, cur_r = halfpoint_idxs[0]

        for l, r in halfpoint_idxs[1:]:
            # overlap or touch?
            if l <= cur_r:
                cur_r = max(cur_r, r)
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r

        merged.append((cur_l, cur_r))
        halfpoint_idxs = merged

    # --- 2) compute raw scores per region ---
    raw_scores: List[float] = []
    for (l_i, r_i) in halfpoint_idxs:
        l_i = max(0, min(int(l_i), n - 1))
        r_i = max(0, min(int(r_i), n - 1))
        if r_i <= l_i:
            raw_scores.append(0.0)
            continue
        win = smoothed[l_i : r_i + 1]
        win = win[np.isfinite(win)]
        if win.size < 3:
            raw_scores.append(0.0)
            continue
        deficit = np.maximum(0.0, float(background_median) - win)
        s = float(np.median(deficit))
        raw_scores.append(s)

    raw_arr = np.asarray(raw_scores, dtype=float)
    finite_raw = raw_arr[np.isfinite(raw_arr) & (raw_arr > 0)]
    # robust scale: map the score_q quantile to 1000
    if finite_raw.size == 0:
        scale = 1.0
    else:
        scale = float(np.quantile(finite_raw, 0.95))
        if scale <= 0 or not np.isfinite(scale):
            scale = float(finite_raw.max()) if finite_raw.size else 1.0
            if scale <= 0 or not np.isfinite(scale):
                scale = 1.0

    def _to_bed_score(s: float) -> int:
        if not np.isfinite(s) or s <= 0:
            return 0
        v = int(round(1000.0 * (s / scale)))
        return max(0, min(1000, v))

    bed_scores = [_to_bed_score(s) for s in raw_arr]

    # --- 3) build BedTable output with scores 0-1000 ---
    out: List[IntervalRecord] = []
    for dip_id, ((l_i, r_i), bed_score) in enumerate(zip(halfpoint_idxs, bed_scores), start=1):
        l_i = max(0, min(int(l_i), n - 1))
        r_i = max(0, min(int(r_i), n - 1))
        if r_i <= l_i:
            continue

        start = int(positions[l_i])
        end = int(positions[r_i])
        if not end_inclusive:
            end = end + 1

        out.append(
            IntervalRecord(
                chrom=chrom,
                start=start,
                end=end,
                name=f"{label}",
                score=int(bed_score),
                strand=".",
                extras=(start, end, color),
            )
        )

    return BedTable(out, inferred_kind="bed", inferred_ncols=6), halfpoint_idxs