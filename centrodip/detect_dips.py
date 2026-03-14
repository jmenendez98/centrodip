from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal, stats

from centrodip.bedtable import BedTable, IntervalRecord


def detectDips(
    chrom: str,
    bedgraph: BedTable,
    *,
    prominence: float,
    height: float,
    enrichment: bool,
    broadness: float,
    score_sensitivity: float,
    label: str = "CDR",
    color: str = "50,50,255",
    debug: bool = False,
) -> BedTable:
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

    chroms = {r.chrom for r in rows}
    chrom_for_output = rows[0].chrom
    positions = np.asarray([r.start for r in rows], dtype=int) # x positions used for reporting dips
    smoothed = np.asarray([_safe_extras(r, 0) for r in rows], dtype=float) # smoothed and slope come from extras
    smoothed_dy = np.asarray([_safe_extras(r, 1) for r in rows], dtype=float)

    # -------------------------
    # Find dip centers and edges
    # -------------------------
    dip_center_idxs = find_dip_centers(smoothed, prominence, height, enrichment) # call dip centers using scipy.find_peaks

    if debug:
        kind = "enrichment peaks" if enrichment else "dip centers"
        print(f"[DEBUG] {chrom}: found {len(dip_center_idxs)} potential {kind}. [prominence={prominence}; height={height}].")

    # find initial edges using simple thresholding
    # at smoothed methylation median
    simple_regions, simple_idxs = find_edges(
        chrom = chrom_for_output,
        smoothed = smoothed,
        positions = positions,
        background_median = np.median(smoothed),
        score_sensitivity = score_sensitivity,
        dip_center_idxs = dip_center_idxs,
        broadness = 1,
        enrichment=enrichment,
        label = label,
        color = "211,211,211",
        debug = False
    )

    if debug:
        print(f"[DEBUG] {chrom}: masking out {np.sum([e - s for s, e in simple_idxs])} CpGs from background.")

    # estimate out of CDR methylation 
    background_median = estimate_bkgrd_median(
        smoothed=smoothed,
        masked_regions=simple_idxs,
    )

    if debug:
        print(f"[DEBUG] {chrom}: estimated background median = {background_median:.2f}.")
        inflection = background_median * (1 + score_sensitivity) if enrichment else background_median * (1 - score_sensitivity)
        print(f"[DEBUG] {chrom}: estimated score inflection (~500) = {inflection:.2f}.")

    # get half-point edges using dip_centers, smoothed, and background median
    dip_regions, halfpoint_idxs = find_edges(
        chrom = chrom_for_output,
        smoothed = smoothed,
        positions = positions,
        background_median = background_median,
        score_sensitivity = score_sensitivity,
        dip_center_idxs = dip_center_idxs,
        broadness = broadness,
        enrichment=enrichment,
        label = label,
        color = color,
        debug = debug
    )

    if debug:
        kind = "enrichment" if enrichment else "dip"
        print(f"[DEBUG] {chrom}: detected {len(dip_regions)} potential {kind} regions. [broadness={broadness}; score_sensitivity={score_sensitivity}].")

    return dip_regions, {"median": background_median, "values": smoothed[np.isfinite(smoothed)]}


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
            wlen=len(smoothed_methylation)
        )
    else:
        centers, _ = signal.find_peaks(
            -smoothed_methylation,
            prominence=data_prominence_threshold,
            height=-np.percentile(smoothed_methylation, q=(height)*100),
            wlen=len(smoothed_methylation)
        )

    return centers.astype(int)

def estimate_bkgrd_median(
    smoothed: np.ndarray,
    masked_regions: list[tuple[int, int]],
):
    """
    Estimate background methylation median after masking a set of potential dip/CDR regions.
    """
    n = len(smoothed)
    mask = np.ones(n, dtype=bool)

    for l, r in masked_regions:
        l = max(0, min(int(l), n - 1))
        r = max(0, min(int(r), n - 1))
        if r <= l:
            continue
        mask[l : r + 1] = False
    good = mask & np.isfinite(smoothed)

    if not np.any(good):
        good = np.isfinite(smoothed)

    bg_vals = smoothed[good]
    return float(np.median(bg_vals)) if bg_vals.size else np.nan

def find_edges(
    chrom: str,
    smoothed: np.ndarray,
    positions: np.ndarray,
    background_median: float,
    score_sensitivity: float,
    dip_center_idxs: np.ndarray,
    broadness: float,
    enrichment: bool,
    label: str,
    color: str, 
    debug: bool = False,
) -> [BedTable, List[Tuple[int, int]]]:
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

    k_consecutive = 5

    if enrichment:
        # For enrichment: scan outward until k consecutive points DROP back to `level`
        def _scan_left(c: int, level: float) -> int:
            i = c
            while i >= 0:
                j0 = max(0, i - (k_consecutive - 1))
                window = smoothed[j0 : i + 1]
                if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window <= level):
                    return j0
                i -= 1
            return 0

        def _scan_right(c: int, level: float) -> int:
            i = c
            while i < n:
                j1 = min(n, i + k_consecutive)
                window = smoothed[i:j1]
                if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window <= level):
                    return j1 - 1
                i += 1
            return n - 1
    else:
        # For dips: scan outward until k consecutive points RISE back to `level`
        def _scan_left(c: int, level: float) -> int:
            i = c
            while i >= 0:
                j0 = max(0, i - (k_consecutive - 1))
                window = smoothed[j0 : i + 1]
                if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window >= level):
                    return j0
                i -= 1
            return 0

        def _scan_right(c: int, level: float) -> int:
            i = c
            while i < n:
                j1 = min(n, i + k_consecutive)
                window = smoothed[i:j1]
                if window.size == k_consecutive and np.all(np.isfinite(window)) and np.all(window >= level):
                    return j1 - 1
                i += 1
            return n - 1

    # --- 1) call edges as indices ---
    halfpoint_idxs: List[Tuple[int, int]] = []
    for c in centers:
        if c < 0 or c >= n:
            continue
        y0 = smoothed[c]
        if not np.isfinite(y0):
            continue

        if enrichment:
            # depth is how far ABOVE background the peak sits
            depth = float(y0 - background_median)
            # level is the point between peak and background (scaled by broadness)
            level = float(y0 - broadness * depth)
        else:
            # depth is how far BELOW background the dip sits
            depth = float(background_median - y0)
            level = float(y0 + broadness * depth)

        li = _scan_left(c, level)
        ri = _scan_right(c, level)
        if ri <= li:
            continue
        halfpoint_idxs.append((li, ri))

    halfpoint_idxs = list(dict.fromkeys(tuple(x) for x in halfpoint_idxs))

    # --- merge overlapping / touching index intervals ---
    if halfpoint_idxs:
        halfpoint_idxs.sort(key=lambda x: (x[0], x[1]))
        merged: List[Tuple[int, int]] = []
        cur_l, cur_r = halfpoint_idxs[0]
        for l, r in halfpoint_idxs[1:]:
            if l <= cur_r:
                cur_r = max(cur_r, r)
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))
        halfpoint_idxs = merged

    # --- 2) compute raw scores per region ---
    # null_surplus / null_deficit: the background level shifted by score_sensitivity
    # For dips:       score rewards values *below* background
    # For enrichment: score rewards values *above* background
    if enrichment:
        null_surplus = background_median * (1 + score_sensitivity)
    else:
        null_deficit = background_median * (1 - score_sensitivity)

    scores: List[float] = []
    for (l_i, r_i) in halfpoint_idxs:
        l_i = max(0, min(int(l_i), n - 1))
        r_i = max(0, min(int(r_i), n - 1))

        if r_i <= l_i:
            scores.append(0.0)
            continue

        region_values = smoothed[l_i : r_i + 1]
        region_values = region_values[np.isfinite(region_values)]

        if region_values.size < 3:
            scores.append(0.0)
            continue

        if enrichment:
            # mirror of the dip formula, but for upward deviation
            a = np.mean(region_values - null_surplus) / background_median
            b = (np.max(region_values) - null_surplus) / background_median
        else:
            a = np.mean(null_deficit - region_values) / background_median
            b = (null_deficit - np.min(region_values)) / background_median

        deficit = np.sign(a) * np.sqrt(np.abs(a) * np.abs(b))
        z = deficit * np.log1p(region_values.size)
        score = round(np.clip(1000.0 / (1 + np.exp(-3 * z)), 0.0, 1000.0))

        if debug:
            print(
                f"[DEBUG] {chrom}:{positions[l_i]}-{positions[r_i]}: "
                f"region_mean={np.mean(region_values):.2f}; deficit={deficit:.2f}; z={z:.2f}; score={score}"
            )

        scores.append(score)

    bed_scores = np.asarray(scores, dtype=float)

    # --- 3) build BedTable output ---
    out: List[IntervalRecord] = []
    for dip_id, ((l_i, r_i), bed_score) in enumerate(zip(halfpoint_idxs, bed_scores), start=1):
        l_i = max(0, min(int(l_i), n - 1))
        r_i = max(0, min(int(r_i), n - 1))
        if r_i <= l_i:
            continue

        start = int(positions[l_i])
        end = int(positions[r_i]) + 1

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