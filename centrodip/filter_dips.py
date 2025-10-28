from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, List, Tuple

import numpy as np


DipRecord = Dict[str, list[int]]
DipResults = Dict[str, DipRecord]
IdxPairs = List[Tuple[int, int]]


def filterDips(
    dips: DipResults,
    dip_idxs,
    fraction_modified,
    min_size: int,
    min_zscore: float,
    cluster_distance,
) -> DipResults:

    # size filter
    filtered, filtered_idxs = filter_by_size(
        dips, dip_idxs, 
        min_size
    ) 

    # zscore filter
    filtered, filtered_idxs = filter_by_zscore(
        filtered, filtered_idxs, 
        fraction_modified, 
        min_zscore
    ) 

    # cluster filter
    filtered, filtered_idxs = filter_by_cluster(
        filtered, filtered_idxs,
        cluster_distance
    )

    return filtered


def _apply_masking(
    record: DipRecord,
    dip_idxs: Optional[IdxPairs],
    mask: Sequence[bool],
) -> Tuple[DipRecord, Optional[IdxPairs]]:
    """Filter DipRecord (starts/ends and any fields with same length) and dip_idxs by mask."""
    filtered: DipRecord = {}
    mlist = list(mask)
    n_mask = len(mlist)
    for key, values in record.items():
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            # filter lists that match the number of dips (e.g., starts/ends and any parallel lists)
            if key in {"starts", "ends"} or len(values) == n_mask:
                filtered[key] = [v for v, keep in zip(values, mlist) if keep]
            else:
                filtered[key] = list(values)
        else:
            filtered[key] = values

    filtered_idxs = None
    if dip_idxs is not None:
        filtered_idxs = [p for p, keep in zip(dip_idxs, mlist) if keep]
    return filtered, filtered_idxs


def filter_by_size(
    dips: DipRecord, 
    dip_idxs: Optional[IdxPairs],
    min_size: int
) -> Tuple[DipRecord, Optional[IdxPairs]]:
    """Keep only dips whose span >= min_size (in bp)."""
    if min_size is None or min_size <= 0:
        return dips, dip_idxs

    starts = list(dips.get("starts", []))
    ends = list(dips.get("ends", []))
    if not starts or not ends or len(starts) != len(ends):
        return dips, dip_idxs

    mask: List[bool] = []
    for s, e in zip(starts, ends):
        mask.append(abs(int(e) - int(s)) >= int(min_size))

    if all(mask):
        return dips, dip_idxs
    return _apply_masking(dips, dip_idxs, mask)


def filter_by_zscore(
    dips: DipRecord,
    dip_idxs: Optional[IdxPairs],
    fraction_modified: Sequence[float],
    min_value_zscore: float
) -> Tuple[DipRecord, Optional[IdxPairs]]:
    """
    Keep dips whose inside-mean is at least `min_value_zscore` SDs lower than the outside mean:
        z = (outside_mean - inside_mean) / outside_std  >=  threshold
    Uses dip_idxs (index pairs into fraction_modified) for fast slicing.
    """
    if min_value_zscore is None or min_value_zscore <= 0:
        return dips, dip_idxs

    if dip_idxs is None:
        # For zscore filtering we require index pairs for efficiency and correctness.
        # You can relax this if you want to fall back to position-based scans.
        return dips, dip_idxs  # or raise ValueError("dip_idxs required for zscore filtering")

    starts = dips.get("starts", [])
    ends = dips.get("ends", [])
    if not starts or not ends or len(starts) != len(ends):
        return dips, dip_idxs

    values = np.asarray(fraction_modified, dtype=float)
    n = values.size
    if n == 0:
        return dips, dip_idxs

    # Precompute global stats once for fallback cases
    global_mean = float(np.mean(values))
    global_std = float(np.std(values))

    mask: List[bool] = []
    for (li, ri), s, e in zip(dip_idxs, starts, ends):
        # sanitize indices
        l = max(0, min(int(li), n - 1))
        r = max(0, min(int(ri), n - 1))
        if l > r:
            l, r = r, l

        inside = values[l:r + 1]
        # outside = all points before l and after r
        if l == 0 and r == n - 1:
            outside = np.array([], dtype=float)
        elif l == 0:
            outside = values[r + 1:]
        elif r == n - 1:
            outside = values[:l]
        else:
            # concatenate without copying huge arrays unnecessarily
            left = values[:l]
            right = values[r + 1:]
            # Avoid making a big copy when both sides are large:
            outside = np.concatenate((left, right)) if left.size and right.size else (left if left.size else right)

        if inside.size == 0:
            # if we can't measure inside, conservatively keep
            mask.append(True)
            continue

        if outside.size == 0:
            # if there's no outside, fall back to global stats (if meaningful)
            if global_std <= 1e-9:
                mask.append(False)  # cannot establish contrast; drop
                continue
            inside_mean = float(np.mean(inside))
            z = (global_mean - inside_mean) / global_std
            mask.append(z >= min_value_zscore)
            continue

        inside_mean = float(np.mean(inside))
        outside_mean = float(np.mean(outside))
        outside_std = float(np.std(outside))

        if outside_std <= 1e-9:
            # if outside is nearly constant, require strictly lower inside mean
            mask.append(inside_mean + 1e-9 < outside_mean)
            continue

        # remove dips with z-score smaller (less sig) than min_value_zscore
        z = (outside_mean - inside_mean) / outside_std
        mask.append(z >= min_value_zscore)

    if all(mask):
        return dips, dip_idxs

    return _apply_masking(dips, dip_idxs, mask)

def filter_by_cluster(
    record: DipRecord,
    dip_idxs: Optional[IdxPairs],
    cluster_distance: int
) -> Tuple[DipRecord, Optional[IdxPairs]]:

    # skip if negative
    if cluster_distance is None or cluster_distance < 0:
        return record, dip_idxs

    starts = list(record.get("starts", []))
    ends = list(record.get("ends", []))
    if not starts or not ends or len(starts) <= 1:
        return record, dip_idxs

    # Build clusters in input order (assumes starts/ends are co-sorted)
    clusters: List[List[tuple[int, int, int]]] = []
    current: List[tuple[int, int, int]] = []
    current_end: Optional[int] = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        s = int(start); e = int(end)
        if current and current_end is not None and s - current_end > int(cluster_distance):
            clusters.append(current)
            current = []
        current.append((s, e, index))
        current_end = max(current_end, e) if current_end is not None else e
    if current:
        clusters.append(current)

    if len(clusters) <= 1:
        return record, dip_idxs

    def cluster_score(cluster: List[tuple[int, int, int]]) -> tuple[int, int, int]:
        # coverage of merged intervals, number of items, tie-breaker by earliest start
        ordered = sorted((s, e) for s, e, _ in cluster)
        coverage = 0
        span_s, span_e = ordered[0]
        for s, e in ordered[1:]:
            if s <= span_e:
                span_e = max(span_e, e)
            else:
                coverage += max(0, span_e - span_s)
                span_s, span_e = s, e
        coverage += max(0, span_e - span_s)
        count = len(cluster)
        first_start = min(s for s, _ in ordered)
        return coverage, count, -first_start

    best_cluster = max(clusters, key=cluster_score)
    keep_indices = {idx for *_unused, idx in best_cluster}
    mask = [(i in keep_indices) for i in range(len(starts))]

    if all(mask):
        return record, dip_idxs
    return _apply_masking(record, dip_idxs, mask)



__all__ = ["filterDips"]