from __future__ import annotations

from typing import Dict, Mapping, Sequence, List, Tuple
import numpy as np

from centrodip.bedtable import BedTable


def filterDips(
    dips: BedTable,
    min_size: int,
    min_score: float,
    cluster_distance: int,
) -> BedTable:

    # size filter - remove dip regions smaller than min_size
    # convert back to BedTable
    size_filtered = BedTable([r for r in dips._records if r.length >= min_size], inferred_kind="bed", inferred_ncols=6)

    # score filter - remove dip regions with score less than min_score
    # convert back to BedTable
    score_filtered = BedTable([r for r in size_filtered._records if (r.score is not None and int(r.score) >= min_score)], inferred_kind="bed", inferred_ncols=6)

    # cluster filter - keep only dips in the largest cluster within cluster_distance
    cluster_filtered = clusterFilter(
        score_filtered,
        cluster_distance
    )

    return cluster_filtered

def clusterFilter(
    dips: BedTable,
    cluster_distance: int,
    *,
    require_min_members: int = 1,     # require at least N dips in chosen cluster
) -> BedTable:
    """
    For each dip i, define a candidate cluster as all dips within cluster_distance
    (on either side) of dip i. Rank candidate clusters by mean(score), then choose
    the single best cluster and return only those dips.

    Assumes dips are on a single chromosome and sorted by start.
    """
    if cluster_distance is None or cluster_distance < 0:
        return dips

    recs = list(dips._records)
    n = len(recs)
    if n <= 1:
        return dips

    # Ensure sorted by start (important for two-pointer expansion)
    recs = sorted(recs, key=lambda r: (r.start, r.end))

    starts = np.array([r.start for r in recs], dtype=int)
    ends   = np.array([r.end   for r in recs], dtype=int)
    scores = np.array([int(r.score) if r.score is not None else 0 for r in recs], dtype=int)

    x = (starts + ends) // 2
    # within window if |xj - xi| <= cluster_distance
    def within(j: int, i: int) -> bool:
        return abs(int(x[j]) - int(x[i])) <= cluster_distance

    # Helper: expand window around i using two pointers (since sorted)
    def neighborhood_indices(i: int) -> Tuple[int, int]:
        li = i
        while li - 1 >= 0 and within(li - 1, i):
            li -= 1
        ri = i
        while ri + 1 < n and within(ri + 1, i):
            ri += 1
        return li, ri

    best = None  # tuple(rank_tuple, li, ri)
    for i in range(n):
        li, ri = neighborhood_indices(i)
        if (ri - li + 1) < require_min_members:
            continue

        window_scores = scores[li : ri + 1]
        mean_score = float(np.mean(window_scores)) if window_scores.size else 0.0

        # optional tie-breakers: more members, larger span, earlier start
        span_bp = int(ends[ri] - starts[li])
        count = int(ri - li + 1)
        first_start = int(starts[li])

        rank = (mean_score, count, span_bp, -first_start)
        if best is None or rank > best[0]:
            best = (rank, li, ri)

    if best is None:
        return BedTable([], inferred_kind="bed", inferred_ncols=6)

    _, li, ri = best
    out = recs[li : ri + 1]
    return BedTable(out, inferred_kind="bed", inferred_ncols=6)