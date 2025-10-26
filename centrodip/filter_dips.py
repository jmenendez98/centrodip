from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


DipRecord = Dict[str, List[int]]
DipResults = Dict[str, DipRecord]


def _filter_record(record: DipRecord, mask: Sequence[bool]) -> DipRecord:
    filtered: DipRecord = {}
    for key, values in record.items():
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if key in {"starts", "ends"} or len(values) == len(mask):
                filtered[key] = [value for value, keep in zip(values, mask) if keep]
            else:
                filtered[key] = list(values)
        else:
            filtered[key] = values
    return filtered


def filter_by_size(
    record: DipRecord, 
    min_size: int
) -> DipRecord:
    """Simple filter based on an entry size"""
    if min_size <= 0:
        return record

    starts = list(record.get("starts", []))
    ends = list(record.get("ends", []))
    if not starts or not ends:
        return record

    mask: List[bool] = []
    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        mask.append(abs(end_i - start_i) >= min_size)

    if all(mask):
        return record
    return _filter_record(record, mask)

def filter_by_value(
    record: DipRecord, 
    min_value_zscore: float
) -> DipRecord:
    """Filter entry based on z-scoring threshold of CpG values within it"""

    # If the threshold is non-positive, do nothing (no filtering).
    if min_value_zscore <= 0:
        return record

    starts = list(record.get("starts", []))
    ends = list(record.get("ends", []))
    if not starts or not ends:
        return record

    # fetch CpG positions within entry
    positions = record.get("positions") or record.get("position")
    if positions is None:
        positions = record.get("cpg_sites")

    # fetch CpG frac mod values within entry
    values = (record.get("lowess_fraction_modified"))
    if positions is None or values is None:
        return record

    positions_list = list(positions)
    values_list = list(values)
    if len(positions_list) != len(values_list) or not positions_list:
        return record

    points = [(int(pos), float(val)) for pos, val in zip(positions_list, values_list)]

    mask: List[bool] = []
    for start, end in zip(starts, ends):
        left, right = sorted((int(start), int(end)))
        inside_values = [value for pos, value in points if left <= pos <= right]
        outside_values = [value for pos, value in points if pos < left or pos > right]
        if not inside_values or not outside_values:
            mask.append(True)
            continue

        # Compute means for inside/outside and stdev for outside.
        inside_mean = sum(inside_values) / float(len(inside_values))
        outside_mean = sum(outside_values) / float(len(outside_values))
        outside_std = np.std(outside_values)

        if outside_std <= 1e-9:
            mask.append(inside_mean + 1e-9 < outside_mean)
            continue

        # mask out entry if inside mean isn't `min_value_zscore` away from outside mean
        z_score = (outside_mean - inside_mean) / outside_std
        mask.append(z_score >= min_value_zscore)

    if all(mask):
        return record
    return _filter_record(record, mask)

def filter_by_cluster(record: DipRecord, cluster_distance: int) -> DipRecord:
    
    # if cluster distance is negative skip filtering
    if cluster_distance < 0:
        return record

    starts = list(record.get("starts", []))
    ends = list(record.get("ends", []))
    if not starts or not ends or len(starts) <= 1: # if there is 0 or 1 entry no need to cluster
        return record 

    clusters: List[List[tuple[int, int, int]]] = []
    current: List[tuple[int, int, int]] = []
    current_end: int | None = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        if current and current_end is not None and start - current_end > cluster_distance:
            clusters.append(current)
            current = []
        current.append((start, end, index))
        current_end = max(current_end, end) if current_end is not None else end

    if current: # add the last cluster to the cluster list
        clusters.append(current)

    if len(clusters) <= 1: # if everything is in one cluster just return it...
        return record

    def cluster_score(cluster: List[tuple[int, int, int]]) -> tuple[int, int, int]:
        indices = [idx for _, _, idx in cluster]
        coverage = 0

        # calculate the spans of each cluster
        ordered = sorted((start, end) for start, end, _ in cluster)
        span_start, span_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= span_end:
                span_end = max(span_end, end)
            else:
                coverage += max(0, span_end - span_start)
                span_start, span_end = start, end
        coverage += max(0, span_end - span_start)

        count = len(cluster) # how many items are in a cluster
        first_start = min(start for start, _ in ordered) # used in order to break a tie in scoring (should be rare) ...
        return coverage, count, -first_start

    best_cluster = max(clusters, key=cluster_score) # chose cluster with highest score
    keep_indices = {idx for *_, idx in best_cluster} 
    mask = [index in keep_indices for index in range(len(starts))] # mask out all of the other ones

    if all(mask):
        return record
    return _filter_record(record, mask)


def filterDips(
    dip_dict: DipResults,
    cluster_distance: int,
    min_size: int,
    *,
    min_value_zscore: float = 1.0,
) -> DipResults:
    filtered: DipResults = {}
    for region, record in (dip_dict or {}).items():
        current = record
        current = filter_by_size(record, min_size)
        current = filter_by_value(current, min_value_zscore)
        current = filter_by_cluster(current, cluster_distance)
        filtered[region] = current
    return filtered


__all__ = ["filterDips"]