from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


DipRecord = Dict[str, List[int]]
DipResults = Dict[str, DipRecord]


def _size_filter(
    record: DipRecord,
    min_size: int,
) -> DipRecord:
    if min_size < 0:
        return record

    filt_dips = {}
    for region, data in record.items():
        filtered_starts, filtered_ends = [], []
        for start, end in zip(data.get("starts", []), data.get("ends", [])):
            if end - start >= min_size:
                filtered_starts.append(start)
                filtered_ends.append(end)
        filt_dips[region] = {"starts": filtered_starts, "ends": filtered_ends}
    
    return filt_dips

def _cluster_filter(
    record: DipRecord,
    cluster_distance: int,
) -> DipResults:
    return


def filterDips(
    dip_dict: DipResults,
    cluster_distance: int,
    min_size: int,
) -> DipResults:

    # filter dips based on size
    size_filt_dips = _size_filter(record=dip_dict, min_size=min_size)

    # add coverage filter here if needed in future
    
    # would add clustering filter here if needed in future
    # cluster_filt_dips = _cluster_filter(record=dip_size_filt_dipsdict, cluster_distance=cluster_distance)

    final_dips = size_filt_dips # not needed currently temporary

    return final_dips


__all__ = ["filterDips"]