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
    filtered_starts = []
    filtered_ends = []
    for start, end in zip(record.get("starts", []), record.get("ends", [])):
        if end - start >= min_size:
            filtered_starts.append(start)
            filtered_ends.append(end)
    return {"starts": filtered_starts, "ends": filtered_ends}

def filterDips(
    dip_dict: DipResults,
    cluster_distance: int = -1,
    min_size: int = -1,
) -> DipResults:

    # filter dips based on size
    size_filt_dips = _size_filter(record=dip_dict, min_size=min_size)

    # add coverage filter here if needed in future

    # would add clustering filter here if needed in future

    final_dips = size_filt_dips # not needed currently temporary
    return final_dips


__all__ = ["filterDips"]