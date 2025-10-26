from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import concurrent.futures
import warnings
import numpy as np
import os


RegionDict = Dict[str, Dict[str, List[int]]]
MethylationDict = Dict[str, Dict[str, List[float]]]


# ---------- LOWESS (bp-window) at module scope so it's picklable ----------
def lowess_smooth(y, x, window_bp):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    if n == 0:
        return np.array([], float), np.array([], float)

    half = float(window_bp) / 2.0
    ys = np.empty(n, float)
    dydx = np.empty(n, float)

    left = right = 0
    for i in range(n):
        xi = x[i]
        while right < n and (x[right] - xi) <= half: right += 1
        while left  < n and (xi - x[left])  >  half: left  += 1

        sl = slice(left, right)
        xs = x[sl]; ys_win = y[sl]
        m = np.isfinite(xs) & np.isfinite(ys_win)
        if m.sum() < 2:
            ys[i] = y[i]
            # fallback: centered finite-diff if possible
            if 0 < i < n-1:
                dydx[i] = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
            else:
                dydx[i] = 0.0
            continue

        xs = xs[m]; ys_loc = ys_win[m]
        dist = np.abs(xs - xi); dmax = dist.max()
        if dmax == 0:
            ys[i] = ys_loc.mean(); dydx[i] = 0.0
            continue

        w = (1.0 - (dist / dmax)**3)**3  # tricube

        # weighted LS for y ~ b0 + b1*x
        X0 = np.ones_like(xs)
        s00 = np.sum(w * X0 * X0)
        s01 = np.sum(w * X0 * xs)
        s11 = np.sum(w * xs * xs)
        t0  = np.sum(w * X0 * ys_loc)
        t1  = np.sum(w * xs * ys_loc)

        det = s00 * s11 - s01 * s01
        if det == 0:
            ys[i] = ys_loc.mean(); dydx[i] = 0.0
            continue

        b0 = ( t0 * s11 - s01 * t1) / det
        b1 = (-t0 * s01 + s00 * t1) / det

        ys[i]   = b0 + b1 * xi      # LOWESS value
        dydx[i] = b1                # LOWESS slope (clean derivative)

    # simple edge fill for first/last if needed
    if n >= 2:
        dydx[0]  = dydx[1]
        dydx[-1] = dydx[-2]
        
    return ys, dydx


def _smooth_region_task(args: Tuple[List[int], List[float], int]) -> List[float]:
    """
    Worker function for parallel smoothing.
    Args: (positions, fraction_modified, window_bp)
    Returns: smoothed fraction list (same order as sorted positions)
    """
    positions, fractions, window_bp = args
    if not positions:
        return [], []

    x = np.asarray(positions, dtype=float)
    y = np.asarray(fractions, dtype=float)

    y_sm, dy_sm = lowess_smooth(y, x, window_bp)

    return y_sm.tolist(), dy_sm.tolist()


class DataHandler:
    """Parser to read in region and methylation bed files for centrodip."""
    def __init__(
        self,
        regions_path: Path | str,
        methylation_path: Path | str,
        mod_code: str,
        bedgraph: bool,
        smooth_window_bp: int = 10000,
        threads: int | None = None,
        debug: bool = False,
    ) -> None:
        self.mod_code = mod_code
        self.bedgraph = bedgraph

        self.smooth_window_bp = smooth_window_bp
        self.threads = threads

        if debug:
            print(f"[DEBUG] Reading regions from {regions_path}...")
        self.region_dict = self.read_regions_bed(regions_path)
        if debug:
            print(f"[DEBUG] Loaded {len(self.region_dict)} regions.")

        if debug:
            print(f"[DEBUG] Reading methylation from {methylation_path}...")
        self.methylation_dict = self.read_methylation_bed(
            methylation_path,
            self.region_dict,
        )
        if debug:
            Ncpg = sum(len(entry["position"]) for entry in self.methylation_dict.values())
            print(f"[DEBUG] Loaded data from {Ncpg} CpG Sites.")

        # set this up to parallel calculate lowess smoothed values using futures
        self._add_lowess_parallel()

    def read_regions_bed(self, regions_path: Path | str) -> RegionDict:
        """ Read and filter regions from a BED file. """

        regions_path = Path(regions_path)
        if not regions_path.exists():
            raise FileNotFoundError(f"File not found: {regions_path}")

        region_dict: RegionDict = defaultdict(lambda: {"starts": [], "ends": []})

        with regions_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 3:
                    raise TypeError(
                        f"Less than 3 columns in {regions_path}. Likely incorrectly formatted bed file."
                    )
                chrom = columns[0]
                start, end = int(columns[1]), int(columns[2])
                region_dict[chrom]["starts"].append(start)
                region_dict[chrom]["ends"].append(end)

        return region_dict
    
    def read_methylation_bed(
        self,
        methylation_path: Path | str,
        region_dict: RegionDict,
    ) -> MethylationDict:
        """ Read and filter methylation data from a BED file. """

        methylation_path = Path(methylation_path)
        if not methylation_path.exists():
            raise FileNotFoundError(f"File not found: {methylation_path}")

        methylation_dict: Dict[str, Dict[str, List[float | int]]] = defaultdict(
            lambda: {
                "position": [],
                "fraction_modified": [],
                "valid_coverage": [],
            }
        )

        region_lookup = {
            chrom: (coords["starts"], coords["ends"])
            for chrom, coords in region_dict.items()
        }

        with methylation_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                columns = line.rstrip("\n").split("\t")
                min_columns = 4 if self.bedgraph else 11
                if len(columns) < min_columns:
                    raise TypeError(
                        f"Insufficient columns in {methylation_path}. "
                        "Likely incorrectly formatted."
                    )
                if self.bedgraph and len(columns) > 4:
                    warnings.warn(
                        f"Warning: {methylation_path} has more than 4 columns, and was "
                        "passed in as bedgraph. Potentially incorrectly formatted bedgraph file.",
                        stacklevel=2,
                    )
                if not self.bedgraph and columns[3] != self.mod_code:
                    continue

                chrom = columns[0]
                methylation_position = int(columns[1])
                if chrom not in region_lookup:
                    continue

                starts, ends = region_lookup[chrom]
                idx = bisect_right(starts, methylation_position) - 1
                if idx < 0:
                    continue
                region_start = starts[idx]
                region_end = ends[idx]
                if not (region_start < methylation_position < region_end):
                    continue

                region_key = f"{chrom}:{region_start}-{region_end}"
                entry = methylation_dict[region_key]
                entry["position"].append(methylation_position)
                entry["fraction_modified"].append(
                    float(columns[3]) if self.bedgraph else float(columns[10])
                )
                entry["valid_coverage"].append(
                    1.0 if self.bedgraph else float(columns[4])
                )

        return methylation_dict

    
    def _add_lowess_parallel(self) -> None:
        """
        Compute LOWESS-smoothed methylation per region in parallel and
        store it under key 'lowess_fraction_modified' in self.methylation_dict[region].
        """
        tasks = []
        keys = []
        for key, entry in self.methylation_dict.items():
            if len(entry["position"]) == 0:
                # still add an empty array for uniformity
                entry["lowess_fraction_modified"] = []
                continue
            keys.append(key)
            tasks.append((entry["position"], entry["fraction_modified"], self.smooth_window_bp))

        if not tasks:
            return

        # Use processes for CPU-bound smoothing; allow caller to override worker count
        max_workers = self.threads or os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_to_key = {
                ex.submit(_smooth_region_task, task): key
                for key, task in zip(keys, tasks)
            }
            for fut in concurrent.futures.as_completed(fut_to_key):
                key = fut_to_key[fut]
                y_sm, dy_sm = fut.result()

                self.methylation_dict[key]["lowess_fraction_modified"] = y_sm
                self.methylation_dict[key]["lowess_fraction_modified_dy"] = dy_sm