from __future__ import annotations

import numpy as np
from centrodip.bedtable import BedTable, IntervalRecord


def bedMethyl_LOWESS(
    bedMethyl: BedTable,
    *,
    window_bp: int,
    cov_conf: float,
    y_col_1based: int = 11,
    cov_col_1based: int = 10,
    x_mode: str = "start",          # "start" or "midpoint"
    drop_nonfinite: bool = True,    # if True, rows with NaN y/x are removed from output
) -> BedTable:
    """
    LOWESS with tricube distance kernel on an already-sorted BedTable.

    Input: BedTable (assumed sorted)
    Output: BedTable (same order), with extras = (smoothedY, dY)

      chrom  start  end  smoothedY  dY

    Notes:
    - No internal sorting is performed.
    - Windowing is done in bp using x coordinates as given.
    - If drop_nonfinite=True, non-finite y/x rows are excluded from BOTH the fit and output.
      If False, they remain in output with NaN extras (and are skipped in fitting windows).
    """

    def _cov_to_weights(coverage, cov_conf, decay_rate: float = 5.0):
        coverage = np.asarray(coverage, dtype=float)
        if cov_conf <= 0:
            raise ValueError("cov_conf must be greater than 0.")
        if np.any(coverage < 0):
            raise ValueError("coverage must be non-negative.")
        weights = 1 - np.exp(-decay_rate * (coverage / cov_conf))
        return np.clip(weights, 0, 1)

    rows = list(bedMethyl)
    if not rows:
        return BedTable([], inferred_kind="bedgraph", inferred_ncols=5)

    # Build x/y/c in the SAME order as input
    if x_mode == "start":
        x_all = np.asarray([r.start for r in rows], dtype=float)
    elif x_mode == "midpoint":
        x_all = np.asarray([(r.start + r.end) / 2.0 for r in rows], dtype=float)
    else:
        raise ValueError("x_mode must be 'start' or 'midpoint'")

    y_all = np.asarray(bedMethyl.get_column_1based(y_col_1based, as_float=True), dtype=float)
    if cov_col_1based is None:
        c_all = np.full(len(rows), cov_conf, dtype=float)
    else:
        c_all = np.asarray(bedMethyl.get_column_1based(cov_col_1based, as_float=True), dtype=float)

    finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
    if drop_nonfinite:
        rows_fit = [rows[i] for i in np.where(finite_mask)[0]]
        x = x_all[finite_mask]
        y = y_all[finite_mask]
        c = c_all[finite_mask]
        if len(x) == 0:
            return BedTable([], inferred_kind="bedgraph", inferred_ncols=5)
    else:
        # Fit uses only finite points, but output keeps all rows
        rows_fit = [rows[i] for i in np.where(finite_mask)[0]]
        x = x_all[finite_mask]
        y = y_all[finite_mask]
        c = c_all[finite_mask]

    pw = _cov_to_weights(c, cov_conf)
    pw = np.clip(np.nan_to_num(pw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    n = len(y)
    half = float(window_bp) / 2.0
    ys = np.empty(n, float)
    dydx = np.empty(n, float)

    left = right = 0
    for i in range(n):
        xi = x[i]

        # Sliding window assumes x is non-decreasing (sorted input)
        while right < n and (x[right] - xi) <= half:
            right += 1
        while left < n and (xi - x[left]) > half:
            left += 1

        sl = slice(left, right)
        xs = x[sl]
        ys_win = y[sl]

        m = np.isfinite(xs) & np.isfinite(ys_win)
        if m.sum() < 2:
            ys[i] = y[i]
            if 0 < i < n - 1 and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1]) and (x[i + 1] != x[i - 1]):
                dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            else:
                dydx[i] = 0.0
            continue

        xs = xs[m]
        ys_loc = ys_win[m]
        dist = np.abs(xs - xi)
        dmax = dist.max()

        if dmax == 0:
            ys[i] = ys_loc.mean()
            dydx[i] = 0.0
            continue

        w = (1.0 - (dist / dmax) ** 3) ** 3
        w *= pw[sl][m]

        if not np.any(w > 0):
            ys[i] = ys_loc.mean()
            dydx[i] = 0.0
            continue

        # weighted LS: y ~ b0 + b1*x
        X0 = np.ones_like(xs)
        s00 = np.sum(w * X0 * X0)
        s01 = np.sum(w * X0 * xs)
        s11 = np.sum(w * xs * xs)
        t0 = np.sum(w * X0 * ys_loc)
        t1 = np.sum(w * xs * ys_loc)

        det = s00 * s11 - s01 * s01
        if det == 0:
            ys[i] = ys_loc.mean()
            dydx[i] = 0.0
            continue

        b0 = (t0 * s11 - s01 * t1) / det
        b1 = (-t0 * s01 + s00 * t1) / det

        ys[i] = b0 + b1 * xi
        dydx[i] = b1

    if n >= 2:
        dydx[0] = dydx[1]
        dydx[-1] = dydx[-2]

    # ---- Build output in SAME order as input ----
    out_records = []

    if drop_nonfinite:
        # output only the fitted rows, preserving their original relative order
        for rec, sm, dyv in zip(rows_fit, ys, dydx):
            out_records.append(
                IntervalRecord(
                    chrom=rec.chrom,
                    start=rec.start,
                    end=rec.end,
                    name=None,
                    score=None,
                    strand=None,
                    extras=(float(sm), float(dyv)),
                )
            )
    else:
        # output all rows, fill NaNs for non-finite ones
        smoothed_all = np.full(len(rows), np.nan, dtype=float)
        dydx_all = np.full(len(rows), np.nan, dtype=float)

        idx_fit = np.where(finite_mask)[0]
        smoothed_all[idx_fit] = ys
        dydx_all[idx_fit] = dydx

        for rec, sm, dyv in zip(rows, smoothed_all, dydx_all):
            out_records.append(
                IntervalRecord(
                    chrom=rec.chrom,
                    start=rec.start,
                    end=rec.end,
                    name=None,
                    score=None,
                    strand=None,
                    extras=(float(sm), float(dyv)),
                )
            )

    return BedTable(out_records, inferred_kind="bedgraph", inferred_ncols=5)
