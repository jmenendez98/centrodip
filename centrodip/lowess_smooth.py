from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import concurrent.futures
import warnings
import numpy as np
import os


def lowessSmooth(
    y, x, c, 
    window_bp, cov_conf
):
    """
    LOWESS with tricube distance kernel.
    If point_weights is provided (length n), they are multiplied into the local
    tricube weights inside each window. Use values in [0, 1].
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    point_weights = _cov_to_weights(c, cov_conf)
    n = len(y)
    if n == 0:
        return np.array([], float), np.array([], float)

    if point_weights is not None:
        pw = np.asarray(point_weights, float)
        if pw.shape != y.shape:
            raise ValueError("point_weights must have same length as y/x")
        # sanitize: clamp to [0, 1] and replace NaNs
        pw = np.clip(np.nan_to_num(pw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        pw = None

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
            if 0 < i < n-1 and np.isfinite(y[i-1]) and np.isfinite(y[i+1]) and (x[i+1] != x[i-1]):
                dydx[i] = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
            else:
                dydx[i] = 0.0
            continue

        xs = xs[m]; ys_loc = ys_win[m]
        dist = np.abs(xs - xi); dmax = dist.max()
        if dmax == 0:
            ys[i] = ys_loc.mean(); dydx[i] = 0.0
            continue

        # tricube distance weights
        w = (1.0 - (dist / dmax)**3)**3

        # multiply by per-point weights if provided
        if pw is not None:
            w *= pw[sl][m]

        # guard against all-zero weights
        if not np.any(w > 0):
            ys[i] = ys_loc.mean(); dydx[i] = 0.0
            continue

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

        ys[i]   = b0 + b1 * xi
        dydx[i] = b1

    if n >= 2:
        dydx[0]  = dydx[1]
        dydx[-1] = dydx[-2]

    return ys, dydx


def _cov_to_weights(
    coverage,
    cov_conf
):
    """Input an array of valid coverage. Returns an array of LOWESS weights based on coverage."""
    coverage = np.asarray(coverage, dtype=int)
    if cov_conf <= 0:
        raise ValueError("cov_conf must be greater than 0.")
    weights = np.minimum(coverage / cov_conf, 1.0)
    return weights