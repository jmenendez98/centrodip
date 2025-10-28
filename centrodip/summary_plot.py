from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


RegionDict = Dict[str, Dict[str, List[int]]]
MethylationRecord = Dict[str, List[float]]
MethylationDict = Dict[str, MethylationRecord]
DipRecord = Dict[str, List[int]]
DipDict = Dict[str, DipRecord]


def _normalise_interval(start: int, end: int) -> Tuple[float, float]:
    left = float(min(start, end))
    right = float(max(start, end))
    if left == right:
        left -= 0.5
        right += 0.5
    return left, right


def _position_edges(positions: Sequence[float]) -> np.ndarray:
    if positions is None:
        return np.asarray([])

    positions_arr = np.asarray(positions, dtype=float).ravel()

    if positions_arr.size == 0:
        return np.asarray([])

    valid_mask = ~np.isnan(positions_arr)
    if not np.any(valid_mask):
        return np.asarray([])

    ordered = np.sort(positions_arr[valid_mask])
    if len(ordered) == 1:
        pos = ordered[0]
        return np.asarray([pos - 0.5, pos + 0.5])

    deltas = np.diff(ordered)
    left_edge = ordered[0] - deltas[0] / 2.0
    right_edge = ordered[-1] + deltas[-1] / 2.0
    midpoints = (ordered[:-1] + ordered[1:]) / 2.0
    return np.concatenate(([left_edge], midpoints, [right_edge]))


def _fraction_window_bins(
    positions: Sequence[float],
    values: Sequence[float],
    region_start: int,
    region_end: int,
    window_size: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if positions is None or values is None:
        return np.asarray([]), np.asarray([])

    pos_arr = np.asarray(positions, dtype=float).ravel()
    val_arr = np.asarray(values, dtype=float).ravel()

    if pos_arr.size == 0 or val_arr.size == 0:
        return np.asarray([]), np.asarray([])

    region_min = float(min(region_start, region_end))
    region_max = float(max(region_start, region_end))
    if region_min == region_max:
        region_min -= 0.5
        region_max += 0.5

    edges = np.arange(region_min, region_max, window_size, dtype=float)
    if edges.size == 0 or edges[-1] < region_max:
        edges = np.append(edges, region_max)
    else:
        edges[-1] = region_max

    if edges[0] != region_min:
        edges = np.insert(edges, 0, region_min)

    means: List[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right < left:
            left, right = right, left
        is_last = np.isclose(right, edges[-1])
        mask = (pos_arr >= left) & (pos_arr < right if not is_last else pos_arr <= right)
        if np.any(mask):
            means.append(float(np.nanmean(val_arr[mask])))
        else:
            means.append(np.nan)

    return edges, np.asarray(means, dtype=float)


def _slice_methylation_record(
    record: MethylationRecord,
    region_start: int,
    region_end: int,
) -> MethylationRecord:
    if not record:
        return {}

    positions = list(record.get("position", []))
    if not positions:
        return {key: [] for key in record.keys()}

    region_min = float(min(region_start, region_end))
    region_max = float(max(region_start, region_end))
    mask = [region_min <= float(pos) <= region_max for pos in positions]
    if not any(mask):
        return {key: [] for key in record.keys()}

    indices = [idx for idx, keep in enumerate(mask) if keep]
    sliced: MethylationRecord = {}
    for key, values in record.items():
        if isinstance(values, list) and len(values) == len(positions):
            sliced[key] = [values[idx] for idx in indices]
        else:
            sliced[key] = list(values) if isinstance(values, list) else []

    # Ensure expected keys exist even if absent in the original record.
    for key in (
        "position",
        "fraction_modified",
        "valid_coverage",
        "lowess_fraction_modified",
    ):
        sliced.setdefault(key, [])

    return sliced


def _filter_dips_for_region(
    dips: DipRecord,
    region_start: int,
    region_end: int,
) -> DipRecord:
    if not dips:
        return {"starts": [], "ends": []}

    region_min = int(min(region_start, region_end))
    region_max = int(max(region_start, region_end))

    starts = dips.get("starts", [])
    ends = dips.get("ends", [])

    region_starts: List[int] = []
    region_ends: List[int] = []

    for dip_start, dip_end in zip(starts, ends):
        dip_start = int(dip_start)
        dip_end = int(dip_end)

        if dip_end < region_min or dip_start > region_max:
            continue

        clipped_start = max(dip_start, region_min)
        clipped_end = min(dip_end, region_max)
        region_starts.append(clipped_start)
        region_ends.append(clipped_end)

    return {"starts": region_starts, "ends": region_ends}


def _plot_band(
    ax: plt.Axes,
    x,
    val,
    y_bottom: float,
    y_top: float,
    norm: Normalize,
    *,
    cmap_name: str = "Greys",
) -> None:
    if x is None or val is None:
        return

    x_arr = np.asarray(x, dtype=float)
    v_arr = np.asarray(val, dtype=float)

    if x_arr.size == 0 or v_arr.size == 0:
        return

    n = min(x_arr.size, v_arr.size)
    if n == 0:
        return

    x_arr = x_arr[:n]
    v_arr = v_arr[:n]

    edges_x = _position_edges(x_arr)
    if edges_x.size < 2:
        return  # nothing to draw

    data = np.ma.masked_invalid(v_arr[np.newaxis, :n])
    edges_y = np.asarray([y_bottom, y_top], dtype=float)

    ax.pcolormesh(
        edges_x,
        edges_y,
        data,
        shading="auto",
        cmap=plt.get_cmap(cmap_name),
        norm=norm,
    )

def _plot_fraction_line(
    ax: plt.Axes,
    xpos,
    val,
    *,
    color: str = "black",
    linewidth: float = 1.0,
    alpha: float = 0.5,
) -> None:
    if xpos is None or val is None:
        return

    positions = np.asarray(xpos, dtype=float)
    values = np.asarray(val, dtype=float)

    if positions.size == 0 or values.size == 0:
        return

    n = min(positions.size, values.size)
    if n == 0:
        return

    positions = positions[:n]
    values = values[:n]

    y_values = 1.75 + values / 100.0
    ax.plot(positions, y_values, color=color, linewidth=linewidth, alpha=alpha)


def _plot_dips(
    ax: plt.Axes,
    starts=None,
    ends=None,
    y_bottom: float = 3.0,
    height: float = 0.5,
    color: str = "black",
    alpha: float = 0.7,
    zorder: float | None = None,
) -> None:
    if starts is None or ends is None:
        return

    starts_arr = np.asarray(starts, dtype=float)
    ends_arr = np.asarray(ends, dtype=float)

    n = min(starts_arr.size, ends_arr.size)
    if n == 0:
        return

    for start, end in zip(starts_arr[:n], ends_arr[:n]):
        left, right = _normalise_interval(int(start), int(end))
        rect = Rectangle(
            (left, y_bottom),
            right - left,
            height,
            facecolor=color,
            alpha=alpha,
            edgecolor="none",
            zorder=zorder,
        )
        ax.add_patch(rect)


def _add_track_legends(
    ax: plt.Axes,
    coverage_norm: Normalize,
) -> None:
    total_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_width = 0.12
    bar_height = 0.08
    coverage_centre = 1.25
    bottom = coverage_centre / total_height - bar_height / 2

    coverage_ax = ax.inset_axes([1.02, bottom, bar_width, bar_height])
    coverage_cbar = plt.colorbar(
        ScalarMappable(norm=coverage_norm, cmap="Greys"),
        cax=coverage_ax,
        orientation="horizontal",
    )
    coverage_cbar.ax.tick_params(labelsize=6, pad=1)
    coverage_cbar.set_ticks([0, 10])
    coverage_cbar.ax.set_xlabel("Cov", fontsize=6, labelpad=-3)


def centrodipSummaryPlot(
    results,
    output_path: Path | str,
    panel_height: float = 2.0,
    figure_width: float = 16.0,
) -> Path:

    chromosomes: List[str] = []
    for chrom in results.keys():
        chromosomes.append(chrom)

    chromosomes.sort()

    if not chromosomes:
        raise ValueError("No regions available to plot.")

    figure_height = max(panel_height * len(chromosomes), panel_height)
    fig, axes = plt.subplots(
        nrows=len(chromosomes),
        ncols=1,
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=False,
    )

    coverage_norm = Normalize(vmin=0, vmax=10, clip=True)

    for axis_row, chrom in zip(axes, chromosomes):
        ax = axis_row[0]
        chrom_results = results.get(chrom, {})

        cpg_pos = chrom_results.get("cpg_pos", [])
        cpg_coverage = chrom_results.get("valid_coverage", [])
        frac_mod = chrom_results.get("fraction_modified", [])
        lowess_frac_mod = chrom_results.get("lowess_fraction_modified", [])

        if cpg_pos is None or len(cpg_pos) == 0:
            ax.set_axis_off()
            continue

        x_min = min(cpg_pos)
        x_max = max(cpg_pos)
        if x_min == x_max:
            x_min -= 10000
            x_max += 10000

        # _plot_regions(ax, regions)


        # plot coverage bar
        _plot_band(
            ax=ax,
            x=cpg_pos,
            val=cpg_coverage,
            y_bottom=1.0, y_top=1.5,
            norm=coverage_norm,
        )

        # plot the raw fraction modified values as a line
        _plot_fraction_line(
            ax=ax,
            xpos=cpg_pos,
            val=frac_mod,
            alpha=0.25,
            color="black",
        )

        # plot the smoothed LOWESS fraction modified line
        _plot_fraction_line(
            ax=ax,
            xpos=cpg_pos,
            val=lowess_frac_mod,
            alpha=0.75,
            color="orange",
        )

        # plot unfiltered dips
        unfiltered_dip_starts = chrom_results.get("unfiltered_dip_starts", [])
        unfiltered_dip_ends = chrom_results.get("unfiltered_dip_ends", [])
        _plot_dips(
            ax=ax,
            starts=unfiltered_dip_starts,
            ends=unfiltered_dip_ends,
            y_bottom=3.6,
            height=0.35,
            color="tab:blue",
            alpha=0.4,
            zorder=3,
        )

        # plot final dips
        dip_starts = chrom_results.get("dip_starts", [])
        dip_ends = chrom_results.get("dip_ends", [])
        _plot_dips(
            ax=ax,
            starts=dip_starts,
            ends=dip_ends,
            y_bottom=3.0,
            height=0.5,
            color="black",
            alpha=0.7,
            zorder=2,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 4.2)
        ax.set_yticks([0.5, 1.25, 2.25, 3.25, 3.775])
        ax.set_yticklabels([
            "Regions",
            "Coverage",
            "FracMod",
            "Filtered dips",
            "Unfiltered dips",
        ])

        ax.set_ylabel(f"{chrom}")
        ax.tick_params(axis=u'y', which=u'both', length=0)
        ax.grid(False)
        _add_track_legends(ax, coverage_norm)

        ax.axhline(
            y=1.75,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.axhline(
            y=2.75,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        secax = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y: (np.asarray(y) - 1.75) * 100.0,
                lambda p: np.asarray(p) / 100.0 + 1.75,
            ),
        )
        secax.set_yticks([0, 50, 100])
        secax.set_yticklabels(["0%", "50%", "100%"])
        secax.set_ylabel("")

    axes[-1][0].set_xlabel("Genomic position (bp)")

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


__all__ = ["centrodip_summary_plot"]