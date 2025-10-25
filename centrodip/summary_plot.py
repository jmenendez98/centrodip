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


def _region_key(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start}-{end}"


def _normalise_interval(start: int, end: int) -> Tuple[float, float]:
    left = float(min(start, end))
    right = float(max(start, end))
    if left == right:
        left -= 0.5
        right += 0.5
    return left, right


def _ordered_positions(
    record: MethylationRecord, key: str
) -> Tuple[Sequence[float], Sequence[float]]:
    positions = list(record.get("position", []))
    values = list(record.get(key, []))

    if not positions or not values:
        return [], []

    length = min(len(positions), len(values))
    order = sorted(range(length), key=positions.__getitem__)
    return (
        [positions[idx] for idx in order],
        [values[idx] for idx in order],
    )


def _position_edges(positions: Sequence[float]) -> np.ndarray:
    if not positions:
        return np.asarray([])

    ordered = np.asarray(sorted(float(pos) for pos in positions))
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
    *,
    window_size: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not positions or not values:
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

    pos_arr = np.asarray(positions, dtype=float)
    val_arr = np.asarray(values, dtype=float)

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


def _plot_band(
    ax: plt.Axes,
    record: MethylationRecord,
    region_start: int,
    region_end: int,
    value_key: str,
    y_bottom: float,
    y_top: float,
    norm: Normalize,
) -> None:
    positions, values = _ordered_positions(record, value_key)
    if not positions or not values:
        return

    edges_x = _position_edges(positions)
    if len(edges_x) < 2:
        return

    data = np.ma.masked_invalid(np.asarray(values, dtype=float)[np.newaxis, :])
    edges_y = np.asarray([y_bottom, y_top], dtype=float)

    cmap = plt.get_cmap("Greys")
    ax.pcolormesh(
        edges_x,
        edges_y,
        data,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )


def _plot_fraction_line(
    ax: plt.Axes,
    record: MethylationRecord,
    column: str,
    region_start: int,
    region_end: int,
    *,
    window_size: float = 1000.0,
    color: str = "black",
    linewidth: float = 1,
    alpha : float = 0.5,
) -> None:
    positions, values = _ordered_positions(record, column)
    if not positions or not values:
        return

    y_values = 1.75 + np.asarray(values, dtype=float) / 100.0
    ax.plot(positions, y_values, color=color, linewidth=linewidth, alpha=alpha)


def _plot_regions(ax: plt.Axes, regions: Iterable[Tuple[int, int]]) -> None:
    for start, end in regions:
        start = int(start)
        end = int(end)
        left, right = _normalise_interval(start, end)
        rect = Rectangle(
            (left, 0),
            right - left,
            1,
            facecolor=(153/255,0/255,0/255),
            alpha=1,
            edgecolor="none",
        )
        ax.add_patch(rect)


def _plot_dips(
    ax: plt.Axes,
    dips: DipRecord,
    *,
    y_bottom: float = 3.0,
    height: float = 0.5,
    color: str = "black",
    alpha: float = 0.7,
    zorder: float | None = None,
) -> None:
    starts = dips.get("starts", []) if dips else []
    ends = dips.get("ends", []) if dips else []

    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)
        left, right = _normalise_interval(start, end)
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


def centrodip_summary_plot(
    regions_per_chrom: RegionDict,
    methylation_per_region: MethylationDict,
    final_dips: DipDict,
    output_path: Path | str,
    *,
    unfiltered_dips: DipDict | None = None,
    panel_height: float = 2.0,
    figure_width: float = 12.0,
) -> Path:

    chromosomes: List[str] = []
    for chrom, coords in regions_per_chrom.items():
        starts = coords.get("starts", []) if coords else []
        ends = coords.get("ends", []) if coords else []
        if starts and ends:
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
    plot_unfiltered = bool(unfiltered_dips)

    for axis_row, chrom in zip(axes, chromosomes):
        ax = axis_row[0]
        coords = regions_per_chrom.get(chrom, {})
        starts = coords.get("starts", []) if coords else []
        ends = coords.get("ends", []) if coords else []
        regions = [(int(start), int(end)) for start, end in zip(starts, ends)]
        if not regions:
            ax.set_visible(False)
            continue

        x_min = min(min(start, end) for start, end in regions)
        x_max = max(max(start, end) for start, end in regions)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        _plot_regions(ax, regions)

        for start, end in regions:
            region_key = _region_key(chrom, start, end)
            record = methylation_per_region.get(region_key, {})

            # plot coverage bar
            _plot_band(
                ax,
                record,
                start,
                end,
                "valid_coverage",
                1.0,
                1.5,
                coverage_norm,
            )

            # plot the raw fraction modified values as a line
            _plot_fraction_line(
                ax=ax,
                record=record,
                column="fraction_modified",
                region_start=start,
                region_end=end,
                alpha=0.25,
                color="black"
            )

            # plot the smoothed LOWESS fraction modified line
            _plot_fraction_line(
                ax=ax,
                record=record,
                column="lowess_fraction_modified",
                region_start=start,
                region_end=end,
                alpha=0.75,
                color="orange"
            )

            _plot_dips(
                ax,
                final_dips.get(region_key, {}),
                y_bottom=3.0,
                height=0.5,
                color="black",
                alpha=0.7,
                zorder=2,
            )

            if plot_unfiltered:
                unfiltered_record = (
                    unfiltered_dips.get(region_key, {}) if unfiltered_dips else {}
                )

                _plot_dips(
                    ax,
                    unfiltered_record,
                    y_bottom=3.6,
                    height=0.35,
                    color="tab:blue",
                    alpha=0.4,
                    zorder=3,
                )

        ax.set_xlim(x_min, x_max)
        if plot_unfiltered:
            ax.set_ylim(0, 4.2)
            ax.set_yticks([0.5, 1.25, 2.25, 3.25, 3.775])
            ax.set_yticklabels([
                "Regions",
                "Coverage",
                "FracMod",
                "Filtered dips",
                "Unfiltered dips",
            ])
        else:
            ax.set_ylim(0, 4)
            ax.set_yticks([0.5, 1.25, 2.25, 3.25])
            ax.set_yticklabels([
                "Regions",
                "Coverage",
                "FracMod",
                "Dips",
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