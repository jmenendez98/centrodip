from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    """Build the canonical region key ``chrom:start-end``."""

    return f"{chrom}:{start}-{end}"


def _normalise_interval(start: int, end: int) -> Tuple[float, float]:
    """Return the inclusive interval bounds in ascending order.

    Zero-length intervals are expanded by 0.5 bp on either side to ensure they
    remain visible once plotted while keeping the rectangle centred on the
    genomic coordinate.
    """

    left = float(min(start, end))
    right = float(max(start, end))
    if left == right:
        left -= 0.5
        right += 0.5
    return left, right


def _sorted_by_position(record: MethylationRecord, key: str) -> Sequence[float]:
    """Return ``record[key]`` sorted by genomic position.

    The methylation records keep the underlying ``position`` array sorted, but
    the helper defensively reorders the requested value list by the same
    ordering in case a pre-processed dictionary was passed in.
    """

    positions = list(record.get("position", []))
    values = list(record.get(key, []))

    if not positions or not values:
        return []

    length = min(len(positions), len(values))
    order = sorted(range(length), key=positions.__getitem__)
    return [values[idx] for idx in order]


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
    """Draw a horizontal heatmap band for a methylation metric."""

    values = _sorted_by_position(record, value_key)
    if not values:
        return

    data = np.asarray(values, dtype=float)[np.newaxis, :]

    region_left, region_right = _normalise_interval(region_start, region_end)
    extent = (region_left, region_right, y_bottom, y_top)

    cmap = plt.get_cmap("Greys")
    ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=extent,
    )


def _plot_regions(ax: plt.Axes, regions: Iterable[Tuple[int, int]]) -> None:
    """Draw the search regions at the bottom of the panel."""

    for start, end in regions:
        start = int(start)
        end = int(end)
        left, right = _normalise_interval(start, end)
        rect = Rectangle(
            (left, 0),
            right - left,
            1,
            facecolor="red",
            alpha=0.4,
            edgecolor="none",
        )
        ax.add_patch(rect)


def _plot_dips(ax: plt.Axes, dips: DipRecord) -> None:
    """Visualise dip intervals as black rectangles in the top track."""

    starts = dips.get("starts", []) if dips else []
    ends = dips.get("ends", []) if dips else []

    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)
        left, right = _normalise_interval(start, end)
        rect = Rectangle(
            (left, 3),
            right - left,
            1,
            facecolor="black",
            alpha=0.7,
            edgecolor="none",
        )
        ax.add_patch(rect)


def _add_track_legends(
    ax: plt.Axes,
    coverage_norm: Normalize,
    fraction_norm: Normalize,
) -> None:
    """Attach compact horizontal colour bars for the heatmap tracks."""

    track_height = 1.0 / 4.0
    bar_height = track_height * 0.35
    bar_width = 0.12

    def _legend_axes(track_index: int) -> plt.Axes:
        bottom = track_index * track_height + (track_height - bar_height) / 2
        return ax.inset_axes([1.02, bottom, bar_width, bar_height])

    # Coverage legend (grayscale bar capped at 10)
    coverage_ax = _legend_axes(1)
    coverage_cbar = plt.colorbar(
        ScalarMappable(norm=coverage_norm, cmap="Greys"),
        cax=coverage_ax,
        orientation="horizontal",
    )
    coverage_cbar.ax.tick_params(labelsize=6, pad=1)
    coverage_cbar.set_ticks([0, 10])
    coverage_cbar.ax.set_xlabel("Cov", fontsize=6, labelpad=-3)

    # Fraction modified legend (0-100 gradient)
    fraction_ax = _legend_axes(2)
    fraction_cbar = plt.colorbar(
        ScalarMappable(norm=fraction_norm, cmap="Greys"),
        cax=fraction_ax,
        orientation="horizontal",
    )
    fraction_cbar.ax.tick_params(labelsize=6, pad=1)
    fraction_cbar.set_ticks([0, 100])
    fraction_cbar.ax.set_xlabel("Frac%", fontsize=6, labelpad=-3)


def create_summary_plot(
    regions_per_chrom: RegionDict,
    methylation_per_region: MethylationDict,
    dip_results: DipDict,
    output_path: Path | str,
    *,
    panel_height: float = 2.0,
    figure_width: float = 12.0,
) -> Path:
    """Create a vertically stacked summary plot of the centrodip results.

    Parameters
    ----------
    regions_per_chrom:
        Dictionary mapping chromosome names to region coordinate lists as
        produced by :class:`~centrodip.parse.Parser`.
    methylation_per_region:
        Dictionary of methylation metrics per region key from
        :class:`~centrodip.dip_detect.DipDetector`.
    dip_results:
        Dictionary containing dip coordinates per region key returned by the
        detector.
    output_path:
        File path where the generated figure will be written.
    panel_height:
        Height of each chromosome subplot in inches. Defaults to ``3``.
    figure_width:
        Width of the full figure in inches. Defaults to ``12``.

    Returns
    -------
    Path
        The path where the plot image was written.
    """

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
    fraction_norm = Normalize(vmin=0, vmax=100, clip=True)

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
            _plot_band(
                ax,
                record,
                start,
                end,
                "valid_coverage",
                1,
                2,
                coverage_norm,
            )
            _plot_band(
                ax,
                record,
                start,
                end,
                "fraction_modified",
                2,
                3,
                fraction_norm,
            )

            _plot_dips(ax, dip_results.get(region_key, {}))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 4)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels([
            "Regions",
            "Coverage",
            "% FracMod",
            "Dips",
        ])
        ax.set_ylabel(f"{chrom}")
        ax.grid(False)
        _add_track_legends(ax, coverage_norm, fraction_norm)

    axes[-1][0].set_xlabel("Genomic position (bp)")

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


__all__ = ["create_summary_plot"]