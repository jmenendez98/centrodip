from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import math
import numpy as np

from centrodip.bedtable import BedTable, IntervalRecord

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


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


def _plot_coverage_band(
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
        return

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

    # Your values are in percent (0..100) in your current pipeline
    y_values = 1.75 + values / 100.0
    ax.plot(positions, y_values, color=color, linewidth=linewidth, alpha=alpha)


def _plot_regions(
    ax: plt.Axes,
    regions: Optional[BedTable],
    *,
    y_bottom: float = 0.95,
    height: float = 0.18,
    color: str = "tab:green",
    alpha: float = 0.25,
    zorder: float | None = None,
) -> None:
    if regions is None or len(regions._records) == 0:
        return
    for r in regions._records:
        left, right = _normalise_interval(int(r.start), int(r.end))
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
    *,
    cbar_width_in: float = 0.25,
    cbar_height_in: float = 0.5,
    coverage_center_y_data: float = 1.225,
    line_legend_y: float = 0.535,
    x_offset_in: float = 0.75,   # distance from axes right edge
) -> None:
    """
    Adds:
      1) A vertical coverage colorbar with fixed physical size (inches), placed to the right.
      2) A small legend for raw vs smoothed methylation lines.
    """

    fig = ax.figure

    # --- Figure and axes sizes in inches ---
    fig_w_in, fig_h_in = fig.get_size_inches()
    bbox = ax.get_position()  # axes bbox in figure-relative coords
    ax_w_in = bbox.width * fig_w_in
    ax_h_in = bbox.height * fig_h_in

    # --- Convert inch sizes to axes-relative fractions ---
    w_frac = cbar_width_in / ax_w_in
    h_frac = cbar_height_in / ax_h_in

    # --- Vertical placement: convert DATA y to axes-fraction y ---
    y0, y1 = ax.get_ylim()
    if y1 == y0:
        return
    y_center_axes = (coverage_center_y_data - y0) / (y1 - y0)
    y_frac = y_center_axes - h_frac / 2.0

    # clamp so it stays inside the axes box
    y_frac = max(0.0, min(1.0 - h_frac, y_frac))

    # --- Horizontal placement: fixed offset from right edge in inches ---
    x_frac = 1.0 + (x_offset_in / ax_w_in)

    # --- Create inset axes for the colorbar ---
    coverage_ax = ax.inset_axes([x_frac, y_frac, w_frac, h_frac])

    coverage_cbar = plt.colorbar(
        ScalarMappable(norm=coverage_norm, cmap="Greys"),
        cax=coverage_ax,
        orientation="vertical",
    )

    # Ticks/labels tuned for vertical
    coverage_cbar.ax.tick_params(labelsize=7, pad=2, length=2, width=0.6)
    coverage_cbar.set_ticks([0, 10])
    coverage_cbar.set_label("Coverage", fontsize=9, labelpad=4, rotation=90)

    # --- Add raw vs smoothed line legend ---
    handles = [
        Line2D([0], [0], color="black", lw=2, alpha=0.25, label="Raw"),
        Line2D([0], [0], color="orange", lw=2, alpha=0.75, label="LOWESS"),
    ]
    ax.legend(
        handles=handles,
        title="",
        fontsize=11,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(x_frac, line_legend_y),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.3,
    )


# ---------- BedTable helpers ----------

def _bt_filter_chrom(bt: Optional[BedTable], chrom: str) -> Optional[BedTable]:
    if bt is None:
        return None
    recs = [r for r in bt._records if r.chrom == chrom]
    if not recs:
        return BedTable([], inferred_kind=bt.inferred_kind, inferred_ncols=bt.inferred_ncols)
    return BedTable(recs, inferred_kind=bt.inferred_kind, inferred_ncols=bt.inferred_ncols)


def _bt_positions(bt: BedTable, *, x_mode: str = "start") -> np.ndarray:
    if x_mode == "start":
        return np.asarray([r.start for r in bt._records], dtype=int)
    if x_mode == "midpoint":
        return np.asarray([(r.start + r.end) // 2 for r in bt._records], dtype=int)
    raise ValueError("x_mode must be 'start' or 'midpoint'")


def _bt_get_col_1based(bt: BedTable, col_1based: int, *, as_float: bool = True) -> np.ndarray:
    """
    Uses BedTable.get_column_1based if you have it; otherwise falls back to extras access.
    """
    if hasattr(bt, "get_column_1based"):
        vals = bt.get_column_1based(col_1based, as_float=as_float)
        return np.asarray(vals, dtype=float if as_float else object)

    # Fallback: columns 1..6 are BED; 7+ are extras
    idx0 = col_1based - 1
    out = []
    for r in bt._records:
        if idx0 == 0:
            out.append(r.chrom)
        elif idx0 == 1:
            out.append(r.start)
        elif idx0 == 2:
            out.append(r.end)
        elif idx0 == 3:
            out.append(r.name)
        elif idx0 == 4:
            out.append(r.score)
        elif idx0 == 5:
            out.append(r.strand)
        else:
            ex_i = idx0 - 6
            if r.extras is None or ex_i >= len(r.extras):
                out.append(np.nan if as_float else None)
            else:
                out.append(r.extras[ex_i])
    return np.asarray(out, dtype=float if as_float else object)


def _lowess_smoothed_from_bedgraph(lowess_bg: BedTable) -> Tuple[np.ndarray, np.ndarray]:
    """
    lowess_bg records have extras=(smoothedY, dY).
    Returns:
      x_positions (start)
      smoothedY
    """
    xs = np.asarray([r.start for r in lowess_bg._records], dtype=int)
    ys = []
    for r in lowess_bg._records:
        if r.extras is None or len(r.extras) < 1:
            ys.append(np.nan)
        else:
            ys.append(float(r.extras[0]))
    return xs, np.asarray(ys, dtype=float)


def _dips_to_starts_ends(dips: Optional[BedTable]) -> Tuple[List[int], List[int]]:
    if dips is None:
        return [], []
    starts = [int(r.start) for r in dips._records]
    ends = [int(r.end) for r in dips._records]
    return starts, ends


# ---------- Main plotting entry ----------
def centrodipSummaryPlot_bedtable(
    bedMethyl: BedTable,
    regions: BedTable,
    lowess_bg: BedTable,
    dips_final: Optional[BedTable],
    dips_unfiltered: Optional[BedTable],
    output_path: Path | str,
    *,
    cov_col_1based: int = 10,
    frac_mod_col_1based: int = 11,
    x_mode: str = "start",
    panel_height: float = 5.5,
    bottom_margin_in: float = 0.5,
    left_margin_in: float = 1.5,
    right_margin_in: float = 2,
    bp_per_inch: float = 250_000.0,   # 250kb per inch
) -> Path:
    """
    ...
    """

    chroms = sorted({r.chrom for r in bedMethyl._records})
    if not chroms:
        raise ValueError("bedMethyl has no records.")

    # ----- Determine span (in bp) for THIS plot -----
    # In your usage, bedMethyl is already per-chrom/per-region for this plot.
    if x_mode == "start":
        all_pos = np.asarray([r.start for r in bedMethyl._records], dtype=int)
    else:
        all_pos = np.asarray([(r.start + r.end) // 2 for r in bedMethyl._records], dtype=int)

    x_min_global = int(np.min(all_pos))
    x_max_global = int(np.max(all_pos))
    span_bp = max(1, x_max_global - x_min_global)

    # ----- Convert span to *panel* width in inches -----
    panel_width_in = span_bp / float(bp_per_inch)

    # ----- Total figure width includes fixed margins -----
    fig_width_in = left_margin_in + panel_width_in + right_margin_in
    fig_height_in = panel_height + bottom_margin_in

    fig, axes = plt.subplots(
        nrows=1, ncols=1,
        figsize=(fig_width_in, fig_height_in),
        squeeze=False,
        sharex=False,
    )

    # ----- Enforce 1 inch margins by placing axes explicitly -----
    # Fractions of figure width:
    left_frac = left_margin_in / fig_width_in
    right_frac = 1.0 - (right_margin_in / fig_width_in)

    # Vertical margins (still as fractions; keep modest and stable)
    bottom_frac = bottom_margin_in / fig_height_in
    top_frac = 0.95

    coverage_norm = Normalize(vmin=0, vmax=10, clip=True)

    for axis_row, chrom in zip(axes, chroms):
        ax = axis_row[0]

        bm_chr = _bt_filter_chrom(bedMethyl, chrom)
        rg_chr = _bt_filter_chrom(regions, chrom) if regions is not None else None
        lw_chr = _bt_filter_chrom(lowess_bg, chrom)
        df_chr = _bt_filter_chrom(dips_final, chrom) if dips_final is not None else None
        du_chr = _bt_filter_chrom(dips_unfiltered, chrom) if dips_unfiltered is not None else None

        if bm_chr is None or len(bm_chr._records) == 0:
            ax.set_axis_off()
            continue

        cpg_pos = _bt_positions(bm_chr, x_mode=x_mode)
        cpg_coverage = _bt_get_col_1based(bm_chr, cov_col_1based, as_float=True)
        frac_mod = _bt_get_col_1based(bm_chr, frac_mod_col_1based, as_float=True)

        # LOWESS line from bedGraph-like output
        lowess_x, lowess_y = (np.asarray([]), np.asarray([]))
        if lw_chr is not None and len(lw_chr._records) > 0:
            lowess_x, lowess_y = _lowess_smoothed_from_bedgraph(lw_chr)

        x_min = int(np.min(cpg_pos))
        x_max = int(np.max(cpg_pos))
        if x_min == x_max:
            x_min -= 10_000
            x_max += 10_000

        # coverage band
        _plot_coverage_band(
            ax=ax,
            x=cpg_pos,
            val=cpg_coverage,
            y_bottom=1.0,
            y_top=1.5,
            norm=coverage_norm,
        )

        # raw frac/percent modified
        _plot_fraction_line(
            ax=ax,
            xpos=cpg_pos,
            val=frac_mod,
            alpha=0.25,
            color="black",
        )

        # smoothed line
        if lowess_x.size and lowess_y.size:
            _plot_fraction_line(
                ax=ax,
                xpos=lowess_x,
                val=lowess_y,
                alpha=0.75,
                color="orange",
            )

        # unfiltered dips
        u_starts, u_ends = _dips_to_starts_ends(du_chr)
        _plot_regions(
            ax=ax,
            regions=du_chr,
            y_bottom=3.6,
            height=0.35,
            color="black",
            alpha=0.25,
            zorder=3,
        )

        # final dips
        f_starts, f_ends = _dips_to_starts_ends(df_chr)
        _plot_regions(
            ax=ax,
            regions=df_chr,
            y_bottom=3.0,
            height=0.5,
            color="black",
            alpha=1,
            zorder=3,
        )

        # regions track (BED intervals)
        _plot_regions(
            ax=ax,
            regions=rg_chr,
            y_bottom=0.5,
            height=0.18,
            color=(153/255,0/255,0/255),
            alpha=1,
            zorder=1,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 4.2)
        ax.set_yticks([0.59, 1.25, 2.25, 3.25, 3.775])
        ax.set_yticklabels([
            "Regions",
            "Coverage",
            "Fraction\nModified",
            "Filtered dips",
            "Unfiltered dips",
        ])

        # xticks are each even 500kb in range xmin,xmax
        span = x_max - x_min
        if span <= 1_000_000:
            step = 100_000
        elif span <= 5_000_000:
            step = 500_000
        elif span <= 20_000_000:
            step = 1_000_000
        else:
            step = 5_000_000

        tick_start = int(math.ceil(x_min / step) * step)   # first tick >= x_min
        tick_end   = int(math.floor(x_max / step) * step)  # last tick  <= x_max

        if tick_end >= tick_start:
            ax.set_xticks(np.arange(tick_start, tick_end + 1, step))
        else:
            ax.set_xticks([x_min, x_max])  # fallback for tiny windows

        xticks = np.arange(tick_start, tick_end + 1, step)
        xticks_labs = [f"{x / 1_000_000:.2f} Mb" for x in xticks] # float 1Mb but with 2 decimal places

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labs, fontsize=8)

        ax.set_ylabel(f"{chrom}")
        ax.tick_params(axis="y", which="both", length=0)
        ax.grid(False)
        _add_track_legends(ax, coverage_norm)

        ax.axhline(y=1.75, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(y=2.75, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

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
    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


__all__ = ["centrodipSummaryPlot_bedtable"]
