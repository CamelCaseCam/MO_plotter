#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

import matplotlib.pyplot as plt


def truthy(val: str) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


@dataclass(frozen=True)
class Level:
    energy: float
    is_homo: bool
    is_diene: bool
    id_tag: Optional[str]  # None means no ID


def read_levels_csv(path: str) -> List[Level]:
    levels: List[Level] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Energy", "Is_HOMO", "Is_diene", "ID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=2):
            try:
                energy = float(row["Energy"])
            except Exception as e:
                raise ValueError(f"Bad Energy value on line {i}: {row.get('Energy')!r}") from e

            is_homo = truthy(row.get("Is_HOMO", ""))
            is_diene = truthy(row.get("Is_diene", ""))

            raw_id = (row.get("ID") or "").strip()
            id_tag = None if raw_id in {"---", ""} else raw_id

            levels.append(Level(energy=energy, is_homo=is_homo, is_diene=is_diene, id_tag=id_tag))
    return levels


def _shift_text_display_pixels(ax, text, dy_pixels: float) -> None:
    """Shift a Text object vertically by dy_pixels in display coords, preserving x in data coords."""
    x_data, y_data = text.get_position()
    x_disp, y_disp = ax.transData.transform((x_data, y_data))
    x2_data, y2_data = ax.transData.inverted().transform((x_disp, y_disp + dy_pixels))
    text.set_position((x_data, y2_data))


def _bboxes_overlap(a, b) -> bool:
    return (
        a.x0 < b.x1 and a.x1 > b.x0 and
        a.y0 < b.y1 and a.y1 > b.y0
    )


def relax_text_y(
    fig,
    ax,
    texts: Sequence,
    step_px: float = 6.0,
    max_iter: int = 300,
) -> None:
    """
    Simple label-repulsion: iteratively nudge overlapping text boxes up/down in pixel space.
    Works well for small N (like ID letters).
    """
    if len(texts) < 2:
        return

    # Need a draw so extents are valid.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bboxes = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.05) for t in texts]
        moved = False

        # Accumulate per-text motion each iteration to avoid order bias.
        dy = [0.0 for _ in texts]

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not _bboxes_overlap(bboxes[i], bboxes[j]):
                    continue

                # Push apart based on vertical ordering in display space
                ci = 0.5 * (bboxes[i].y0 + bboxes[i].y1)
                cj = 0.5 * (bboxes[j].y0 + bboxes[j].y1)

                if ci <= cj:
                    dy[i] -= step_px / 2.0
                    dy[j] += step_px / 2.0
                else:
                    dy[i] += step_px / 2.0
                    dy[j] -= step_px / 2.0

                moved = True

        if not moved:
            break

        for t, d in zip(texts, dy):
            if d != 0.0:
                _shift_text_display_pixels(ax, t, d)


def plot_diagram(
    levels: List[Level],
    out_path: str,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.2, 5.0),
    show_ids: bool = False,
    level_halfwidth = 0.22,
    level_height=2.6
) -> None:
    x_left = -1.0
    x_right = +1.0

    colors = {
        ("diene", "HOMO"): "#d30f0f",
        ("diene", "LUMO"): "#ffd000",
        ("dienophile", "HOMO"): "#0400ff",
        ("dienophile", "LUMO"): "#11b111",
    }
    rail_colors = {"diene": "#d62728", "dienophile": "#2ca02c"}

    energies = [lv.energy for lv in levels]
    if not energies:
        raise ValueError("No rows found in the CSV.")

    y_min = min(energies)
    y_max = max(energies)
    pad = max(0.06, 0.12 * (y_max - y_min if y_max > y_min else 1.0))
    y0 = y_min - pad
    y1 = y_max + pad

    fig, ax = plt.subplots(figsize=figsize)

    # Rails
    ax.plot([x_left, x_left], [y0, y1], lw=2.5, color=rail_colors["diene"])
    ax.plot([x_right, x_right], [y0, y1], lw=2.5, color=rail_colors["dienophile"])

    texts_left = []
    texts_right = []

    # Levels (+ optional per-level ID labels)
    for lv in sorted(levels, key=lambda t: t.energy, reverse=True):
        side = "diene" if lv.is_diene else "dienophile"
        kind = "HOMO" if lv.is_homo else "LUMO"
        color = colors[(side, kind)]

        xc = x_left if side == "diene" else x_right
        x_start = xc - level_halfwidth
        x_end = xc + level_halfwidth
        ax.plot([x_start, x_end], [lv.energy, lv.energy], lw=level_height, color=color, solid_capstyle="butt")

        if show_ids and lv.id_tag is not None:
            # Put ID just inside the diagram (toward the center)
            x_text = x_end + 0.06 if side == "diene" else x_start - 0.06
            ha = "left" if side == "diene" else "right"
            t = ax.text(
                x_text, lv.energy, lv.id_tag,
                ha=ha, va="center",
                fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )
            (texts_left if side == "diene" else texts_right).append(t)

    # De-overlap ID labels (per side)
    if show_ids:
        relax_text_y(fig, ax, texts_left, step_px=7.0, max_iter=400)
        relax_text_y(fig, ax, texts_right, step_px=7.0, max_iter=400)

    # Column labels
    ax.text(x_left, y0 - 0.06 * (y1 - y0), "dienes", ha="center", va="top", fontsize=12)
    ax.text(x_right, y0 - 0.06 * (y1 - y0), "dienophiles", ha="center", va="top", fontsize=12)

    # HOMO/LUMO labels
    homos = [lv.energy for lv in levels if lv.is_homo]
    lumos = [lv.energy for lv in levels if not lv.is_homo]
    if lumos:
        yl = sorted(lumos)[len(lumos) // 2]
        ax.text(0.0, min(y1 - 0.02 * (y1 - y0), yl + 0.12 * (y1 - y0)),
                "LUMO", color="#1f77b4", ha="center", va="bottom", fontsize=20, fontweight="bold")
    if homos:
        yh = sorted(homos)[len(homos) // 2]
        ax.text(0.0, max(y0 + 0.02 * (y1 - y0), yh - 0.12 * (y1 - y0)),
                "HOMO", color="#d62728", ha="center", va="top", fontsize=20, fontweight="bold")

    # Axes styling
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])

    # One axis, mirrored ticks on both sides => always same scale
    ax.set_ylabel("Energy (Hartree)")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both", right=True, labelright=True, direction="out")

    # Make sure the right spine is visible (since we're not using twinx anymore)
    ax.spines["right"].set_visible(True)

    # Right-side y-label (manual, since matplotlib only gives the axis label on the left for a single Axes)
    ax.text(
        1.06, 0.5, "Energy (Hartree)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left",
    )

    # Spines cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)



    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a two-column HOMO/LUMO diagram from CSV.")
    ap.add_argument("csv_path", help="Input CSV with columns: Energy,Is_HOMO,Is_diene,ID")
    ap.add_argument("-o", "--out", default="diagram.png", help="Output image path (png/pdf/svg)")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI (for raster formats)")
    ap.add_argument("--w", type=float, default=6.2, help="Figure width (inches)")
    ap.add_argument("--h", type=float, default=5.0, help="Figure height (inches)")
    ap.add_argument("--lw", type=float, default=0.44, help="Level marker width")
    ap.add_argument("--lh", type=float, default=2.6, help="Level marker height")
    ap.add_argument("--show-ids", action="store_true",
                    help="Show ID letters next to each level (no connectors).")
    args = ap.parse_args()

    levels = read_levels_csv(args.csv_path)
    plot_diagram(
        levels,
        out_path=args.out,
        dpi=args.dpi,
        figsize=(args.w, args.h),
        show_ids=args.show_ids,
        level_halfwidth=args.lw/2,
        level_height=args.lh
    )


if __name__ == "__main__":
    main()
