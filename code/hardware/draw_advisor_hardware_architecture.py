"""Render the advisor-facing final hardware dataflow figure.

The figure is intentionally self-contained and uses only Matplotlib so the SVG,
PDF, and PNG exports can be regenerated on the project server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BG = "#F7F9FC"
INK = "#152238"
MUTED = "#53657A"
LINE = "#253B57"
SPARSE = "#DDF3E8"
SPARSE_EDGE = "#23825A"
REGULAR = "#DFEAFA"
REGULAR_EDGE = "#356FB5"
HEAD = "#EEE5FA"
HEAD_EDGE = "#7551A6"
MEMORY = "#FFF0CF"
MEMORY_EDGE = "#B7791F"
OUTPUT = "#FDE3DF"
OUTPUT_EDGE = "#C65146"
NEUTRAL = "#EDF1F5"
NEUTRAL_EDGE = "#68798D"
WHITE = "#FFFFFF"


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9.0,
    weight: str = "normal",
    radius: float = 0.12,
    linewidth: float = 1.5,
    text_color: str = INK,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
        zorder=zorder + 1,
    )
    return patch


def group_box(ax, x, y, w, h, title, edge, face):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.16",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
        linestyle=(0, (5, 3)),
        alpha=0.48,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.18,
        y + h - 0.20,
        title,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=edge,
        zorder=2,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=LINE,
    width=1.55,
    style="-|>",
    mutation=13,
    connection="arc3,rad=0",
    zorder=5,
    dashed=False,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        linestyle="--" if dashed else "-",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, text, face, edge, fontsize=7.7):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=edge,
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.45",
            "facecolor": face,
            "edgecolor": edge,
            "linewidth": 0.9,
        },
        zorder=8,
    )


def render(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": BG,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=False)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.42,
        8.64,
        "EBV-Eye SNN Accelerator",
        fontsize=21,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    ax.text(
        0.43,
        8.28,
        "Event-driven L1/L2 + per-layer adjacent t0/t1 execution",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="center",
    )
    pill(ax, 13.36, 8.60, "400 events", NEUTRAL, NEUTRAL_EDGE)
    pill(ax, 14.27, 8.60, "T = 2", NEUTRAL, NEUTRAL_EDGE)
    pill(ax, 15.13, 8.60, "S16 / S24", NEUTRAL, NEUTRAL_EDGE)

    # Panel A: complete accelerator.
    ax.text(0.43, 7.86, "A", fontsize=14, fontweight="bold", color=WHITE,
            ha="center", va="center",
            bbox={"boxstyle": "circle,pad=0.24", "facecolor": INK, "edgecolor": INK})
    ax.text(0.82, 7.86, "End-to-end accelerator", fontsize=12.2,
            fontweight="bold", color=INK, ha="left", va="center")

    group_box(ax, 2.10, 4.78, 4.45, 2.58, "EVENT-DRIVEN FRONT END", SPARSE_EDGE, SPARSE)
    group_box(ax, 6.75, 4.78, 4.22, 2.58, "REGULAR SHARED BACK END", REGULAR_EDGE, REGULAR)
    group_box(ax, 11.16, 4.78, 2.55, 2.58, "DENSE HEAD", HEAD_EDGE, HEAD)

    rounded_box(
        ax, 0.42, 5.28, 1.42, 1.45,
        "Sorted event\nstream\n\n(pol, row, col)",
        face=NEUTRAL, edge=NEUTRAL_EDGE, fontsize=9.1, weight="bold",
    )
    ax.text(1.13, 5.08, "400 unit events", fontsize=7.5, color=MUTED,
            ha="center", va="top")

    rounded_box(
        ax, 2.36, 5.18, 1.72, 1.63,
        "L1 event scatter\n\n72 LUT add lanes\n3 x 3 x 8",
        face=WHITE, edge=SPARSE_EDGE, fontsize=8.8, weight="bold",
    )
    rounded_box(
        ax, 4.55, 5.18, 1.72, 1.63,
        "L2 count-MAC\n\n48 DSP lanes\n3 taps x 16 OC",
        face=WHITE, edge=SPARSE_EDGE, fontsize=8.8, weight="bold",
    )
    ax.text(3.22, 4.96, "3-row / 9-bank S40 psum", fontsize=7.4,
            color=SPARSE_EDGE, ha="center", va="top")
    ax.text(5.41, 4.96, "Pool -> bias -> clamp -> LIF", fontsize=7.4,
            color=SPARSE_EDGE, ha="center", va="top")

    rounded_box(
        ax, 7.03, 5.18, 2.18, 1.63,
        "L3-L8 shared Conv\n\n288 DSP lanes\n8 OC x 4 IC x 3 x 3",
        face=WHITE, edge=REGULAR_EDGE, fontsize=8.7, weight="bold",
    )
    rounded_box(
        ax, 9.48, 5.18, 1.20, 1.63,
        "Local\nbuffers\n\n2 psums\nS16 state",
        face=WHITE, edge=REGULAR_EDGE, fontsize=8.0, weight="bold",
    )
    ax.text(8.12, 4.96, "one physical core, reused by all layers", fontsize=7.4,
            color=REGULAR_EDGE, ha="center", va="top")
    ax.text(10.08, 4.96, "L3 I tile: 4.5 KiB", fontsize=7.4,
            color=REGULAR_EDGE, ha="center", va="top")

    rounded_box(
        ax, 11.42, 5.18, 1.20, 1.63,
        "H1 / H2\n\npaired\n16 DSP",
        face=WHITE, edge=HEAD_EDGE, fontsize=8.6, weight="bold",
    )
    rounded_box(
        ax, 12.82, 5.18, 0.66, 1.63,
        "Read-\nout\n\n2\nDSP",
        face=WHITE, edge=HEAD_EDGE, fontsize=8.2, weight="bold",
    )
    rounded_box(
        ax, 14.02, 5.28, 1.48, 1.45,
        "Continuous\ngaze output\n\n(x, y)",
        face=OUTPUT, edge=OUTPUT_EDGE, fontsize=9.1, weight="bold",
    )

    arrow(ax, (1.84, 6.00), (2.36, 6.00))
    arrow(ax, (4.08, 6.00), (4.55, 6.00))
    arrow(ax, (6.27, 6.00), (7.03, 6.00))
    arrow(ax, (9.21, 6.00), (9.48, 6.00))
    arrow(ax, (10.68, 6.00), (11.42, 6.00))
    arrow(ax, (12.62, 6.00), (12.82, 6.00))
    arrow(ax, (13.48, 6.00), (14.02, 6.00))

    rounded_box(
        ax, 5.62, 7.46, 4.43, 0.47,
        "On-chip weight BRAM  ->  256-bit tile loader  ->  PE-local registers",
        face=MEMORY, edge=MEMORY_EDGE, fontsize=8.6, weight="bold", radius=0.09,
    )
    for x_to in (3.22, 5.41, 8.12, 12.04, 13.15):
        arrow(ax, (7.84, 7.46), (x_to, 6.84), color=MEMORY_EDGE,
              width=1.05, mutation=10, dashed=True,
              connection="arc3,rad=0.05" if x_to < 7.84 else "arc3,rad=-0.05",
              zorder=2)

    rounded_box(
        ax, 5.62, 4.22, 4.43, 0.38,
        "Layer controller / address generator / ready-valid scheduler",
        face=NEUTRAL, edge=NEUTRAL_EDGE, fontsize=8.0, weight="bold", radius=0.08,
    )
    arrow(ax, (7.84, 4.60), (7.84, 5.17), color=NEUTRAL_EDGE, width=1.05,
          mutation=10, dashed=True)

    # Panel divider.
    ax.plot([0.42, 15.55], [4.02, 4.02], color="#CDD6E1", linewidth=1.1)

    # Panel B: schedule detail.
    ax.text(0.43, 3.70, "B", fontsize=14, fontweight="bold", color=WHITE,
            ha="center", va="center",
            bbox={"boxstyle": "circle,pad=0.24", "facecolor": INK, "edgecolor": INK})
    ax.text(0.82, 3.70, "Deep-layer schedule: keep weights and state local across t0 / t1",
            fontsize=12.2, fontweight="bold", color=INK, ha="left", va="center")

    group_box(ax, 0.42, 0.78, 6.55, 2.55, "L3 SPECIAL CASE: IDENTICAL INPUT AT BOTH STEPS",
              REGULAR_EDGE, REGULAR)
    rounded_box(ax, 0.70, 1.42, 1.05, 1.02, "L2 spike\nplane",
                face=WHITE, edge=REGULAR_EDGE, fontsize=8.4, weight="bold")
    rounded_box(ax, 2.00, 1.42, 1.25, 1.02, "Conv L3\nonce",
                face=WHITE, edge=REGULAR_EDGE, fontsize=8.8, weight="bold")
    rounded_box(ax, 3.50, 1.42, 1.25, 1.02, "S24 current\ntile 4.5 KiB",
                face=MEMORY, edge=MEMORY_EDGE, fontsize=8.1, weight="bold")
    rounded_box(ax, 5.06, 2.12, 1.28, 0.66, "LIF t0\nwrite state",
                face=WHITE, edge=REGULAR_EDGE, fontsize=8.1, weight="bold")
    rounded_box(ax, 5.06, 1.08, 1.28, 0.66, "LIF t1\nread state",
                face=WHITE, edge=REGULAR_EDGE, fontsize=8.1, weight="bold")
    arrow(ax, (1.75, 1.93), (2.00, 1.93), color=REGULAR_EDGE)
    arrow(ax, (3.25, 1.93), (3.50, 1.93), color=REGULAR_EDGE)
    arrow(ax, (4.75, 2.05), (5.06, 2.43), color=REGULAR_EDGE,
          connection="arc3,rad=-0.12")
    arrow(ax, (4.75, 1.80), (5.06, 1.41), color=REGULAR_EDGE,
          connection="arc3,rad=0.12")
    arrow(ax, (5.70, 2.12), (5.70, 1.74), color=REGULAR_EDGE, width=1.2,
          mutation=10)
    ax.text(3.55, 0.96, "same current is replayed; no second L3 convolution",
            fontsize=7.5, color=REGULAR_EDGE, ha="center", va="center")

    group_box(ax, 7.20, 0.78, 8.35, 2.55,
              "L4-L8 AND H1/H2: ONE WEIGHT LOAD SERVES BOTH STEPS",
              HEAD_EDGE, HEAD)
    rounded_box(ax, 7.53, 1.42, 1.54, 1.02, "Load weight\ntile once",
                face=MEMORY, edge=MEMORY_EDGE, fontsize=8.5, weight="bold")
    rounded_box(ax, 9.42, 1.42, 1.32, 1.02, "Scan t0\npsum bank 0",
                face=WHITE, edge=HEAD_EDGE, fontsize=8.3, weight="bold")
    rounded_box(ax, 11.09, 1.42, 1.32, 1.02, "Scan t1\npsum bank 1",
                face=WHITE, edge=HEAD_EDGE, fontsize=8.3, weight="bold")
    rounded_box(ax, 12.76, 1.42, 1.26, 1.02, "LIF / state\nstays local",
                face=WHITE, edge=HEAD_EDGE, fontsize=8.2, weight="bold")
    rounded_box(ax, 14.37, 1.42, 0.87, 1.02, "Next\ntile",
                face=NEUTRAL, edge=NEUTRAL_EDGE, fontsize=8.4, weight="bold")
    arrow(ax, (9.07, 1.93), (9.42, 1.93), color=HEAD_EDGE)
    arrow(ax, (10.74, 1.93), (11.09, 1.93), color=HEAD_EDGE)
    arrow(ax, (12.41, 1.93), (12.76, 1.93), color=HEAD_EDGE)
    arrow(ax, (14.02, 1.93), (14.37, 1.93), color=HEAD_EDGE)
    ax.text(11.38, 1.01,
            "t0/t1 are adjacent within each tile; layers and samples remain sequential",
            fontsize=7.5, color=HEAD_EDGE, ha="center", va="center")

    # Footer evidence strip.
    rounded_box(
        ax, 0.42, 0.14, 15.13, 0.42,
        "RTL evidence  |  7.911M scheduled lanes  |  67,942.9 cycles/sample  |  "
        "345,080,448 checked values, 0 mismatch  |  Yosys: 354 DSP + 80.5 BRAM36-equivalent",
        face=INK, edge=INK, fontsize=8.5, weight="bold", radius=0.08,
        text_color=WHITE,
    )

    stem = "ebv_eye_event_driven_layer_adjacent_architecture"
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.10)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.10)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.10, dpi=220)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/hardware/figures"),
        help="Directory for SVG, PDF, and PNG exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in render(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
