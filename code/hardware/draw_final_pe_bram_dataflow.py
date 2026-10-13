"""Render the detailed PE and on-chip-memory dataflow of the final RTL.

The overview figure in ``draw_advisor_hardware_architecture.py`` explains the
algorithm-level schedule.  This companion figure exposes the processing
elements, logical on-chip memories, tile loader, and paired-time data path that
are implemented by ``pursuit_sparse_layer_major_accelerator_top.sv``.

Only Matplotlib is required so SVG, PDF, and PNG exports can be regenerated on
the project server and in the standalone hardware repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BG = "#F6F8FB"
INK = "#16243A"
MUTED = "#56677A"
LINE = "#263D59"
WHITE = "#FFFFFF"
FRONT = "#DDF3E8"
FRONT_EDGE = "#23825A"
DEEP = "#DFEAFA"
DEEP_EDGE = "#356FB5"
DENSE = "#EEE5FA"
DENSE_EDGE = "#7551A6"
MEM = "#FFF0CF"
MEM_EDGE = "#B7791F"
DATA = "#EAF0F5"
DATA_EDGE = "#64778C"
CURRENT = "#FCE5D8"
CURRENT_EDGE = "#C96836"
OUTPUT = "#FDE3DF"
OUTPUT_EDGE = "#C65146"
SOFT_LINE = "#CBD5E1"


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = WHITE,
    edge: str = LINE,
    fontsize: float = 8.2,
    weight: str = "normal",
    radius: float = 0.10,
    linewidth: float = 1.35,
    color: str = INK,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.22,
        zorder=zorder + 1,
    )
    return patch


def group(ax, x, y, w, h, title, *, face, edge, fontsize=9.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.15",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.35,
        linestyle=(0, (5, 3)),
        alpha=0.48,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.16,
        y + h - 0.16,
        title,
        ha="left",
        va="top",
        fontsize=fontsize,
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
    linewidth=1.45,
    mutation=12,
    connection="arc3,rad=0",
    dashed=False,
    zorder=6,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        color=color,
        linewidth=linewidth,
        connectionstyle=connection,
        linestyle="--" if dashed else "-",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, x, y, letter, title):
    ax.text(
        x,
        y,
        letter,
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=WHITE,
        bbox={
            "boxstyle": "circle,pad=0.22",
            "facecolor": INK,
            "edgecolor": INK,
        },
        zorder=10,
    )
    ax.text(
        x + 0.43,
        y,
        title,
        ha="left",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color=INK,
        zorder=10,
    )


def pill(ax, x, y, text, *, face=DATA, edge=DATA_EDGE):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=edge,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.40",
            "facecolor": face,
            "edgecolor": edge,
            "linewidth": 0.9,
        },
        zorder=10,
    )


def render(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "ebv_eye_final_pe_bram_dataflow",
            "pdf.fonttype": 42,
            "savefig.facecolor": BG,
        }
    )

    fig, ax = plt.subplots(figsize=(18, 10.5), constrained_layout=False)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        0.42,
        10.16,
        "EBV-Eye Final RTL: PE and On-Chip-Memory Dataflow",
        ha="left",
        va="center",
        fontsize=20.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.43,
        9.82,
        "Event-driven L1/L2, one shared 288-lane deep core, and adjacent t0/t1 execution",
        ha="left",
        va="center",
        fontsize=11.2,
        color=MUTED,
    )
    pill(ax, 14.55, 10.15, "400 events")
    pill(ax, 15.55, 10.15, "T = 2")
    pill(ax, 16.50, 10.15, "S16 W")
    pill(ax, 17.35, 10.15, "S40 -> S24")

    # ------------------------------------------------------------------
    # Panel A: full accelerator and the logical on-chip memory hierarchy.
    # ------------------------------------------------------------------
    panel_label(ax, 0.44, 9.40, "A", "Complete sample flow and memory hierarchy")

    group(
        ax,
        1.85,
        5.93,
        5.08,
        2.82,
        "EVENT-DRIVEN FRONT END",
        face=FRONT,
        edge=FRONT_EDGE,
    )
    group(
        ax,
        8.20,
        5.93,
        5.66,
        2.82,
        "SHARED PAIRED-TIME BACK END",
        face=DEEP,
        edge=DEEP_EDGE,
    )
    group(
        ax,
        14.06,
        5.93,
        2.26,
        2.82,
        "DENSE HEAD",
        face=DENSE,
        edge=DENSE_EDGE,
    )

    box(
        ax,
        0.30,
        6.48,
        1.26,
        1.50,
        "Sorted unit\nevents\n\n(pol, row, col)",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=8.4,
        weight="bold",
    )
    ax.text(
        0.93,
        6.29,
        "400 records",
        ha="center",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    box(
        ax,
        2.10,
        6.57,
        1.70,
        1.43,
        "L1 event PE array\n\n72 LUT add lanes\n9 taps x 8 OC",
        face=WHITE,
        edge=FRONT_EDGE,
        fontsize=8.2,
        weight="bold",
    )
    box(
        ax,
        3.98,
        6.79,
        0.88,
        1.00,
        "Nonzero\nU8 packet\nFIFO",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=7.2,
        weight="bold",
    )
    box(
        ax,
        5.04,
        6.57,
        1.63,
        1.43,
        "L2 count-MAC PEs\n\n48 DSP lanes\n3 taps x 16 OC",
        face=WHITE,
        edge=FRONT_EDGE,
        fontsize=8.0,
        weight="bold",
    )
    box(
        ax,
        2.10,
        6.05,
        1.70,
        0.34,
        "3-row / 9-bank S40 psum ring",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.7,
        weight="bold",
        radius=0.06,
    )
    box(
        ax,
        5.04,
        6.05,
        1.63,
        0.34,
        "3-row / 9-bank S40 psum ring",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.5,
        weight="bold",
        radius=0.06,
    )
    ax.text(
        4.39,
        8.17,
        "Pool -> constant -> S24 clamp -> stateless LIF",
        ha="center",
        va="center",
        fontsize=7.2,
        color=FRONT_EDGE,
        fontweight="bold",
    )

    box(
        ax,
        7.08,
        6.46,
        0.91,
        1.64,
        "L2 full plane\n\n16 x 24 x 32\nU8\n\n8 x (2 U8)\nbanks",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.0,
        weight="bold",
    )
    ax.text(
        7.54,
        6.24,
        "L3 reads once;\nreplay current",
        ha="center",
        va="top",
        fontsize=5.9,
        color=MEM_EDGE,
    )

    box(
        ax,
        8.42,
        6.54,
        1.74,
        1.48,
        "L3-L8 shared Conv PEs\n\n8 output PEs\n36 MAC lanes / PE\n2 x S40 psum banks\n= 288 DSP",
        face=WHITE,
        edge=DEEP_EDGE,
        fontsize=7.5,
        weight="bold",
    )
    box(
        ax,
        10.38,
        6.54,
        1.55,
        1.48,
        "Pool + constant\n+ S24 clamp + LIF\n\nS16 t0 state\nL3: 4.5 KiB I tile",
        face=WHITE,
        edge=DEEP_EDGE,
        fontsize=7.2,
        weight="bold",
    )
    box(
        ax,
        12.15,
        6.54,
        1.43,
        1.48,
        "Paired U8\nfeature store\n\nt0/t1 x A/B\n4 sets x 8 banks\n576 words/bank",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.9,
        weight="bold",
    )
    ax.text(
        11.03,
        8.20,
        "one physical core reused sequentially: L3 -> L4 -> ... -> L8",
        ha="center",
        va="center",
        fontsize=6.9,
        color=DEEP_EDGE,
        fontweight="bold",
    )

    box(
        ax,
        14.30,
        6.54,
        1.10,
        1.48,
        "H1 / H2\npaired engine\n\n16 DSP",
        face=WHITE,
        edge=DENSE_EDGE,
        fontsize=7.8,
        weight="bold",
    )
    box(
        ax,
        15.56,
        6.54,
        0.52,
        1.48,
        "Read-\nout\n\n2\nDSP",
        face=WHITE,
        edge=DENSE_EDGE,
        fontsize=7.3,
        weight="bold",
    )
    box(
        ax,
        16.65,
        6.48,
        1.05,
        1.50,
        "Continuous\ngaze\n\n(x, y)",
        face=OUTPUT,
        edge=OUTPUT_EDGE,
        fontsize=8.2,
        weight="bold",
    )

    arrow(ax, (1.56, 7.23), (2.10, 7.23))
    arrow(ax, (3.80, 7.23), (3.98, 7.23), color=FRONT_EDGE)
    arrow(ax, (4.86, 7.23), (5.04, 7.23), color=FRONT_EDGE)
    arrow(ax, (6.67, 7.23), (7.08, 7.23), color=FRONT_EDGE)
    arrow(ax, (7.99, 7.23), (8.42, 7.23), color=DEEP_EDGE)
    arrow(ax, (10.16, 7.23), (10.38, 7.23), color=DEEP_EDGE)
    arrow(ax, (11.93, 7.23), (12.15, 7.23), color=DEEP_EDGE)
    arrow(ax, (13.58, 7.23), (14.30, 7.23))
    arrow(ax, (15.40, 7.23), (15.56, 7.23), color=DENSE_EDGE)
    arrow(ax, (16.08, 7.23), (16.65, 7.23))

    # The paired store receives every deep-layer spike map.  Its A/B sets are
    # read back by L4--L8; after L8 the same store is read by H1.
    arrow(
        ax,
        (12.86, 6.54),
        (9.28, 6.54),
        color=DEEP_EDGE,
        linewidth=1.15,
        mutation=10,
        connection="arc3,rad=0.18",
    )
    ax.text(
        11.08,
        6.02,
        "L4-L8: read previous A/B set; write the other set",
        ha="center",
        va="center",
        fontsize=6.4,
        color=DEEP_EDGE,
        fontweight="bold",
    )

    # Weight/config stores above the datapaths.
    box(
        ax,
        2.08,
        8.90,
        4.58,
        0.40,
        "Front-end weight/config storage: L1 144 + L2 1,152 S16 weights",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.3,
        weight="bold",
        radius=0.07,
    )
    box(
        ax,
        8.46,
        8.90,
        7.62,
        0.40,
        "On-chip weight BRAMs -> 256-bit tile loader; parameter RAM -> postprocess",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.4,
        weight="bold",
        radius=0.07,
    )
    for x0, x1 in ((3.00, 2.95), (5.75, 5.85)):
        arrow(
            ax,
            (x0, 8.90),
            (x1, 8.01),
            color=MEM_EDGE,
            linewidth=1.0,
            mutation=9,
            dashed=True,
            zorder=2,
        )
    for x0, x1 in ((10.10, 9.29), (12.30, 11.15), (14.60, 14.85), (15.25, 15.82)):
        arrow(
            ax,
            (x0, 8.90),
            (x1, 8.02),
            color=MEM_EDGE,
            linewidth=1.0,
            mutation=9,
            dashed=True,
            zorder=2,
        )

    box(
        ax,
        6.14,
        5.54,
        5.72,
        0.27,
        "Layer controller / address generator / ready-valid scheduler",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=7.2,
        weight="bold",
        radius=0.05,
    )
    arrow(
        ax,
        (9.00, 5.81),
        (9.00, 6.52),
        color=DATA_EDGE,
        linewidth=1.0,
        mutation=9,
        dashed=True,
        zorder=2,
    )

    ax.plot([0.30, 17.70], [5.32, 5.32], color=SOFT_LINE, linewidth=1.1)

    # ------------------------------------------------------------------
    # Panel B: detailed deep convolution tile.
    # ------------------------------------------------------------------
    panel_label(ax, 0.44, 5.02, "B", "One deep-convolution tile: PE lanes and local buffers")
    group(
        ax,
        0.30,
        0.82,
        10.72,
        3.78,
        "",
        face=DEEP,
        edge=DEEP_EDGE,
        fontsize=8.8,
    )

    box(
        ax,
        0.55,
        2.47,
        1.32,
        1.12,
        "L2 plane or\npaired store\n\n4 x U8 / read",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.5,
        weight="bold",
    )
    box(
        ax,
        2.13,
        2.47,
        1.46,
        1.12,
        "4-channel\n3-row buffer\n\n36-count window",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=7.4,
        weight="bold",
    )
    box(
        ax,
        4.06,
        2.17,
        2.22,
        1.72,
        "288-lane count-MAC array\n\nPE0: 36 lanes -> S40 add tree\n...\nPE7: 36 lanes -> S40 add tree",
        face=WHITE,
        edge=DEEP_EDGE,
        fontsize=7.8,
        weight="bold",
    )
    box(
        ax,
        6.72,
        2.24,
        1.32,
        1.58,
        "Dual S40\npsum memories\n\nt0 bank\nt1 bank",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.5,
        weight="bold",
    )
    box(
        ax,
        8.45,
        2.47,
        1.22,
        1.12,
        "2x2 Pool\n+ constant\n+ S24 clamp",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=7.3,
        weight="bold",
    )
    box(
        ax,
        9.91,
        2.47,
        0.84,
        1.12,
        "8-lane\nLIF array\n\n8 x U8\nto A/B",
        face=WHITE,
        edge=DEEP_EDGE,
        fontsize=6.9,
        weight="bold",
    )

    arrow(ax, (1.87, 3.03), (2.13, 3.03), color=DEEP_EDGE)
    arrow(ax, (3.59, 3.03), (4.06, 3.03), color=DEEP_EDGE)
    arrow(ax, (6.28, 3.03), (6.72, 3.03), color=DEEP_EDGE)
    arrow(ax, (8.04, 3.03), (8.45, 3.03), color=DEEP_EDGE)
    arrow(ax, (9.67, 3.03), (9.91, 3.03), color=DEEP_EDGE)

    box(
        ax,
        3.91,
        4.03,
        2.52,
        0.35,
        "288 S16 tile registers\n18 x 256-bit beats",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.5,
        weight="bold",
        radius=0.06,
    )
    arrow(
        ax,
        (5.17, 4.03),
        (5.17, 3.89),
        color=MEM_EDGE,
        linewidth=1.05,
        mutation=9,
    )
    box(
        ax,
        0.55,
        4.03,
        2.72,
        0.35,
        "256-bit weight BRAM port\nload once per tile",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.4,
        weight="bold",
        radius=0.06,
    )
    arrow(ax, (3.27, 4.20), (3.91, 4.20), color=MEM_EDGE)

    box(
        ax,
        8.18,
        1.16,
        1.78,
        0.66,
        "L3 only: 192 x 8 x S24\ncurrent tile = 4.5 KiB",
        face=CURRENT,
        edge=CURRENT_EDGE,
        fontsize=7.1,
        weight="bold",
    )
    box(
        ax,
        9.97,
        1.16,
        0.78,
        0.66,
        "Local S16\nt0 state",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=6.9,
        weight="bold",
    )
    arrow(
        ax,
        (9.06, 2.47),
        (9.06, 1.82),
        color=CURRENT_EDGE,
        linewidth=1.05,
        mutation=9,
        connection="arc3,rad=0.06",
    )
    arrow(
        ax,
        (9.96, 1.49),
        (9.97, 1.49),
        color=CURRENT_EDGE,
        linewidth=1.05,
        mutation=9,
    )
    arrow(
        ax,
        (10.36, 1.82),
        (10.36, 2.47),
        color=MEM_EDGE,
        linewidth=1.05,
        mutation=9,
    )
    ax.text(
        5.15,
        1.49,
        "L4-L8: scan t0 and t1 with the same loaded tile\n"
        "L3: compute current once, then replay it through LIF t0 -> t1",
        ha="center",
        va="center",
        fontsize=7.4,
        color=DEEP_EDGE,
        fontweight="bold",
        linespacing=1.35,
    )

    # ------------------------------------------------------------------
    # Panel C: exact tile schedule and evidence.
    # ------------------------------------------------------------------
    panel_label(ax, 11.52, 5.02, "C", "Per-tile execution order")
    group(
        ax,
        11.28,
        0.82,
        6.42,
        3.78,
        "WEIGHT / STATE LOCALITY ACROSS t0 AND t1",
        face=DENSE,
        edge=DENSE_EDGE,
        fontsize=8.8,
    )

    box(
        ax,
        11.60,
        3.02,
        1.05,
        0.86,
        "1  Load\n18 beats",
        face=MEM,
        edge=MEM_EDGE,
        fontsize=7.7,
        weight="bold",
    )
    box(
        ax,
        12.92,
        3.02,
        1.24,
        0.86,
        "2  Scan t0\nMAC + LIF\nwrite t0 spike/state",
        face=WHITE,
        edge=DENSE_EDGE,
        fontsize=6.8,
        weight="bold",
    )
    box(
        ax,
        14.44,
        3.02,
        1.24,
        0.86,
        "3  Scan t1\nMAC + LIF\nreuse t0 state",
        face=WHITE,
        edge=DENSE_EDGE,
        fontsize=6.8,
        weight="bold",
    )
    box(
        ax,
        15.96,
        3.02,
        1.10,
        0.86,
        "Next weight\ntile",
        face=DATA,
        edge=DATA_EDGE,
        fontsize=7.3,
        weight="bold",
    )
    arrow(ax, (12.65, 3.45), (12.92, 3.45), color=DENSE_EDGE)
    arrow(ax, (14.16, 3.45), (14.44, 3.45), color=DENSE_EDGE)
    arrow(ax, (15.68, 3.45), (15.96, 3.45), color=DENSE_EDGE)

    box(
        ax,
        11.62,
        1.95,
        5.74,
        0.62,
        "L3 special case: Conv once -> 4.5 KiB current tile -> LIF t0\n"
        "save state/spike -> replay current -> LIF t1",
        face=CURRENT,
        edge=CURRENT_EDGE,
        fontsize=6.9,
        weight="bold",
    )
    box(
        ax,
        11.62,
        1.10,
        5.74,
        0.56,
        "H1/H2: one 256-bit weight beat serves t0 then t1; readout uses 2 DSP",
        face=WHITE,
        edge=DENSE_EDGE,
        fontsize=7.6,
        weight="bold",
    )
    ax.text(
        14.49,
        4.13,
        "Layers and samples remain sequential; this is not a cross-layer wavefront",
        ha="center",
        va="center",
        fontsize=7.1,
        color=DENSE_EDGE,
        fontweight="bold",
    )

    # Footer: measured scope and the memory-label interpretation.
    box(
        ax,
        0.30,
        0.19,
        17.40,
        0.43,
        "Measured RTL/Yosys evidence | 354 DSP = 48 + 288 + 16 + 2 | "
        "6,060 runtime 256-bit weight reads | 67,942.9 cycles/sample | "
        "80.5 BRAM36-equivalent | 159,408 checked values, 0 mismatch",
        face=INK,
        edge=INK,
        fontsize=8.0,
        weight="bold",
        color=WHITE,
        radius=0.07,
    )
    ax.text(
        17.68,
        0.70,
        "Memory boxes are RTL logical on-chip buffers; final BRAM/LUTRAM packing requires Vivado.",
        ha="right",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )

    stem = "ebv_eye_final_pe_bram_dataflow"
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(
        outputs[0],
        bbox_inches="tight",
        pad_inches=0.10,
        metadata={"Date": None},
    )
    # Matplotlib writes path commands with trailing spaces.  Normalizing them
    # keeps ``git diff --check`` clean and makes the checked-in SVG stable.
    svg_text = outputs[0].read_text(encoding="utf-8")
    outputs[0].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        outputs[1],
        bbox_inches="tight",
        pad_inches=0.10,
        metadata={"CreationDate": None, "ModDate": None},
    )
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
