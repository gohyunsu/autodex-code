#!/usr/bin/env python3
"""Generate reproducible figures for the precision-insertion thesis draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SOURCE = ASSETS / "source"

NAVY = "#111c2c"
PANEL = "#172538"
TEXT = "#e8eef5"
MUTED = "#a9b7c8"
CYAN = "#18d4e8"
GREEN = "#3df59c"
ORANGE = "#ffae42"
MAGENTA = "#ef65d5"
BLUE = "#78a8ff"
RED = "#ff6275"


def setup() -> None:
    korean_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(korean_font)
    family = font_manager.FontProperties(fname=korean_font).get_name()
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 4.2), dpi=180)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        ("Session", "socket pose\n1회 측정"),
        ("Perception", "key pose 및\ntabletop pose"),
        ("Grasp", "고정 후보\ntable/0/20195"),
        ("Transfer", "lift 확인 후\npre-insertion"),
        ("Observe", "3-view key-tip\n추정"),
        ("Align", "bounded XY\nJacobian 이동"),
        ("Insert", "축 방향 20 mm\nrelease + label"),
    ]
    colors = ["#354a67", "#2e5670", "#3f526c", "#445064",
              "#225f6b", "#246752", "#305c48"]
    xs = np.linspace(0.25, 13.3, len(stages))
    for i, ((title, body), x, color) in enumerate(zip(stages, xs, colors)):
        box = FancyBboxPatch((x, 1.10), 1.75, 1.95,
                             boxstyle="round,pad=0.06,rounding_size=0.16",
                             linewidth=1.4, edgecolor="#d5deea",
                             facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.875, 2.52, title, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(x + 0.875, 1.78, body, ha="center", va="center",
                color="#eef4f8", fontsize=10.5, linespacing=1.35)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.78, 2.08),
                                         (xs[i + 1] - 0.04, 2.08),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color="#708198", linewidth=1.8))
    # Closed-loop feedback arrow.
    ax.add_patch(FancyArrowPatch((11.55, 1.00), (9.25, 1.00),
                                 connectionstyle="arc3,rad=-0.35",
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=ORANGE, linewidth=2.4))
    ax.text(10.4, 0.25, "재촬영 · 재추정 · 같은 시행 내 반복",
            ha="center", color="#a85f00", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(ASSETS / "pipeline_overview.png", bbox_inches="tight")
    plt.close(fig)


def error_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.1), dpi=180)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    boxes = [
        (0.4, 2.8, 3.0, 1.35, "Wrist pose error", "#41536a"),
        (0.4, 0.8, 3.0, 1.35, "Object-in-hand error", "#5b4962"),
        (5.0, 1.8, 3.35, 1.55, "Observed key pose", "#2f6570"),
        (9.85, 1.8, 3.15, 1.55, "Insertion residual", "#35664f"),
    ]
    for x, y, w, h, title, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.07,rounding_size=0.12",
                                    facecolor=color, edgecolor="#d8e1eb",
                                    linewidth=1.5))
        ax.text(x+w/2, y+h*0.50, title, color="white", ha="center",
                va="center", fontsize=13, fontweight="bold")
    for start in [(3.4, 3.48), (3.4, 1.48)]:
        ax.add_patch(FancyArrowPatch(start, (4.95, 2.58), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=2,
                                     color="#6c7d92"))
    ax.add_patch(FancyArrowPatch((8.37, 2.58), (9.8, 2.58), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=2,
                                 color="#6c7d92"))
    ax.text(6.7, 4.55,
            r"$T^{B}_{K}=T^{B}_{H}(q)\,T^{H}_{K}$",
            ha="center", fontsize=19, color="#1f2d3d")
    fig.tight_layout(pad=0.2)
    fig.savefig(ASSETS / "error_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def convergence() -> None:
    iterations = np.arange(6)
    error = np.asarray([76.3271, 14.5884, 11.0297, 7.9111, 4.9049, 1.6065])
    cmd = np.asarray([3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 0.0000])
    measured = np.asarray([3.0704, 2.9681, 2.9703, 2.9938, 2.9307, 0.7102])

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=180)
    ax.plot(iterations, error, color=BLUE, linewidth=2.5, marker="o", zorder=2)
    ax.scatter([0], [error[0]], s=90, color=RED, zorder=3)
    ax.plot(iterations[1:], error[1:], color=GREEN, linewidth=3,
            marker="o", markersize=7, zorder=4,
            )
    for x, y in zip(iterations, error):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(2.5, color="#58687a", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Closed-loop iteration")
    ax.set_ylabel("Estimated XY residual (mm)")
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(ASSETS / "xy_residual.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    width = 0.36
    ax.bar(iterations - width/2, cmd, width, color=ORANGE)
    ax.bar(iterations + width/2, measured, width, color=CYAN)
    ax.set_xlabel("Closed-loop iteration")
    ax.set_ylabel("XY motion norm (mm)")
    ax.set_xticks(iterations)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(ASSETS / "xy_motion.png", bbox_inches="tight")
    plt.close(fig)


def split_camera_evidence() -> None:
    """Separate camera evidence from the presentation-style composite plots."""
    panel_x = {"25322639": 277, "25322642": 741, "26053248": 1205}
    for iteration in (1, 5):
        source = Image.open(SOURCE / f"closed_loop_iteration_{iteration:02d}.png")
        for camera, x0 in panel_x.items():
            # The top 52 px only contain the presentation renderer's status
            # badge. Camera IDs and used/rejected semantics belong in the
            # report caption, not in the evidence image itself.
            source.crop((x0, 222, x0 + 438, 608)).save(
                ASSETS / f"iteration_{iteration:02d}_{camera}_overlay.png")

    final = Image.open(SOURCE / "insertion_result.png")
    # The final clear-view composite uses 438x326 camera panels at y=226.
    for camera, x0 in panel_x.items():
        final.crop((x0, 226, x0 + 438, 552)).save(
            ASSETS / f"final_{camera}.png")


def timing() -> None:
    names = [
        "Pickup / squeeze", "Lift", "Post-lift settle", "Transfer",
        "6× visual estimate", "5× XY motion", "Axial replan",
        "20 mm insertion", "Release + clear view",
    ]
    seconds = [6.11, 6.80, 2.00, 3.50, 9.29, 6.71, 1.25, 3.74, 6.98]
    colors = ["#556b83", "#556b83", "#6f7d8f", "#556b83",
              "#2f7180", "#3d7f69", "#2f7180", "#3d7f69", "#556b83"]
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=180)
    y = np.arange(len(names))
    bars = ax.barh(y, seconds, color=colors)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Inclusive wall time (s)")
    ax.grid(True, axis="x", alpha=0.2)
    for bar, value in zip(bars, seconds):
        ax.text(value + 0.16, bar.get_y() + bar.get_height()/2,
                f"{value:.2f}", va="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "success_timing.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    setup()
    pipeline_overview()
    error_decomposition()
    convergence()
    split_camera_evidence()
    timing()


if __name__ == "__main__":
    main()
