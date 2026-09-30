#!/usr/bin/env python3
"""Render a Slack-friendly overview of AutoDex motion planning internals.

The right panel reuses a real attached_container approach/lift replay. The
left panel is explicitly labeled as a conceptual view: IK branches, PRM graph
growth, trajectory optimization, and dense validation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS = (960, 540)
DEFAULT_FPS = 10
FRAMES_PER_STAGE = 18
STAGES = (
    ("IK solving", "Find valid joint configurations for the target wrist pose"),
    ("Graph search", "Build a collision-free route through configuration space"),
    ("Trajectory optimization", "Smooth and shorten the graph seed under constraints"),
    ("Final validation", "Interpolate, retime, and check the exact dense trajectory"),
)

BG = "#F6F8FB"
INK = "#172033"
MUTED = "#667085"
BLUE = "#2878D0"
LIGHT_BLUE = "#A8D4FF"
GREEN = "#1F9D68"
LIGHT_GREEN = "#B9E8D2"
ORANGE = "#F07C2E"
RED = "#D94B4B"
GRAY = "#BBC3CF"
PANEL = "#FFFFFF"
GRID = "#E7EBF1"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


F_TITLE = font(26, True)
F_STAGE = font(22, True)
F_BODY = font(14)
F_SMALL = font(12)
F_SMALL_BOLD = font(12, True)


def load_replay(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as gif:
        index = 0
        while True:
            try:
                gif.seek(index)
            except EOFError:
                break
            frame = gif.convert("RGB")
            # Existing replay has a 72 px caption; this GIF supplies its own.
            frame = frame.crop((0, 72, frame.width, frame.height))
            frames.append(frame)
            index += 1
    if not frames:
        raise ValueError(f"no frames found in {path}")
    return frames


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#DDE3EC", width=2)


def progress_ease(t: float) -> float:
    return 3 * t * t - 2 * t * t * t


def polyline_point(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    distance = max(0.0, min(1.0, t)) * total
    for (a, b), length in zip(zip(points, points[1:]), lengths):
        if distance <= length:
            u = distance / max(length, 1e-9)
            return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        distance -= length
    return points[-1]


def catmull_rom(points: list[tuple[float, float]], samples: int = 100) -> list[tuple[float, float]]:
    pts = np.asarray([points[0], *points, points[-1]], dtype=float)
    output = []
    segments = len(points) - 1
    for i in range(segments):
        p0, p1, p2, p3 = pts[i:i + 4]
        for u in np.linspace(0, 1, max(2, samples // segments), endpoint=False):
            p = 0.5 * ((2 * p1) + (-p0 + p2) * u
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)
            output.append(tuple(p))
    output.append(points[-1])
    return output


def draw_arm(draw: ImageDraw.ImageDraw, joints: list[tuple[int, int]],
             color: str, width: int = 11, alpha: int = 255,
             spheres: bool = False,
             collision_box: tuple[int, int, int, int] | None = None) -> bool:
    # Alpha is retained for call-site clarity; drawing occurs on an opaque canvas.
    _ = alpha
    draw.line(joints, fill=color, width=width, joint="curve")
    for i, xy in enumerate(joints):
        r = 8 if i else 10
        draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r),
                     fill="#FFFFFF", outline=color, width=3)
    has_collision = False
    if spheres:
        for a, b in zip(joints, joints[1:]):
            for u in (0.20, 0.45, 0.70):
                x = round(a[0] + (b[0] - a[0]) * u)
                y = round(a[1] + (b[1] - a[1]) * u)
                r = 13
                colliding = False
                if collision_box is not None:
                    left, top, right, bottom = collision_box
                    nearest_x = min(max(x, left), right)
                    nearest_y = min(max(y, top), bottom)
                    colliding = (x-nearest_x) ** 2 + (y-nearest_y) ** 2 <= r ** 2
                has_collision |= colliding
                sphere_fill = "#FFD7D7" if colliding else "#D9ECFF"
                sphere_outline = RED if colliding else BLUE
                draw.ellipse((x-r, y-r, x+r, y+r), fill=sphere_fill,
                             outline=sphere_outline, width=2)
    return has_collision


def draw_collision(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = (40, 154, 548, 422)
    card(draw, box)
    draw.text((58, 169), "Robot workspace", font=F_SMALL_BOLD, fill=MUTED)
    obstacle_box = (380, 246, 485, 356)
    draw.rounded_rectangle(obstacle_box, radius=12,
                           fill="#DDE3EA", outline="#9AA6B5", width=2)
    draw.text((401, 292), "object", font=F_SMALL, fill=MUTED)
    starts = [(100, 362), (157, 335), (207, 277), (287, 252), (355, 276)]
    ends = [(100, 362), (174, 354), (247, 330), (323, 306), (395, 292)]
    u = progress_ease(t)
    joints = [(round(a[0] + (b[0]-a[0])*u), round(a[1] + (b[1]-a[1])*u))
              for a, b in zip(starts, ends)]
    colliding = draw_arm(draw, joints, BLUE, spheres=True,
                         collision_box=obstacle_box)
    status = "collision detected" if colliding else "collision-free"
    color = RED if colliding else GREEN
    draw.rounded_rectangle((58, 382, 196, 408), radius=13, fill=color)
    draw.text((72, 388), status, font=F_SMALL_BOLD, fill="white")
    draw.text((290, 388), "FK  →  sphere distances", font=F_SMALL, fill=MUTED)


def draw_ik(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = (40, 154, 548, 422)
    card(draw, box)
    draw.text((58, 169), "Parallel IK: three seeds, one wrist target",
              font=F_SMALL_BOLD, fill=MUTED)
    target = (446, 278)
    draw.line((target[0]-14, target[1], target[0]+14, target[1]), fill=ORANGE, width=3)
    draw.line((target[0], target[1]-14, target[0], target[1]+14), fill=ORANGE, width=3)
    draw.ellipse((target[0]-5, target[1]-5, target[0]+5, target[1]+5), fill=ORANGE)
    final_branches = [
        ([(96, 363), (165, 337), (225, 278), (340, 235), target], GRAY, "joint limit"),
        ([(96, 363), (165, 313), (246, 342), (357, 326), target], RED, "collision"),
        ([(96, 363), (164, 347), (238, 296), (350, 274), target], GREEN, "selected"),
    ]
    initial_seeds = [
        [(96, 363), (140, 323), (197, 307), (275, 325), (348, 314)],
        [(96, 363), (153, 354), (224, 368), (305, 346), (374, 327)],
        [(96, 363), (145, 301), (220, 247), (302, 229), (381, 250)],
    ]
    seed_colors = ("#9BCBFA", "#66ACEF", BLUE)
    iterations = 6
    iteration = min(iterations, int(t * iterations) + 1)
    amount = progress_ease((iteration - 1) / (iterations - 1))
    previous_amount = progress_ease(max(0, iteration - 2) / (iterations - 1))
    solved = iteration == iterations

    draw.rounded_rectangle((385, 190, 525, 217), radius=13, fill="#EAF3FD")
    draw.text((455, 203), f"iteration {iteration}/{iterations}",
              font=F_SMALL_BOLD, fill=BLUE, anchor="mm")
    residual_mm = round(82 * (1 - amount) + 2 * amount)
    draw.text((388, 222), f"target residual ≈ {residual_mm} mm",
              font=F_SMALL, fill=MUTED)

    for idx, ((final_joints, result_color, result_label), initial_joints) in enumerate(
            zip(final_branches, initial_seeds)):
        initial = np.asarray(initial_joints, dtype=float)
        final = np.asarray(final_joints, dtype=float)
        previous = initial * (1 - previous_amount) + final * previous_amount
        current = initial * (1 - amount) + final * amount
        if iteration > 1:
            draw_arm(draw, [tuple(map(round, p)) for p in previous],
                     "#DCE5EF", width=4)
        color = result_color if solved else seed_colors[idx]
        draw_arm(draw, [tuple(map(round, p)) for p in current], color,
                 width=5 if idx < 2 else 8)
        x, y = (76, 195 + idx * 30)
        draw.text((x, y), f"seed {idx + 1}", font=F_SMALL, fill=MUTED)
        draw.rounded_rectangle((125, y+2, 197, y+11), radius=4, fill="#E3E9F1")
        draw.rounded_rectangle((125, y+2, 125 + round(72 * amount), y+11),
                               radius=4, fill=BLUE)
        label = result_label if solved else "iterating"
        label_color = result_color if solved else BLUE
        draw.text((207, y), label, font=F_SMALL, fill=label_color)
    draw.text((300, 389), "prefer the branch near q₀", font=F_SMALL, fill=MUTED)


GRAPH_NODES = [
    (78, 363), (117, 316), (140, 379), (162, 266), (184, 337), (202, 213),
    (232, 290), (244, 382), (274, 239), (297, 346), (326, 194), (350, 263),
    (374, 375), (402, 215), (431, 326), (470, 267), (507, 205), (512, 362),
]
GRAPH_EDGES = [(0,1),(0,2),(1,3),(1,4),(2,4),(3,5),(3,6),(4,6),(4,7),
               (5,8),(6,8),(6,9),(7,9),(8,10),(8,11),(9,11),(9,12),
               (10,13),(11,13),(11,14),(12,14),(13,15),(13,16),(14,15),
               (14,17),(15,16),(15,17)]
PATH_IDS = [0, 1, 3, 5, 8, 10, 13, 16]


def draw_cspace_base(draw: ImageDraw.ImageDraw) -> None:
    box = (40, 154, 548, 422)
    card(draw, box)
    draw.text((58, 169), "Conceptual joint-space view", font=F_SMALL_BOLD, fill=MUTED)
    draw.text((69, 391), "q₁", font=F_SMALL, fill=MUTED)
    draw.text((511, 391), "q₂", font=F_SMALL, fill=MUTED)
    draw.ellipse((238, 252, 382, 367), fill="#F6D7D7", outline="#E99A9A", width=2)
    draw.text((269, 302), "collision\nregion", font=F_SMALL, fill=RED, align="center")


def draw_graph(draw: ImageDraw.ImageDraw, t: float) -> None:
    draw_cspace_base(draw)
    count = max(2, round(len(GRAPH_NODES) * min(1.0, t * 1.25)))
    for a, b in GRAPH_EDGES:
        if a < count and b < count:
            draw.line((*GRAPH_NODES[a], *GRAPH_NODES[b]), fill="#CDD5DF", width=2)
    for xy in GRAPH_NODES[:count]:
        draw.ellipse((xy[0]-4, xy[1]-4, xy[0]+4, xy[1]+4), fill=BLUE)
    if t > 0.62:
        path = [GRAPH_NODES[i] for i in PATH_IDS]
        draw.line(path, fill=ORANGE, width=6, joint="curve")
        draw.text((390, 389), "shortest valid route", font=F_SMALL, fill=ORANGE)


def draw_trajopt(draw: ImageDraw.ImageDraw, t: float) -> None:
    draw_cspace_base(draw)
    graph_path = [GRAPH_NODES[i] for i in PATH_IDS]
    smooth_controls = [(78, 363), (142, 281), (211, 210), (324, 189),
                       (425, 203), (507, 205)]
    samples = 120
    graph = np.asarray([polyline_point(graph_path, i / (samples - 1))
                        for i in range(samples)])
    smooth_path = catmull_rom(smooth_controls, samples)
    smooth = np.asarray([polyline_point(smooth_path, i / (samples - 1))
                         for i in range(samples)])

    # Six discrete updates make the iterative nature of TrajOpt visible. Keep
    # the prior iterate as a faint trace so each update reads as refinement.
    iterations = 6
    iteration = min(iterations, int(t * iterations) + 1)
    amount = (iteration - 1) / (iterations - 1)
    previous_amount = max(0.0, (iteration - 2) / (iterations - 1))
    previous = graph * (1 - previous_amount) + smooth * previous_amount
    current = graph * (1 - amount) + smooth * amount
    draw.line([tuple(p) for p in graph], fill="#F6BE97", width=4, joint="curve")
    if iteration > 1:
        draw.line([tuple(p) for p in previous], fill="#BFD8CD", width=5,
                  joint="curve")
    draw.line([tuple(p) for p in current], fill=GREEN, width=7, joint="curve")

    draw.rounded_rectangle((394, 174, 525, 201), radius=13, fill="#EAF7F1")
    draw.text((459, 187), f"iteration {iteration}/{iterations}",
              font=F_SMALL_BOLD, fill=GREEN, anchor="mm")
    path_score = 1.00 - 0.18 * amount
    smooth_score = 1.00 - 0.55 * amount
    draw.text((65, 389), "PRM seed", font=F_SMALL, fill=ORANGE)
    draw.text((145, 389), f"length {path_score:.2f}", font=F_SMALL, fill=GREEN)
    draw.text((256, 389), f"smoothness {smooth_score:.2f}", font=F_SMALL, fill=GREEN)


def draw_validation(draw: ImageDraw.ImageDraw, t: float) -> None:
    draw_cspace_base(draw)
    smooth = catmull_rom([(78, 363), (142, 281), (211, 210), (324, 189),
                          (425, 203), (507, 205)], 120)
    draw.line(smooth, fill=GREEN, width=6)
    dense = [polyline_point(smooth, i / 31) for i in range(32)]
    checked = max(1, round(len(dense) * t))
    for idx, (x, y) in enumerate(dense):
        color = GREEN if idx < checked else "#D5DBE4"
        r = 5 if idx < checked else 3
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    labels = ("collision", "joint limits", "velocity / acceleration", "goal")
    x = 58
    for label in labels:
        color = GREEN
        w = draw.textlength(label, font=F_SMALL)
        draw.ellipse((x, 389, x+14, 403), fill=color)
        draw.text((x+2, 386), "✓", font=F_SMALL_BOLD, fill="white")
        draw.text((x+20, 388), label, font=F_SMALL, fill=color)
        x += round(w) + 45


def paste_scene(canvas: Image.Image, replay: list[Image.Image], stage: int,
                t: float) -> None:
    draw = ImageDraw.Draw(canvas)
    box = (574, 154, 920, 422)
    card(draw, box)
    if stage == 3:
        source_index = min(len(replay)-1, round(t * (len(replay)-1)))
    elif stage == 0:
        source_index = min(len(replay)-1, round(0.58 * (len(replay)-1)))
    else:
        source_index = min(len(replay)-1, round(0.18 * (len(replay)-1)))
    scene = replay[source_index].copy().resize((326, 244), Image.Resampling.LANCZOS)
    if stage < 4:
        veil = Image.new("RGBA", scene.size, (255, 255, 255, 35))
        scene = Image.alpha_composite(scene.convert("RGBA"), veil).convert("RGB")
    canvas.paste(scene, (584, 166))
    badge = "AUTODEX REPLAY"
    draw.rounded_rectangle((593, 375, 731, 401), radius=13, fill="#172033")
    draw.text((662, 388), badge, font=F_SMALL_BOLD, fill="white", anchor="mm")
    if stage == 3:
        phase = "approach" if source_index < len(replay) * 0.42 else "10 cm Jacobian lift"
        draw.rounded_rectangle((743, 375, 901, 401), radius=13, fill="#FFF4EA")
        draw.text((822, 388), phase, font=F_SMALL_BOLD,
                  fill=ORANGE if "lift" in phase else BLUE, anchor="mm")
    else:
        draw.text((822, 388), "attached_container", font=F_SMALL, fill=MUTED,
                  anchor="mm")


def draw_timeline(draw: ImageDraw.ImageDraw, stage: int) -> None:
    y = 488
    xs = np.linspace(80, 880, len(STAGES)).astype(int)
    draw.line((xs[0], y, xs[-1], y), fill="#D6DCE5", width=4)
    if stage:
        draw.line((xs[0], y, xs[stage], y), fill=GREEN, width=4)
    labels = ("IK", "PRM", "TrajOpt", "Validate")
    for idx, (x, label) in enumerate(zip(xs, labels)):
        fill = GREEN if idx <= stage else "white"
        outline = GREEN if idx <= stage else GRAY
        draw.ellipse((x-10, y-10, x+10, y+10), fill=fill, outline=outline, width=3)
        draw.text((x, y+18), label, font=F_SMALL, fill=INK if idx == stage else MUTED,
                  anchor="ma")


def render_frame(replay: list[Image.Image], stage: int, local_index: int) -> Image.Image:
    t = local_index / (FRAMES_PER_STAGE - 1)
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 26), "AutoDex motion planning", font=F_TITLE, fill=INK)
    draw.text((40, 62), "From a grasp target to an executable trajectory",
              font=F_BODY, fill=MUTED)
    draw.rounded_rectangle((40, 103, 72, 135), radius=16, fill=BLUE)
    draw.text((56, 119), str(stage + 1), font=F_SMALL_BOLD, fill="white", anchor="mm")
    draw.text((84, 106), STAGES[stage][0], font=F_STAGE, fill=INK)
    draw.text((84, 133), STAGES[stage][1], font=F_BODY, fill=MUTED)

    if stage == 0:
        draw_ik(draw, t)
    elif stage == 1:
        draw_graph(draw, t)
    elif stage == 2:
        draw_trajopt(draw, t)
    else:
        draw_validation(draw, t)
    paste_scene(canvas, replay, stage, t)
    draw_timeline(draw, stage)
    return canvas


def main() -> int:
    default_replay = Path(
        "/home/robot/shared_data/AutoDex/reachability/inspire/attached_container/"
        "pipeline_lift/xarm/v8/000/attached_container_000_xarm_per_grasp/"
        "greedy_trajectory_replays/animations_mesh/"
        "01_shelf_5_n1000_s123_97_mesh.gif"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=default_replay)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/visualizations/autodex_motion_planning.gif"))
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="GIF playback speed (default: 10)")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    replay = load_replay(args.replay.expanduser().resolve())
    frames = [render_frame(replay, stage, frame)
              for stage in range(len(STAGES))
              for frame in range(FRAMES_PER_STAGE)]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(1000 / args.fps), loop=0, optimize=True,
                   disposal=2)
    print(f"saved {output} ({len(frames) / args.fps:.1f}s, "
          f"{CANVAS[0]}x{CANVAS[1]}, {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
