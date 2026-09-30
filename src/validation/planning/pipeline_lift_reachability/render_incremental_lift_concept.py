#!/usr/bin/env python3
"""Render a quiet, visual explanation of incremental Jacobian lift planning.

One 5 mm increment is shown slowly: target, seed, solve, chord check, accept.
The remaining increments are then accumulated before the checked samples are
smoothed, retimed, and replayed for final validation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS = (1200, 675)
FPS = 12

BG = "#F7F9FC"
INK = "#182230"
MUTED = "#8490A3"
FAINT = "#DCE3EC"
BLUE = "#2F7DD1"
PALE_GREEN = "#DDF5E9"
GREEN = "#18A36B"
ORANGE = "#F07C2E"
PALE_ORANGE = "#FFF0E5"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


F_STAGE = font(34, True)
F_COUNT = font(25, True)
F_LABEL = font(14, True)
F_SMALL = font(13)
F_TINY = font(11, True)


@dataclass(frozen=True)
class Scene:
    name: str
    seconds: float
    kind: str


SCENES = (
    Scene("LIFT", 2.0, "goal"),
    Scene("TARGET", 1.8, "target"),
    Scene("SEED", 1.8, "seed"),
    Scene("SOLVE", 3.2, "solve"),
    Scene("CHECK", 2.6, "check"),
    Scene("ACCEPT", 1.4, "accept"),
    Scene("CONTINUE", 5.7, "continue"),
    Scene("SMOOTH + RETIME", 2.8, "smooth"),
    Scene("VALIDATE", 3.6, "validate"),
    Scene("READY", 2.0, "ready"),
)

STAGES = ("TARGET", "SEED", "SOLVE", "CHECK", "ACCEPT", "SMOOTH", "VALIDATE")
STAGE_INDEX = {
    "target": 0, "seed": 1, "solve": 2, "check": 3, "accept": 4,
    "continue": 4, "smooth": 5, "validate": 6, "ready": 6,
}


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def ease(value: float) -> float:
    value = clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def arm_joints(progress: float, *, correction: float = 0.0) -> list[tuple[int, int]]:
    """Plausible planar arm pose with a vertical wrist trajectory."""
    p = ease(progress)
    start = np.array([
        [105, 505], [180, 473], [270, 430], [363, 466], [456, 492], [542, 498],
    ], dtype=float)
    end = np.array([
        [105, 505], [176, 467], [246, 380], [340, 318], [445, 323], [542, 318],
    ], dtype=float)
    current = (1.0 - p) * start + p * end
    bow = math.sin(math.pi * p)
    current[1, 0] -= 9 * bow
    current[2, 0] -= 15 * bow
    current[3, 0] += 8 * bow
    current[2, 0] += 9 * correction
    current[3, 0] += 13 * correction
    current[4, 1] += 9 * correction
    current[5, 0] += 24 * correction
    current[5, 1] += 13 * correction
    return [tuple(np.rint(point).astype(int)) for point in current]


def draw_arm(draw: ImageDraw.ImageDraw, joints: list[tuple[int, int]], *,
             color: str = BLUE, fill: str = "white", width: int = 14,
             collision_samples: bool = False) -> None:
    draw.line(joints, fill=color, width=width, joint="curve")
    for index, (x, y) in enumerate(joints):
        radius = 11 if index == 0 else 7
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=fill, outline=color, width=3)
    if collision_samples:
        for a, b in zip(joints, joints[1:]):
            for fraction in (0.28, 0.56, 0.84):
                x = round(a[0] + fraction * (b[0] - a[0]))
                y = round(a[1] + fraction * (b[1] - a[1]))
                draw.ellipse((x-8, y-8, x+8, y+8), fill=PALE_GREEN,
                             outline=GREEN, width=2)


def draw_payload(draw: ImageDraw.ImageDraw, wrist: tuple[int, int], *,
                 color: str = ORANGE, fill: str = PALE_ORANGE) -> None:
    x, y = wrist
    draw.rounded_rectangle((x-12, y-11, x+13, y+12), radius=5, fill=color)
    draw.line((x+9, y-7, x+25, y-15), fill=color, width=5)
    draw.line((x+9, y+7, x+25, y+15), fill=color, width=5)
    draw.rounded_rectangle((x+24, y-21, x+70, y+21), radius=7,
                           fill=fill, outline=color, width=3)


def draw_robot(draw: ImageDraw.ImageDraw, progress: float, *, correction: float = 0.0,
               ghosts: list[float] | None = None, collision_samples: bool = False) -> None:
    for ghost_progress in ghosts or []:
        ghost = arm_joints(ghost_progress)
        draw_arm(draw, ghost, color="#C7D1DE", fill=BG, width=7)
        draw_payload(draw, ghost[-1], color="#BCC7D5", fill="#EEF2F6")
    joints = arm_joints(progress, correction=correction)
    draw_arm(draw, joints, collision_samples=collision_samples)
    draw_payload(draw, joints[-1])


def draw_world(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((60, 520, 650, 545), fill="#C9D1DC")
    draw.rectangle((72, 545, 638, 563), fill="#E4E9F0")
    draw.rounded_rectangle((71, 493, 139, 524), radius=7, fill="#8D99A8")


def rail_y(node: float) -> float:
    return 498.0 - 180.0 * node / 20.0


def draw_rail(draw: ImageDraw.ImageDraw, accepted: int, *, target_node: int | None,
              tentative: bool = False, pulse: float = 0.0) -> None:
    x = 692
    draw.line((x, rail_y(0), x, rail_y(20)), fill="#B8C2CF", width=3)
    draw.polygon([(x, rail_y(20)-13), (x-7, rail_y(20)), (x+7, rail_y(20))],
                 fill=GREEN)
    for node in range(21):
        y = round(rail_y(node))
        if node == 0:
            color, radius = BLUE, 5
        elif node <= accepted:
            color, radius = GREEN, 5
        else:
            color, radius = FAINT, 3
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    if target_node is not None:
        y = round(rail_y(target_node))
        radius = round(8 + 3 * pulse)
        color = ORANGE if tentative else GREEN
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=BG, outline=color, width=3)
        draw.line((x+radius+5, y, x+48, y), fill=color, width=3)
    for node, label in ((0, "0"), (10, "50"), (20, "100 mm")):
        draw.text((x+59, rail_y(node)), label, font=F_SMALL,
                  fill=INK if node == 20 else MUTED, anchor="lm")


def draw_header(draw: ImageDraw.ImageDraw, scene: Scene, step: int,
                accepted: int) -> None:
    draw.text((60, 42), scene.name, font=F_STAGE,
              fill=GREEN if scene.kind == "ready" else INK)
    if scene.kind not in {"goal", "smooth", "validate", "ready"}:
        draw.text((1135, 57), f"{step:02d}/20", font=F_COUNT, fill=INK, anchor="rm")
    elif scene.kind == "smooth":
        draw.text((1135, 57), "21 NODES", font=F_COUNT, fill=INK, anchor="rm")
    elif scene.kind == "validate":
        draw.text((1135, 57), "10 ms", font=F_COUNT, fill=INK, anchor="rm")
    elif scene.kind == "ready":
        draw.text((1135, 57), "100 mm", font=F_COUNT, fill=GREEN, anchor="rm")
    draw.rounded_rectangle((60, 91, 1140, 99), radius=4, fill="#E4E9F0")
    width = round(1080 * accepted / 20)
    if width:
        draw.rounded_rectangle((60, 91, 60 + width, 99), radius=4, fill=GREEN)


def draw_stage_strip(draw: ImageDraw.ImageDraw, scene: Scene) -> None:
    if scene.kind == "goal":
        return
    active = STAGE_INDEX[scene.kind]
    xs = np.linspace(104, 1096, len(STAGES))
    y = 620
    draw.line((xs[0], y, xs[-1], y), fill=FAINT, width=3)
    for index, (x, label) in enumerate(zip(xs, STAGES)):
        if index < active or scene.kind == "ready":
            color, fill = GREEN, PALE_GREEN
        elif index == active:
            color, fill = ORANGE, PALE_ORANGE
        else:
            color, fill = "#AAB5C3", BG
        draw.ellipse((x-7, y-7, x+7, y+7), fill=fill, outline=color, width=2)
        draw.text((x, y+20), label, font=F_TINY, fill=color, anchor="ma")


def draw_error(draw: ImageDraw.ImageDraw, current: tuple[int, int],
               target: tuple[int, int]) -> None:
    draw.line((*current, *target), fill=ORANGE, width=4)
    tx, ty = target
    draw.ellipse((tx-7, ty-7, tx+7, ty+7), fill=PALE_ORANGE,
                 outline=ORANGE, width=3)


def draw_spline_panel(draw: ImageDraw.ImageDraw, t: float) -> None:
    x0, y0, x1, y1 = 790, 205, 1125, 485
    draw.line((x0, y1, x1, y1), fill="#BEC8D5", width=2)
    draw.line((x0, y0, x0, y1), fill="#BEC8D5", width=2)
    draw.text((x0-18, y0), "q", font=F_LABEL, fill=MUTED, anchor="mm")
    draw.text((x1, y1+17), "t", font=F_LABEL, fill=MUTED, anchor="mm")
    u = np.linspace(0.0, 1.0, 21)
    for color, phase in zip((BLUE, GREEN, ORANGE), (0.1, 1.7, 3.2)):
        values = 0.50 + 0.30 * np.sin(1.2 * math.pi * u + phase) * (0.35 + 0.65*u)
        points = [(round(x0 + value * (x1-x0)), round(y1 - q * (y1-y0)))
                  for value, q in zip(u, values)]
        for x, y in points:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        reveal = max(2, round(2 + ease(t) * (len(points)-2)))
        dense_u = np.linspace(0.0, u[reveal-1], 140)
        dense_v = 0.50 + 0.30 * np.sin(1.2 * math.pi * dense_u + phase) * (0.35 + 0.65*dense_u)
        curve = [(round(x0 + value * (x1-x0)), round(y1 - q * (y1-y0)))
                 for value, q in zip(dense_u, dense_v)]
        draw.line(curve, fill=color, width=4)


def render(scene: Scene, t: float) -> Image.Image:
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)
    t = clamp01(t)
    if scene.kind == "goal":
        step, accepted, progress = 1, 0, 0.0
    elif scene.kind in {"target", "seed", "solve", "check"}:
        step, accepted, progress = 1, 0, 0.0
    elif scene.kind == "accept":
        step, accepted, progress = 1, int(t > 0.25), 1.0/20.0
    elif scene.kind == "continue":
        accepted = min(20, 1 + int(t * 20))
        step = min(20, accepted + 1)
        progress = accepted / 20.0
    else:
        step, accepted, progress = 20, 20, 1.0

    draw_header(draw, scene, step, accepted)
    draw_world(draw)
    if scene.kind == "goal":
        draw_robot(draw, 0.0, ghosts=[1.0] if t > 0.25 else [])
        draw_rail(draw, 0, target_node=20 if t > 0.35 else None,
                  tentative=True, pulse=0.5 + 0.5*math.sin(2*math.pi*t))
    elif scene.kind == "target":
        draw_robot(draw, 0.0)
        draw_rail(draw, 0, target_node=1, tentative=True,
                  pulse=0.5 + 0.5*math.sin(4*math.pi*t))
    elif scene.kind == "seed":
        draw_robot(draw, 0.0)
        draw_rail(draw, 0, target_node=1, tentative=True)
        wrist = arm_joints(0.0)[-1]
        radius = round(18 + 6 * math.sin(math.pi*t))
        draw.ellipse((wrist[0]-radius, wrist[1]-radius,
                      wrist[0]+radius, wrist[1]+radius), outline=BLUE, width=3)
    elif scene.kind == "solve":
        solve = ease(t)
        correction = 0.50 * (1.0-solve) * math.sin(3.0*math.pi*(1.0-t))
        current_progress = solve / 20.0
        draw_robot(draw, current_progress, correction=correction,
                   ghosts=[0.0] if t > 0.08 else [])
        draw_rail(draw, 0, target_node=1, tentative=True)
        draw_error(draw, arm_joints(current_progress, correction=correction)[-1],
                   (542, round(rail_y(1))))
    elif scene.kind == "check":
        scan = ease(t)
        draw_robot(draw, scan / 20.0, ghosts=[0.0], collision_samples=True)
        draw_rail(draw, 0, target_node=1, tentative=True)
        wrist = arm_joints(scan / 20.0)[-1]
        draw.ellipse((wrist[0]-26, wrist[1]-26, wrist[0]+26, wrist[1]+26),
                     outline=GREEN, width=4)
    elif scene.kind == "accept":
        draw_robot(draw, 1.0/20.0, ghosts=[0.0])
        draw_rail(draw, 1 if t > 0.25 else 0, target_node=1,
                  tentative=t <= 0.25, pulse=1.0-t)
    elif scene.kind == "continue":
        previous = max(0, accepted-1) / 20.0
        draw_robot(draw, progress, ghosts=[previous] if accepted < 20 else [])
        draw_rail(draw, accepted, target_node=step if accepted < 20 else None,
                  tentative=accepted < 20)
    elif scene.kind == "smooth":
        draw_robot(draw, 1.0)
        draw_rail(draw, 20, target_node=None)
        draw_spline_panel(draw, t)
    elif scene.kind == "validate":
        cycle = 0.5 - 0.5 * math.cos(2.0*math.pi*t)
        draw_robot(draw, cycle, collision_samples=True)
        draw_rail(draw, 20, target_node=None)
        wrist = arm_joints(cycle)[-1]
        draw.ellipse((wrist[0]-29, wrist[1]-29, wrist[0]+29, wrist[1]+29),
                     outline=GREEN, width=4)
        for index, symbol in enumerate(("↥", "◇", "✓")):
            cx = 860 + index*105
            draw.ellipse((cx-25, 410, cx+25, 460), fill=PALE_GREEN,
                         outline=GREEN, width=3)
            draw.text((cx, 435), symbol, font=F_COUNT, fill=GREEN, anchor="mm")
    else:
        draw_robot(draw, 1.0, ghosts=[0.0])
        draw_rail(draw, 20, target_node=None)
        cx, cy = 930, 355
        draw.ellipse((cx-42, cy-42, cx+42, cy+42), fill=PALE_GREEN,
                     outline=GREEN, width=4)
        draw.text((cx, cy), "✓", font=font(45, True), fill=GREEN, anchor="mm")
    draw_stage_strip(draw, scene)
    return image


def build_frames(fps: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for scene in SCENES:
        count = max(2, round(scene.seconds * fps))
        frames.extend(render(scene, index/(count-1)) for index in range(count))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/visualizations/autodex_incremental_lift.gif"))
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    frames = build_frames(args.fps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(1000/args.fps), loop=0,
                   optimize=True, disposal=2)
    poster = output.with_suffix(".png")
    frames[-1].save(poster)
    print(f"saved {output} ({len(frames)/args.fps:.1f}s, "
          f"{CANVAS[0]}x{CANVAS[1]}, {args.fps} fps)")
    print(f"saved {poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
