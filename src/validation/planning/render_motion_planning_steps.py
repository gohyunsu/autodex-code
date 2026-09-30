#!/usr/bin/env python3
"""Render concise, configuration-faithful AutoDex planning explainer GIFs.

These are deliberately conceptual diagrams.  Their parameters and timings are
tied to the Inspire/Pepsi pipeline configuration and the warmed v8/008 Pepsi
pipeline replay (108 plans), rather than to a particular playback trajectory.
"""
from __future__ import annotations

import math
import heapq
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (960, 480)
FPS = 10
N_FRAMES = 36

BG, INK, MUTED = "#F6F8FB", "#172033", "#667085"
BLUE, GREEN, ORANGE, RED = "#2878D0", "#1F9D68", "#F07C2E", "#D94B4B"
PANEL, BORDER, GRID, PALE_BLUE = "#FFFFFF", "#DDE3EC", "#E7EBF1", "#EAF3FD"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{face}", size)


F_TITLE, F_STAGE = font(25, True), font(20, True)
F_BODY, F_SMALL, F_SMALL_BOLD = font(14), font(12), font(12, True)


STAGES = ("Solve IK", "Graph search", "Trajectory optimization")


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 3 * t * t - 2 * t * t * t


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=BORDER, width=2)


def chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: str = BLUE) -> int:
    box = draw.textbbox((0, 0), label, font=F_SMALL_BOLD)
    width = box[2] - box[0] + 22
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 27), radius=13, fill="#FFFFFF",
                           outline=color, width=2)
    draw.text((x + 11, y + 7), label, font=F_SMALL_BOLD, fill=color)
    return x + width


def pipeline(draw: ImageDraw.ImageDraw, active: int) -> None:
    x0, width, gap, y = 44, 270, 30, 28
    for index, label in enumerate(STAGES):
        x = x0 + index * (width + gap)
        if index < active:
            fill, outline, text_color = "#E7F6EE", GREEN, GREEN
        elif index == active:
            fill, outline, text_color = PALE_BLUE, BLUE, BLUE
        else:
            fill, outline, text_color = "#FFFFFF", BORDER, MUTED
        draw.rounded_rectangle((x, y, x + width, y + 48), radius=14, fill=fill,
                               outline=outline, width=3 if index == active else 2)
        draw.text((x + 18, y + 15), f"{index + 1}", font=F_SMALL_BOLD, fill=text_color)
        draw.text((x + 43, y + 14), label, font=F_SMALL_BOLD, fill=text_color)
        if index < len(STAGES) - 1:
            ax = x + width + 8
            draw.line((ax, y + 24, ax + gap - 12, y + 24), fill="#AAB5C5", width=3)
            draw.polygon([(ax + gap - 12, y + 24), (ax + gap - 19, y + 19),
                          (ax + gap - 19, y + 29)], fill="#AAB5C5")


def draw_arm(draw: ImageDraw.ImageDraw, joints: list[tuple[float, float]], color: str,
             width: int = 6) -> None:
    points = [(round(x), round(y)) for x, y in joints]
    draw.line(points, fill=color, width=width, joint="curve")
    for i, (x, y) in enumerate(points):
        radius = 6 if i else 8
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#FFFFFF",
                     outline=color, width=2)


IK_BASE = np.array([115.0, 360.0])
IK_TARGET = np.array([750.0, 230.0])
IK_LINKS = np.array([150.0, 160.0, 170.0, 195.0])
IK_SEEDS = np.array([
    [-1.0, 1.6, -1.2, 0.8],
    [0.8, -1.5, 1.3, -0.7],
    [-1.8, 1.3, 1.2, -0.5],
])


def arm_points(q: np.ndarray) -> list[tuple[float, float]]:
    angles = np.cumsum(q)
    steps = np.column_stack((IK_LINKS * np.cos(angles), IK_LINKS * np.sin(angles)))
    points = np.vstack((IK_BASE, IK_BASE + np.cumsum(steps, axis=0)))
    return [tuple(point) for point in points]


def simulate_ik() -> np.ndarray:
    """Run a small damped-least-squares IK problem for each visible seed."""
    histories = []
    for initial in IK_SEEDS:
        q = initial.copy()
        states = [q.copy()]
        for _ in range(200):
            angles = np.cumsum(q)
            endpoint = np.asarray(arm_points(q)[-1])
            error = IK_TARGET - endpoint
            jacobian = np.array([
                [-(IK_LINKS[j:] * np.sin(angles[j:])).sum() for j in range(len(q))],
                [ (IK_LINKS[j:] * np.cos(angles[j:])).sum() for j in range(len(q))],
            ])
            damping = 18.0
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(2), error)
            norm = np.linalg.norm(update)
            if norm > 0.18:
                update *= 0.18 / norm
            q += 0.06 * update
            states.append(q.copy())
        histories.append(states)
    return np.asarray(histories)


IK_HISTORY = simulate_ik()


def workspace(draw: ImageDraw.ImageDraw) -> tuple[int, int]:
    card(draw, (40, 154, 920, 424))
    target = tuple(IK_TARGET)
    draw.line((target[0]-14, target[1], target[0]+14, target[1]), fill=ORANGE, width=3)
    draw.line((target[0], target[1]-14, target[0], target[1]+14), fill=ORANGE, width=3)
    draw.ellipse((target[0]-4, target[1]-4, target[0]+4, target[1]+4), fill=ORANGE)
    return target


def ik_frame(draw: ImageDraw.ImageDraw, t: float) -> None:
    workspace(draw)
    shown = round(t * 200)
    draw.rounded_rectangle((62, 174, 245, 204), radius=15, fill=PALE_BLUE)
    draw.text((153, 182), f"iteration  {shown} / 200", anchor="ma",
              font=F_SMALL_BOLD, fill=BLUE)
    iteration = min(200, shown)
    colors = ("#8BC5F5", "#4B91D8", "#736FD0")
    for index, color in enumerate(colors):
        if iteration == 200:
            color = GREEN
        draw_arm(draw, arm_points(IK_HISTORY[index, iteration]), color, 6)


GRAPH_CENTER = np.array([474.0, 303.5])
GRAPH_RADII = np.array([111.0, 70.0])


def graph_point_free(point: np.ndarray, clearance: float = 1.0) -> bool:
    return float(np.sum(((point - GRAPH_CENTER) / GRAPH_RADII) ** 2)) > clearance ** 2


def graph_edge_free(a: np.ndarray, b: np.ndarray) -> bool:
    for amount in np.linspace(0.0, 1.0, 36):
        if not graph_point_free(a * (1.0 - amount) + b * amount, 1.05):
            return False
    return True


def build_prm() -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    """Build a deterministic, collision-checked miniature PRM and shortest path."""
    rng = np.random.default_rng(31)
    start = np.array([105.0, 370.0])
    goal = np.array([835.0, 218.0])
    samples = [start]
    while len(samples) < 47:
        point = rng.uniform([90.0, 190.0], [855.0, 385.0])
        if graph_point_free(point, 1.08):
            samples.append(point)
    samples.append(goal)
    nodes = np.asarray(samples)
    edges_set: set[tuple[int, int]] = set()
    for i, point in enumerate(nodes):
        distances = np.linalg.norm(nodes - point, axis=1)
        for j in np.argsort(distances)[1:16]:
            edge = (min(i, int(j)), max(i, int(j)))
            if edge not in edges_set and graph_edge_free(point, nodes[j]):
                edges_set.add(edge)
    edges = sorted(edges_set)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for i, j in edges:
        distance = float(np.linalg.norm(nodes[i] - nodes[j]))
        adjacency[i].append((j, distance))
        adjacency[j].append((i, distance))
    distances = [math.inf] * len(nodes)
    previous = [-1] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    path = []
    node = len(nodes) - 1
    while node >= 0:
        path.append(node)
        node = previous[node]
    path.reverse()
    if not path or path[0] != 0:
        raise RuntimeError("deterministic PRM did not connect start and goal")
    return nodes, edges, path


NODES, EDGES, PATH = build_prm()


def graph_frame(draw: ImageDraw.ImageDraw, t: float) -> None:
    card(draw, (40, 154, 920, 424))
    draw.text((62, 171), "Joint space (two coordinates shown)", font=F_SMALL_BOLD, fill=MUTED)
    draw.text((78, 397), "q₁", font=F_SMALL, fill=MUTED)
    draw.text((860, 397), "q₂", font=F_SMALL, fill=MUTED)
    draw.ellipse((370, 244, 577, 363), fill="#F9DEDE", outline="#E99A9A", width=2)
    draw.text((474, 298), "collision", anchor="mm", font=F_SMALL, fill=RED)
    count = max(2, round(len(NODES) * min(1, t / .72)))
    for i, j in EDGES:
        if i < count and j < count:
            draw.line((*NODES[i], *NODES[j]), fill="#CAD3DE", width=2)
    for x, y in NODES[:count]:
        draw.ellipse((x-4,y-4,x+4,y+4), fill=BLUE)
    if t > .72:
        p = min(1, (t-.72)/.28)
        path = [NODES[i] for i in PATH]
        n = max(2, round(1 + p * (len(path) - 1)))
        draw.line([tuple(point) for point in path[:n]], fill=ORANGE, width=6, joint="curve")
        if n == len(path):
            draw.text((661, 382), "trajectory seed", font=F_SMALL_BOLD, fill=ORANGE)


TRAJ_SEED_CONTROLS = np.array([
    (110, 374), (201, 307), (304, 227), (389, 207),
    (501, 214), (632, 196), (807, 222),
], dtype=float)


def sample_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, distance[-1], samples)
    return np.column_stack([
        np.interp(targets, distance, points[:, axis]) for axis in range(2)
    ])


def simulate_trajopt() -> np.ndarray:
    """Optimize a 64-knot path using smoothing plus collision projection."""
    trajectory = sample_polyline(TRAJ_SEED_CONTROLS, 64)
    center = np.array([500.0, 300.0])
    clearance_radii = np.array([103.0, 72.0])
    history = [trajectory.copy()]
    for iteration in range(600):
        # Neighbor averaging is the discrete smoothness-gradient update. The
        # endpoints remain fixed, as they do for a start/goal trajectory.
        step = 0.48 * (1.0 - 0.5 * iteration / 599.0)
        update = np.zeros_like(trajectory)
        update[1:-1] = ((trajectory[:-2] + trajectory[2:]) * 0.5
                        - trajectory[1:-1])
        trajectory[1:-1] += step * update[1:-1]

        # Project any violating knot to the clearance boundary. This is a
        # compact stand-in for cuRobo's differentiable collision cost.
        normalized = (trajectory - center) / clearance_radii
        distance = np.sqrt(np.sum(normalized * normalized, axis=1))
        for knot in np.flatnonzero(distance < 1.08):
            if knot not in (0, len(trajectory) - 1):
                trajectory[knot] = center + (
                    (trajectory[knot] - center) * (1.08 / max(distance[knot], 1e-9)))
        history.append(trajectory.copy())
    return np.asarray(history)


TRAJOPT_HISTORY = simulate_trajopt()


def trajopt_frame(draw: ImageDraw.ImageDraw, t: float) -> None:
    card(draw, (40, 154, 920, 424))
    draw.text((62, 171), "Joint-space trajectories", font=F_SMALL_BOLD, fill=MUTED)
    obstacle = (417, 245, 582, 355)
    draw.ellipse(obstacle, fill="#F9DEDE", outline="#E99A9A", width=2)
    draw.text((500, 300), "collision", anchor="mm", font=F_SMALL, fill=RED)
    iteration = round(t * 600)
    seed = TRAJOPT_HISTORY[0]
    current = TRAJOPT_HISTORY[iteration]
    draw.line([tuple(point) for point in seed], fill=ORANGE, width=3, joint="curve")
    current_color = GREEN if iteration == 600 else BLUE
    draw.line([tuple(point) for point in current], fill=current_color, width=7, joint="curve")
    for x, y in current[::max(1, len(current)//8)]:
        draw.ellipse((x-3,y-3,x+3,y+3), fill="#FFFFFF", outline=current_color, width=2)
    draw.line((676, 375, 704, 375), fill=ORANGE, width=3)
    draw.text((713, 367), "graph seed", font=F_SMALL, fill=MUTED)
    draw.line((676, 396, 704, 396), fill=current_color, width=6)
    draw.text((713, 388), "optimized path", font=F_SMALL, fill=MUTED)


def info_boxes(draw: ImageDraw.ImageDraw, parameters: tuple[str, ...], timing: str) -> None:
    card(draw, (40, 397, 700, 467))
    card(draw, (720, 397, 920, 467))
    draw.text((62, 409), "PARAMETERS", font=F_SMALL_BOLD, fill=MUTED)
    draw.text((742, 409), "AVG TIME", font=F_SMALL_BOLD, fill=MUTED)
    x = 62
    for value in parameters:
        x = chip(draw, (x, 431), value) + 9
    draw.text((742, 438), timing, font=F_STAGE, fill=BLUE)


def render(stage: int, frame: int) -> Image.Image:
    t = frame / (N_FRAMES - 1)
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    pipeline(draw, stage)
    # The detailed diagrams retain a convenient local coordinate system, then
    # are packed directly under the pipeline header.
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    if stage == 0:
        ik_frame(layer_draw, t)
        parameters, timing = ("50 targets / batch", "32 seeds / target", "200 fixed iterations"), "0.80 s"
    elif stage == 1:
        graph_frame(layer_draw, t)
        parameters, timing = ("PRM*", "1,500 samples", "k = 15"), "0.043 s"
    else:
        trajopt_frame(layer_draw, t)
        parameters, timing = ("32 trajectory seeds", "64 knots", "200 + 400 fixed iterations"), "0.40 s"
    image.paste(layer.crop((0, 154, 960, 424)), (0, 105), layer.crop((0, 154, 960, 424)))
    info_boxes(draw, parameters, timing)
    return image


def main() -> int:
    out = Path("outputs/visualizations")
    out.mkdir(parents=True, exist_ok=True)
    names = ("autodex_motion_planning_01_solve_ik.gif",
             "autodex_motion_planning_02_graph_search.gif",
             "autodex_motion_planning_03_trajectory_optimization.gif")
    for stage, name in enumerate(names):
        frames = [render(stage, f) for f in range(N_FRAMES)]
        path = out / name
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=round(1000 / FPS), loop=0, optimize=True, disposal=2)
        print(f"saved {path} ({N_FRAMES / FPS:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
