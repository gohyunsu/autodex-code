#!/usr/bin/env python3
"""Operator-paced container demo: pick out of a container, put it somewhere else.

The take has two kinds of turn.

**Turn 1 — measure the containers.**  The runner says it is ready; put *only*
the containers on the table and press Enter.  Each one (``open_box``,
``smallbowl``, ...) is a normal AutoDex object, so its 6-D pose comes from the
same distributed FoundPose init the picked object uses.  Its mesh then gives
the wall/floor collision cuboids, the floor height an object inside it rests
on, the rim it is released over, and the interior region a detection has to
fall inside (``fixtures.py``).

**Turn 2 — the step sequence.**  Each step names one object, where it is picked
from, and where it goes:

    --fixture bowl=smallbowl \\
    --step apple:bowl:tray --step banana:table:bowl

The runner asks for each object in turn; place it and press Enter.  It then
runs the proven single-inference motion
(``src/demo/inference/run_demo.run_once``) in the measured world:

    perceive -> grasp (inside the container, if that is the source) -> lift ->
    rotate J0 -> lay the object down / release it over the target container ->
    retreat

A ``tray`` target is not perceived: it is thin, it sits wherever
``--place-turn-deg`` points, and only its surface height matters
(``--tray-top-z``).  A container target *is* measured, so the object is carried
to its bearing and released just above its rim.

After the last step the sequence can be repeated, so one container measurement,
one camera session, one planner and one robot connection serve the whole video.

    /home/robot/anaconda3/envs/planner/bin/python \\
      src/demo/box_to_tray/run_demo.py --fixture bowl=smallbowl \\
      --step apple:bowl:tray --step banana:table:bowl --execute

Without ``--execute`` the same flow runs as a dry run: it perceives and plans
but sends no motion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _paradex_root in (os.environ.get("AUTODEX_PARADEX_ROOT"), str(Path.home() / "paradex")):
    _path = Path(_paradex_root).expanduser() if _paradex_root else None
    if _path is not None and (_path / "paradex").is_dir():
        sys.path.insert(0, str(_path))
        break

from src.demo.box_to_tray.fixtures import (
    DEFAULT_BOX_OBJ,
    ContainerFixture,
    build_fixture,
)
from src.demo.box_to_tray.planning import make_scene_hook
from src.demo.inference.run_demo import (
    GRASP_ASSET_VERSION,
    parse_args as inference_parse_args,
    run_once,
)

EXPERIMENT_NAME = "box_to_tray"
# How far J0 turns FROM THE MEASURED POST-LIFT CONFIGURATION when the target is
# the tray, in degrees.  It is a relative sweep, never a joint or bearing
# target: -30 means "having lifted, turn 30 degrees clockwise", whatever J0
# happened to be at.  A container target ignores this and aims at its measured
# bearing instead.
DEFAULT_PLACE_TURN_DEG = -30.0
TABLE_SOURCE = "table"
TRAY_TARGET = "tray"
# A step target of ``goal`` means: in turn 1 the operator puts that object where
# it should end up, its pose is measured there, and the step later carries the
# object back onto that measured pose.
GOAL_TARGET = "goal"
# Planner table top; a thin tray lying on it is only a few millimetres higher.
TABLE_SURFACE_Z = 0.040


@dataclass(frozen=True)
class Step:
    """One pick: ``obj`` goes from ``source`` to ``target``."""

    obj: str
    source: str
    target: str

    def describe(self) -> str:
        where = "on the table" if self.source == TABLE_SOURCE else f"in the {self.source}"
        goes = {TRAY_TARGET: "on the tray",
                GOAL_TARGET: "back onto its measured goal pose"}.get(
                    self.target, f"into the {self.target}")
        return f"{self.obj} {where} -> {goes}"


def parse_fixture(value: str) -> tuple:
    """``name=asset`` or bare ``asset`` (which names itself)."""
    name, sep, asset = value.partition("=")
    name, asset = name.strip(), asset.strip()
    if not sep:
        name, asset = name, name
    if not name or not asset:
        raise argparse.ArgumentTypeError(f"invalid --fixture {value!r}; use name=asset")
    return name, asset


def parse_step(value: str) -> Step:
    parts = [p.strip() for p in value.split(":")]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"invalid --step {value!r}; use object:source:target, e.g. apple:bowl:tray")
    return Step(*parts)


def parse_prompt(value: str) -> tuple:
    name, sep, text = value.partition("=")
    if not sep or not name.strip() or not text.strip():
        raise argparse.ArgumentTypeError(
            f"invalid --prompt {value!r}; use name=text, e.g. apple='red apple'")
    return name.strip(), text.strip()


def default_prompt(name: str) -> str:
    """SAM3 reads natural language, so an asset name needs its underscores out."""
    return name.replace("_", " ")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", action="append", type=parse_fixture, default=[],
                    metavar="NAME=ASSET",
                    help="a container to measure in turn 1, e.g. bowl=smallbowl. "
                         "Repeatable; NAME is how steps refer to it")
    ap.add_argument("--step", action="append", type=parse_step, default=[],
                    metavar="OBJ:SOURCE:TARGET",
                    help=f"one pick, e.g. apple:bowl:goal. SOURCE is a fixture name or "
                         f"{TABLE_SOURCE!r}; TARGET is a fixture name, {TRAY_TARGET!r}, "
                         f"or {GOAL_TARGET!r} (the pose that object was measured at in "
                         "turn 1). Repeatable and run in order")
    ap.add_argument("--obj", default=None,
                    help="single-step shorthand: pick this object out of --box-obj and "
                         "lay it on the tray")
    ap.add_argument("--box-obj", default=DEFAULT_BOX_OBJ,
                    help="container asset for the --obj shorthand (default: open_box)")
    ap.add_argument("--prompt", action="append", type=parse_prompt, default=[],
                    metavar="NAME=TEXT",
                    help="SAM3 text prompt for an object or fixture (default: its name "
                         "with underscores replaced by spaces). Repeatable")
    ap.add_argument("--exclude-grasp", action="append", default=[], metavar="PATTERN",
                    help="drop success-library grasps whose episode path or source "
                         "contains PATTERN, e.g. a specific episode timestamp that "
                         "never actually grips, or 'selected_100' for a whole store. "
                         "Repeatable. It filters this run only and never touches the "
                         "shared dataset the other demos read")
    ap.add_argument("--grasp-order", choices=["shuffle", "library"], default="shuffle",
                    help="shuffle (default) permutes the success library per run, so a "
                         "grasp that failed physically is unlikely to come up first — "
                         "but the run is then different every time. library keeps the "
                         "fixed newest-first order, which replays the same grasp")
    ap.add_argument("--grasp-seed", type=int, default=None,
                    help="seed for the --grasp-order shuffle. Pass the seed a good run "
                         "recorded in result.json (candidates.grasp_seed) to repeat it")
    ap.add_argument("--max-grasp-attempts", type=int, default=4,
                    help="how many candidates the success library may spend before the "
                         "run hands over to the v8 candidate pool")
    ap.add_argument("--max-v8-attempts", type=int, default=0,
                    help="cap on v8-pool attempts after the library budget is spent "
                         "(0 = keep drawing until the pool is empty)")
    ap.add_argument("--arm", choices=["xarm", "franka"], default="franka")
    ap.add_argument("--execute", action="store_true",
                    help="drive the robot; without it every step stops after the dry run")
    ap.add_argument("--fixtures-json", default=None,
                    help="reuse fixtures.json from an earlier turn 1 instead of measuring")
    ap.add_argument("--goals-json", default=None,
                    help="reuse goals.json from an earlier turn 1 instead of measuring "
                         "the goal poses")
    place = ap.add_mutually_exclusive_group()
    place.add_argument("--place-turn-deg", type=float, default=DEFAULT_PLACE_TURN_DEG,
                       help="tray steps only: how far J0 turns FROM the measured "
                            "post-lift pose (deg). Relative sweep, not a target")
    place.add_argument("--place-bearing-deg", type=float, default=None,
                       help="tray steps only: release at this absolute robot-frame "
                            "bearing instead (+x forward, +CCW); J0 then takes the "
                            "shorter sweep to it, a different number from the turn")
    ap.add_argument("--tray-top-z", type=float, default=TABLE_SURFACE_Z + 0.01,
                    help="height of the thin tray's top surface in robot z (m)")
    ap.add_argument("--tray-radius", type=float, default=0.5,
                    help="distance from the base to the tray (m). J0-only transfer "
                         "controls the bearing, never the radius, so this only feeds "
                         "the reported release-point diagnostic")
    ap.add_argument("--lay-down-clearance", type=float, default=0.01,
                    help="height above the tray surface at which the hand opens (m)")
    ap.add_argument("--into-gap", type=float, default=0.10,
                    help="height above a target container's rim at which the object is "
                         "released, so it drops in instead of the hand reaching inside")
    ap.add_argument("--goal-pose-mode", choices=["yaw", "full"], default="yaw",
                    help="how closely a goal step reproduces the measured attitude: "
                         "yaw (default) keeps the attitude the object was picked with "
                         "and matches only the rotation about z; full commands the "
                         "whole recorded rotation and is harder to reach")
    ap.add_argument("--goal-clearance", type=float, default=0.005,
                    help="height above the measured goal contact at which the hand "
                         "opens (m)")
    ap.add_argument("--max-descend", type=float, default=0.35,
                    help="cap on the lay-down descent (m); the actual descent is "
                         "measured from the held object's mesh bottom to the target "
                         "surface, so this is only an upper bound")
    ap.add_argument("--lift-height", type=float, default=0.15,
                    help="lift this far above the grasp before rotating (m). A step "
                         "targeting a container lifts further when its rim needs it")
    ap.add_argument("--max-lift-height", type=float, default=0.40,
                    help="cap on the rim-driven lift of a container step (m)")
    ap.add_argument("--carry-clearance", type=float, default=0.05,
                    help="minimum object clearance above the table during the rotation (m)")
    ap.add_argument("--retreat-h", type=float, default=0.15,
                    help="straight-up climb after a lay-down release (m), before the "
                         "return home")
    ap.add_argument("--drop-retreat-h", type=float, default=0.0,
                    help="the same climb after a DROP into a container (m). It defaults "
                         "to 0 because that release already happens at the carry "
                         "height: the executor's own post-release lift and the reset's "
                         "lift then stack on top of it, which is what made the return "
                         "home look like three separate climbs")
    ap.add_argument("--return-floor", type=float, default=0.15,
                    help="virtual floor height used ONLY for the return home: the table "
                         "obstacle is raised to it so the arm comes back above the "
                         "setup. 0 keeps the real table")
    ap.add_argument("--no-source-check", action="store_true",
                    help="do not require the estimated pose to be inside the source container")
    ap.add_argument("--container-model", choices=["mesh", "cuboid", "both"],
                    default="mesh",
                    help="how a measured container enters the planning world. mesh "
                         "(default) uses its own mesh, which is exact for a round bowl; "
                         "cuRobo drops meshes from the lift/carry/retreat worlds, so "
                         "those phases plan without it. cuboid uses the fitted wall/floor "
                         "boxes, which survive every phase but over-approximate a round "
                         "container. both adds each")
    ap.add_argument("--no-container-floor", action="store_true",
                    help="omit the inner floor cuboid of every container")
    ap.add_argument("--interior-inset", type=float, default=0.015,
                    help="safety band removed from a measured interior footprint (m)")
    ap.add_argument("--floor-percentile", type=float, default=5.0,
                    help="percentile of the interior probe heights taken as the floor. "
                         "Low on purpose: a bowl's cavity is curved, so the median sits "
                         "halfway up the curve and would bury a resting object inside "
                         "the floor obstacle")
    ap.add_argument("--hold-retract-orientation", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="after a release, travel back over the home position with the "
                         "wrist attitude held and reorient only there. The plain "
                         "joint-space retract is collision-free but unconstrained in "
                         "attitude, so it can roll the wrist palm-down halfway home")
    ap.add_argument("--home-pose", choices=["clear_view", "init"], default="clear_view",
                    help="which home the arm perceives from and retracts to. "
                         "clear_view (default) is FR3_INIT with J0 rotated -40 deg so "
                         "the arm leaves the cameras' view; execute() then swings back "
                         "to FR3_INIT for its approach, which is the J0 hop you see "
                         "twice per step. init parks at FR3_INIT itself and removes "
                         "that hop — use it only if the arm at FR3_INIT does not "
                         "occlude the object")
    ap.add_argument("--home-before-step", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="move to the clear-view home before a step's perception, so "
                         "the arm does not occlude the object. The previous step's "
                         "reset already parks it there, so --no-home-before-step skips "
                         "a redundant free-space move on every step after the first")
    ap.add_argument("--stop-on-step-failure", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="end the round when a step fails instead of moving on to the "
                         "next object. On by default: a silently skipped step looks "
                         "like the robot ignoring it and grabbing the next thing")
    ap.add_argument("--rim-tolerance", type=float, default=0.02,
                    help="a downward probe landing this far below the rim counts as "
                         "interior; lower it for a shallow bowl")
    ap.add_argument("--snap-max-m", type=float, default=0.02,
                    help="largest correction applied to put an object on a container floor")
    ap.add_argument("--fixture-sil-iters", type=int, default=100,
                    help="silhouette refinement iterations for a container pose; 0 keeps "
                         "the IoU-selected pose without refining it")
    ap.add_argument("--fixture-sil-loss-max", type=float, default=0.01,
                    help="reject a container pose above this silhouette loss. The object "
                         "default is 0.003, too strict for a large open container whose "
                         "mask carries the cavity, the far rim and the floor")
    ap.add_argument("--fixture-selection", choices=["iou", "quality"], default="iou",
                    help="iou = render-based view selection then silhouette refine "
                         "(default). quality = FoundPose's own best-scoring view, no "
                         "rendering and no silhouette")
    ap.add_argument("--pc-list", nargs="+", default=None)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--stream-fps", type=int, default=10)
    ap.add_argument("--stream-warmup-s", type=float, default=2.0)
    ap.add_argument("--init-timeout-s", type=float, default=60.0)
    ap.add_argument("--sil-iters", type=int, default=100)
    ap.add_argument("--auto-steps", action="store_true",
                    help="run the whole step list after a single Enter, instead of "
                         "asking before each step. Use it when every object is already "
                         "on the table at the start and nothing is placed mid-take. The "
                         "containers are then measured in that same pass, so there is no "
                         "separate container-only turn")
    ap.add_argument("--reuse-fixtures", action="store_true",
                    help="with --auto-steps, measure the containers once instead of at "
                         "the start of every round. Use it only if nothing moves them")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="stop after this many passes through the step list "
                         "(0 = until you answer q)")
    args = ap.parse_args(argv)

    fixtures = dict(args.fixture)
    steps: List[Step] = list(args.step)
    if args.obj:
        if steps:
            ap.error("--obj is a single-step shorthand; do not mix it with --step")
        fixtures.setdefault("box", args.box_obj)
        steps = [Step(args.obj, "box", TRAY_TARGET)]
    if not steps:
        ap.error("nothing to do: pass --step OBJ:SOURCE:TARGET (or the --obj shorthand)")
    for step in steps:
        if step.source != TABLE_SOURCE and step.source not in fixtures:
            ap.error(f"step {step.obj!r} picks from unknown container {step.source!r}; "
                     f"add --fixture {step.source}=<asset>")
        if step.target not in (TRAY_TARGET, GOAL_TARGET) and step.target not in fixtures:
            ap.error(f"step {step.obj!r} targets unknown container {step.target!r}; "
                     f"add --fixture {step.target}=<asset>")
    args.fixtures = fixtures
    args.steps = steps
    args.prompts = dict(args.prompt)
    return args


def filter_grasps(pool, patterns) -> tuple:
    """Drop library grasps matching any blacklist pattern; return (kept, dropped)."""
    if not patterns:
        return list(pool), []
    kept, dropped = [], []
    for grasp in pool:
        hay = f"{grasp.source} {grasp.episode}"
        (dropped if any(p in hay for p in patterns) else kept).append(grasp)
    return kept, dropped


def prompt_for(args, name: str) -> str:
    return args.prompts.get(name, default_prompt(name))


def container_lift_height(args, rim_z: float) -> float:
    """Lift high enough that the object clears a target container's rim by --into-gap.

    The release into a container is a drop from the carry height, so that height
    has to be reached by the lift itself.  The default 15 cm lift is measured
    from a grasp on the table, which for a tall bowl leaves the object below the
    intended release height and turns the drop into a descent into the bowl.
    """
    needed = float(rim_z) + float(args.into_gap) - TABLE_SURFACE_Z
    return float(min(max(args.lift_height, needed), args.max_lift_height))


def build_inference_args(args, *, obj: str, prompt: str, bearing_deg: float, pc_list,
                         lift_height: Optional[float] = None,
                         retreat_h: Optional[float] = None):
    """Reuse the inference runner's own parser so its defaults stay in one place."""
    argv = [
        "--obj", obj,
        "--arm", args.arm,
        "--transfer-mode", "joint0-arc",
        "--drop-target", "fixed-box",
        "--drop-mode", "controlled",
        "--joint0-drop-bearing-deg", str(bearing_deg),
        "--drop-h", str(args.max_descend),
        "--lift-height", str(args.lift_height if lift_height is None else lift_height),
        "--carry-clearance", str(args.carry_clearance),
        "--grasp-order", args.grasp_order,
        "--max-grasp-attempts", str(args.max_grasp_attempts),
        "--max-v8-attempts", str(args.max_v8_attempts),
        "--retreat-h", str(args.retreat_h if retreat_h is None else retreat_h),
        "--return-floor", str(args.return_floor),
        "--init-timeout-s", str(args.init_timeout_s),
        "--sil-iters", str(args.sil_iters),
        "--stream-fps", str(args.stream_fps),
        "--stream-warmup-s", str(args.stream_warmup_s),
        "--prompt", prompt,
        "--pc-list", *pc_list,
    ]
    if args.grasp_seed is not None:
        argv += ["--grasp-seed", str(args.grasp_seed)]
    if args.calib_dir:
        argv += ["--calib-dir", args.calib_dir]
    if args.execute:
        argv.append("--execute")
    return inference_parse_args(argv)


def _prompt(message: str) -> str:
    """Blocking operator prompt; treat EOF as a request to stop."""
    try:
        return input(message).strip().lower()
    except EOFError:
        print()
        return "q"


def measure_fixture(orch, args, *, name: str, asset: str, mesh_path: Path,
                    assets_root: Path, intrinsics, extrinsics, image_hw, pc_serials,
                    out_dir: Path, c2r: np.ndarray) -> ContainerFixture:
    """Estimate one container's pose with FoundPose and measure its geometry.

    Silhouette refinement is kept on: a container is measured once per take, so
    its accuracy matters more than its latency.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_for(args, name if name in args.prompts else asset)
    orch.init_object(obj_name=asset, mesh_path=str(mesh_path),
                     assets_root=str(assets_root),
                     intrinsics_full=intrinsics, extrinsics_full=extrinsics,
                     image_hw=image_hw, mode="live", pc_serials=pc_serials)
    masks, poses, collect_timing = orch.collect_payloads(
        prompt=prompt, timeout_s=args.init_timeout_s, save_capture_dir=str(out_dir))
    pose_world, select_timing = orch.refine_from_payloads(
        masks, poses, sil_iters=args.fixture_sil_iters,
        sil_loss_threshold=args.fixture_sil_loss_max,
        selection_mode=args.fixture_selection, save_capture_dir=str(out_dir))
    timing = {**collect_timing, **select_timing, "prompt": prompt}
    (out_dir / "perception.json").write_text(json.dumps(timing, indent=2, default=str))
    if pose_world is None:
        # Name the actual rejection: "no reliable pose" is three different
        # failures with three different fixes.
        reason = str(select_timing.get("reason", "unknown"))
        hint = {
            "no_candidates_or_masks":
                f"no capture PC returned a usable {asset} pose/mask. Check the prompt "
                f"(now {prompt!r}; set another with --prompt {name}=...) and that the "
                f"container is visible.",
            "iou_select_failed":
                "the rendered IoU selection failed for every candidate view.",
        }.get(reason)
        if hint is None and reason.startswith("sil_loss_too_high"):
            hint = (f"the silhouette refinement rejected the pose at "
                    f"{select_timing.get('sil_loss')} > --fixture-sil-loss-max "
                    f"{args.fixture_sil_loss_max}. A large open container legitimately "
                    f"scores worse than a small convex object: raise the limit, use "
                    f"--fixture-sil-iters 0 to keep the IoU-selected pose, or "
                    f"--fixture-selection quality. Check the pose before trusting one "
                    f"accepted this way.")
        raise RuntimeError(
            f"no reliable {asset} pose ({reason}); "
            f"{select_timing.get('n_candidates', 0)} pose candidate(s), "
            f"{select_timing.get('n_masks', 0)} mask(s). "
            + (hint or f"diagnostics saved to {out_dir / 'perception.json'}"))
    np.save(out_dir / "pose_world.npy", pose_world)
    pose_robot = np.linalg.inv(np.asarray(c2r, dtype=np.float64)) @ pose_world
    fixture = build_fixture(name, asset, str(mesh_path), pose_robot, perception=timing,
                            interior_inset=args.interior_inset,
                            rim_tolerance=args.rim_tolerance,
                            floor_percentile=args.floor_percentile,
                            include_floor=not args.no_container_floor)
    m = fixture.measure
    if select_timing.get("sil_loss") is not None:
        print(f"[{name}] sil_loss={select_timing['sil_loss']:.5f} "
              f"(limit {args.fixture_sil_loss_max}), view={select_timing.get('best_serial')}")
    print(f"[{name}] {asset} at xyz={np.round(pose_robot[:3, 3], 3).tolist()}, "
          f"bearing {fixture.bearing_deg:.1f} deg")
    print(f"[{name}] outer {m['outer_size_xy'][0]:.3f}x{m['outer_size_xy'][1]:.3f} m, "
          f"interior {m['interior_size_xy'][0]:.3f}x{m['interior_size_xy'][1]:.3f} m "
          f"({m['n_interior_probes']} probes), floor z={fixture.floor_z:.3f}, "
          f"rim z={fixture.rim_z:.3f}")
    (out_dir / "fixture.json").write_text(json.dumps(fixture.to_json(), indent=2, default=str))
    return fixture


def measure_goal_pose(orch, args, *, obj: str, mesh_path: Path, assets_root: Path,
                      intrinsics, extrinsics, image_hw, pc_serials,
                      out_dir: Path, c2r: np.ndarray) -> dict:
    """Turn 1: record where the operator wants this object to end up.

    The object is standing at its goal spot, so a plain object init measures it
    exactly like a pick would.  Both the pose and the mesh's contact height are
    stored: the later lay-down descends until the object's own mesh bottom is
    back at the height it had here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_for(args, obj)
    orch.init_object(obj_name=obj, mesh_path=str(mesh_path), assets_root=str(assets_root),
                     intrinsics_full=intrinsics, extrinsics_full=extrinsics,
                     image_hw=image_hw, mode="live", pc_serials=pc_serials)
    masks, poses, collect_timing = orch.collect_payloads(
        prompt=prompt, timeout_s=args.init_timeout_s, save_capture_dir=str(out_dir))
    pose_world, select_timing = orch.refine_from_payloads(
        masks, poses, sil_iters=args.sil_iters, save_capture_dir=str(out_dir))
    timing = {**collect_timing, **select_timing, "prompt": prompt}
    (out_dir / "perception.json").write_text(json.dumps(timing, indent=2, default=str))
    if pose_world is None:
        raise RuntimeError(
            f"no reliable {obj} pose at its goal spot "
            f"({select_timing.get('reason', 'unknown')}); "
            f"{select_timing.get('n_candidates', 0)} candidate(s), "
            f"{select_timing.get('n_masks', 0)} mask(s). Check the prompt "
            f"(now {prompt!r}) and that the object is visible there.")
    np.save(out_dir / "pose_world.npy", pose_world)
    pose_robot = np.linalg.inv(np.asarray(c2r, dtype=np.float64)) @ pose_world

    from src.demo.continuous_basket.tabletop import load_mesh_vertices, mesh_bottom_z

    contact_z = float(mesh_bottom_z(pose_robot, load_mesh_vertices(str(mesh_path))))
    goal = {"obj": obj, "pose_robot": pose_robot.tolist(), "contact_z": contact_z,
            "perception": timing}
    (out_dir / "goal.json").write_text(json.dumps(goal, indent=2, default=str))
    print(f"[goal] {obj} at xyz={np.round(pose_robot[:3, 3], 3).tolist()}, "
          f"contact z={contact_z:.3f} m")
    return goal


def main(argv: Optional[list] = None) -> None:
    from paradex.calibration.utils import load_current_C2R
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.system import get_camera_list, get_pc_ip

    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.planner import GraspPlanner
    from autodex.utils.path import get_obj_root, project_dir

    from src.demo.banana_test.run_demo import (
        ASSETS_BASE,
        CAM_PARAM_ROOT,
        DEFAULT_PC_LIST,
        _clear_camera_errors,
        _ensure_camera_lock,
        _load_calib,
        _planner_robot,
        _rcc_start,
        _safe,
        _stop_with_timeout,
        _warn_if_not_streaming,
        quiet_curobo,
    )
    from src.demo.inference.grasp_library import load_demo_grasps
    from src.execution.scene_cfg import check_mesh_frame_match

    args = parse_args(argv)
    pc_list = args.pc_list or DEFAULT_PC_LIST
    obj_asset_root = Path(get_obj_root(GRASP_ASSET_VERSION))

    def asset_paths(name: str) -> tuple:
        return (obj_asset_root / name / "raw_mesh" / f"{name}.obj", ASSETS_BASE / name)

    objects = list(dict.fromkeys(step.obj for step in args.steps))
    for name in objects + list(args.fixtures.values()):
        mesh, assets = asset_paths(name)
        if not mesh.is_file():
            raise SystemExit(f"mesh not found: {mesh}")
        if not (assets / "object_repre/v1" / name / "1/repre.pth").is_file():
            raise SystemExit(f"FoundPose representation missing for {name} under {assets}")
    for name in objects:
        mesh, _ = asset_paths(name)
        frame_ok, frame_msg = check_mesh_frame_match(name, str(mesh), str(obj_asset_root))
        if not frame_ok:
            raise SystemExit(f"[mesh_frame] {frame_msg}")
    grasps = {}
    for name in objects:
        kept, dropped = filter_grasps(load_demo_grasps(name), args.exclude_grasp)
        grasps[name] = kept
        note = f" ({len(dropped)} excluded)" if dropped else ""
        print(f"[library] {len(kept)} fixed Inspire successes for {name}{note}")
        for grasp in dropped:
            print(f"[library]   excluded {grasp.source}: {grasp.episode}")
        if not kept:
            raise SystemExit(
                f"every success-library grasp for {name} was excluded; the run would "
                f"fall straight through to the v8 candidate pool. Relax --exclude-grasp.")
    print("[plan] " + "; ".join(step.describe() for step in args.steps))

    calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    intrinsics, extrinsics, height, width = _load_calib(calib_dir)
    pc_ips = [get_pc_ip(pc) for pc in pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in pc_list}
    active = {serial for pc in pc_list for serial in pc_serials[pc]}
    intrinsics = {s: v for s, v in intrinsics.items() if s in active}
    extrinsics = {s: v for s, v in extrinsics.items() if s in active}
    print(f"calib: {calib_dir.name}  ({len(intrinsics)} cams, {height}x{width})")
    c2r = load_current_C2R()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = (Path(project_dir) / "experiment" / EXPERIMENT_NAME / "inspire"
                   / "_".join(objects) / stamp)
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[session] {session_dir}")

    rcc = remote_camera_controller("box_to_tray", pc_list=pc_list, stall_timeout=15.0)
    if not _ensure_camera_lock(rcc):
        _safe("rcc.end", rcc.end)
        raise SystemExit("camera daemons are owned by another controller; "
                         "no robot motion was sent")
    if not _clear_camera_errors(rcc):
        _safe("rcc.end", rcc.end)
        raise SystemExit("capture cameras remain in an error state; "
                         "no robot motion was sent")
    print(f"[stream] start @ {args.stream_fps} FPS...")
    try:
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    except Exception as exc:
        _safe("rcc.end", rcc.end)
        raise SystemExit(f"could not start the camera stream; no robot motion "
                         f"was sent: {exc!r}") from None
    time.sleep(args.stream_warmup_s)
    if not _warn_if_not_streaming(rcc):
        _safe("rcc.stop", rcc.stop)
        _safe("rcc.end", rcc.end)
        raise SystemExit("camera stream did not start; no robot motion was sent")

    orch = None
    executor = None
    fixtures: Dict[str, ContainerFixture] = {}
    goals: Dict[str, dict] = {}
    goal_objects = [step.obj for step in args.steps if step.target == GOAL_TARGET]
    takes = 0
    measurements = 0
    try:
        print(f"[orch] connecting to the init daemons on {pc_list}...")
        orch = InitOrchestrator(pc_list=pc_list, capture_ips=pc_ips)

        def measure_all(index: int) -> Dict[str, ContainerFixture]:
            out: Dict[str, ContainerFixture] = {}
            for name, asset in args.fixtures.items():
                mesh, assets = asset_paths(asset)
                out[name] = measure_fixture(
                    orch, args, name=name, asset=asset, mesh_path=mesh,
                    assets_root=assets, intrinsics=intrinsics, extrinsics=extrinsics,
                    image_hw=(height, width), pc_serials=pc_serials,
                    out_dir=session_dir / "fixtures" / f"{index:02d}" / name, c2r=c2r)
            return out

        def measure_goals(index: int) -> Dict[str, dict]:
            out: Dict[str, dict] = {}
            for obj in dict.fromkeys(goal_objects):
                mesh, assets = asset_paths(obj)
                print(f"\nPlace the {obj} where it should END UP, then press Enter.")
                if _prompt("        (this pose is recorded as its goal; q to quit): ") == "q":
                    raise KeyboardInterrupt
                out[obj] = measure_goal_pose(
                    orch, args, obj=obj, mesh_path=mesh, assets_root=assets,
                    intrinsics=intrinsics, extrinsics=extrinsics,
                    image_hw=(height, width), pc_serials=pc_serials,
                    out_dir=session_dir / "goals" / f"{index:02d}" / obj, c2r=c2r)
            return out

        def save_goals() -> None:
            (session_dir / "goals.json").write_text(
                json.dumps(goals, indent=2, default=str))

        def save_fixtures() -> None:
            (session_dir / "fixtures.json").write_text(json.dumps(
                {name: f.to_json() for name, f in fixtures.items()},
                indent=2, default=str))

        # ── turn 1: measure the containers ──────────────────────────────────
        # With --auto-steps everything is on the table from the start, so the
        # containers are measured inside the run pass instead of in a turn of
        # their own.  An object standing in a container occludes part of it,
        # which is exactly the situation the run has to cope with anyway.
        measure_with_run = args.auto_steps and not args.fixtures_json
        if args.fixtures_json:
            data = json.loads(Path(args.fixtures_json).expanduser().read_text())
            fixtures = {name: ContainerFixture.from_json(v) for name, v in data.items()}
            missing = set(args.fixtures) - set(fixtures)
            if missing:
                raise SystemExit(f"{args.fixtures_json} has no entry for {sorted(missing)}")
            print(f"[fixtures] reusing {args.fixtures_json}: "
                  + ", ".join(f"{n} floor {f.floor_z:.3f} rim {f.rim_z:.3f}"
                              for n, f in fixtures.items()))
        while not fixtures and not measure_with_run:
            names = ", ".join(f"{n} ({a})" for n, a in args.fixtures.items())
            print(f"\nREADY — place the containers on the table: {names}.")
            print("        Nothing else: the step objects are placed later, one "
                  "at a time.")
            answer = _prompt("        press Enter to measure them (q to quit): ")
            if answer == "q":
                return
            try:
                fixtures = measure_all(measurements)
            except Exception as exc:
                print(f"[fixtures] measurement failed: {exc}")
                print("[fixtures] adjust the containers or the prompts, then try again")
            measurements += 1
        if fixtures:
            save_fixtures()

        # ── turn 1b: record the goal poses, then let the objects be moved ────
        if goal_objects:
            if args.goals_json:
                goals = json.loads(Path(args.goals_json).expanduser().read_text())
                missing = set(goal_objects) - set(goals)
                if missing:
                    raise SystemExit(f"{args.goals_json} has no goal for {sorted(missing)}")
                print(f"[goals] reusing {args.goals_json}")
            else:
                try:
                    goals = measure_goals(measurements)
                except KeyboardInterrupt:
                    return
                save_goals()
                moves = "; ".join(
                    f"{step.obj} -> "
                    + ("the table" if step.source == TABLE_SOURCE else f"the {step.source}")
                    for step in args.steps if step.target == GOAL_TARGET)
                print(f"\nGoal pose(s) recorded.  Now move: {moves}.")
                if _prompt("        press Enter when the objects are in place "
                           "(q to quit): ") == "q":
                    return

        # ── planner and robot ───────────────────────────────────────────────
        planner_robot = _planner_robot(args.arm, "inspire")
        print(f"[planner] warmup ({planner_robot})...")
        planner = GraspPlanner(hand=planner_robot)
        quiet_curobo()
        print(f"[executor] connect ({args.arm})...")
        if args.arm == "franka":
            from src.execution.franka_executor import FrankaExecutor
            executor = FrankaExecutor(hand_name="inspire")
            executor.set_speed_profile_planner(planner)
        else:
            from autodex.executor.real import RealExecutor
            executor = RealExecutor(hand_name="inspire")
        if args.hold_retract_orientation and hasattr(executor, "retract_hold_orientation"):
            executor.retract_hold_orientation = True
            print("[executor] post-release return holds the wrist attitude until home")
        if args.home_pose == "init" and hasattr(executor, "retract_goal"):
            # Park where the approach starts, so the retract does not undo
            # itself at the next step.
            executor.retract_goal = executor._arm_init
            print("[executor] home/retract pose: FR3_INIT (no clear-view J0 swing)")

        # ── turn 2: run the step sequence ───────────────────────────────────
        current_object = None
        rounds = 0
        stop = False
        while not stop:
            rounds += 1
            if args.auto_steps:
                names = ", ".join(f"{n} ({a})" for n, a in args.fixtures.items())
                print("\n" + "; ".join(step.describe() for step in args.steps))
                if measure_with_run:
                    print(f"        Everything on the table, including: {names}.")
                answer = _prompt("        Enter = run the whole sequence, "
                                 "r = re-measure the containers, q = quit: ")
                if answer == "q":
                    break
                remeasure = answer == "r" or (
                    measure_with_run and (not fixtures or not args.reuse_fixtures))
                if remeasure:
                    try:
                        fixtures = measure_all(measurements)
                        save_fixtures()
                    except Exception as exc:
                        print(f"[fixtures] measurement failed: {exc}")
                    measurements += 1
                    current_object = None      # the daemons hold a container now
                    if answer == "r" or not fixtures:
                        rounds -= 1            # this pass did not run any step
                        continue
            for step in args.steps:
                if not args.auto_steps:
                    where = ("on the table" if step.source == TABLE_SOURCE
                             else f"in the {step.source}")
                    print(f"\nPlace the {step.obj} {where}.  ({step.describe()})")
                    answer = _prompt("        Enter = run this step, "
                                     "r = re-measure the containers, s = skip, q = quit: ")
                    if answer == "q":
                        stop = True
                        break
                    if answer == "s":
                        continue
                    if answer == "r":
                        try:
                            fixtures = measure_all(measurements)
                            save_fixtures()
                        except Exception as exc:
                            print(f"[fixtures] measurement failed: {exc}")
                        measurements += 1
                        current_object = None  # the daemons hold a container now
                        continue
                else:
                    print(f"\n[step] {step.describe()}")

                mesh, assets = asset_paths(step.obj)
                if current_object != step.obj:
                    print(f"[orch] init for {step.obj}...")
                    orch.init_object(obj_name=step.obj, mesh_path=str(mesh),
                                     assets_root=str(assets),
                                     intrinsics_full=intrinsics, extrinsics_full=extrinsics,
                                     image_hw=(height, width), mode="live",
                                     pc_serials=pc_serials)
                    current_object = step.obj

                source = None if step.source == TABLE_SOURCE else fixtures[step.source]
                goal_pose = None
                lift_height = args.lift_height
                retreat_h = args.retreat_h
                if step.target == GOAL_TARGET:
                    goal = goals[step.obj]
                    goal_pose = np.asarray(goal["pose_robot"], dtype=np.float64)
                    turn_deg = None
                    bearing = float(np.degrees(np.arctan2(goal_pose[1, 3], goal_pose[0, 3])))
                    # Land the object's mesh bottom back on the height it had
                    # when the goal was recorded.
                    surface_z = float(goal["contact_z"])
                    clearance = args.goal_clearance
                    release_xy = goal_pose[:2, 3]
                elif step.target == TRAY_TARGET:
                    turn_deg = (None if args.place_bearing_deg is not None
                                else args.place_turn_deg)
                    bearing = float(args.place_bearing_deg or 0.0)
                    surface_z = args.tray_top_z
                    clearance = args.lay_down_clearance
                    release_xy = args.tray_radius * np.array(
                        [np.cos(np.deg2rad(bearing)), np.sin(np.deg2rad(bearing))])
                else:
                    target = fixtures[step.target]
                    turn_deg = None
                    bearing = target.bearing_deg
                    # Drop from the carry height rather than descending: the
                    # hand must never enter a container it is not picking from.
                    # The lift is what puts the object --into-gap over the rim,
                    # and the release is direct (place_surface_z stays None, so
                    # run_once keeps its fixed-box direct release).
                    surface_z = None
                    clearance = 0.0
                    lift_height = container_lift_height(args, target.rim_z)
                    retreat_h = args.drop_retreat_h
                    release_xy = np.asarray(target.center_xy, dtype=np.float64)
                if goal_pose is not None:
                    print(f"[place] measured goal pose "
                          f"{np.round(goal_pose[:3, 3], 3).tolist()} "
                          f"({args.goal_pose_mode} attitude), contact z={surface_z:.3f} m")
                elif turn_deg is not None:
                    print(f"[place] J0 turn {turn_deg:+.1f} deg, laying the object down "
                          f"{clearance * 100:.1f} cm above z={surface_z:.3f} m")
                else:
                    print(f"[place] bearing {bearing:.1f} deg, lift {lift_height * 100:.0f} cm "
                          f"then release over the {step.target} rim "
                          f"({fixtures[step.target].rim_z:.3f} m + "
                          f"{args.into_gap * 100:.0f} cm)")

                takes += 1
                run_dir = session_dir / "takes" / f"{takes:03d}_{step.obj}"
                run_dir.mkdir(parents=True, exist_ok=True)
                inf_args = build_inference_args(
                    args, obj=step.obj, prompt=prompt_for(args, step.obj),
                    bearing_deg=bearing, pc_list=pc_list, lift_height=lift_height,
                    retreat_h=retreat_h)
                if args.execute and (args.home_before_step or takes == 1):
                    # The first step has to reach the home pose from wherever
                    # the arm started; later ones are already parked there by
                    # the previous step's reset.
                    clear_view = args.home_pose == "clear_view"
                    print(f"[executor] moving to home ({args.home_pose})")
                    executor.home(clear_view=clear_view)
                hook = make_scene_hook(list(fixtures.values()), source=source,
                                       require_in_source=not args.no_source_check,
                                       snap_max_m=args.snap_max_m,
                                       model=args.container_model)
                try:
                    record = run_once(
                        inf_args, orch=orch, planner=planner, executor=executor, rcc=rcc,
                        target_xyz=np.array([
                            release_xy[0], release_xy[1],
                            surface_z if surface_z is not None else args.tray_top_z]),
                        run_dir=run_dir, grasps=grasps[step.obj],
                        scene_cfg_hook=hook, place_surface_z=surface_z,
                        place_clearance=clearance, joint0_turn_deg=turn_deg,
                        place_object_pose=goal_pose,
                        place_pose_mode=args.goal_pose_mode,
                    )
                except KeyboardInterrupt:
                    record = {"success": False, "reason": "interrupted"}
                    print("\n[interrupted]")
                except Exception as exc:
                    # Never issue an automatic release or home after an unknown
                    # error: the hand may still hold the object.
                    record = {"success": False, "reason": "fatal", "error": repr(exc),
                              "action": "stopped_without_robot_reset"}
                    print(f"\n[FATAL] {exc!r}\n        stopping this step without a robot reset")
                record["step"] = {"object": step.obj, "source": step.source,
                                  "target": step.target, "round": rounds}
                record["place_target"] = {
                    "surface_z": surface_z, "clearance_m": clearance,
                    "lift_height_m": lift_height, "retreat_h_m": retreat_h,
                    "release_mode": ("direct_over_rim" if surface_z is None
                                     else "lay_down"),
                    "joint0_turn_deg": turn_deg,
                    "release_bearing_deg": (None if turn_deg is not None else bearing),
                    "goal_pose_robot": (None if goal_pose is None else goal_pose.tolist()),
                    "goal_pose_mode": (None if goal_pose is None else args.goal_pose_mode)}
                record["fixtures"] = {n: {"obj": f.obj, "floor_z": f.floor_z,
                                          "rim_z": f.rim_z}
                                      for n, f in fixtures.items()}
                (run_dir / "result.json").write_text(json.dumps(record, indent=2, default=str))
                status = ("SUCCESS" if record.get("success") else
                          ("DRY-RUN" if record.get("success") is None
                           else record.get("reason", "FAIL")))
                print(f"\nSTEP {takes} ({step.describe()}): {status}   "
                      f"saved to {run_dir}/result.json")
                if record.get("reason") in ("fatal", "interrupted"):
                    stop = True
                    break
                if not record.get("success") and args.stop_on_step_failure:
                    print(f"        step failed ({record.get('reason')}); stopping this "
                          "round instead of moving on to the next object "
                          "(--no-stop-on-step-failure to continue)")
                    stop = True
                    break
            if stop:
                break
            if args.max_rounds and rounds >= args.max_rounds:
                print(f"[session] reached --max-rounds {args.max_rounds}")
                break
            print(f"\nRound {rounds} done ({takes} step(s) run).")
            if _prompt("        Enter = run the sequence again, q = quit: ") == "q":
                break
    finally:
        if executor is not None:
            _safe("executor.shutdown", executor.shutdown)
        if orch is not None:
            _safe("orch.close", orch.close)
        for fn, name in ((rcc.stop, "rcc.stop"), (rcc.end, "rcc.end")):
            _stop_with_timeout(name, fn)
        print(f"[session] {takes} step(s) -> {session_dir}")


if __name__ == "__main__":
    main()
