#!/usr/bin/env python3
"""Interactive, executor-free test of the 5 mm Jacobian lift primitive.

The default session measures an empty Charuco board once, then repeatedly
asks for a constructed scenario.  ``--object-source perception`` instead
asks only for an object name and obtains its complete 6D pose using the same
FoundPose and scene-conversion calls as ``src/execution/run_auto.py``.

No code here instantiates ``FrankaExecutor``/``RealExecutor`` or sends a robot
motion command.  Camera acquisition in live modes is intentionally retained.

Choose the geometric arm model with ``--arm``.  Both modes use the same
Inspire candidate assets and camera/scene pipeline; only the arm kinematics,
collision model, and saved replay robot differ.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.demo.lift_test.board import (load_proxy, point_in_polygon,
                                      proxy_from_charuco_measurement,
                                      save_json, y_interval_at_x)
from src.demo.lift_test.candidate_policy import (CandidatePolicy,
                                                 coverage_metadata,
                                                 make_candidate_policy,
                                                 order_coverage_keys)
from src.demo.lift_test.execution_trajectory import (ExecutionProfile,
                                                      build_execution_trajectory,
                                                      profile_for_arm)
from src.demo.lift_test.jacobian_lift import (LiftOptions, continue_vertical_lift,
                                              options_as_dict, save_step_csv)
from src.demo.lift_test.batch_validation import (
    check_robot_states_batch as _check_robot_states_batch,
    fk_wrist_batch as _fk_wrist_batch,
    object_bottom_z_batch as _object_bottom_z_batch,
)
from autodex.timing import TimingRecorder

# Keep ``--help`` and board-only source inspection usable in a lightweight base
# environment.  Curobo/trimesh/FoundPose are resolved only after argparse has
# accepted an actual run request.
GraspPlanner = None
trimesh = None


# The candidate catalogue is keyed by hand, not arm: both modes use the same
# right-Inspire candidate set.  Keeping this mapping here makes the arm switch
# explicit at the one place where it affects planning.
ARM_TO_PLANNER_ROBOT = {
    "franka": "fr3_inspire",
    "xarm": "inspire",
}


def _planner_robot_for_arm(arm: str) -> str:
    """Return the GraspPlanner hand/configuration key for an arm mode."""
    try:
        return ARM_TO_PLANNER_ROBOT[arm]
    except KeyError as exc:
        raise ValueError(f"unsupported arm mode: {arm!r}") from exc


def _load_runtime_dependencies() -> None:
    global GraspPlanner, trimesh, _to_curobo_world, _without_target_mesh
    global cart2se3, se32cart, get_candidate_path, get_obj_root, project_dir
    global get_cyl_axis_local, get_cyl_yaw_grid, table_cuboid
    global classify_tabletop_pose, check_mesh_frame_match, find_planning_mesh
    global pose_world_to_scene_cfg
    global load_candidate, load_openpose_for_candidates, _expand_candidates_cyl, _to_curobo_pose
    import trimesh as _trimesh
    from autodex.planner import GraspPlanner as _GraspPlanner
    from autodex.planner.planner import (_expand_candidates_cyl as _expand,
                                         _to_curobo_pose as _pose,
                                         _to_curobo_world as _world,
                                         _without_target_mesh as _without)
    from autodex.utils.conversion import cart2se3 as _cart2se3, se32cart as _se32cart
    from autodex.utils.path import (get_candidate_path as _candidate_path,
                                    get_obj_root as _obj_root,
                                    load_candidate as _load_candidate,
                                    load_openpose_for_candidates as _load_openpose,
                                    project_dir as _project_dir)
    from autodex.utils.symmetry import get_cyl_axis_local as _cyl_axis, get_cyl_yaw_grid as _cyl_grid
    from autodex.utils.tabletop_geometry import table_cuboid as _table_cuboid
    from src.experiment.reset.tabletop_pose import classify_tabletop_pose as _classify
    from src.execution.scene_cfg import (check_mesh_frame_match as _frame_match,
                                         find_planning_mesh as _planning_mesh,
                                         pose_world_to_scene_cfg as _scene_from_pose)
    GraspPlanner, trimesh = _GraspPlanner, _trimesh
    _to_curobo_world, _without_target_mesh = _world, _without
    _expand_candidates_cyl, _to_curobo_pose = _expand, _pose
    cart2se3, se32cart = _cart2se3, _se32cart
    get_candidate_path, get_obj_root, project_dir = _candidate_path, _obj_root, _project_dir
    get_cyl_axis_local, get_cyl_yaw_grid, table_cuboid = _cyl_axis, _cyl_grid, _table_cuboid
    classify_tabletop_pose = _classify
    check_mesh_frame_match, find_planning_mesh, pose_world_to_scene_cfg = _frame_match, _planning_mesh, _scene_from_pose
    load_candidate, load_openpose_for_candidates = _load_candidate, _load_openpose


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def _load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    if trimesh is None:
        _load_runtime_dependencies()
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _mesh_bottom_z(mesh_path: str | Path, T: np.ndarray) -> float:
    mesh = _load_mesh(mesh_path)
    h = np.c_[np.asarray(mesh.vertices), np.ones(len(mesh.vertices))]
    return float((np.asarray(T) @ h.T).T[:, 2].min())


class UserQuit(RuntimeError):
    pass


class CaptureContext:
    """Camera calibration and optional live perception lifecycle.

    The calibration layout, capture-PC list, FoundPose assets, and camera
    startup helpers are imported from the current runner rather than copied,
    so a live trial uses the same calibration semantics as AutoDex.
    """
    def __init__(self, args: argparse.Namespace):
        from paradex.utils.system import get_camera_list, get_pc_ip
        from src.execution.run_auto import (ASSETS_BASE, CAM_PARAM_ROOT,
                                            DEFAULT_PC_LIST, _load_calib)

        self.args = args
        self.assets_base = Path(ASSETS_BASE)
        self.pc_list = list(args.pc_list or DEFAULT_PC_LIST)
        calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                     else sorted(Path(CAM_PARAM_ROOT).iterdir())[-1])
        self.calib_dir = calib_dir
        intr, extr, self.height, self.width = _load_calib(calib_dir)
        self.capture_ips = [get_pc_ip(pc) for pc in self.pc_list]
        self.pc_serials = {pc: get_camera_list(pc) for pc in self.pc_list}
        active = {serial for pc in self.pc_list for serial in self.pc_serials[pc]}
        self.intrinsics = {s: v for s, v in intr.items() if s in active}
        self.extrinsics = {s: v for s, v in extr.items() if s in active}
        if not self.intrinsics or set(self.intrinsics) != set(self.extrinsics):
            raise RuntimeError("active camera calibration is incomplete")
        self._orch = None
        self._rcc = None

    def measure_board(self, session_dir: Path) -> tuple[dict, dict]:
        """Run the same empty-board Charuco measurement as pipeline preflight."""
        from paradex.calibration.utils import load_current_C2R
        from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator
        from src.execution.charuco_tabletop import (measure_tabletop_from_images,
                                                    save_tabletop_measurement)

        out = session_dir / "board"
        images_dir = out / "images"
        out.mkdir(parents=True, exist_ok=True)
        # Match run_pipeline's ordering: ownership/error recovery and a
        # warmed camera stream must exist before SnapshotOrchestrator sends a
        # one-shot ``snap`` request. Otherwise a daemon may accept ``snap``
        # while no cameras are acquiring and return a misleading 0/N frame set.
        self._ensure_camera_stream()
        snap = SnapshotOrchestrator(
            pc_list=self.pc_list, capture_ips=self.capture_ips,
            port_snap=self.args.port_snap, port_cmd=self.args.port_snap_cmd,
        )
        try:
            t0 = perf_counter()
            payloads, snap_timing = snap.snap(
                n_expected=len(self.intrinsics), timeout_s=self.args.snapshot_timeout_s,
                save_dir_local=str(images_dir), decode=True,
            )
            snapshot_s = perf_counter() - t0
            images = {serial: item["image"] for serial, item in payloads.items()
                      if item.get("image") is not None}
            print(f"[board] snapshot {len(images)}/{len(self.intrinsics)} cameras", flush=True)
            if len(images) != len(self.intrinsics):
                diagnostics = {
                    "snapshot_timing": snap_timing,
                    "received_serials": sorted(payloads),
                    "decoded_serials": sorted(images),
                }
                try:
                    diagnostics["camera_status"] = self._rcc.get_status()
                except Exception as exc:
                    diagnostics["camera_status_error"] = repr(exc)
                _write_json(out / "snapshot_failure.json", diagnostics)
                raise RuntimeError(
                    "board measurement needs all active cameras; got "
                    f"{len(images)}/{len(self.intrinsics)} "
                    f"(diagnostics: {out / 'snapshot_failure.json'})")
            t0 = perf_counter()
            c2r = np.asarray(load_current_C2R(), dtype=np.float64)
            measurement = measure_tabletop_from_images(
                images, self.intrinsics, self.extrinsics, c2r)
            measurement_s = perf_counter() - t0
            measurement["snapshot_timing"] = snap_timing
            measurement["timing_s"] = {
                "snapshot_s": round(snapshot_s, 3),
                "measurement_s": round(measurement_s, 3),
                "total_s": round(snapshot_s + measurement_s, 3),
            }
            np.save(out / "C2R.npy", c2r)
            save_tabletop_measurement(measurement, out)
            proxy = proxy_from_charuco_measurement(measurement)
            save_json(out / "board_proxy.json", proxy)
            metrics = measurement.get("metrics", {})
            print(
                f"[board] Charuco corners={metrics.get('corners_triangulated', '?')}/"
                f"{metrics.get('corners_expected', '?')} "
                f"plane_rms={metrics.get('plane_rms_mm', float('nan')):.2f}mm "
                f"table_z={measurement['table_surface_z_m']:.4f}m",
                flush=True,
            )
            return measurement, proxy
        finally:
            try:
                snap.close()
            except Exception:
                pass

    def _ensure_camera_stream(self) -> None:
        """Start the production camera lifecycle for snapshots and FoundPose."""
        if self._rcc is not None:
            return
        from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
        from src.execution.run_auto import (_clear_camera_errors,
                                            _ensure_camera_lock, _rcc_start)

        self._rcc = remote_camera_controller("lift_test", pc_list=self.pc_list,
                                              stall_timeout=15.0)
        if not _ensure_camera_lock(self._rcc):
            raise RuntimeError("camera controller ownership could not be established")
        if not _clear_camera_errors(self._rcc):
            raise RuntimeError("capture cameras remain in an error state")
        _rcc_start(self._rcc, "stream", False, fps=self.args.stream_fps)
        if self.args.stream_warmup_s > 0:
            time.sleep(self.args.stream_warmup_s)

    def _ensure_live_perception(self) -> None:
        self._ensure_camera_stream()
        if self._orch is not None:
            return
        from autodex.perception.init_orchestrator import InitOrchestrator

        self._orch = InitOrchestrator(
            pc_list=self.pc_list, capture_ips=self.capture_ips,
            port_mask=self.args.port_mask, port_pose=self.args.port_pose,
            port_cmd=self.args.port_cmd,
        )

    def perceive_object(self, obj_name: str, grasp_version: str,
                        tabletop_geometry: Mapping[str, Any], out_dir: Path) -> dict:
        """Use the production FoundPose and production scene conversion path."""
        from paradex.calibration.utils import load_current_C2R

        self._ensure_live_perception()
        obj_root = Path(get_obj_root(grasp_version))
        mesh_path = obj_root / obj_name / "raw_mesh" / f"{obj_name}.obj"
        assets_root = self.assets_base / obj_name
        if not mesh_path.exists():
            raise FileNotFoundError(f"planning/perception mesh missing: {mesh_path}")
        repre = assets_root / "object_repre/v1" / obj_name / "1/repre.pth"
        if not repre.exists():
            raise FileNotFoundError(f"FoundPose representation missing: {repre}")
        ok, msg = check_mesh_frame_match(obj_name, str(mesh_path), str(obj_root))
        if not ok:
            raise RuntimeError(f"[mesh_frame] {msg}")

        self._orch.init_object(
            obj_name=obj_name, mesh_path=str(mesh_path), assets_root=str(assets_root),
            intrinsics_full=self.intrinsics, extrinsics_full=self.extrinsics,
            image_hw=(self.height, self.width), mode="live", pc_serials=self.pc_serials,
        )
        threshold = (float("inf") if self.args.perception_mode == "ignore_sil_loss"
                     else 0.003)
        t0 = perf_counter()
        pose_world, timing = self._orch.trigger_init(
            prompt=obj_name, save_capture_dir=str(out_dir / "perception" / "init_capture"),
            sil_iters=self.args.sil_iters, sil_lr=self.args.sil_lr,
            timeout_s=self.args.init_timeout_s, sil_loss_threshold=threshold,
        )
        elapsed = perf_counter() - t0
        if pose_world is None:
            return {"success": False, "reason": (timing or {}).get("reason", "perception_failed"),
                    "timing": timing, "perception_s": elapsed, "mesh_frame": msg}

        c2r = np.asarray(load_current_C2R(), dtype=np.float64)
        raw_robot = np.linalg.inv(c2r) @ np.asarray(pose_world, dtype=np.float64)
        # This is intentionally the current pipeline helper.  It applies the
        # cylinder/sphere orientation policy and, with a measured table,
        # raises only a mesh that would sit below the table surface.
        scene_cfg = pose_world_to_scene_cfg(
            np.asarray(pose_world, dtype=np.float64), c2r, obj_name, str(obj_root),
            tabletop_geometry=dict(tabletop_geometry),
        )
        scene_robot = cart2se3(np.asarray(scene_cfg["mesh"]["target"]["pose"], dtype=float))
        planning_mesh = find_planning_mesh(obj_name, str(obj_root))
        return {
            "success": True,
            "scene_cfg": scene_cfg,
            "pose_world": np.asarray(pose_world, dtype=np.float64),
            "c2r": c2r,
            "pose_robot_raw": raw_robot,
            "pose_robot_scene": scene_robot,
            "table_snap_delta_m": float(scene_robot[2, 3] - raw_robot[2, 3]),
            "mesh_bottom_z_raw_m": _mesh_bottom_z(planning_mesh, raw_robot),
            "mesh_bottom_z_scene_m": _mesh_bottom_z(planning_mesh, scene_robot),
            "timing": timing,
            "perception_s": elapsed,
            "mesh_frame": msg,
        }

    def close(self) -> None:
        for obj, method in ((self._orch, "close"), (self._rcc, "stop"),
                            (self._rcc, "end")):
            if obj is None:
                continue
            try:
                getattr(obj, method)()
            except Exception:
                pass


def _candidate_objects(hand: str, version: str) -> list[str]:
    root = Path(get_candidate_path(hand)) / version
    if not root.is_dir():
        return []
    names: set[str] = set()
    for p in root.iterdir():
        if p.is_dir():
            names.add(p.name)
        elif p.name.endswith(".tar.gz"):
            names.add(p.name[:-7])
        elif p.suffix == ".tgz":
            names.add(p.stem)
        elif p.suffix == ".zip":
            names.add(p.stem)
    return sorted(names)


def _tabletop_files(obj: str, version: str) -> list[Path]:
    root = Path(get_obj_root(version)) / obj / "processed_data" / "info" / "tabletop"
    return sorted(root.glob("*.npy")) if root.is_dir() else []


def _input(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise UserQuit
    if value.lower() in {"q", "quit"}:
        raise UserQuit
    return default if not value and default is not None else value


def _prompt_xy(proxy: Mapping[str, Any]) -> np.ndarray:
    vertices = np.asarray(proxy["vertices_xy_m"], dtype=np.float64)
    center = np.asarray(proxy["center_xy_m"], dtype=np.float64)
    bounds = proxy["xy_bounds_m"]
    print("\n[XY] Charuco proxy (CCW, m):")
    for i, p in enumerate(vertices):
        print(f"  v{i}: ({p[0]:.4f}, {p[1]:.4f})")
    print(f"  x range={bounds['x'][0]:.4f}..{bounds['x'][1]:.4f}, "
          f"y range={bounds['y'][0]:.4f}..{bounds['y'][1]:.4f}")
    while True:
        x_text = _input("  x [m], Enter=proxy center", default=f"{center[0]:.4f}")
        try:
            x = float(x_text)
        except ValueError:
            print("  x must be a number.")
            continue
        interval = y_interval_at_x(vertices, x)
        if interval is None:
            print("  x is outside the proxy. Try again.")
            continue
        y_default = center[1] if interval[0] <= center[1] <= interval[1] else sum(interval) / 2.0
        y_text = _input(f"  y [m], allowed {interval[0]:.4f}..{interval[1]:.4f}",
                        default=f"{y_default:.4f}")
        try:
            point = np.array([x, float(y_text)], dtype=np.float64)
        except ValueError:
            print("  y must be a number.")
            continue
        if point_in_polygon(point, vertices):
            return point
        print("  point lies outside the proxy. Try again.")


def _prompt_object(hand: str, version: str, last: str | None) -> str:
    names = _candidate_objects(hand, version)
    if not names:
        raise RuntimeError(f"no candidate objects in {get_candidate_path(hand)}/{version}")
    print("\n[object] available candidate objects:")
    print("  " + ", ".join(names))
    default = last if last in names else None
    while True:
        name = _input("  object name", default=default)
        if name in names:
            return name
        print("  not in the available-object list; choose exactly one listed name.")


def _prompt_tabletop_pose(obj: str, version: str, last: int | None) -> tuple[int, Path]:
    files = _tabletop_files(obj, version)
    if not files:
        raise RuntimeError(f"{obj}: no tabletop pose files under {get_obj_root(version)}")
    print(f"\n[tabletop pose] {obj}:")
    print("  " + ", ".join(f"{i}:{p.stem}" for i, p in enumerate(files)))
    default = str(last) if last is not None and 0 <= last < len(files) else None
    while True:
        text = _input("  pose number", default=default)
        try:
            index = int(text)
        except ValueError:
            print("  pose must be an integer shown above.")
            continue
        if 0 <= index < len(files):
            return index, files[index]
        print("  pose number is outside the displayed range.")


def _tabletop_transform(file_path: Path, xy: np.ndarray, table_z: float) -> np.ndarray:
    raw = np.asarray(np.load(file_path), dtype=np.float64)
    if raw.shape == (3, 3):
        T = np.eye(4)
        T[:3, :3] = raw
    elif raw.shape == (4, 4):
        T = raw.copy()
    else:
        raise ValueError(f"unsupported tabletop pose shape {raw.shape}: {file_path}")
    T[:3, 3] += np.array([float(xy[0]), float(xy[1]), float(table_z)])
    return T


def _constructed_scenario(obj: str, version: str, xy: np.ndarray,
                          pose_index: int, pose_file: Path,
                          tabletop_geometry: Mapping[str, Any]) -> dict:
    table_z = float(tabletop_geometry["table_surface_z_m"])
    T = _tabletop_transform(pose_file, xy, table_z)
    mesh_path = find_planning_mesh(obj, get_obj_root(version))
    scene_cfg = {
        "mesh": {"target": {"pose": se32cart(T).tolist(), "file_path": mesh_path}},
        "cuboid": {"table": table_cuboid(dict(tabletop_geometry))},
    }
    return {
        "success": True,
        "source": "constructed",
        "scene_cfg": scene_cfg,
        "pose_robot_raw": T.copy(),
        "pose_robot_scene": T.copy(),
        "selected_xy_m": np.asarray(xy, dtype=float),
        "selected_tabletop_pose": {"idx": pose_index, "filename": pose_file.name},
        "table_snap_delta_m": 0.0,
        "mesh_bottom_z_raw_m": _mesh_bottom_z(mesh_path, T),
        "mesh_bottom_z_scene_m": _mesh_bottom_z(mesh_path, T),
        "timing": {},
    }


def _pipeline_candidate_scene_filter(args: argparse.Namespace) -> str | None:
    """Mirror ``run_auto``'s v8 scene-type filtering decision."""
    return (args.candidate_scene_type or
            (args.scene if args.scene in ("wall", "shelf", "box") else None))


def _apply_pipeline_obstacles(scene_cfg: Mapping[str, Any], *, args: argparse.Namespace,
                              tabletop_geometry: Mapping[str, Any]) -> dict:
    """Build exactly the production obstacle family for a parity trial.

    ``add_obstacles`` mutates its argument.  A deep copy keeps the raw
    perception scene stored by :class:`CaptureContext` auditable alongside the
    exact production-style planning scene.
    """
    from autodex.planner.obstacles import add_obstacles

    return add_obstacles(
        deepcopy(dict(scene_cfg)), args.scene,
        wall_gap=args.wall_gap, wall_angle=args.wall_angle,
        seed=args.clutter_seed,
        clutter_min_dist=args.clutter_min_dist,
        clutter_max_dist=args.clutter_max_dist,
        clutter_n=args.clutter_n,
        shelf_width=args.shelf_width, shelf_depth=args.shelf_depth,
        shelf_height=args.shelf_height, shelf_gap=args.shelf_gap,
        shelf_back=not args.no_shelf_back,
        shelf_sides=not args.no_shelf_sides,
        shelf_top=not args.no_shelf_top,
        tabletop_geometry=dict(tabletop_geometry),
    )


def _pipeline_parity_gate(*, args: argparse.Namespace, obj: str,
                          version: str, hand: str,
                          tabletop: Mapping[str, Any] | None) -> dict | None:
    """Return an early production decision that precedes ``planner.plan``.

    A normal v8 ``run_auto`` trial does not enter candidate planning when a
    tabletop scene already has a successful grasp, or when its coverage is
    exhausted and a reorientation should happen first.  Reporting those
    outcomes here avoids falsely calling an empty Jacobian candidate list a
    lift-planning failure.
    """
    pose_stem = str((tabletop or {}).get("filename", "")).removesuffix(".npy")
    if not pose_stem:
        return None
    scene_id = str((tabletop or {}).get("idx"))
    scene_type = args.scene if args.scene != "table" else "table"
    from src.execution.run_auto import _scene_has_success

    if _scene_has_success(hand, version, obj, scene_type, scene_id):
        return {
            "reason": "pipeline_scene_already_done",
            "failure_code": "pipeline_scene_already_done",
            "scene_type": scene_type,
            "scene_id": scene_id,
            "production_decision": "skip_before_planning",
        }

    from autodex.utils.coverage import uncovered_scenes, pick_reorient_target
    remaining = uncovered_scenes(obj, pose_stem, hand=hand, version=version)
    if remaining is not None and len(remaining) == 0:
        target = pick_reorient_target(
            obj, pose_stem, hand=hand, version=version,
            obj_root=get_obj_root(version), success_root=None)
        gate = {
            "reason": "pipeline_reorient_needed",
            "failure_code": "pipeline_reorient_needed",
            "tabletop_pose_stem": pose_stem,
            "production_decision": "reorient_before_planning",
        }
        if target is not None:
            target_idx, target_stem, remaining_count = target
            gate.update({
                "reorient_target_idx": int(target_idx),
                "reorient_target_stem": str(target_stem),
                "reorient_uncovered_n": int(remaining_count),
            })
        else:
            gate["failure_code"] = "pipeline_reorient_target_absent"
        return gate
    return None


def _pipeline_parity_contract(args: argparse.Namespace, *, tabletop: Mapping[str, Any],
                              candidate_scene_type_filter: str | None) -> dict:
    """Persist the exact production-side choices matched by this experiment."""
    return {
        "schema_version": 1,
        "mode": "pipeline_parity_live_v1",
        "perception": "live_foundpose_then_pose_world_to_scene_cfg",
        "tabletop_classification_pose": "raw_robot_pose_before_scene_z_snap",
        "candidate_state": "shared_candidate_result_json",
        "skip_done": True,
        "skip_scenes_with_success": True,
        "success_only": False,
        "candidate_selection": (
            "run_auto_direct_scene_type_loading" if candidate_scene_type_filter is not None
            else "run_auto_remaining_coverage_order"),
        "scene": {
            "type": args.scene,
            "candidate_scene_type_filter": candidate_scene_type_filter,
            "wall_gap": args.wall_gap,
            "wall_angle": args.wall_angle,
            "clutter_seed": args.clutter_seed,
            "clutter_min_dist": args.clutter_min_dist,
            "clutter_max_dist": args.clutter_max_dist,
            "clutter_n": args.clutter_n,
            "shelf_width": args.shelf_width,
            "shelf_depth": args.shelf_depth,
            "shelf_height": args.shelf_height,
            "shelf_gap": args.shelf_gap,
            "shelf_back": not args.no_shelf_back,
            "shelf_sides": not args.no_shelf_sides,
            "shelf_top": not args.no_shelf_top,
        },
        "tabletop": dict(tabletop),
        "legacy_replacement": "only_legacy_lift_preflight_is_replaced_by_jacobian_continuation",
        "excluded": [
            "robot_execution_and_live_joint_tracking",
            "post_grasp_object_attached_collision_geometry",
            "run_auto_label_writeback",
        ],
    }


def _verify_execution_trajectory(
        planner: GraspPlanner, *, execution: Mapping[str, Any], scene_cfg: Mapping[str, Any],
        lift_start_qpos: np.ndarray, object_pose_at_grasp: np.ndarray,
        mesh_vertices: np.ndarray, table_surface_z_m: float,
        lift_options: LiftOptions,
) -> dict[str, Any]:
    """Validate the exact timestamped q samples intended for an executor.

    The approach keeps MotionGen's already interpolated q samples.  Squeeze
    and lift use the held-object world; each lift sample also proves that the
    C2 retiming remained a vertical, attached-object lift rather than merely a
    smooth joint-space curve.
    """
    from scipy.spatial.transform import Rotation

    qpos = np.asarray(execution["qpos"], dtype=np.float32)
    time_s = np.asarray(execution["time_s"], dtype=np.float64)
    phases = np.asarray(execution["phase"])
    n_arm = int(planner._n_arm)
    if (qpos.ndim != 2 or len(qpos) < 2 or len(time_s) != len(qpos)
            or len(phases) != len(qpos) or qpos.shape[1] != len(lift_start_qpos)):
        return {"success": False, "failure_code": "execution_trajectory_shape_invalid"}
    if not np.isfinite(qpos).all() or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0):
        return {"success": False, "failure_code": "execution_trajectory_time_invalid"}
    profile = execution.get("profile", {})
    expected_dt = float(profile.get("sample_dt_s", float("nan")))
    if (not np.isfinite(expected_dt) or expected_dt <= 0.0
            or not np.allclose(np.diff(time_s), expected_dt, atol=1.0e-8, rtol=0.0)):
        return {"success": False, "failure_code": "execution_trajectory_not_uniformly_timestamped"}
    if not np.allclose(qpos[phases == "lift", n_arm:],
                       np.asarray(lift_start_qpos)[n_arm:], atol=1.0e-6):
        return {"success": False, "failure_code": "execution_lift_hand_not_fixed"}

    # The saved path is sampled at fixed dt but follows the C2 reference
    # linearly between samples in the adapter.  Check finite-difference arm
    # velocity/acceleration under that exact streaming interpretation.  Finger
    # dynamics are controller-specific; the arm limits are the common planner
    # contract, and the squeeze has no arm motion.
    velocity_limit = float(profile.get("max_joint_velocity_rad_s", float("nan")))
    acceleration_limit = float(profile.get("max_joint_acceleration_rad_s2", float("nan")))
    held_scale = float(profile.get("held_object_speed_scale", float("nan")))
    if (not np.isfinite(velocity_limit) or not np.isfinite(acceleration_limit)
            or not np.isfinite(held_scale) or velocity_limit <= 0.0
            or acceleration_limit <= 0.0 or not 0.0 < held_scale <= 1.0):
        return {"success": False, "failure_code": "execution_trajectory_profile_invalid"}
    arm_velocity = np.diff(qpos[:, :n_arm], axis=0) / expected_dt
    interval_phase = np.asarray(phases[1:]).astype(str)
    interval_scale = np.where(interval_phase == "lift", held_scale, 1.0)
    allowed_velocity = velocity_limit * interval_scale[:, None]
    bad_velocity = np.where(np.abs(arm_velocity) > allowed_velocity + 1.0e-5)
    if len(bad_velocity[0]):
        return {
            "success": False, "failure_code": "execution_trajectory_velocity_limit",
            "sample_index": int(bad_velocity[0][0] + 1),
            "joint_index": int(bad_velocity[1][0]),
        }
    arm_acceleration = np.diff(arm_velocity, axis=0) / expected_dt
    if len(arm_acceleration):
        acceleration_scale = np.minimum(interval_scale[:-1], interval_scale[1:])
        allowed_acceleration = acceleration_limit * acceleration_scale[:, None]
        bad_acceleration = np.where(
            np.abs(arm_acceleration) > allowed_acceleration + 1.0e-4)
        if len(bad_acceleration[0]):
            return {
                "success": False, "failure_code": "execution_trajectory_acceleration_limit",
                "sample_index": int(bad_acceleration[0][0] + 2),
                "joint_index": int(bad_acceleration[1][0]),
            }

    limits = planner._motion_gen.kinematics.get_joint_limits().position.detach().cpu().numpy()
    limit_lo = np.asarray(limits[0, :qpos.shape[1]], dtype=np.float32)
    limit_hi = np.asarray(limits[1, :qpos.shape[1]], dtype=np.float32)
    bad_limit = np.where((qpos < limit_lo - 1.0e-6) | (qpos > limit_hi + 1.0e-6))
    if len(bad_limit[0]):
        return {
            "success": False, "failure_code": "execution_trajectory_joint_limit",
            "sample_index": int(bad_limit[0][0]), "joint_index": int(bad_limit[1][0]),
        }

    world_cfg = _to_curobo_world(dict(scene_cfg))
    held_world = _without_target_mesh(world_cfg)
    anchor_wrist = planner.fk_wrist(np.asarray(lift_start_qpos, dtype=np.float32))
    T_obj_in_wrist = np.linalg.inv(anchor_wrist) @ np.asarray(object_pose_at_grasp, dtype=np.float64)
    phase_names = phases.astype(str)
    known_phases = np.array(["approach", "squeeze", "lift"])
    invalid_phase = np.flatnonzero(~np.isin(phase_names, known_phases))
    if len(invalid_phase):
        index = int(invalid_phase[0])
        return {"success": False, "failure_code": "execution_trajectory_phase_invalid",
                "sample_index": index, "phase": str(phase_names[index])}

    approach_idx = np.flatnonzero(phase_names == "approach")
    held_idx = np.flatnonzero(phase_names != "approach")
    lift_idx = np.flatnonzero(phase_names == "lift")
    checked_by_phase = {
        name: int(np.count_nonzero(phase_names == name))
        for name in ("approach", "squeeze", "lift")
    }
    validation_started = perf_counter()
    collision_s = 0.0
    fk_s = 0.0
    object_clearance_s = 0.0
    world_update_s = 0.0
    collision_batches: dict[str, Any] = {}

    def _validation_metadata() -> dict[str, Any]:
        return {
            "collision_backend": "curobo_rollout_batch",
            "collision_batches": collision_batches,
            "validation_timing_s": {
                "robot_collision_batch_s": collision_s,
                "wrist_fk_batch_s": fk_s,
                "object_clearance_batch_s": object_clearance_s,
                "world_update_s": world_update_s,
                "total_s": perf_counter() - validation_started,
            },
        }

    try:
        # A collision query is a static-state operation.  Use trajectory
        # samples as the batch dimension, splitting only where the collision
        # world changes: target-present approach vs target-removed held phases.
        collision_failures: list[tuple[int, dict[str, Any]]] = []
        if len(approach_idx):
            t0 = perf_counter()
            planner._set_motion_world(world_cfg)
            world_update_s += perf_counter() - t0
            t0 = perf_counter()
            approach_valid, approach_status, approach_batch = _check_robot_states_batch(
                planner, qpos[approach_idx])
            collision_s += perf_counter() - t0
            collision_batches["approach"] = approach_batch
            invalid = np.flatnonzero(~approach_valid)
            if len(invalid):
                global_index = int(approach_idx[int(invalid[0])])
                collision_failures.append((global_index, {
                    "success": False,
                    "failure_code": "execution_trajectory_robot_collision",
                    "sample_index": global_index,
                    "phase": str(phase_names[global_index]),
                    "status": approach_status,
                }))

        if len(held_idx):
            t0 = perf_counter()
            planner._set_motion_world(held_world)
            world_update_s += perf_counter() - t0
            t0 = perf_counter()
            held_valid, held_status, held_batch = _check_robot_states_batch(
                planner, qpos[held_idx])
            collision_s += perf_counter() - t0
            collision_batches["held"] = held_batch
            invalid = np.flatnonzero(~held_valid)
            if len(invalid):
                global_index = int(held_idx[int(invalid[0])])
                collision_failures.append((global_index, {
                    "success": False,
                    "failure_code": "execution_trajectory_robot_collision",
                    "sample_index": global_index,
                    "phase": str(phase_names[global_index]),
                    "status": held_status,
                }))

        if not len(lift_idx):
            return {"success": False, "failure_code": "execution_lift_samples_missing"}

        t0 = perf_counter()
        lift_wrist = _fk_wrist_batch(planner, qpos[lift_idx])
        fk_s += perf_counter() - t0
        lateral_error = np.linalg.norm(
            lift_wrist[:, :2, 3] - anchor_wrist[None, :2, 3], axis=1)
        relative_rotation = (
            lift_wrist[:, :3, :3] @ anchor_wrist[None, :3, :3].transpose(0, 2, 1))
        rotation_error = np.linalg.norm(
            Rotation.from_matrix(relative_rotation).as_rotvec(), axis=1)
        lift_z = lift_wrist[:, 2, 3]

        t0 = perf_counter()
        object_transforms = lift_wrist @ T_obj_in_wrist
        object_bottom = _object_bottom_z_batch(mesh_vertices, object_transforms)
        object_clearance_s += perf_counter() - t0

        # Choose the earliest invalid trajectory sample, retaining the scalar
        # check's precedence at one sample: robot, lateral, rotation,
        # monotonic-z, then carried-object clearance.
        failures: list[tuple[int, int, dict[str, Any]]] = [
            (index, 0, payload) for index, payload in collision_failures
        ]
        for local in np.flatnonzero(lateral_error > lift_options.position_tolerance_m):
            index = int(lift_idx[int(local)])
            failures.append((index, 1, {
                "success": False, "failure_code": "execution_lift_lateral_deviation",
                "sample_index": index,
                "lateral_error_m": float(lateral_error[int(local)]),
            }))
        for local in np.flatnonzero(rotation_error > lift_options.orientation_tolerance_rad):
            index = int(lift_idx[int(local)])
            failures.append((index, 2, {
                "success": False, "failure_code": "execution_lift_orientation_deviation",
                "sample_index": index,
                "orientation_error_rad": float(rotation_error[int(local)]),
            }))
        nonmonotonic = np.flatnonzero(np.diff(lift_z) < -2.0e-4) + 1
        for local in nonmonotonic:
            index = int(lift_idx[int(local)])
            failures.append((index, 3, {
                "success": False, "failure_code": "execution_lift_nonmonotonic_z",
                "sample_index": index,
                "previous_z_m": float(lift_z[int(local) - 1]),
                "z_m": float(lift_z[int(local)]),
            }))
        below_table = np.flatnonzero(
            object_bottom < table_surface_z_m - lift_options.table_clearance_tolerance_m)
        for local in below_table:
            index = int(lift_idx[int(local)])
            failures.append((index, 4, {
                "success": False, "failure_code": "execution_lift_object_below_table",
                "sample_index": index,
                "object_bottom_z_m": float(object_bottom[int(local)]),
            }))
        if failures:
            _index, _precedence, failure = min(failures, key=lambda item: (item[0], item[1]))
            failure.update(_validation_metadata())
            return failure

        max_lift_q_delta = (
            0.0 if len(lift_idx) < 2 else float(np.max(np.abs(np.diff(
                qpos[lift_idx, :n_arm], axis=0)))))
        max_lateral_error = float(np.max(lateral_error))
        max_rotation_error = float(np.max(rotation_error))
        final_z_error = abs(
            (float(lift_z[-1]) - float(anchor_wrist[2, 3])) - lift_options.height_m)
        if final_z_error > lift_options.position_tolerance_m:
            failure = {
                "success": False, "failure_code": "execution_lift_final_height_deviation",
                "final_height_error_m": final_z_error,
            }
            failure.update(_validation_metadata())
            return failure
        success_result = {
            "success": True, "failure_code": None,
            "checked_samples_by_phase": checked_by_phase,
            "max_lift_lateral_error_m": max_lateral_error,
            "max_lift_orientation_error_rad": max_rotation_error,
            "max_lift_sample_joint_delta_rad": max_lift_q_delta,
            "final_lift_height_error_m": final_z_error,
            "max_arm_velocity_rad_s": float(np.max(np.abs(arm_velocity))),
            "max_arm_acceleration_rad_s2": (
                0.0 if not len(arm_acceleration)
                else float(np.max(np.abs(arm_acceleration)))),
        }
        success_result.update(_validation_metadata())
        return success_result
    finally:
        # The candidate loop expects the held-object world before a lift and
        # resets the free world itself before its next approach attempt.
        planner._set_motion_world(held_world)


def _coverage_order(*, obj: str, version: str, hand: str, pose_stem: str | None,
                    policy: CandidatePolicy) -> tuple[list[tuple[str, str, str]], dict]:
    """Return the same v8 remaining-coverage order used by ``run_auto``."""
    if version != "v8":
        raise ValueError("pipeline-comparable lift_test currently supports only v8")
    from autodex.utils.coverage import load_coverage_map

    coverage_map = load_coverage_map(
        obj, tabletop_pose_stem=pose_stem, hand=hand, version=version,
        success_root=policy.success_root)
    if coverage_map is None:
        raise RuntimeError(
            f"coverage JSON missing for {obj}/{version}; build the production v8 coverage first")
    order = order_coverage_keys(coverage_map)
    return order, {
        "policy": policy.as_dict(),
        "tabletop_pose_stem": pose_stem,
        **coverage_metadata(coverage_map, order),
    }


def _annotate_candidate_source(source_info: Mapping[str, Any],
                               policy: CandidatePolicy) -> dict:
    """State the independent ranking and record-membership contracts."""
    return {
        **source_info,
        "policy": policy.as_dict(),
        "candidate_record_gate": (
            "result_json_success_true" if policy.success_only else "all_candidate_records"),
        "coverage_ranking_source": policy.mutable_state_source,
    }


@dataclass(frozen=True)
class PreparedCandidateCatalogue:
    """Object-frame v8 candidate data reusable across translated grid cells.

    ``load_candidate`` normally left-multiplies every wrist target by the
    current object pose.  In a grid experiment the object orientation and
    candidate policy are fixed, with only its XY translation changing.  Keep
    the wrist targets in object coordinates here and materialize them against
    each cell's object pose later.  This is algebraically identical to a
    fresh ``load_candidate`` call, while avoiding repeated NAS/disk reads.
    """

    obj: str
    version: str
    hand: str
    pose_stem: str | None
    source_info: dict
    wrist_object: np.ndarray
    pregrasp: np.ndarray
    grasp: np.ndarray
    openpose: list[Any]
    scene_info: list


def prepare_candidate_catalogue(*, obj: str, version: str, candidate_hand: str,
                                pose_stem: str | None,
                                candidate_policy: CandidatePolicy) -> PreparedCandidateCatalogue:
    """Load the immutable candidate data once for a constructed XY grid.

    The caller must use the returned catalogue only with the same object,
    v8 version, hand, tabletop-pose stem, and candidate-state policy.
    """
    candidate_order, source_info = _coverage_order(
        obj=obj, version=version, hand=candidate_hand, pose_stem=pose_stem,
        policy=candidate_policy)
    source_info = _annotate_candidate_source(source_info, candidate_policy)
    if not candidate_order:
        return PreparedCandidateCatalogue(
            obj=obj, version=version, hand=candidate_hand, pose_stem=pose_stem,
            source_info=source_info,
            wrist_object=np.empty((0, 4, 4), dtype=np.float64),
            pregrasp=np.empty((0, 0), dtype=np.float32),
            grasp=np.empty((0, 0), dtype=np.float32), openpose=[], scene_info=[])

    # With an identity object pose, load_candidate returns its stored
    # object-frame wrist targets unchanged.  Cylinder expansion is likewise
    # evaluated in object coordinates, so later multiplication by each cell's
    # pose is exactly the ordinary world-frame result.
    wrist, pregrasp, grasp, scene_info = load_candidate(
        obj, np.eye(4), version, shuffle=True,
        skip_done=candidate_policy.skip_done,
        success_only=candidate_policy.success_only,
        hand=candidate_hand, scene_id=None, scene_type_filter=None,
        skip_scenes_with_success=candidate_policy.skip_scenes_with_success,
        tabletop_pose_stem=pose_stem, candidate_order=candidate_order)
    source_info = {
        **source_info,
        "n_candidate_records_after_record_gate": int(len(wrist)),
    }
    if len(wrist):
        openpose = (load_openpose_for_candidates(
            obj, scene_info, candidate_hand, version, pose_stem)
                    if pose_stem else [None] * len(pregrasp))
    else:
        openpose = []
    cyl_axis, cyl_grid = get_cyl_axis_local(obj), get_cyl_yaw_grid(obj)
    wrist, pregrasp, grasp, openpose, scene_info = _expand_candidates_cyl(
        wrist, pregrasp, grasp, openpose, scene_info, np.eye(4), cyl_axis, cyl_grid)
    source_info = {
        **source_info,
        "n_candidate_records_after_symmetry_expansion": int(len(wrist)),
    }
    return PreparedCandidateCatalogue(
        obj=obj, version=version, hand=candidate_hand, pose_stem=pose_stem,
        source_info=source_info,
        wrist_object=np.asarray(wrist, dtype=np.float64),
        pregrasp=np.asarray(pregrasp, dtype=np.float32),
        grasp=np.asarray(grasp, dtype=np.float32), openpose=list(openpose),
        scene_info=list(scene_info))


def _candidate_key(scene_info: tuple | list) -> tuple[str, str, str]:
    return tuple(str(value) for value in scene_info)  # type: ignore[return-value]


def _timed_begin(timing: TimingRecorder | None, *, parent_id: str | None,
                 kind: str, name: str, **attrs: Any) -> str | None:
    if timing is None:
        return None
    return timing.begin(phase="planning", kind=kind, name=name,
                        parent_id=parent_id, **attrs)


def _timed_end(timing: TimingRecorder | None, span_id: str | None, *,
               outcome: str = "success", **attrs: Any) -> None:
    if timing is not None and span_id is not None:
        timing.end(span_id, outcome=outcome, **attrs)


def _lift_progress_printer(*, candidate_attempt: int, candidate_total: int,
                           total_steps: int, start_z_m: float):
    """Return a compact, human-readable per-5-mm lift progress callback."""
    def _print(record: dict[str, Any]) -> None:
        dz_mm = (float(record["target_z_m"]) - start_z_m) * 1000.0
        pos_mm = float(record["position_error_m"]) * 1000.0
        rot_deg = float(np.degrees(record["orientation_error_rad"]))
        sigma = float(record["min_singular_value"])
        cond = float(record["condition_number"])
        status = "ok" if record["success"] else str(record["failure_code"])
        print(
            f"    [lift {candidate_attempt}/{candidate_total}] "
            f"step={int(record['step'])}/{total_steps} dz={dz_mm:+.1f}mm "
            f"pos={pos_mm:.3f}mm rot={rot_deg:.3f}deg "
            f"sigma={sigma:.3e} cond={cond:.1f} "
            f"segment={record['segment_samples']} {status}",
            flush=True,
        )
    return _print


def _approach_and_lift(planner: GraspPlanner, *, scene_cfg: dict, obj: str,
                       version: str, candidate_hand: str, pose_stem: str | None,
                       options: LiftOptions, candidate_policy: CandidatePolicy,
                       candidate_budget: int | None,
                       timing: TimingRecorder | None = None,
                       timing_parent_id: str | None = None,
                       candidate_catalogue: PreparedCandidateCatalogue | None = None,
                       candidate_scene_type_filter: str | None = None,
                       use_coverage_order: bool = True,
                       execution_profile: ExecutionProfile | None = None,
                       verbose: bool = True) -> dict:
    """Production-equivalent candidate funnel with only lift backend replaced.

    This deliberately duplicates the *candidate stage* of ``GraspPlanner.plan``
    instead of calling that public method: its legacy lift preflight would
    remove candidates before this experiment can evaluate the Jacobian method.
    Geometry loading, coverage ordering, collision filter, endpoint IK, and
    approach ``plan_single_js`` all retain production semantics.
    """
    import torch
    from autodex.utils.path import load_openpose_for_candidates

    t_total = perf_counter()
    if execution_profile is None:
        execution_profile = profile_for_arm(
            "franka" if int(planner._n_arm) == 7 else "xarm")
    object_pose = cart2se3(np.asarray(scene_cfg["mesh"]["target"]["pose"], dtype=float))
    source_span = _timed_begin(timing, parent_id=timing_parent_id, kind="decision",
                               name="candidate_source", object=obj,
                               tabletop_pose_stem=pose_stem,
                               policy=candidate_policy.name)
    if candidate_catalogue is None:
        candidate_order, source_info = _coverage_order(
            obj=obj, version=version, hand=candidate_hand, pose_stem=pose_stem,
            policy=candidate_policy)
        source_info = _annotate_candidate_source(source_info, candidate_policy)
        source_info = {
            **source_info,
            "selection_mode": ("coverage_order" if use_coverage_order
                               else "direct_scene_type_loading"),
            "candidate_scene_type_filter": candidate_scene_type_filter,
        }
        if use_coverage_order and not candidate_order:
            _timed_end(timing, source_span, outcome="failure", ordered_count=0)
            return {
                "success": False, "reason": "coverage_exhausted",
                "failure_code": "candidate_coverage_exhausted",
                "candidate_source": source_info, "candidate_metadata": [],
                "candidate_attempts": [], "timing": {"total_s": perf_counter() - t_total},
            }
        _timed_end(timing, source_span, outcome="success",
                   ordered_count=len(candidate_order), cache="miss")
        # 1. Load the same candidate keys in the same coverage order as run_auto.
        load_span = _timed_begin(timing, parent_id=timing_parent_id, kind="io",
                                 name="candidate_catalogue_load", candidate_count=len(candidate_order))
        wrist, pregrasp, grasp, scene_info = load_candidate(
            obj, object_pose, version, shuffle=True,
            skip_done=candidate_policy.skip_done,
            success_only=candidate_policy.success_only,
            hand=candidate_hand, scene_id=None,
            scene_type_filter=candidate_scene_type_filter,
            skip_scenes_with_success=candidate_policy.skip_scenes_with_success,
            tabletop_pose_stem=pose_stem,
            candidate_order=(candidate_order if use_coverage_order else None))
        source_info = {
            **source_info,
            "n_candidate_records_after_record_gate": int(len(wrist)),
        }
        if len(wrist):
            openpose = load_openpose_for_candidates(
                obj, scene_info, candidate_hand, version, pose_stem) if pose_stem else [None] * len(pregrasp)
        else:
            openpose = []
        cyl_axis, cyl_grid = get_cyl_axis_local(obj), get_cyl_yaw_grid(obj)
        wrist, pregrasp, grasp, openpose, scene_info = _expand_candidates_cyl(
            wrist, pregrasp, grasp, openpose, scene_info, object_pose, cyl_axis, cyl_grid)
        source_info = {
            **source_info,
            "n_candidate_records_after_symmetry_expansion": int(len(wrist)),
        }
        _timed_end(timing, load_span, outcome="success", loaded_count=len(wrist), cache="miss")
    else:
        if not use_coverage_order or candidate_scene_type_filter is not None:
            raise ValueError(
                "prepared candidate catalogues are only valid for coverage-ordered "
                "candidate loading")
        expected = (obj, version, candidate_hand, pose_stem)
        actual = (candidate_catalogue.obj, candidate_catalogue.version,
                  candidate_catalogue.hand, candidate_catalogue.pose_stem)
        if actual != expected:
            raise ValueError(
                "candidate catalogue does not match object/version/hand/tabletop pose: "
                f"expected {expected}, got {actual}")
        source_info = dict(candidate_catalogue.source_info)
        candidate_order = [tuple(key) for key in source_info.get("ordered_keys", [])]
        _timed_end(timing, source_span, outcome="success",
                   ordered_count=len(candidate_order), cache="hit")
        load_span = _timed_begin(timing, parent_id=timing_parent_id, kind="io",
                                 name="candidate_catalogue_load", candidate_count=len(candidate_order))
        wrist = np.matmul(object_pose, candidate_catalogue.wrist_object)
        pregrasp = candidate_catalogue.pregrasp
        grasp = candidate_catalogue.grasp
        openpose = candidate_catalogue.openpose
        scene_info = candidate_catalogue.scene_info
        _timed_end(timing, load_span, outcome="success", loaded_count=len(wrist), cache="hit")
    if use_coverage_order and not candidate_order:
        return {
            "success": False, "reason": "coverage_exhausted",
            "failure_code": "candidate_coverage_exhausted",
            "candidate_source": source_info, "candidate_metadata": [],
            "candidate_attempts": [], "timing": {"total_s": perf_counter() - t_total},
        }
    if verbose:
        print(
            f"[coverage] tabletop={pose_stem or 'unclassified'} policy={candidate_policy.name} "
            f"catalogue={source_info.get('n_coverage_candidates', '?')} "
            f"useful={source_info.get('n_coverage_useful', '?')} "
            f"zero={source_info.get('n_coverage_zero', '?')}",
            flush=True,
        )
        print(
            f"[candidates] loaded={len(wrist)} after tabletop/order/symmetry filters "
            f"(record_gate={source_info['candidate_record_gate']})",
            flush=True,
        )
    if len(wrist) == 0:
        no_records_reason = ("no_verified_candidates" if candidate_policy.success_only
                             else "no_candidates_after_policy_filter")
        no_records_code = ("candidate_verified_pool_empty" if candidate_policy.success_only
                           else "candidate_catalogue_empty")
        return {
            "success": False, "reason": no_records_reason,
            "failure_code": no_records_code,
            "candidate_source": source_info, "candidate_metadata": [],
            "candidate_attempts": [], "timing": {"total_s": perf_counter() - t_total},
        }

    rank = {key: i for i, key in enumerate(candidate_order)}
    coverage = source_info.get("remaining_coverage_by_key", {})
    metadata: list[dict] = []
    for idx, info in enumerate(scene_info):
        key = _candidate_key(info)
        metadata.append({
            "candidate_index": int(idx), "candidate_key": list(key),
            "scene_info": list(key), "coverage_rank": rank.get(key),
            "remaining_coverage": coverage.get("/".join(key)),
            "status": "catalogued",
        })
    approach_fingers = np.asarray([
        op if op is not None else pg for op, pg in zip(openpose, pregrasp)
    ], dtype=np.float32)

    # 2. This is the production collision/IK funnel.  The target object is
    # removed only for free-hand candidate checks and endpoint IK, just as in
    # GraspPlanner.plan(); it is restored for the approach trajectory itself.
    world_cfg = _to_curobo_world(scene_cfg)
    held_world = _without_target_mesh(world_cfg)
    setup_span = _timed_begin(timing, parent_id=timing_parent_id, kind="setup",
                              name="candidate_world_setup")
    planner._set_motion_world(world_cfg)
    planner._set_ik_world(held_world)
    _timed_end(timing, setup_span)

    filter_span = _timed_begin(timing, parent_id=timing_parent_id, kind="check",
                               name="candidate_collision_filter", candidate_count=len(wrist))
    backward = (np.zeros(len(wrist), dtype=bool) if "inspire" in planner._hand
                else (wrist[:, :3, :3] @ planner._link6_y_in_wrist)[:, 2] < 0.3)
    collision, world_collision, self_collision = planner._check_collision(
        held_world, wrist, pregrasp, return_components=True)
    valid = np.where(~(backward | collision))[0]
    for idx in range(len(wrist)):
        row = metadata[idx]
        row.update({
            "backward": bool(backward[idx]), "world_collision": bool(world_collision[idx]),
            "self_collision": bool(self_collision[idx]), "candidate_collision": bool(collision[idx]),
        })
        if backward[idx]:
            row["status"] = "filtered"
            row["failure_code"] = "candidate_filtered_backward"
        elif collision[idx]:
            row["status"] = "filtered"
            row["failure_code"] = ("candidate_filtered_world_collision"
                                   if world_collision[idx] else "candidate_filtered_self_collision")
    _timed_end(timing, filter_span, outcome="success", valid_count=len(valid))
    if verbose:
        print(
            f"[filter] total={len(wrist)} backward={int(backward.sum())} "
            f"collision={int(collision.sum())} "
            f"(world={int(world_collision.sum())}, self={int(self_collision.sum())}) "
            f"pass={len(valid)}",
            flush=True,
        )
    if len(valid) == 0:
        return {
            "success": False, "reason": "all_candidates_filtered",
            "failure_code": "candidate_all_filtered",
            "candidate_source": source_info, "candidate_metadata": metadata,
            "candidate_attempts": [], "timing": {"total_s": perf_counter() - t_total},
        }

    ik_span = _timed_begin(timing, parent_id=timing_parent_id, kind="plan",
                           name="candidate_endpoint_ik", candidate_count=len(valid))
    ik_success = np.zeros(len(wrist), dtype=bool)
    ik_qpos = np.full((len(wrist), len(planner._init_state)), np.nan, dtype=np.float32)
    for chunk_start in range(0, len(valid), planner.BATCH_SIZE):
        chunk_idx = valid[chunk_start:chunk_start + planner.BATCH_SIZE]
        chunk_poses = wrist[chunk_idx]
        actual = len(chunk_poses)
        if actual < planner.BATCH_SIZE:
            chunk_poses = np.concatenate([
                chunk_poses, np.tile(chunk_poses[:1], (planner.BATCH_SIZE - actual, 1, 1))])
        goal = _to_curobo_pose(chunk_poses, planner._tensor_args.device)
        retract = torch.tensor(planner._init_state, dtype=torch.float32,
                               device=planner._tensor_args.device).unsqueeze(0).repeat(len(chunk_poses), 1)
        ik_result = planner._ik_solver.solve_batch(goal, retract_config=retract)
        successes = ik_result.success.detach().cpu().numpy()[:actual]
        q_solutions = ik_result.solution.detach().cpu().numpy()[:actual]
        if q_solutions.ndim == 3:
            q_solutions = q_solutions[:, 0, :]
        for local_idx, candidate_idx in enumerate(chunk_idx):
            if not bool(successes[local_idx]):
                continue
            arm_q = q_solutions[local_idx, :planner._n_arm].copy()
            planner._snap_arm(arm_q, planner._init_state)
            if np.any(np.abs(arm_q) > np.pi):
                metadata[candidate_idx].update({"status": "ik_failed",
                                                "failure_code": "candidate_ik_far_wrap"})
                continue
            ik_success[candidate_idx] = True
            ik_qpos[candidate_idx, :planner._n_arm] = arm_q
            ik_qpos[candidate_idx, planner._n_arm:] = approach_fingers[candidate_idx]
    ik_valid = np.where(ik_success)[0]
    for idx in valid:
        if not ik_success[idx] and "failure_code" not in metadata[idx]:
            metadata[idx].update({"status": "ik_failed", "failure_code": "candidate_ik_failed"})
        elif ik_success[idx]:
            metadata[idx]["status"] = "ik_passed"
    _timed_end(timing, ik_span, outcome="success", success_count=len(ik_valid))
    if verbose:
        print(f"[IK] pass={len(ik_valid)}/{len(valid)}", flush=True)
    if len(ik_valid) == 0:
        return {
            "success": False, "reason": "all_candidates_ik_failed",
            "failure_code": "candidate_all_ik_failed",
            "candidate_source": source_info, "candidate_metadata": metadata,
            "candidate_attempts": [], "timing": {"total_s": perf_counter() - t_total},
        }

    # ``load_candidate(..., candidate_order=...)`` already imposed the v8
    # order.  Do not shuffle here: production also preserves it when supplied.
    object_mesh = _load_mesh(scene_cfg["mesh"]["target"]["file_path"])
    table = scene_cfg["cuboid"]["table"]
    table_z = float(table["pose"][2] + table["dims"][2] / 2.0)
    reports: list[dict] = []
    attempts = 0
    budget_exhausted = False
    total_attemptable = len(ik_valid)
    if candidate_budget is not None:
        total_attemptable = min(total_attemptable, candidate_budget)
    for idx in ik_valid:
        if candidate_budget is not None and attempts >= candidate_budget:
            budget_exhausted = True
            break
        attempts += 1
        key = _candidate_key(scene_info[idx])
        if verbose:
            print(
                f"[candidate {attempts}/{total_attemptable}] key={'/'.join(key)} "
                f"coverage={metadata[idx].get('remaining_coverage')} "
                f"rank={metadata[idx].get('coverage_rank')}",
                flush=True,
            )
        planner._set_motion_world(world_cfg)
        approach_span = _timed_begin(timing, parent_id=timing_parent_id, kind="plan",
                                     name="candidate_approach", candidate_key=list(key),
                                     candidate_index=int(idx), attempt=attempts)
        refined_ok, approach = planner._refine_fingers(planner._init_state, ik_qpos[idx])
        _timed_end(timing, approach_span, outcome="success" if refined_ok else "failure")
        if not refined_ok:
            if verbose:
                print(f"  [approach] failed: approach_plan_failed", flush=True)
            metadata[idx].update({"status": "approach_failed",
                                  "failure_code": "approach_plan_failed"})
            reports.append({"candidate_index": int(idx), "candidate_key": list(key),
                            "success": False, "failure_code": "approach_plan_failed"})
            continue

        if verbose:
            print(f"  [approach] ok: waypoints={len(approach)}", flush=True)

        lift_start = np.asarray(approach[-1], dtype=np.float32).copy()
        lift_start[planner._n_arm:] = grasp[idx]
        planner._set_motion_world(held_world)
        lift_span = _timed_begin(timing, parent_id=timing_parent_id, kind="plan",
                                 name="candidate_jacobian_lift", candidate_key=list(key),
                                 candidate_index=int(idx), attempt=attempts,
                                 step_m=options.step_m, height_m=options.height_m)
        start_wrist = planner.fk_wrist(lift_start)
        n_lift_steps = int(np.ceil(options.height_m / options.step_m))
        if verbose:
            print(
                f"  [lift] Jacobian continuation: {n_lift_steps} x "
                f"{options.step_m * 1000.0:.1f}mm = {options.height_m * 1000.0:.1f}mm",
                flush=True,
            )
        lift, steps, lift_info = continue_vertical_lift(
            planner, lift_start, hand_q=grasp[idx],
            mesh_vertices=np.asarray(object_mesh.vertices),
            object_pose_at_grasp=object_pose, table_surface_z_m=table_z,
            options=options,
            progress_callback=(_lift_progress_printer(
                candidate_attempt=attempts, candidate_total=total_attemptable,
                total_steps=n_lift_steps, start_z_m=float(start_wrist[2, 3]))
                               if verbose else None),
        )
        # ``continue_vertical_lift`` returns these collision-checked chord
        # samples only in memory. Keep JSON reports compact and hand the dense
        # array to the time-parameterization stage instead.
        lift_execution_qpos = lift_info.pop("execution_qpos", None)
        lift_execution_contract = lift_info.pop("execution_sample_contract", None)
        _timed_end(timing, lift_span, outcome="success" if lift is not None else "failure",
                   failure_code=lift_info.get("failure_code"),
                   completed_steps=lift_info.get("completed_steps"))
        report = {
            "candidate_index": int(idx), "candidate_key": list(key),
            "success": lift is not None, "reason": lift_info.get("reason"),
            "failure_code": lift_info.get("failure_code"), "lift": lift_info,
            "jacobian_steps": steps,
        }
        reports.append(report)
        if lift is not None:
            if lift_execution_qpos is None:
                execution_failure = "execution_lift_samples_missing"
                report.update({"success": False, "failure_code": execution_failure,
                               "execution": {"success": False,
                                             "failure_code": execution_failure}})
                metadata[idx].update({"status": "execution_failed",
                                      "failure_code": execution_failure})
                continue
            execution_span = _timed_begin(
                timing, parent_id=timing_parent_id, kind="plan",
                name="candidate_execution_trajectory", candidate_key=list(key),
                candidate_index=int(idx), attempt=attempts,
                profile=execution_profile.as_dict())
            try:
                execution = build_execution_trajectory(
                    approach_qpos=approach, squeeze_qpos=lift_start,
                    lift_execution_qpos=lift_execution_qpos,
                    arm_dof=planner._n_arm, profile=execution_profile)
                execution_validation = _verify_execution_trajectory(
                    planner, execution=execution, scene_cfg=scene_cfg,
                    lift_start_qpos=lift_start, object_pose_at_grasp=object_pose,
                    mesh_vertices=np.asarray(object_mesh.vertices),
                    table_surface_z_m=table_z, lift_options=options)
            except Exception as exc:
                execution = None
                execution_validation = {
                    "success": False, "failure_code": "execution_trajectory_build_exception",
                    "exception": repr(exc),
                }
            _timed_end(
                timing, execution_span,
                outcome="success" if execution_validation.get("success") else "failure",
                failure_code=execution_validation.get("failure_code"))
            report["execution"] = {
                "success": bool(execution_validation.get("success")),
                "failure_code": execution_validation.get("failure_code"),
                "validation": execution_validation,
                "segments": (None if execution is None else execution["segments"]),
                "lift_source": lift_execution_contract,
            }
            if not execution_validation.get("success") or execution is None:
                execution_failure = str(execution_validation.get(
                    "failure_code", "execution_trajectory_invalid"))
                report.update({"success": False, "failure_code": execution_failure})
                metadata[idx].update({"status": "execution_failed",
                                      "failure_code": execution_failure,
                                      "jacobian_completed_steps": lift_info.get("completed_steps")})
                if verbose:
                    print(f"  [execution] rejected: {execution_failure}", flush=True)
                continue
            if verbose:
                print(
                    f"[selected] key={'/'.join(key)} approach_waypoints={len(approach)} "
                    f"lift_waypoints={len(lift)} execution_samples={len(execution['qpos'])} "
                    f"duration={execution['segments']['total_duration_s']:.2f}s",
                    flush=True,
                )
            metadata[idx].update({"status": "selected",
                                  "failure_code": None,
                                  "jacobian_completed_steps": lift_info.get("completed_steps")})
            return {
                "success": True, "reason": None, "failure_code": None,
                "candidate_index": int(idx), "scene_info": scene_info[idx],
                "wrist_se3": np.asarray(wrist[idx], dtype=np.float64),
                "approach": np.asarray(approach, dtype=np.float32),
                "lift_start": lift_start, "lift": lift,
                "grasp_hand": np.asarray(grasp[idx], dtype=np.float32),
                "lift_collision_checked_qpos": np.asarray(lift_execution_qpos, dtype=np.float32),
                "execution": execution,
                "execution_validation": execution_validation,
                "jacobian_steps": steps, "candidate_source": source_info,
                "candidate_metadata": metadata, "candidate_attempts": reports,
                "timing": {"total_s": perf_counter() - t_total,
                           "approach_attempts": attempts,
                           "ik_valid_count": len(ik_valid),
                           "candidate_budget": candidate_budget,
                           "search_truncated": False},
            }
        metadata[idx].update({"status": "jacobian_failed",
                              "failure_code": lift_info.get("failure_code"),
                              "jacobian_completed_steps": lift_info.get("completed_steps")})
        if verbose:
            print(
                f"  [lift] failed after {lift_info.get('completed_steps', 0)}/"
                f"{lift_info.get('requested_steps', n_lift_steps)} steps: "
                f"{lift_info.get('failure_code')}",
                flush=True,
            )

    return {
        "success": False,
        "reason": "candidate_budget_exhausted" if budget_exhausted else "no_candidate_completed_jacobian_lift",
        "failure_code": ("candidate_search_truncated" if budget_exhausted
                         else "candidate_all_jacobian_failed"),
        "candidate_source": source_info, "candidate_metadata": metadata,
        "candidate_attempts": reports,
        "timing": {"total_s": perf_counter() - t_total,
                   "approach_attempts": attempts, "ik_valid_count": len(ik_valid),
                   "candidate_budget": candidate_budget,
                   "search_truncated": budget_exhausted},
    }


def _save_jacobian_lift(path: Path, planning: Mapping[str, Any],
                        options: LiftOptions) -> None:
    """Persist geometry nodes and the collision-checked lift chord samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int32),
        strategy=np.array("jacobian_continuation_v1"),
        qpos=np.asarray(planning["lift"], dtype=np.float32),
        collision_checked_qpos=np.asarray(planning["lift_collision_checked_qpos"], dtype=np.float32),
        start_full_qpos=np.asarray(planning["lift_start"], dtype=np.float32),
        fixed_hand_qpos=np.asarray(planning["grasp_hand"], dtype=np.float32),
        wrist_se3=np.asarray(planning["wrist_se3"], dtype=np.float64),
        options_json=np.array(json.dumps(options_as_dict(options), sort_keys=True)),
        step_records_json=np.array(json.dumps(planning["jacobian_steps"],
                                              default=_json_default, sort_keys=True)),
    )


def _save_execution_trajectory(path: Path, planning: Mapping[str, Any]) -> None:
    """Write the canonical executor-reference artifact for one selected grasp."""
    execution = planning["execution"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int32),
        qpos=np.asarray(execution["qpos"], dtype=np.float32),
        time_s=np.asarray(execution["time_s"], dtype=np.float64),
        phase=np.asarray(execution["phase"]),
        profile_json=np.array(json.dumps(execution["profile"], sort_keys=True)),
        segments_json=np.array(json.dumps(execution["segments"], default=_json_default,
                                          sort_keys=True)),
        validation_json=np.array(json.dumps(planning["execution_validation"],
                                            default=_json_default, sort_keys=True)),
    )


def _save_animation(path: Path, planner: GraspPlanner, scenario: Mapping[str, Any],
                    planning: Mapping[str, Any]) -> None:
    execution = planning.get("execution")
    if execution is None:
        # Compatibility fallback for manually produced legacy episodes.
        approach = np.asarray(planning["approach"], dtype=np.float32)
        lift = np.asarray(planning["lift"], dtype=np.float32)
        squeeze = np.asarray(planning["lift_start"], dtype=np.float32)[None, :]
        qpos = np.concatenate([approach, squeeze, lift[1:]], axis=0)
        phases = np.concatenate([
            np.full(len(approach), "approach", dtype="<U12"),
            np.array(["squeeze"], dtype="<U12"),
            np.full(max(0, len(lift) - 1), "lift", dtype="<U12"),
        ])
        time_s = np.arange(len(qpos), dtype=np.float64) * 0.05
        trajectory_kind = "legacy_geometric_replay"
    else:
        qpos = np.asarray(execution["qpos"], dtype=np.float32)
        phases = np.asarray(execution["phase"])
        time_s = np.asarray(execution["time_s"], dtype=np.float64)
        trajectory_kind = "timestamped_execution_reference"
    T_obj = cart2se3(np.asarray(scenario["scene_cfg"]["mesh"]["target"]["pose"], dtype=float))
    q_grasp = np.asarray(planning["lift_start"], dtype=np.float32)
    T_wrist_grasp = planner.fk_wrist(q_grasp)
    T_obj_in_wrist = np.linalg.inv(T_wrist_grasp) @ T_obj
    wrist = _fk_wrist_batch(planner, qpos)
    object_poses = np.repeat(T_obj[None, :, :], len(qpos), axis=0)
    lift_idx = np.where(phases == "lift")[0]
    if len(lift_idx):
        object_poses[lift_idx] = wrist[lift_idx] @ T_obj_in_wrist
    np.savez_compressed(path, qpos=qpos, phase=phases,
                        wrist_pose=wrist, object_pose=object_poses,
                        object_pose_in_wrist=T_obj_in_wrist,
                        time_s=time_s, trajectory_kind=np.array(trajectory_kind))


def _episode_dir(root: Path, hand: str, obj: str, session_stamp: str,
                 trial_number: int) -> Path:
    path = root / hand / obj / f"{session_stamp}_{trial_number:04d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_last_state(root: Path, hand: str) -> dict:
    path = root / hand / "_last_selection.json"
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_last_state(root: Path, hand: str, state: Mapping[str, Any]) -> None:
    _write_json(root / hand / "_last_selection.json", dict(state))


def _print_next_prompt() -> str:
    try:
        return input("\n[n] new trial  [v] replay last result  [b] remeasure board  [q] quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _available_viewer_port(requested_port: int, *, attempts: int = 32) -> int:
    """Select a local free port before launching Viser.

    Viser silently advances to a different port when its requested one is in
    use. Selecting it here keeps the URL printed by the parent process true.
    """
    for port in range(int(requested_port), int(requested_port) + attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError(
        f"no free Viser port in {requested_port}..{requested_port + attempts - 1}")


def _launch_viewer(episode: Path, port: int) -> int:
    actual_port = _available_viewer_port(port)
    script = Path(__file__).with_name("viser_view.py")
    log_path = episode / "viser.log"
    log = log_path.open("w")
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--episode", str(episode),
             "--port", str(actual_port)],
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    print(f"[viser] started replay process for {episode}")
    if actual_port != port:
        print(f"[viser] requested port {port} was occupied; using {actual_port}")
    print(f"[viser] open: http://localhost:{actual_port}")
    print(f"[viser] log: {log_path}")
    return actual_port


def _print_viewer_hint(episode: Path, port: int) -> None:
    """Print a copy-paste replay command even when auto-launch is disabled."""
    script = Path(__file__).with_name("viser_view.py")
    print("[viser] replay available:")
    print(f"  python {script} --episode {episode} --port {port}")
    print(f"[viser] after launching, open: http://localhost:{port} "
          "(or the port printed by Viser if it is occupied)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object-source", choices=["constructed", "perception"], default="constructed",
                   help="constructed prompts for XY/tabletop pose; perception asks only object and runs FoundPose")
    p.add_argument("--board-source", choices=["live-charuco", "file"], default="live-charuco")
    p.add_argument("--board-json", type=Path, help="board proxy JSON; required with --board-source file")
    p.add_argument("--arm", default="franka", choices=sorted(ARM_TO_PLANNER_ROBOT),
                   help=("arm kinematic/collision model: franka = FR3 + Inspire; "
                         "xarm = XArm6 + the same Inspire hand"))
    p.add_argument("--hand", default="inspire", choices=["inspire"],
                   help="candidate-hand namespace shared by both arm modes")
    p.add_argument("--grasp-version", default="v8", choices=["v8"],
                   help="v8 only: matches the production tabletop/candidate asset contract")
    p.add_argument("--candidate-policy", choices=["clean-state", "current-state", "verified-only",
                                                    "pipeline-parity"],
                   default="clean-state",
                   help=("clean-state ignores legacy outcomes; current-state reads shared "
                         "completion state; verified-only requires candidate result.json success=true; "
                         "pipeline-parity requires live perception and matches run_auto scene/candidate policy"))
    # These names and defaults intentionally mirror run_auto.py. They are
    # consumed only by --candidate-policy pipeline-parity, where they make the
    # constructed collision scene and candidate scene-type filter explicit.
    p.add_argument("--scene", choices=["table", "wall", "shelf", "cluttered"], default="table",
                   help="production obstacle scene for pipeline-parity (default: table)")
    p.add_argument("--candidate-scene-type", choices=["table", "wall", "shelf", "box"],
                   default=None,
                   help="production v8 candidate scene-type filter for pipeline-parity")
    p.add_argument("--wall-gap", type=float, default=0.04)
    p.add_argument("--wall-angle", type=float, default=0.0)
    p.add_argument("--clutter-seed", type=int, default=42)
    p.add_argument("--clutter-min-dist", type=float, default=0.12)
    p.add_argument("--clutter-max-dist", type=float, default=0.20)
    p.add_argument("--clutter-n", type=int, default=4)
    p.add_argument("--shelf-width", type=float, default=0.30)
    p.add_argument("--shelf-depth", type=float, default=0.30)
    p.add_argument("--shelf-height", type=float, default=0.30)
    p.add_argument("--shelf-gap", type=float, default=0.02)
    p.add_argument("--no-shelf-back", action="store_true")
    p.add_argument("--no-shelf-sides", action="store_true")
    p.add_argument("--no-shelf-top", action="store_true")
    p.add_argument("--pc-list", nargs="+", default=None)
    p.add_argument("--calib-dir")
    p.add_argument("--port-mask", type=int, default=5006)
    p.add_argument("--port-pose", type=int, default=5007)
    p.add_argument("--port-cmd", type=int, default=6893)
    p.add_argument("--port-snap", type=int, default=5009)
    p.add_argument("--port-snap-cmd", type=int, default=6894)
    p.add_argument("--snapshot-timeout-s", type=float, default=5.0)
    p.add_argument("--stream-fps", type=int, default=10)
    p.add_argument("--stream-warmup-s", type=float, default=2.0)
    p.add_argument("--prompt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--perception-mode", choices=["iou", "ignore_sil_loss"], default="iou")
    p.add_argument("--sil-iters", type=int, default=100)
    p.add_argument("--sil-lr", type=float, default=0.002)
    p.add_argument("--init-timeout-s", type=float, default=60.0)
    p.add_argument("--max-candidates", type=int, default=0,
                   help="maximum IK-valid candidates to preflight; 0 = all (default)")
    p.add_argument("--lift-height-m", type=float, default=0.10)
    p.add_argument("--lift-step-m", type=float, default=0.005)
    p.add_argument("--damping", type=float, default=0.02)
    p.add_argument("--max-iterations", type=int, default=24)
    p.add_argument("--max-joint-step-rad", type=float, default=0.10)
    p.add_argument("--max-segment-joint-delta-rad", type=float, default=0.02,
                   help="maximum per-joint interpolation increment for collision checks")
    p.add_argument("--execution-dt-s", type=float, default=0.01,
                   help="uniform control period of the saved executable q(t) (default: 10 ms)")
    p.add_argument("--squeeze-duration-s", type=float, default=0.50,
                   help="minimum close-hand duration before the timed lift (default: 0.50 s)")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    p.add_argument("--open-viewer", action="store_true",
                   help="automatically start a separate Viser replay process after success")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.board_source == "file" and args.board_json is None:
        raise SystemExit("--board-json is required with --board-source file")
    if args.max_candidates < 0:
        raise SystemExit("--max-candidates must be >= 0 (0 means all)")
    if args.execution_dt_s <= 0.0 or args.squeeze_duration_s <= 0.0:
        raise SystemExit("--execution-dt-s and --squeeze-duration-s must be positive")
    if (args.candidate_policy == "pipeline-parity"
            and args.object_source != "perception"):
        raise SystemExit(
            "--candidate-policy pipeline-parity requires "
            "--object-source perception; a constructed XY pose cannot represent "
            "the run_auto perception input contract")
    planner_robot = _planner_robot_for_arm(args.arm)
    _load_runtime_dependencies()
    options = LiftOptions(height_m=args.lift_height_m, step_m=args.lift_step_m,
                          damping=args.damping, max_iterations=args.max_iterations,
                          max_joint_step_rad=args.max_joint_step_rad,
                          max_segment_joint_delta_rad=args.max_segment_joint_delta_rad)
    execution_profile = profile_for_arm(
        args.arm, sample_dt_s=args.execution_dt_s,
        squeeze_duration_s=args.squeeze_duration_s)
    output_root = Path(project_dir) / "experiment" / "lift_test"
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / args.hand / "_sessions" / session_stamp
    session_dir.mkdir(parents=True, exist_ok=False)
    candidate_policy = make_candidate_policy(
        args.candidate_policy, clean_state_root=session_dir / "candidate_state_clean")
    state = _load_last_state(output_root, args.hand)
    capture = CaptureContext(args)
    last_episode: Path | None = None
    trial_number = 0

    try:
        if args.board_source == "live-charuco":
            print("[board] Clear Charuco board and press Enter to measure (q to quit).")
            _input("  ready")
            measurement, proxy = capture.measure_board(session_dir)
            print("[board] measured: center="
                  f"({proxy['center_xy_m'][0]:.4f}, {proxy['center_xy_m'][1]:.4f}) m, "
                  f"table z={proxy['table_surface_z_m']:.4f} m")
        else:
            proxy = load_proxy(args.board_json)
            measurement = {
                "source": "loaded_board_proxy",
                "table_surface_z_m": proxy["table_surface_z_m"],
            }
            _write_json(session_dir / "board" / "board_proxy.json", proxy)
        _write_json(session_dir / "session.json", {
            "schema_version": 2, "session_stamp": session_stamp,
            "object_source": args.object_source, "board_source": args.board_source,
            "arm": args.arm, "hand": args.hand, "planner_robot": planner_robot,
            "grasp_version": args.grasp_version,
            "candidate_policy": candidate_policy.as_dict(),
            "lift_options": options_as_dict(options),
            "execution_profile": execution_profile.as_dict(),
            "calib_dir": str(capture.calib_dir),
        })

        print(f"[planner] initializing {planner_robot} for --arm {args.arm} "
              f"(cuda_graph={args.cuda_graph})...", flush=True)
        t_planner_init = perf_counter()
        planner = GraspPlanner(hand=planner_robot,
                               use_cuda_graph=(args.cuda_graph == "on"))
        print(f"[planner] ready in {perf_counter() - t_planner_init:.2f}s", flush=True)
        while True:
            episode: Path | None = None
            trace: TimingRecorder | None = None
            pipeline_contract: dict | None = None
            pipeline_gate: dict | None = None
            candidate_scene_type_filter: str | None = None
            try:
                if args.object_source == "constructed":
                    xy = _prompt_xy(proxy)
                    obj = _prompt_object(args.hand, args.grasp_version,
                                         state.get("last_object"))
                    pose_memory = (state.get("last_pose_by_object") or {}).get(obj)
                    pose_idx, pose_file = _prompt_tabletop_pose(obj, args.grasp_version, pose_memory)
                    trace = TimingRecorder()
                    construct_span = trace.begin(phase="preparation", kind="setup",
                                                name="constructed_scenario")
                    scenario = _constructed_scenario(
                        obj, args.grasp_version, xy, pose_idx, pose_file, measurement)
                    trace.end(construct_span)
                    tabletop = {"idx": pose_idx, "filename": pose_file.name,
                                "rot_err_deg": 0.0, "source": "selected"}
                    trial_number += 1
                    episode = _episode_dir(output_root, args.hand, obj, session_stamp, trial_number)
                else:
                    obj = _prompt_object(args.hand, args.grasp_version,
                                         state.get("last_object"))
                    _input(f"[perception] Place {obj} on the board, then press Enter")
                    trace = TimingRecorder()
                    trial_number += 1
                    episode = _episode_dir(output_root, args.hand, obj, session_stamp, trial_number)
                    perception_span = trace.begin(phase="perception", kind="inference",
                                                   name="foundpose_and_scene_conversion")
                    scenario = capture.perceive_object(obj, args.grasp_version, measurement, episode)
                    trace.end(perception_span,
                              outcome="success" if scenario.get("success") else "failure",
                              reason=scenario.get("reason"))
                    if not scenario.get("success"):
                        _write_json(episode / "result.json", {
                            "schema_version": 2, "success": False, "reason": scenario["reason"],
                            "failure_code": "perception_failed", "object": obj,
                            "arm": args.arm, "planner_robot": planner_robot,
                            "source": "perception", "perception": scenario,
                            "timing_trace": trace.as_dict(),
                        })
                        print(f"[trial] perception failed: {scenario['reason']} -> {episode}")
                        last_episode = episode
                        continue
                    xy = np.asarray(scenario["pose_robot_scene"], dtype=float)[:2, 3]
                    tabletop = classify_tabletop_pose(
                        np.asarray(scenario["pose_robot_raw"], dtype=float), obj,
                        get_obj_root(args.grasp_version))
                    if (args.candidate_policy != "pipeline-parity"
                            and not point_in_polygon(
                                xy, np.asarray(proxy["vertices_xy_m"], dtype=float))):
                        _write_json(episode / "result.json", {
                            "schema_version": 2, "success": False,
                            "reason": "perceived_object_outside_board_proxy",
                            "failure_code": "input_outside_board_proxy", "object": obj,
                            "arm": args.arm, "planner_robot": planner_robot,
                            "source": "perception", "perception": scenario,
                            "board_proxy": proxy, "timing_trace": trace.as_dict(),
                        })
                        print(f"[trial] perceived XY ({xy[0]:.4f}, {xy[1]:.4f}) outside proxy -> {episode}")
                        last_episode = episode
                        continue
                    if args.candidate_policy == "pipeline-parity":
                        candidate_scene_type_filter = _pipeline_candidate_scene_filter(args)
                        scenario["scene_cfg"] = _apply_pipeline_obstacles(
                            scenario["scene_cfg"], args=args,
                            tabletop_geometry=measurement)
                        pipeline_contract = _pipeline_parity_contract(
                            args, tabletop=tabletop or {},
                            candidate_scene_type_filter=candidate_scene_type_filter)

                assert scenario.get("success") and episode is not None and trace is not None
                plan_dir = episode / "plan"
                artifact_span = trace.begin(phase="artifacts", kind="io",
                                            name="episode_input_artifacts")
                _write_json(episode / "board_proxy.json", proxy)
                _write_json(episode / "request.json", {
                    "object": obj, "source": args.object_source,
                    "arm": args.arm, "planner_robot": planner_robot,
                    "xy_m": np.asarray(xy).tolist(), "tabletop": tabletop,
                    "lift_options": options_as_dict(options),
                    "execution_profile": execution_profile.as_dict(),
                    "candidate_policy": candidate_policy.as_dict(),
                    "candidate_budget": (None if args.max_candidates == 0 else args.max_candidates),
                    "start_q_source": f"GraspPlanner.{planner_robot} default init state",
                    "lift_strategy": "jacobian_continuation_v1",
                    "pipeline_parity": pipeline_contract,
                })
                if "pose_world" in scenario:
                    np.save(episode / "pose_world.npy", scenario["pose_world"])
                    np.save(episode / "C2R.npy", scenario["c2r"])
                np.save(episode / "pose_robot_raw.npy", scenario["pose_robot_raw"])
                np.save(episode / "pose_robot_scene.npy", scenario["pose_robot_scene"])
                _write_json(episode / "scene_cfg.json", scenario["scene_cfg"])
                trace.end(artifact_span)

                pose_stem = tabletop.get("filename", "").replace(".npy", "") if tabletop else None
                if args.candidate_policy == "pipeline-parity":
                    gate_span = trace.begin(
                        phase="planning", kind="decision", name="pipeline_parity_gate",
                        tabletop_pose_stem=pose_stem, scene=args.scene)
                    pipeline_gate = _pipeline_parity_gate(
                        args=args, obj=obj, version=args.grasp_version,
                        hand=args.hand, tabletop=tabletop)
                    trace.end(gate_span,
                              outcome="skipped" if pipeline_gate is not None else "success",
                              **(pipeline_gate or {}))
                planning_span = trace.begin(
                    phase="planning", kind="plan", name="candidate_preflight",
                    candidate_policy=candidate_policy.name, tabletop_pose_stem=pose_stem,
                    lift_strategy="jacobian_continuation_v1")
                if pipeline_gate is not None:
                    planning = {
                        "success": False,
                        "reason": pipeline_gate["reason"],
                        "failure_code": pipeline_gate["failure_code"],
                        "candidate_source": {
                            "policy": candidate_policy.as_dict(),
                            "selection_mode": "not_entered_due_to_pipeline_gate",
                        },
                        "candidate_metadata": [], "candidate_attempts": [],
                        "timing": {"total_s": 0.0, "approach_attempts": 0,
                                   "ik_valid_count": 0, "candidate_budget": None,
                                   "search_truncated": False},
                    }
                else:
                    planning = _approach_and_lift(
                        planner, scene_cfg=scenario["scene_cfg"], obj=obj,
                        version=args.grasp_version, candidate_hand=args.hand,
                        pose_stem=pose_stem, options=options,
                        candidate_policy=candidate_policy,
                        candidate_budget=(None if args.max_candidates == 0 else args.max_candidates),
                        timing=trace, timing_parent_id=planning_span,
                        candidate_scene_type_filter=candidate_scene_type_filter,
                        use_coverage_order=(candidate_scene_type_filter is None),
                        execution_profile=execution_profile)
                trace.end(planning_span,
                          outcome=("skipped" if pipeline_gate is not None
                                   else ("success" if planning["success"] else "failure")),
                          reason=planning.get("reason"),
                          failure_code=planning.get("failure_code"))

                artifact_span = trace.begin(phase="artifacts", kind="io",
                                            name="episode_planning_artifacts")
                _write_json(episode / "candidate_source.json", planning.get("candidate_source", {}))
                _write_json(episode / "candidate_metadata.json", planning.get("candidate_metadata", []))
                _write_json(episode / "candidate_attempts.json", planning.get("candidate_attempts", []))
                if planning.get("jacobian_steps"):
                    save_step_csv(episode / "jacobian_steps.csv", planning["jacobian_steps"])
                if planning["success"]:
                    plan_dir.mkdir(parents=True, exist_ok=True)
                    np.save(plan_dir / "traj.npy", planning["approach"])
                    np.save(plan_dir / "wrist_se3.npy", planning["wrist_se3"])
                    _save_jacobian_lift(plan_dir / "lift_jacobian.npz", planning, options)
                    _save_execution_trajectory(
                        plan_dir / "execution_trajectory.npz", planning)
                    _save_animation(episode / "animation.npz", planner, scenario, planning)
                trace.end(artifact_span)

                timing_trace = trace.as_dict()
                plan_dir.mkdir(parents=True, exist_ok=True)
                _write_json(plan_dir / "timing.json", timing_trace)
                compact_attempts = [
                    {key: value for key, value in report.items() if key != "jacobian_steps"}
                    for report in planning.get("candidate_attempts", [])
                ]
                trial_success = None if pipeline_gate is not None else bool(planning["success"])
                result = {
                    "schema_version": 3, "success": trial_success,
                    "reason": planning.get("reason"), "failure_code": planning.get("failure_code"),
                    "object": obj, "source": args.object_source,
                    "arm": args.arm, "planner_robot": planner_robot,
                    "input_contract": ("pipeline_parity_live" if pipeline_contract is not None
                                       else ("pipeline_live" if args.object_source == "perception"
                                             else "constructed")),
                    "xy_m": np.asarray(xy).tolist(), "tabletop": tabletop,
                    "board_proxy": proxy, "lift_strategy": "jacobian_continuation_v1",
                    "candidate_policy": planning.get("candidate_source", {}).get("policy"),
                    "candidate_source": "candidate_source.json",
                    "candidate_metadata": "candidate_metadata.json",
                    "candidate_attempts": compact_attempts,
                    "selected_candidate": (
                        {"index": planning["candidate_index"],
                         "scene_info": planning["scene_info"],
                         "wrist_se3_file": "plan/wrist_se3.npy"}
                        if planning.get("success") else None),
                    "perception": {
                        "table_snap_delta_m": scenario.get("table_snap_delta_m"),
                        "mesh_bottom_z_raw_m": scenario.get("mesh_bottom_z_raw_m"),
                        "mesh_bottom_z_scene_m": scenario.get("mesh_bottom_z_scene_m"),
                        "detail": scenario.get("timing", {}),
                    },
                    "planning": planning.get("timing", {}),
                    "timing_trace": timing_trace,
                }
                if pipeline_contract is not None:
                    result["pipeline_parity"] = {
                        "contract": pipeline_contract,
                        "early_gate": pipeline_gate,
                    }
                if planning.get("jacobian_steps"):
                    result["jacobian_lift"] = {
                        "artifact": "plan/lift_jacobian.npz",
                        "step_csv": "jacobian_steps.csv",
                        "step_count": len(planning["jacobian_steps"]),
                        "final": planning["jacobian_steps"][-1],
                    }
                if planning.get("success"):
                    execution = planning["execution"]
                    result["execution_trajectory"] = {
                        "artifact": "plan/execution_trajectory.npz",
                        "schema_version": execution["schema_version"],
                        "sample_count": int(len(execution["qpos"])),
                        "duration_s": execution["segments"]["total_duration_s"],
                        "sample_dt_s": execution["profile"]["sample_dt_s"],
                        "segments": execution["segments"],
                        "validation": planning["execution_validation"],
                        "animation": "animation.npz",
                    }
                _write_json(episode / "result.json", result)
                with (session_dir / "trial_index.jsonl").open("a") as f:
                    f.write(json.dumps({"episode": str(episode), "success": result["success"],
                                        "object": obj, "source": args.object_source,
                                        "arm": args.arm, "planner_robot": planner_robot,
                                        "reason": result["reason"],
                                        "failure_code": result["failure_code"]}) + "\n")
                state["last_object"] = obj
                if args.object_source == "constructed":
                    state.setdefault("last_pose_by_object", {})[obj] = int(tabletop["idx"])
                _save_last_state(output_root, args.hand, state)
                last_episode = episode
                print(f"[trial] success={result['success']} reason={result['reason']} -> {episode}")
                if result["success"]:
                    if args.open_viewer:
                        _launch_viewer(episode, args.port)
                    else:
                        _print_viewer_hint(episode, args.port)
            except UserQuit:
                break
            except Exception as exc:
                if episode is not None:
                    exception_result = {
                        "schema_version": 2, "success": False, "reason": "lift_test_exception",
                        "failure_code": "lift_test_exception", "exception": repr(exc),
                        "arm": args.arm, "planner_robot": planner_robot,
                    }
                    if trace is not None:
                        trace.close_open_spans(outcome="failure")
                        exception_result["timing_trace"] = trace.as_dict()
                    _write_json(episode / "result.json", exception_result)
                    last_episode = episode
                print(f"[trial] ERROR: {exc!r}")

            cmd = _print_next_prompt()
            if cmd == "q":
                break
            if cmd == "v":
                if last_episode is None:
                    print("[viser] no completed trial yet")
                elif not (last_episode / "animation.npz").exists():
                    print("[viser] latest trial has no successful animation")
                else:
                    _launch_viewer(last_episode, args.port)
                continue
            if cmd == "b":
                if args.board_source != "live-charuco":
                    print("[board] remeasurement requires --board-source live-charuco")
                    continue
                try:
                    _input("[board] Clear board and press Enter to remeasure")
                    measurement, proxy = capture.measure_board(session_dir)
                    print("[board] proxy updated for future trials")
                except UserQuit:
                    break
                except Exception as exc:
                    print(f"[board] measurement failed: {exc!r}")
                continue
            if cmd not in {"", "n"}:
                print("[session] choose n, v, b, or q")
    finally:
        capture.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
