#!/usr/bin/env python3
"""Automated mode: distributed FoundPose init -> planning -> execute -> label.

Per trial we run one FoundPose init across
the capture PCs and treat that pose as the object pose for planning.

Prerequisites:
    bash scripts/init_daemons.sh start    # init_daemon on capture1-3, 5, 6
    bash scripts/init_daemons.sh status   # 1 daemon per PC

Usage:
    python src/execution/run_auto.py --obj attached_container
    python src/execution/run_auto.py --obj brown_ramen --scene wall --wall_angle 0
    python src/execution/run_auto.py --obj brown_ramen --success_only --viz
"""
from __future__ import annotations

import argparse
import datetime
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller import get_arm, get_hand  # noqa: F401  (warm import)
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, load_c2r

from autodex.utils.path import (
    project_dir, get_obj_root, get_candidate_path, RESET_RELEASE_HEIGHTS_CM,
)
from autodex.planner import (
    CudaPlanningFault,
    GraspPlanner,
    raise_cuda_planning_fault,
)
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.executor.lift_policy import LiftExecutionError
from autodex.pipeline_trace import PipelineTrace, ScopedPipelineTrace
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator
from autodex.tasks import (
    LiftTask,
    TaskContext,
    TaskInterface,
    attach_task_outcome,
)

from autodex.utils.coverage import (
    experiment_candidate_state_root,
    write_candidate_result,
)
from autodex.utils.robot_config import (
    CHARUCO_BOARD_11_CENTER_XY,
    CHARUCO_BOARD_CENTER_X_OFFSETS_M,
)
from src.demo.continuous_basket.recording import resolve_signal_generator_params
from src.execution.scene_cfg import (
    add_fixed_mesh_fixtures,
    pose_world_to_scene_cfg,
)
from src.execution.handeye import save_arm_C2R
from src.execution.label import auto_label_charuco, get_label

# Board id lives in src/execution/label.py — one place to swap.
from src.execution.label import CHARUCO_BOARD  # noqa: E402

# Candidate pools driven by precomputed scene coverage: the runner ranks
# candidates by remaining-uncovered scenes, bounces to a reorient target when
# the current tabletop is fully covered, and stops when nothing is left.
COVERAGE_VERSIONS = ("v8",)


def _pipeline_result_value(value):
    """Build a compact episode result without a second timing authority.

    ``events.jsonl`` is the only persisted clock. Runtime return objects still
    carry local diagnostic dictionaries because the control flow consumes a
    few counters, but no key containing ``timing`` crosses this persistence
    boundary. Motion arrays are referenced by the normal plan artifacts and
    represented here only by shape so recovery results cannot balloon JSON.
    """
    if dataclasses.is_dataclass(value):
        result = {
            "type": type(value).__name__,
            "success": bool(getattr(value, "success", False)),
            "scene_info": _pipeline_result_value(
                getattr(value, "scene_info", None)),
        }
        local_timing = getattr(value, "timing", None)
        if isinstance(local_timing, dict) and "candidate_idx" in local_timing:
            result["selected_candidate_index"] = int(
                local_timing["candidate_idx"])
        return result
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): _pipeline_result_value(item)
            for key, item in value.items()
            if "timing" not in str(key).lower()
        }
    if isinstance(value, (tuple, list, set)):
        return [_pipeline_result_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_coverage_pool(version: str) -> bool:
    return version in COVERAGE_VERSIONS


def _rotate_streak_count(state: Optional[Dict[str, object]]) -> int:
    """Return the active run of successful rotate recoveries.

    A yaw recovery can legitimately classify as a different tabletop asset on
    the next perception pass.  This must therefore be one session-level
    streak, rather than a counter indexed by the latest classified pose.
    """
    if state is None:
        return 0
    try:
        return max(0, int(state.get("count", 0)))
    except (TypeError, ValueError):
        return 0


def _record_successful_rotate(
        state: Optional[Dict[str, object]], *, tabletop_stem: Optional[str]) -> int:
    """Extend the active physical-rotation streak and return its new count."""
    count = _rotate_streak_count(state) + 1
    if state is not None:
        if count == 1:
            state["sequence_start_tabletop_stem"] = tabletop_stem
        state["count"] = count
        state["last_rotate_tabletop_stem"] = tabletop_stem
    return count


def _reset_rotate_streak(
        state: Optional[Dict[str, object]], *, reason: str,
        tabletop_stem: Optional[str]) -> Optional[dict]:
    """End a rotation-recovery streak, returning an audit event if one existed."""
    count = _rotate_streak_count(state)
    if state is None or count == 0:
        return None
    event = {
        "count_before": count,
        "reason": reason,
        "tabletop_stem": tabletop_stem,
        "sequence_start_tabletop_stem": state.get("sequence_start_tabletop_stem"),
        "last_rotate_tabletop_stem": state.get("last_rotate_tabletop_stem"),
    }
    state["count"] = 0
    state["sequence_start_tabletop_stem"] = None
    state["last_rotate_tabletop_stem"] = None
    state["last_reset"] = event
    return event


def _reorient_target_absent_result(*, obj: str, hand: str, version: str,
                                   obj_root: str, dir_idx: str,
                                   scene_type: str, timing: dict,
                                   success_root: Optional[str] = None) -> dict:
    """Give an accurate terminal reason when reset candidates cannot move us onward.

    A missing target is *not* equivalent to all tabletops being covered: it
    can mean that coverage remains on another v8 tabletop but no direct
    strictly mapped legacy ``reset_<height>/<obj>/reorient_<height>/`` cell
    was generated.
    """
    from autodex.utils.coverage import uncovered_tabletop_counts

    counts = uncovered_tabletop_counts(
        obj, hand, version, obj_root, success_root=success_root)
    remaining = {stem: n for stem, n in (counts or {}).items() if n > 0}
    if remaining:
        runtime_root = get_candidate_path(hand)
        from autodex.utils.tabletop_map import describe_map
        print(f"    [reorient] validated legacy reset candidates are missing for remaining "
              f"tabletops: {remaining}.")
        print(f"    [reorient] tabletop map: {describe_map(obj, version)}")
        print(f"    [reorient] expected mapped legacy cells under "
              f"{runtime_root}/reset_{{{','.join(map(str, RESET_RELEASE_HEIGHTS_CM))}}}/"
              f"{obj}/reorient_<height>/<legacy-current>_<legacy-target>/")
        return {
            "dir_idx": dir_idx, "scene_type": scene_type, "success": None,
            "reason": "reorient_legacy_candidates_missing", "all_done": True,
            "remaining_uncovered_by_tabletop": remaining,
            "reset_candidate_root": runtime_root, "timing": timing,
        }
    print(f"    All v8 tabletops covered — nothing left for {obj}.")
    return {
        "dir_idx": dir_idx, "scene_type": scene_type, "success": None,
        "reason": "all_tabletops_covered", "all_done": True,
        "timing": timing,
    }


def _stop_with_timeout(name: str, fn, timeout: float = 20.0) -> bool:
    """Run a shutdown call with a deadline, naming it if it does not return.

    Every stop() on this path ends in an untimed Event.wait() -- rcc waits on
    `sending_event`, the timestamp monitor on `event["stop"]` -- so a device
    that died mid-capture hangs the whole trial with no clue which one it was.
    An untimed wait is also not Ctrl-C interruptible, so the operator can only
    kill the process. Run each in its own thread and move on if it overruns:
    a leaked stop is far cheaper than a wedged experiment.

    Returns True if it returned in time.
    """
    import threading
    done = threading.Event()
    err = []

    def _run():
        try:
            fn()
        except Exception as exc:      # a failing stop must not kill the trial
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name=f"stop-{name}").start()
    if not done.wait(timeout):
        print(f"[shutdown] {name}.stop() did not return in {timeout:.0f}s — "
              f"leaving it and continuing")
        return False
    if err:
        print(f"[shutdown] {name}.stop() raised: {err[0]!r}")
    return True


def _safe_timestamp_start(tsm, save_path) -> bool:
    """Start the sync-timestamp monitor, refusing to block on a dead one.

    TimestampMonitor.run() gives up when its camera cannot be opened: it logs
    "continuing WITHOUT sync timestamps", sets error+connection+stop, and the
    thread RETURNS. It guards stop() against that ("stop() blocks on this;
    without it a later stop() would hang forever") but not start() -- which
    ends in `self.event["acquisition"].wait()` with no timeout, waiting on a
    thread that is already gone. The wait is not Ctrl-C interruptible, so the
    trial hangs until the process is killed.

    The error flag is one-shot (start() clears it via the "is in ERROR state"
    branch), which is why the hang lands on the SECOND trial rather than the
    first. Check the thread itself instead: no live capture thread means no
    one will ever set `acquisition`.

    Returns True if the monitor was started, False if it was skipped.
    """
    th = getattr(tsm, "capture_thread", None)
    alive = th.is_alive() if th is not None else False
    cam_ok = getattr(tsm, "camera", None) is not None
    if not alive or not cam_ok:
        print(f"[timestamp] monitor is dead (thread_alive={alive} "
              f"camera={'ok' if cam_ok else 'None'}) — skipping, "
              f"recording WITHOUT sync timestamps")
        tsm._autodex_started = False
        return False
    tsm.start(save_path)
    tsm._autodex_started = True
    return True


def _safe_timestamp_stop(tsm) -> None:
    """Stop the monitor only if we actually started it.

    stop() ends in an untimed event["stop"].wait(). When start() was skipped
    there is nothing to stop and nobody left to set that event, so calling it
    burns the full shutdown timeout on every trial for no reason.
    """
    if not getattr(tsm, "_autodex_started", False):
        return
    _stop_with_timeout("timestamp_monitor", tsm.stop)


def _emit_external_sync_cue(sync_generator, trace: PipelineTrace, *,
                            label: str, enabled: bool,
                            duration_s: float, fps: int) -> None:
    """Emit a short trigger burst for an LED visible to an external camera.

    The signal generator's TTL output should be split to a small LED placed at
    the edge of the external camera frame.  The call is opt-in because labs
    without that LED should not receive extra trigger pulses between captures.
    """
    if not enabled:
        return
    cue_span = trace.begin(
        phase="sync", kind="cue", name="external_video_sync_cue",
        label=label, requested_duration_s=duration_s, fps=fps)
    try:
        sync_generator.start(fps=fps)
        trace.event(
            "sync.visual_cue", phase="sync", kind="cue",
            parent_id=cue_span, label=label, fps=fps,
            requested_duration_s=duration_s)
        time.sleep(duration_s)
        sync_generator.stop()
    except Exception as exc:
        try:
            sync_generator.stop()
        except Exception:
            pass
        trace.end(cue_span, outcome="failure", exception=repr(exc))
        raise
    else:
        trace.end(cue_span, outcome="success")


def _prompt_or_auto(args, prompt: str) -> str:
    """Return Enter-equivalent in ``--auto``; prompt only in supervised mode."""
    if args.auto:
        print(f"{prompt} --auto: continuing without operator confirmation")
        return ""
    try:
        return input(prompt).strip().lower()
    except KeyboardInterrupt:
        return "q"


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture now arms in 'acquire' and its
    outputs are toggled as SINKS (set_stream / set_record). The capture PCs
    reject the old names outright --
        invalid mode 'stream': use 'image' (single frame) or 'acquire' + ...
    -- which silently left every trial without frames. Keeping the old call
    shape here means the trial flow below reads unchanged.
    """
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
        _warn_if_not_streaming(rcc)
    elif mode == "full":
        # "full" was video AVI + SHM stream at once (snapshot_daemon reads the
        # stream while the AVI records). Both are just sinks now.
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
        rcc.set_stream(True)
    elif mode == "video":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
    else:                       # 'image' is still a real capture mode
        rcc.start(mode, sync_mode, save_path, fps=fps)


def _warn_if_not_streaming(rcc, timeout_s: float = 4.0, poll_s: float = 0.5) -> bool:
    """Warn if the cameras are not actually capturing after the stream is armed.

    The failure this catches is silent: without a running capture the init
    pipeline just sits on 0/20 masks until it times out. Poll rather than read
    one status snapshot — ``running`` is reported by the daemons' health PUB and
    lags the sink command by a beat, so a single read right after set_stream
    reports False on healthy cameras.
    """
    deadline = time.time() + timeout_s
    dead: list = []
    while time.time() < deadline:
        try:
            dead = [pc for pc, s in (rcc.get_status().get("pc") or {}).items()
                    if not s.get("running")]
        except Exception as exc:
            print(f"[rcc] status check failed: {exc!r}")
            return True                      # don't block the run on telemetry
        if not dead:
            return True
        time.sleep(poll_s)
    print(f"[rcc] WARNING stream armed but not capturing on: {dead}")
    return False


def _ensure_camera_lock(rcc, settle_s: float = 1.5) -> bool:
    """Make sure THIS controller owns the daemons, taking over if it does not.

    A crashed run leaves its lock behind, and the next ``register`` is refused
    ("locked by run_auto_<earlier>"). ``register()`` only logs that and sets
    ``_registered = True`` anyway, so every later arm/set_stream is silently
    dropped by the daemon: cameras never capture, the init pipeline waits out
    its timeout on 0/20 masks, and nothing says why. Check ownership against
    the daemons' own report instead of trusting registration.
    """
    time.sleep(settle_s)
    try:
        pcs = (rcc.get_status().get("pc") or {})
        foreign = {pc: s.get("controller") for pc, s in pcs.items()
                   if s.get("controller") and s.get("controller") != rcc.name}
        if not foreign:
            return True
        print(f"[rcc] daemons held by another controller: {foreign}")
        print("[rcc] forcing takeover")
        rcc.force_takeover()
        time.sleep(1.0)
        pcs = (rcc.get_status().get("pc") or {})
        still = {pc: s.get("controller") for pc, s in pcs.items()
                 if s.get("controller") and s.get("controller") != rcc.name}
        if still:
            print(f"[rcc] TAKEOVER FAILED, still held by: {still}")
            return False
        print("[rcc] takeover ok")
        return True
    except Exception as exc:
        print(f"[rcc] ownership check failed: {exc!r}")
        return False


def _clear_camera_errors(rcc, settle_s: float = 1.5, reload_wait_s: float = 6.0,
                         attempts: int = 2) -> bool:
    """Reload the capture daemons' cameras if they are stuck in an error state.

    A camera that failed to start latches its error and never clears it: on the
    error path ``Camera.start()`` returns BEFORE setting ``event["start"]``,
    while ``error_reset()`` only fires from ``stop()`` when that same event was
    set. So one bad start (e.g. a retired capture mode) poisons the camera for
    every later run, and every ``start`` after it returns early — trials then
    run with no frames at all. Reloading rebuilds the daemon's CameraLoader,
    which is the only thing that clears it.

    Returns True if the cameras are healthy when this returns.
    """
    time.sleep(settle_s)
    if not rcc.is_error():
        return True
    for i in range(attempts):
        print(f"[rcc] cameras in error state — reloading "
              f"({i + 1}/{attempts})")
        try:
            rcc.force_takeover()      # a dead session may still hold the lock
        except Exception as exc:
            print(f"[rcc] force_takeover failed: {exc!r}")
        try:
            rcc.reload_cameras()
        except Exception as exc:
            print(f"[rcc] reload_cameras failed: {exc!r}")
        time.sleep(reload_wait_s)
        if not rcc.is_error():
            print("[rcc] cameras recovered")
            return True
    print("[rcc] STILL in error after reload — check the capture PCs:\n"
          f"      {rcc.get_status()}")
    return False



def _candidate_state_root(args, hand: str, obj: str) -> Optional[str]:
    """Return the active outcome store for this run, if it is private."""
    if not getattr(args, "isolate_experiment", False):
        return None
    return experiment_candidate_state_root(
        args.exp_name, hand, args.grasp_version, obj)


def _candidate_state_grasp_dir(args, hand: str, obj: str, scene_info) -> Optional[str]:
    """Resolve one grasp's mutable outcome directory for this campaign."""
    if not (isinstance(scene_info, (list, tuple)) and len(scene_info) == 3):
        return None
    base = _candidate_state_root(args, hand, obj)
    if base is None:
        base = os.path.join(get_candidate_path(hand), args.grasp_version, obj)
    return os.path.join(base, scene_info[0], scene_info[1], scene_info[2])


def _scene_has_success(hand: str, version: str, obj: str,
                        scene_type: str, scene_id: str,
                        candidate_state_root: Optional[str] = None) -> bool:
    """Return True if any grasp candidate under
    candidates/{hand}/{version}/{obj}/{scene_type}/{scene_id}/ already has
    a result.json marking success=True. Used to skip a whole scene once any
    grasp in it has worked (user policy: 한 scene 성공하면 그 scene 통째로 제외)."""
    from autodex.utils.path import get_candidate_path
    base = (Path(candidate_state_root) if candidate_state_root is not None
            else Path(get_candidate_path(hand)) / version / obj)
    if scene_type:
        base = base / scene_type / scene_id
    else:
        base = base / scene_id
    if not base.is_dir():
        return False
    for grasp_dir in base.iterdir():
        if not grasp_dir.is_dir():
            continue
        rj = grasp_dir / "result.json"
        if not rj.exists():
            continue
        try:
            with open(rj) as f:
                if json.load(f).get("success", False):
                    return True
        except Exception:
            continue
    return False


def _write_candidate_outcome(args, hand: str, obj: str, scene_info,
                             payload: dict) -> bool:
    """Persist a grasp outcome in shared or experiment-private state.

    Geometry stays in ``candidates/<hand>/<version>``.  Isolated campaigns
    mirror only their small mutable ``result.json`` records under
    ``experiment/<exp_name>/candidate_state`` so coverage is fully private.
    """
    grasp_dir = _candidate_state_grasp_dir(args, hand, obj, scene_info)
    if grasp_dir is None:
        return False
    os.makedirs(grasp_dir, exist_ok=True)
    cand_result_path = os.path.join(grasp_dir, "result.json")
    write_candidate_result(cand_result_path, payload)
    if args.isolate_experiment:
        print("    [experiment] candidate outcome -> "
              f"{os.path.relpath(cand_result_path, project_dir)}")
    return True


def _write_experiment_coverage_progress(args, hand: str, obj: str,
                                        obj_root: str) -> Optional[dict]:
    """Persist the isolated campaign's remaining-scene count per tabletop."""
    if (not args.isolate_experiment or args.ignore_coverage
            or not _is_coverage_pool(args.grasp_version)):
        return None
    from autodex.utils.coverage import uncovered_tabletop_counts

    state_root = _candidate_state_root(args, hand, obj)
    remaining = uncovered_tabletop_counts(
        obj, hand, args.grasp_version, obj_root,
        success_root=state_root,
    )
    if remaining is None:
        return None
    total = sum(remaining.values())
    payload = {
        "exp_name": args.exp_name,
        "hand": hand,
        "arm": args.arm,
        "grasp_version": args.grasp_version,
        "candidate_state_root": state_root,
        "updated_at": datetime.datetime.now().isoformat(),
        "remaining_uncovered_by_tabletop": remaining,
        "total_remaining": total,
    }
    progress_path = os.path.join(
        project_dir, "experiment", args.exp_name, "coverage",
        hand, args.grasp_version, f"{obj}.json",
    )
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"    [experiment coverage] {total} scenes remaining; "
          f"state -> {os.path.relpath(progress_path, project_dir)}")
    return payload


logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def quiet_curobo(level=logging.WARNING) -> None:
    """Silence cuRobo's per-solve chatter ("Updating problem kernel", "Ran TO",
    "breaking reference", ...).

    setLevel alone is not enough here: whichever library configures logging
    first owns the root handler, and cuRobo's records still reach it by
    propagation. Cutting propagate keeps them off stdout no matter who set up
    the root, and the NullHandler stops the "no handlers" fallback from
    printing them anyway. A single plan emits dozens of these lines, which bury
    the [planner]/[place] output that actually says what happened.
    """
    lg = logging.getLogger("curobo")
    lg.setLevel(level)
    lg.propagate = False
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())


quiet_curobo()

def _planner_robot(arm: str, hand: str) -> str:
    """cuRobo robot config name for an (arm, hand) pair.

    The xarm configs are keyed by hand alone (``xarm_allegro`` etc. are selected
    inside GraspPlanner from the hand string), so xarm keeps passing the hand
    through unchanged. The FR3 has its own config per hand.
    """
    if arm != "franka":
        return hand
    if hand == "inspire":
        return "fr3_inspire"
    # NOTE: GraspPlanner.HAND_CONFIGS.get(hand, allegro) SILENTLY falls back to
    # the allegro xarm config for an unknown key, so an unsupported combination
    # must be rejected here rather than planned with the wrong robot.
    raise SystemExit(
        f"--arm franka does not support --hand {hand}: no fr3 config for it "
        f"(GraspPlanner.HAND_CONFIGS has fr3_inspire only, and "
        f"{project_dir}/content/configs/robot/ ships fr3_inspire.yml only). "
        f"Use --hand inspire.")


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


def _wait_for_object_on_table(rcc, args, scene_prefix: str, trial_idx: int,
                                required_board: str = CHARUCO_BOARD) -> bool:
    """Block until charuco `required_board` is NOT fully visible (= obj covers it).
    Used as a pre-flight check before each --auto trial.

    Returns True to proceed with trial, False if user pressed 'q' to quit.
    """
    sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
    attempt = 0
    while True:
        attempt += 1
        check_rel = os.path.join(
            "shared_data", "AutoDex", "experiment", args.exp_name, sub,
            args.obj, "_precheck", f"trial{trial_idx:03d}_{attempt:02d}", "raw"
        )
        check_abs = os.path.join(
            project_dir, "experiment", args.exp_name, sub, args.obj,
            "_precheck", f"trial{trial_idx:03d}_{attempt:02d}", "raw", "images"
        )
        _stop_with_timeout("rcc", rcc.stop)
        rcc.start("image", False, check_rel)
        rcc.stop()
        time.sleep(0.3)
        board_visible, info = auto_label_charuco(check_abs, required_board=required_board)
        # board_visible=True → board fully detected = no obj on table → prompt.
        # board_visible=False → board partially hidden = obj on table → start.
        # board_visible=None → no images / board not in cfg → start anyway (don't block).
        if board_visible is None:
            print(f"[precheck] {info.get('reason', 'unknown')} — proceeding without check")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            return True
        if not board_visible:
            print(f"[precheck] obj on table (board covered "
                  f"{info.get('covered')}/{info.get('expected')}). Starting trial.")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            return True
        print(f"[precheck] charuco fully visible "
              f"({info.get('covered')}/{info.get('expected')}) — no obj on table.")
        if args.auto:
            print("[precheck] --auto: object placement needs an operator; "
                  "ending instead of waiting for input")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            return False
        try:
            cmd = input("    Place obj then press Enter (q to quit): ").strip().lower()
        except KeyboardInterrupt:
            return False
        if cmd == "q":
            return False


# ── calibration ──────────────────────────────────────────────────────────────

def _load_calib(calib_dir: Path):
    """Read intrinsics.json + extrinsics.json into the dict shape InitOrchestrator wants."""
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a

    first = next(iter(intrinsics_full.values()))
    return intrinsics_full, extrinsics_full, int(first["height"]), int(first["width"])


# ── single trial ─────────────────────────────────────────────────────────────

_active_vis: Optional[ScenePlanVisualizer] = None


def run_single_trial(
    args,
    *,
    scene_prefix: str,
    orch: InitOrchestrator,
    planner: GraspPlanner,
    executor,          # RealExecutor (xarm) or FrankaExecutor (fr3)
    rcc,
    sync_generator,
    timestamp_monitor,
    pose_adjust_handler=None,
    reorient_handler=None,
    tabletop_geometry=None,
    fixed_fixtures=None,
    rotate_recovery_state: Optional[Dict[str, object]] = None,
    pipeline_trace: Optional[PipelineTrace] = None,
    attempt_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    session_attempted_candidates: Optional[set] = None,
    task: Optional[TaskInterface] = None,
) -> dict:
    global _active_vis
    if _active_vis is not None:
        try:
            _active_vis.server.stop()
        except Exception:
            pass
        _active_vis = None

    obj = args.obj
    hand = args.hand
    session_excluded = {
        tuple(str(part) for part in key)
        for key in (session_attempted_candidates or ())
    }
    candidate_state_root = _candidate_state_root(args, hand, obj)
    # xarm = 6, FR3 = 7. Every arm/hand column split below uses this instead of
    # a literal 6, so the same trial body drives both arms.
    adof = getattr(executor, "arm_dof", 6)
    # Microseconds plus the run-level attempt id prevent recovery retries from
    # reusing a directory when they start inside the same wall-clock second.
    dir_idx = episode_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if episode_id is None and attempt_id:
        dir_idx = f"{dir_idx}_{attempt_id}"
    sub = f"{scene_prefix}/{hand}" if scene_prefix else hand
    img_dir = os.path.join(project_dir, "experiment", args.exp_name, sub, obj, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    task = task or LiftTask()
    # This dict is execution-local control data only.  It is never persisted;
    # the run-level append-only trace below is the sole timing authority.
    timing: dict = {}
    trial_scope: Optional[ScopedPipelineTrace] = None
    episode_span = None
    episode_started_s = None
    if pipeline_trace is not None:
        trial_scope = pipeline_trace.scoped(
            episode_id=dir_idx, attempt_id=attempt_id)
        episode_started_s = pipeline_trace.now_s()
        episode_span = trial_scope.begin(
            phase="episode", kind="lifecycle", name="episode",
            directory=os.path.relpath(img_dir, project_dir),
            scene_type=args.scene, object=obj, hand=hand, arm=args.arm,
        )
    if hasattr(planner, "set_timing_recorder"):
        planner.set_timing_recorder(trial_scope)
    if hasattr(executor, "set_timing_recorder"):
        executor.set_timing_recorder(trial_scope)

    execution_trigger_active = False

    def _start_execution_trigger(fps: int) -> None:
        nonlocal execution_trigger_active
        sync_generator.start(fps=fps)
        execution_trigger_active = True
        if trial_scope is not None:
            trial_scope.event(
                "capture.trigger_started", phase="capture", kind="sync",
                parent_id=episode_span, fps=fps,
                purpose="internal_multicamera_execution")

    def _stop_execution_trigger(reason: str) -> bool:
        nonlocal execution_trigger_active
        if not execution_trigger_active:
            return True
        stopped = _stop_with_timeout("sync_generator", sync_generator.stop)
        execution_trigger_active = False
        if trial_scope is not None:
            trial_scope.event(
                "capture.trigger_stopped", phase="capture", kind="sync",
                parent_id=episode_span,
                outcome=("success" if stopped else "failure"), reason=reason)
        return stopped

    def _ts() -> str:
        return datetime.datetime.now().isoformat()

    episode_finalized = False

    def _with_task_outcome(
        record: dict,
        grasp_success: Optional[bool],
        *,
        grasp_evidence: Optional[dict] = None,
    ) -> dict:
        scene_info = record.get("scene_info")
        context = TaskContext(
            object_name=obj,
            arm=args.arm,
            hand=hand,
            trial_dir=img_dir,
            scene_info=(tuple(str(part) for part in scene_info)
                        if scene_info is not None else None),
            metadata={
                "scene_type": args.scene,
                "candidate_result_scope": record.get("candidate_result_scope"),
                "fixed_fixtures": fixed_fixtures or {},
            },
        )
        outcome = task.evaluate(
            context=context,
            grasp_success=grasp_success,
            grasp_evidence=grasp_evidence,
        )
        return attach_task_outcome(
            record, grasp_success=grasp_success, task_outcome=outcome)

    def _stamp_end(result_dict):
        """Link this episode to the one canonical run-level timeline."""
        nonlocal episode_finalized
        if episode_finalized:
            return result_dict
        timing["trial_end"] = _ts()
        result_dict.pop("timing", None)
        if pipeline_trace is not None and trial_scope is not None:
            success = result_dict.get("success")
            grasp_success = result_dict.get("grasp_success", success)
            outcome = ("success" if success is True else
                       "failure" if success is False else "skipped")
            if result_dict.get("retry_current_trial"):
                outcome = ("success" if result_dict.get("reason") in (
                    "reoriented", "reoriented_manual", "pose_adjusted",
                    "planning_retry_feasible") else outcome)
            recovery_action = None
            if result_dict.get("rotation") is not None or result_dict.get(
                    "pose_adjust") is not None:
                recovery_action = "rotation"
            elif result_dict.get("reorientation") is not None or result_dict.get(
                    "reorient_target_j") is not None:
                recovery_action = "reorientation"
            pipeline_trace.close_episode_spans(
                dir_idx, exclude={episode_span} if episode_span else set(),
                reason=f"episode_finalized:{result_dict.get('reason')}",
            )
            trial_scope.event(
                "episode.result", phase="episode", kind="result",
                parent_id=episode_span, outcome=outcome,
                success=success, reason=result_dict.get("reason"),
                grasp_success=result_dict.get("grasp_success"),
                task_success=result_dict.get("task_success"),
                task_name=(result_dict.get("task") or {}).get("name"),
                scene_info=result_dict.get("scene_info"),
                retry_current_trial=bool(result_dict.get("retry_current_trial")),
                recovery_action=recovery_action,
                pose_adjust=result_dict.get("pose_adjust"),
                reorient_target_j=result_dict.get("reorient_target_j"),
                reorient_target_stem=result_dict.get("reorient_target_stem"),
            )
            if result_dict.get("scene_info") is not None:
                grasp_outcome = (
                    "success" if grasp_success is True else
                    "failure" if grasp_success is False else "skipped"
                )
                trial_scope.event(
                    "grasp.execution_result", phase="validation", kind="result",
                    parent_id=episode_span, outcome=grasp_outcome,
                    scene_info=result_dict.get("scene_info"),
                    success=grasp_success, reason=result_dict.get("reason"),
                )
            ended = (trial_scope.end(
                episode_span, outcome=outcome,
                reason=result_dict.get("reason"), success=success)
                if episode_span is not None else None)
            result_dict["pipeline_trace"] = {
                "schema_version": 1,
                "run_id": pipeline_trace.run_id,
                "episode_id": dir_idx,
                "attempt_id": attempt_id,
                "episode_span_id": episode_span,
                "run_timeline": os.path.relpath(
                    pipeline_trace.output_dir / "events.jsonl", img_dir),
                "episode_timeline": os.path.relpath(
                    pipeline_trace.output_dir /
                    pipeline_trace.episode_timeline_path(dir_idx),
                    img_dir),
            }
            pipeline_trace.add_episode({
                "episode_id": dir_idx,
                "attempt_id": attempt_id,
                "episode_span_id": episode_span,
                "directory": os.path.relpath(img_dir, project_dir),
                "timeline": pipeline_trace.episode_timeline_path(
                    dir_idx).as_posix(),
                "start_pipeline_s": episode_started_s,
                "end_pipeline_s": (ended or {}).get("pipeline_time_s"),
                "duration_s": (ended or {}).get("duration_s"),
                "outcome": outcome,
                "success": success,
                "grasp_success": result_dict.get("grasp_success"),
                "task_success": result_dict.get("task_success"),
                "task_name": (result_dict.get("task") or {}).get("name"),
                "reason": result_dict.get("reason"),
                "retry_current_trial": bool(result_dict.get("retry_current_trial")),
                "scene_info": result_dict.get("scene_info"),
                "recovery_action": recovery_action,
                "reorient_target_j": result_dict.get("reorient_target_j"),
                "reorient_target_stem": result_dict.get("reorient_target_stem"),
            })
        episode_finalized = True
        return result_dict

    def _save_result(result_dict):
        """Persist a compact episode index linked to the global timeline."""
        _stamp_end(result_dict)
        persisted = _pipeline_result_value(result_dict)
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(persisted, f, indent=2)
        return persisted

    timing["trial_start"] = _ts()

    # ── 1. prepare ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[1/6] Trial dir -> {dir_idx}")
    preparation_span = (trial_scope.begin(
        phase="preparation", kind="setup", name="episode_preparation",
        parent_id=episode_span) if trial_scope is not None else None)
    t_preparation = time.perf_counter()
    save_arm_C2R(img_dir, args.arm)
    save_current_camparam(img_dir)
    timing["preparation_calibration_s"] = round(
        time.perf_counter() - t_preparation, 3)
    timing["preparation_s"] = timing["preparation_calibration_s"]
    if preparation_span is not None:
        trial_scope.end(preparation_span)

    # ── 2. Distributed FoundPose init ───────────────────────────────────────
    print(f"[2/6] Init pipeline (FoundPose distributed)...")
    timing["perception_start"] = _ts()
    perception_span = (trial_scope.begin(
        phase="perception", kind="inference", name="foundpose",
        parent_id=episode_span) if trial_scope is not None else None)
    t0 = time.perf_counter()
    save_capture_dir = os.path.join(img_dir, "init_capture")
    pose_world, perc_timing = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=save_capture_dir,
        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        timeout_s=args.init_timeout_s,
        sil_loss_threshold=(float("inf") if args.perception_mode == "ignore_sil_loss"
                            else 0.003),
    )
    timing["perception_s"] = round(time.perf_counter() - t0, 3)
    if perc_timing:
        timing["perception_detail"] = perc_timing

    if pose_world is None:
        reason = (perc_timing or {}).get("reason", "perception_failed")
        if perception_span is not None:
            trial_scope.end(
                perception_span, outcome="failure", reason=reason,
                detail=perc_timing)
        print(f"    Perception FAILED ({reason})")
        chime.error()
        # Pause for human — bad pose estimate likely needs operator to
        # reposition object / lights / camera before next attempt.
        cmd = _prompt_or_auto(
            args, "    [perception_failed] Fix the scene then press Enter "
                  "(q to quit): ")
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": reason, "timing": timing}
        if cmd == "q":
            fail["all_done"] = True
            fail["reason"] = "user_quit_perception_failed"
        return _save_result(fail)

    if perception_span is not None:
        trial_scope.end(
            perception_span, outcome="success", detail=perc_timing,
            pose_world=np.asarray(pose_world).tolist())
    print(f"    Perception: {timing['perception_s']}s")
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 2.5 Reposition detection ─────────────────────────────────────────────
    # If charuco board "1" is fully visible RIGHT AFTER perception succeeded,
    # the obj is somewhere but NOT covering the board → enter reposition mode
    # (grasp obj, place at r=0.4, y=0 on the board).
    reposition_mode = False
    reposition_span = (trial_scope.begin(
        phase="perception", kind="decision", name="reposition_detection",
        parent_id=episode_span) if trial_scope is not None else None)
    t_reposition_detection = time.perf_counter()
    if args.auto:
        _stop_with_timeout("rcc", rcc.stop)
        repo_check_rel = os.path.join(
            "shared_data", "AutoDex", "experiment", args.exp_name, sub, obj,
            dir_idx, "_repo_check", "raw"
        )
        repo_check_abs = os.path.join(img_dir, "_repo_check", "raw", "images")
        rcc.start("image", False, repo_check_rel)
        rcc.stop()
        time.sleep(0.3)
        board_vis, board_info = auto_label_charuco(
            repo_check_abs, required_board=CHARUCO_BOARD
        )
        timing["repo_charuco_before"] = board_info
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if board_vis is True:
            print(f"\n    [reposition] charuco "
                  f"{board_info.get('covered')}/{board_info.get('expected')} "
                  f"fully visible → reposition mode")
            reposition_mode = True
    timing["reposition_detection_s"] = round(
        time.perf_counter() - t_reposition_detection, 3)
    if reposition_span is not None:
        trial_scope.end(reposition_span, reposition_mode=reposition_mode,
                        charuco=timing.get("repo_charuco_before"))

    # ── 3. scene_cfg + plan ──────────────────────────────────────────────────
    print(f"[3/6] Planning (version={args.grasp_version}, scene={args.scene})...")
    timing["planning_start"] = _ts()
    t_planning = time.perf_counter()
    scene_span = (trial_scope.begin(
        phase="planning", kind="setup", name="scene_construction",
        parent_id=episode_span) if trial_scope is not None else None)
    t_scene_construction = time.perf_counter()
    c2r = load_c2r(img_dir)
    # Planning mesh and tabletop poses are both resolved from the v8
    # object_processing asset tree, so their pose stems cannot diverge.
    obj_root = get_obj_root(args.grasp_version)
    if tabletop_geometry is not None:
        timing["tabletop_geometry"] = tabletop_geometry
    scene_cfg = pose_world_to_scene_cfg(
        pose_world, c2r, obj, obj_root, tabletop_geometry=tabletop_geometry)
    scene_cfg = add_obstacles(
        scene_cfg, args.scene,
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
        tabletop_geometry=tabletop_geometry,
    )
    scene_cfg = add_fixed_mesh_fixtures(scene_cfg, fixed_fixtures)
    timing["scene_construction_s"] = round(
        time.perf_counter() - t_scene_construction, 3)
    if scene_span is not None:
        scene_artifact = trial_scope.write_artifact_json(
            f"artifacts/scene/{attempt_id or dir_idx}.json", scene_cfg)
        trial_scope.end(scene_span, scene_type=args.scene,
                        artifact=scene_artifact)
    t_candidate_selection = time.perf_counter()

    def _run_reorient_handler(target_j: int) -> dict:
        """Run a pipeline-owned reorientation without replacing live state.

        The normal ``run_auto.py`` entry point intentionally remains manual:
        it supplies no handler.  ``run_pipeline.py`` installs one which gets
        this exact perception result and the same live hardware objects.
        """
        try:
            return reorient_handler(
                args=args, obj=obj, hand=hand, target_j=target_j,
                planner=planner, executor=executor, rcc=rcc,
                scene_cfg=scene_cfg, pose_world=pose_world, c2r=c2r,
                obj_root=obj_root, img_dir=img_dir, scene_prefix=scene_prefix,
                tabletop_geometry=tabletop_geometry,
                fixed_fixtures=fixed_fixtures,
                pipeline_trace=trial_scope,
            )
        except CudaPlanningFault as reorient_exc:
            # CUDA contexts are process-scoped.  Do not turn this into an
            # ordinary candidate miss and then start another trial whose
            # perception will necessarily fail on the poisoned device.
            print(f"    [pipeline] fatal CUDA planning fault: {reorient_exc!r}")
            return {
                "success": False,
                "reason": "cuda_planning_fault",
                "exception": repr(reorient_exc),
                "fatal_cuda_planning_fault": True,
            }
        except Exception as reorient_exc:
            # Keep this outer boundary as a backstop for direct CUDA users in
            # reset code. New helper code must not be able to downgrade a
            # poisoned CUDA context into an ordinary reorient candidate miss.
            try:
                raise_cuda_planning_fault(
                    "inprocess_reorient_handler", reorient_exc,
                    context={"target_j": int(target_j)},
                )
            except CudaPlanningFault as cuda_fault:
                print(f"    [pipeline] fatal CUDA planning fault: {cuda_fault!r}")
                return {
                    "success": False,
                    "reason": "cuda_planning_fault",
                    "exception": repr(cuda_fault),
                    "fatal_cuda_planning_fault": True,
                }
            print(f"    [pipeline] reorient exception: {reorient_exc!r}")
            return {
                "success": False,
                "reason": "inprocess_reorient_exception",
                "exception": repr(reorient_exc),
            }
    # ── tabletop classification + cylinder freedom (mirrors run_debug) ─────
    from src.experiment.reset.tabletop_pose import classify_tabletop_pose
    from autodex.utils.symmetry import get_cyl_axis_local
    pose_robot = np.linalg.inv(c2r) @ pose_world
    tb_before = classify_tabletop_pose(pose_robot, obj, obj_root)
    timing["tabletop_before"] = tb_before
    if trial_scope is not None:
        trial_scope.event(
            "scene.tabletop_classified", phase="perception", kind="result",
            parent_id=episode_span,
            outcome=("success" if tb_before is not None else "failure"),
            classification=tb_before,
        )
    if tb_before is not None:
        scene_id = str(tb_before["idx"])
        pose_stem = tb_before["filename"].replace(".npy", "")
        print(f"    [tabletop] idx={tb_before['idx']} "
              f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}°")
        # Policy: once any grasp in this scene has succeeded, skip the WHOLE
        # scene (including the successful grasp itself). Only applies for the
        # table scene (other scenes don't persist per-candidate result.json).
        scene_type_for_check = args.scene if args.scene != "table" else "table"
        if (not args.isolate_experiment and _scene_has_success(
                hand, args.grasp_version, obj, scene_type_for_check, scene_id,
                candidate_state_root=candidate_state_root)):
            print(f"    [scene_skip] scene_type={scene_type_for_check} "
                  f"scene_id={scene_id} already has a successful grasp — "
                  f"skipping this trial")
            skip = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "scene_already_done",
                    "scene_id": scene_id, "timing": timing}
            return _save_result(skip)
    else:
        scene_id = None
        pose_stem = None

    def _mark_grasp_validation_started(stage: str) -> None:
        """Break the rotate streak when an executed grasp reaches validation.

        This is intentionally later than planning and execution start: neither
        proves that the gripper closed and completed the lift. Conversely it
        is intentionally independent of the label result. A successful
        rotation followed by a real grasp-validation attempt is no longer a
        sequence of rotations, even if that validation subsequently fails.
        """
        reset_event = _reset_rotate_streak(
            rotate_recovery_state, reason=stage, tabletop_stem=pose_stem)
        if trial_scope is not None:
            trial_scope.event(
                "grasp.validation_started", phase="validation", kind="state",
                parent_id=episode_span, scene_info=getattr(
                    result, "scene_info", None), stage=stage,
            )
        if reset_event is not None:
            timing["rotate_recovery_reset"] = reset_event
            print("    [rotate] grasp validation started; reset consecutive "
                  f"recovery count ({reset_event['count_before']} -> 0)")

    # Symmetry-axis enumerate. Continuous-revolute → 8 angles. Discrete
    # (e.g. blue_alarm 180°) → order-N grid from pose_symmetry.json.
    from autodex.utils.symmetry import get_cyl_yaw_grid as _get_cyl_yaw_grid
    _cyl_axis = get_cyl_axis_local(obj)
    _cyl_grid = _get_cyl_yaw_grid(obj)
    # Reposition mode keeps the same v8 candidate/tabletop contract; it only
    # changes the eventual object placement, never the grasp asset pool.
    if reposition_mode:
        _eff_grasp_version = args.grasp_version
        _plan_scene_id = None
        _plan_scene_type_filter = args.candidate_scene_type or (
            args.scene if args.scene in ("wall", "shelf", "box") else None
        )
        _plan_tabletop_stem = pose_stem
        _plan_candidate_order = None
        _plan_priority_map = None
        print("    [reposition] using the current tabletop's v8 candidates")
    elif _is_coverage_pool(args.grasp_version):
        _eff_grasp_version = args.grasp_version
        _plan_scene_id = None
        # A newly generated tabletop pool has no coverage JSON yet. Select it
        # directly but keep skip-done and the one-success-per-tabletop safety
        # policy, so real Franka collection does not repeat a known failure or
        # overwrite a successful record.
        _plan_scene_type_filter = args.candidate_scene_type or (
            args.scene if args.scene in ("wall", "shelf", "box") else None
        )
        # Trim by tabletop pose: keep only candidates whose scene
        # meta.pose_idx == current tabletop stem.
        _plan_tabletop_stem = pose_stem
        # NO pre-filter (= all tabletop-matching candidates loaded). After
        # IK+collision in planner.plan, sort survivors by coverage count
        # desc via priority_map, then plan_single_js in that order.
        if args.candidate_scene_type is not None:
            _plan_candidate_order = None
            _plan_priority_map = None
            print(f"    [collection] direct candidate scene type="
                  f"{args.candidate_scene_type}; coverage ordering deferred")
        elif args.ignore_coverage:
            _plan_candidate_order = None
            _plan_priority_map = None
            print(f"    [coverage] IGNORED (--ignore_coverage) — full pool")
        else:
            from autodex.utils.coverage import load_coverage_map
            _cov = load_coverage_map(
                obj, tabletop_pose_stem=pose_stem,
                hand=hand, version=args.grasp_version,
                success_root=candidate_state_root)
            if _cov is None:
                # No coverage json at all. Without this the run reads as
                # "0/0 candidates -> all scenes done" and stops, which looks
                # identical to a finished object. It is a missing precompute.
                sys.exit(
                    f"[coverage] no coverage json for {obj}/{args.grasp_version}.\n"
                    f"  expected: {project_dir}/experiment/{args.grasp_version}"
                    f"/coverage/cov_{args.grasp_version}_cand_{obj}.json\n"
                    f"  build it with:\n"
                    f"    python src/dataset/compute_v8_coverage.py --obj {obj} "
                    f"--hand {hand} --version {args.grasp_version}\n"
                    f"  (and make sure candidates/{hand}/{args.grasp_version}/{obj}/ "
                    f"is extracted from its .tar.gz first)")
            _n_session_excluded = sum(
                1 for key, value in _cov.items()
                if value > 0 and key in session_excluded)
            _useful = {
                key: value for key, value in _cov.items()
                if value > 0 and key not in session_excluded
            }
            _n_empty = sum(1 for value in _cov.values() if value == 0)
            _plan_candidate_order = sorted(_useful, key=lambda k: -_useful[k])
            _plan_priority_map = None
            _coverage_policy_rows = [
                {
                    "rank": rank,
                    "candidate_key": [str(part) for part in key],
                    "uncovered_scene_gain": int(_useful[key]),
                }
                for rank, key in enumerate(_plan_candidate_order)
            ]
            if trial_scope is not None:
                policy_artifact = trial_scope.write_artifact_json(
                    f"artifacts/coverage/{attempt_id or dir_idx}_grasp_policy.json",
                    {
                        "tabletop_pose": pose_stem,
                        "selection_rule": (
                            "descending uncovered-scene gain, then planner "
                            "collision/IK/approach/lift feasibility"),
                        "candidates": _coverage_policy_rows,
                        "dropped_zero_gain": _n_empty,
                        "excluded_attempted_this_session": (
                            _n_session_excluded),
                    },
                )
                trial_scope.event(
                    "coverage.grasp_policy_ranked", phase="coverage",
                    kind="decision", parent_id=episode_span,
                    artifact=policy_artifact,
                    tabletop_pose=pose_stem,
                    ranked_candidate_count=len(_coverage_policy_rows),
                    dropped_zero_gain=_n_empty,
                    excluded_attempted_this_session=_n_session_excluded,
                )
            # Dropped = remaining-uncovered == 0. Early on that is NOT
            # "already covered" — those are grasps whose `covers` list is
            # empty, i.e. collision-free in none of this tabletop's scenes.
            # Only once successes accumulate does the count mean progress.
            _drop_reasons = []
            if _n_empty:
                _drop_reasons.append(f"{_n_empty}: cover nothing left")
            if _n_session_excluded:
                _drop_reasons.append(
                    f"{_n_session_excluded}: attempted this session")
            print(f"    [coverage] {len(_useful)}/{len(_cov)} candidates open "
                  f"uncovered scenes"
                  + (f" (dropped {', '.join(_drop_reasons)})"
                     if _drop_reasons else ""))
        # Pre-plan reorient check (skipped under --ignore_coverage).
        from autodex.utils.coverage import uncovered_scenes, pick_reorient_target
        _rem = (None if args.ignore_coverage else
                uncovered_scenes(obj, pose_stem, hand=hand,
                                  version=args.grasp_version,
                                  success_root=candidate_state_root))
        if _rem is not None and len(_rem) == 0:
            target = pick_reorient_target(
                obj, pose_stem, hand=hand, version=args.grasp_version,
                obj_root=obj_root, success_root=candidate_state_root)
            print(f"\n    [reorient] tabletop {pose_stem} fully covered.")
            if target is None:
                done = _reorient_target_absent_result(
                    obj=obj, hand=hand, version=args.grasp_version,
                    obj_root=obj_root, dir_idx=dir_idx,
                    scene_type=args.scene, timing=timing,
                    success_root=candidate_state_root,
                )
                return _save_result(done)
            j_int, stem, n_rem = target
            print(f"    target_j={j_int} (pose {stem}) has {n_rem} uncovered scenes.")
            _reorient_cmd = (f"python src/experiment/reset/reorient.py "
                             f"--obj {obj} --hand {hand} --arm {args.arm} "
                             f"--target_j {j_int} --auto "
                             f"--version {args.grasp_version}")
            print(f"    Suggested:\n      {_reorient_cmd}")
            _prompt = ("    Press Enter to RUN integrated reorient now, "
                       if reorient_handler is not None else
                       "    Press Enter to RUN reorient now, ")
            _cmd = _prompt_or_auto(
                args, _prompt + "'s' to skip-and-continue (you ran it manually), "
                "'q' to quit: ")
            done = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "reorient_needed",
                    "reorient_target_j": j_int,
                    "reorient_target_stem": stem,
                    "reorient_uncovered_n": n_rem,
                    "timing": timing}
            if _cmd == "q":
                done["all_done"] = True
                done["reason"] = "user_quit_reorient"
            elif _cmd == "":
                if reorient_handler is None:
                    # Standalone run_auto remains a human-supervised handoff.
                    print(f"    [manual] auto-launch disabled — run it yourself:\n"
                          f"      {_reorient_cmd}")
                else:
                    print("    [pipeline] reorienting in the current process...")
                    reorient_info = _run_reorient_handler(j_int)
                    done["reorientation"] = reorient_info
                    if reorient_info.get("fatal_cuda_planning_fault"):
                        done["reason"] = "cuda_planning_fault"
                        done["fatal_cuda_planning_fault"] = True
                    elif reorient_info.get("success"):
                        done["reason"] = "reoriented"
                        done["retry_current_trial"] = True
            if _cmd == "s" and not args.auto:
                # 's' explicitly means that the operator already completed
                # the standalone reorientation.
                done["reason"] = "reoriented_manual"
                done["manual_reorient_confirmed"] = True
                done["retry_current_trial"] = True
            return _save_result(done)
    else:
        _eff_grasp_version = args.grasp_version
        _plan_scene_id = scene_id
        _plan_scene_type_filter = None
        _plan_tabletop_stem = None
        _plan_candidate_order = None
        _plan_priority_map = None
    # Reposition retries stats-ranked candidates.  An isolated campaign keeps
    # mutable records outside the geometry tree, so its coverage-derived
    # ``candidate_order`` is the only valid filter; letting load_candidate
    # inspect shared result.json files would leak v8 collection history.
    _skip_done_eff = False if (reposition_mode or args.ignore_coverage
                               or args.isolate_experiment) else True
    _skip_scenes_eff = False if (reposition_mode or args.ignore_coverage
                                 or args.isolate_experiment) else True
    timing["candidate_selection_s"] = round(
        time.perf_counter() - t_candidate_selection, 3)
    t_grasp_search = time.perf_counter()
    grasp_search_span = (trial_scope.begin(
        phase="planning", kind="plan", name="grasp_search",
        parent_id=episode_span, tabletop_pose=pose_stem,
        candidate_order_count=(len(_plan_candidate_order)
                               if _plan_candidate_order is not None else None))
        if trial_scope is not None else None)
    result = planner.plan(
        scene_cfg, obj, _eff_grasp_version,
        skip_done=_skip_done_eff,
        success_only=args.success_only, hand=hand,
        scene_id=_plan_scene_id,
        scene_type_filter=_plan_scene_type_filter,
        skip_scenes_with_success=_skip_scenes_eff,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=_cyl_axis,
        cyl_yaw_grid=_cyl_grid,
        tabletop_pose_stem=_plan_tabletop_stem,
        candidate_order=_plan_candidate_order,
        excluded_candidates=session_excluded,
        priority_map=_plan_priority_map,
    )
    timing["grasp_search_s"] = round(time.perf_counter() - t_grasp_search, 3)
    timing["grasp_search_detail"] = dict(result.timing or {})
    timing["planning_total_s"] = round(time.perf_counter() - t_planning, 3)
    timing["plan_s"] = timing["planning_total_s"]
    if grasp_search_span is not None:
        trial_scope.end(
            grasp_search_span,
            outcome=("success" if result.success else "failure"),
            scene_info=result.scene_info,
            diagnostics=result.timing,
        )
    print(f"    Plan: {timing['plan_s']}s  success={result.success}")

    if not result.success:
        # n_total == 0 means load_candidate returned empty — every scene is
        # already done OR no grasp candidate matches the current tabletop.
        # Either way we need to switch to a different tabletop pose.
        n_total = (result.timing or {}).get("n_total", -1)
        if n_total == 0:
            print(f"    No grasp candidates left at tabletop {pose_stem} "
                  f"for {obj} ({args.grasp_version}).")
            if (_is_coverage_pool(args.grasp_version)
                    and not args.ignore_coverage):
                from autodex.utils.coverage import pick_reorient_target
                target = pick_reorient_target(
                    obj, pose_stem, hand=hand, version=args.grasp_version,
                    obj_root=obj_root, success_root=candidate_state_root)
                if target is None:
                    done = _reorient_target_absent_result(
                        obj=obj, hand=hand, version=args.grasp_version,
                        obj_root=obj_root, dir_idx=dir_idx,
                        scene_type=args.scene, timing=timing,
                        success_root=candidate_state_root,
                    )
                    return _save_result(done)
                j_int, stem, n_rem = target
                print(f"    [reorient] target_j={j_int} (pose {stem}) "
                      f"has {n_rem} uncovered scenes.")
                print(f"    Suggested:")
                print(f"      python src/experiment/reset/reorient.py "
                      f"--obj {obj} --hand {hand} --arm {args.arm} "
                      f"--target_j {j_int} --auto "
                      f"--version {args.grasp_version}")
                _prompt = ("    Press Enter to RUN integrated reorient "
                           "(q to quit): " if reorient_handler is not None
                           else "    Run reorient then press Enter (q to quit): ")
                _cmd = _prompt_or_auto(args, _prompt)
                done = {"dir_idx": dir_idx, "scene_type": args.scene,
                        "success": None, "reason": "reorient_needed",
                        "reorient_target_j": j_int,
                        "reorient_target_stem": stem,
                        "reorient_uncovered_n": n_rem,
                        "timing": timing}
                if _cmd == "q":
                    done["all_done"] = True
                    done["reason"] = "user_quit_reorient"
                elif reorient_handler is not None:
                    print("    [pipeline] reorienting in the current process...")
                    reorient_info = _run_reorient_handler(j_int)
                    done["reorientation"] = reorient_info
                    if reorient_info.get("fatal_cuda_planning_fault"):
                        done["reason"] = "cuda_planning_fault"
                        done["fatal_cuda_planning_fault"] = True
                    elif reorient_info.get("success"):
                        done["reason"] = "reoriented"
                        done["retry_current_trial"] = True
                elif _cmd == "" and not args.auto:
                    # This no-handler prompt asks the operator to run the
                    # reset before pressing Enter, so Enter is confirmation.
                    done["reason"] = "reoriented_manual"
                    done["manual_reorient_confirmed"] = True
                    done["retry_current_trial"] = True
                return _save_result(done)
            if args.ignore_coverage:
                print("    No candidates available in the unrestricted pool.")
            else:
                print("    All scenes already done — nothing left to try.")
            done = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None,
                    "reason": ("no_candidates" if args.ignore_coverage
                               else "all_scenes_done"),
                    "all_done": True, "timing": timing}
            return _save_result(done)
        # cuRobo IK is stochastic — random seeds occasionally find 0
        # feasible at an obj pose where the next run finds several. Before
        # declaring "reorient needed", retry the plan up to 2 more times.
        for _retry in range(1, 3):
            print(f"    Planning failed (attempt {_retry}/2 retry)...")
            retry_span = (trial_scope.begin(
                phase="planning", kind="plan", name="grasp_search_retry",
                parent_id=episode_span, retry_index=_retry)
                if trial_scope is not None else None)
            t_re = time.perf_counter()
            result = planner.plan(
                scene_cfg, obj, _eff_grasp_version,
                skip_done=_skip_done_eff,
                success_only=args.success_only, hand=hand,
                scene_id=_plan_scene_id,
                scene_type_filter=_plan_scene_type_filter,
                skip_scenes_with_success=_skip_scenes_eff,
                openpose_pose_stem=pose_stem,
                cyl_axis_local=_cyl_axis,
                cyl_yaw_grid=_cyl_grid,
                tabletop_pose_stem=_plan_tabletop_stem,
                candidate_order=_plan_candidate_order,
                excluded_candidates=session_excluded,
                priority_map=_plan_priority_map,
            )
            timing[f"plan_retry_{_retry}_s"] = round(
                time.perf_counter() - t_re, 3)
            if retry_span is not None:
                trial_scope.end(
                    retry_span,
                    outcome=("success" if result.success else "failure"),
                    scene_info=result.scene_info,
                    diagnostics=result.timing,
                )
            if result.success:
                print(f"    Plan retry #{_retry} success after {timing[f'plan_retry_{_retry}_s']}s")
                break
        timing["plan_retry_total_s"] = round(sum(
            float(timing.get(f"plan_retry_{i}_s", 0.0))
            for i in range(1, 3)), 3)
        timing["planning_total_s"] = round(
            float(timing.get("planning_total_s", 0.0))
            + timing["plan_retry_total_s"], 3)
        if result.success:
            timing["plan_s"] = timing["planning_total_s"]
            # The recovery body below is only for a still-infeasible plan.
            # It used to run even on this successful retry, which could launch
            # a physical rotate/reorient despite already having a valid plan.
            # The normal execution body cannot safely reuse a retry plan after
            # the next perception snapshot, so restart from perception without
            # moving the robot or changing the rotate streak.
            retry = {
                "dir_idx": dir_idx,
                "scene_type": args.scene,
                "success": None,
                "reason": "planning_retry_feasible",
                "retry_current_trial": True,
                "tabletop_before": tb_before,
                "timing": timing,
            }
            return _save_result(retry)
        else:
            if args.auto:
                print("    Planning FAILED after retries — automatic recovery; "
                      "skipping visualizer.")
            else:
                print("    Planning FAILED after retries — launching visualizer to inspect...")
        # Match planner.plan()'s actual candidate pool: skip_done=True and
        # skip_scenes_with_success=True so the viewer shows only what was
        # actually attempted (not the full disk pool).
        wrist_se3, _, grasp_pose, filtered, ik_failed = planner.get_candidates(
            scene_cfg, obj, _eff_grasp_version,
            success_only=args.success_only,
            skip_done=_skip_done_eff, hand=hand, run_ik=True,
            scene_id=_plan_scene_id,
            scene_type_filter=_plan_scene_type_filter,
            tabletop_pose_stem=_plan_tabletop_stem,
            candidate_order=_plan_candidate_order,
            excluded_candidates=session_excluded,
            cyl_axis_local=_cyl_axis,
            cyl_yaw_grid=_cyl_grid,
            skip_scenes_with_success=_skip_scenes_eff,
        )
        if not args.auto:
            fv = ScenePlanVisualizer(scene_cfg, None, port=8080,
                                     hand=_planner_robot(args.arm, hand))
            fv.add_candidates(wrist_se3, grasp_pose, filtered, ik_failed=ik_failed)
            fv.start_viewer(use_thread=True)
            _active_vis = fv
        chime.error()
        # Before bouncing to reorient, try an (x, yaw) sweep at the
        # CURRENT tabletop — same obj orientation, just translate /
        # rotate around vertical. If any (r, yaw) makes ≥1 candidate
        # IK-feasible, prefer that (rotate_obj_yaw) over reorienting
        # to a different tabletop.
        _rotate_count = _rotate_streak_count(rotate_recovery_state)
        _rotate_limit = int(args.max_consecutive_rotates)
        _rotate_allowed = _rotate_count < _rotate_limit
        _rotate_cap_info = {
            "tabletop_stem": pose_stem,
            "count": _rotate_count,
            "limit": _rotate_limit,
            "sequence_start_tabletop_stem": (
                rotate_recovery_state.get("sequence_start_tabletop_stem")
                if rotate_recovery_state is not None else None),
            "last_rotate_tabletop_stem": (
                rotate_recovery_state.get("last_rotate_tabletop_stem")
                if rotate_recovery_state is not None else None),
        }
        if (_is_coverage_pool(args.grasp_version)
                and not args.ignore_coverage and not _rotate_allowed):
            print(f"    [rotate] consecutive cap reached "
                  f"({_rotate_count}/{_rotate_limit}; current tabletop "
                  f"{pose_stem}); "
                  "skipping rotate and proceeding to reorient.")
        if (_rotate_allowed and _is_coverage_pool(args.grasp_version)
                and not args.ignore_coverage):
            _ros_yaw = None
            _ros_x = None
            _target_x = None
            _target_y = None
            _pose_grid_rows = []
            _pose_search_error = None
            try:
                from autodex.utils.conversion import cart2se3 as _cart2se3
                T_obj_now = _cart2se3(scene_cfg["mesh"]["target"]["pose"])
                obj_z_now = float(T_obj_now[2, 3])
                R_obj = T_obj_now[:3, :3]
                # wrist_se3 already in WORLD frame (transformed by obj pose
                # via load_candidate). Recover obj-local wrist = inv(T_obj_now) @ world.
                _wrist_obj_local = np.einsum(
                    "ij,Njk->Nik", np.linalg.inv(T_obj_now), wrist_se3)
                # Prefer the physical centre of board 11.  Use the same
                # symmetric +/-5 cm and +/-10 cm X fallbacks as reorient when
                # the centre cannot make a useful grasp IK-feasible.
                _target_x = float(CHARUCO_BOARD_11_CENTER_XY[0])
                _target_y = float(CHARUCO_BOARD_11_CENTER_XY[1])
                if tabletop_geometry is not None:
                    _board_center = np.asarray(
                        tabletop_geometry["center_robot_m"], dtype=np.float64)
                    _target_x = float(_board_center[0])
                    _target_y = float(_board_center[1])
                _xs = _target_x + CHARUCO_BOARD_CENTER_X_OFFSETS_M
                _yaws = np.deg2rad(np.arange(0, 360, 30))
                _combos = [(float(x), float(y)) for x in _xs for y in _yaws]
                # For each (x, yaw), build all wrists and IK batch-check.
                # Cap candidates to first 32 to keep grid fast.
                _cand_cap = min(32, len(_wrist_obj_local))
                _wlocal = _wrist_obj_local[:_cand_cap]
                _best_pick = None   # (n_ok, x, yaw)
                for _x, _yaw in _combos:
                    c, s = np.cos(_yaw), np.sin(_yaw)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    T_new = np.eye(4)
                    T_new[:3, :3] = Rz @ R_obj
                    T_new[:3, 3] = [_x, _target_y, obj_z_now]
                    _world_wrists = T_new[None] @ _wlocal
                    _succ = planner.ik_pose_batch(_world_wrists)
                    _n = int(_succ.sum())
                    _pose_grid_rows.append({
                        "x_m": _x,
                        "y_m": _target_y,
                        "yaw_deg": float(np.degrees(_yaw)),
                        "ik_feasible_count": _n,
                        "candidate_count": _cand_cap,
                    })
                    if (_best_pick is None
                            or _n > _best_pick[0]
                            or (_n == _best_pick[0]
                                and abs(_x - _target_x)
                                < abs(_best_pick[1] - _target_x))):
                        _best_pick = (_n, _x, float(np.degrees(_yaw)))
                if _best_pick is not None and _best_pick[0] > 0:
                    _, _ros_x, _ros_yaw = _best_pick
                    print(f"\n    [pose_search] move obj to "
                          f"(x={_ros_x:.2f}, y={_target_y:.3f}) "
                          f"+ rotate {_ros_yaw:.0f}° → {_best_pick[0]}/{_cand_cap} "
                          f"v8 candidates IK-feasible at CURRENT tabletop")
            except Exception as _se:
                _pose_search_error = repr(_se)
                print(f"    [pose_search] failed: {_se!r}")
            if trial_scope is not None:
                grid_artifact = trial_scope.write_artifact_json(
                    f"artifacts/recovery/{attempt_id or dir_idx}_pose_grid.json",
                    {
                        "tabletop_pose": pose_stem,
                        "grid": _pose_grid_rows,
                        "selected": (
                            None if _ros_yaw is None else
                            {"x_m": _ros_x, "y_m": _target_y,
                             "yaw_deg": _ros_yaw}),
                        "exception": _pose_search_error,
                    },
                )
                trial_scope.event(
                    "recovery.rotation_pose_grid", phase="recovery",
                    kind="decision",
                    outcome=("success" if _ros_yaw is not None else "failure"),
                    artifact=grid_artifact,
                    selected=(
                        None if _ros_yaw is None else
                        {"x_m": _ros_x, "y_m": _target_y,
                         "yaw_deg": _ros_yaw}),
                )
            if _ros_yaw is not None:
                # --arm MUST be forwarded: rotate_obj_yaw defaults to xarm,
                # so without it an FR3 run launches XArmController and dies on
                # "connect socket failed" (there is no xarm on this setup).
                _cmd_ros = (f"python src/execution/rotate_obj_yaw.py "
                             f"--obj {obj} --hand {hand} --arm {args.arm} "
                             f"--target_yaw_deg {_ros_yaw:.0f} "
                             f"--target_x {_ros_x:.2f} "
                             f"--target_y {_target_y:.3f} "
                             f"--grasp_version {args.grasp_version}")
                print(f"    Suggested (same-tabletop, no reorient):\n      {_cmd_ros}")
                _cmd = _prompt_or_auto(
                    args, "    Press Enter to RUN rotate_obj_yaw now, "
                    "'s' to skip-and-continue, 'q' to quit: ")
                fail_record = {"dir_idx": dir_idx, "scene_type": args.scene,
                               "success": False,
                               "reason": "pose_adjust_needed",
                               "pose_adjust": {"x": _ros_x, "y": _target_y,
                                               "yaw_deg": _ros_yaw},
                               "rotate_recovery": _rotate_cap_info,
                               "timing": timing}
                if _cmd == "q":
                    fail_record["all_done"] = True
                    fail_record["reason"] = "user_quit_pose_adjust"
                elif _cmd == "":
                    if pose_adjust_handler is None:
                        import subprocess
                        print(f"    [auto] running: {_cmd_ros}")
                        rc = subprocess.call(_cmd_ros, shell=True)
                        fail_record["rotate_obj_yaw_rc"] = rc
                        rotate_info = {
                            "success": rc == 0,
                            "reason": (None if rc == 0
                                       else "rotate_obj_yaw_nonzero_exit"),
                            "returncode": rc,
                        }
                    else:
                        # ``run_pipeline.py`` supplies an in-process handler.
                        # It receives the perception result and already-live
                        # hardware/planner objects, avoiding a second camera
                        # registration, FoundPose init, CUDA warmup, and robot
                        # connection merely to rotate the object.
                        print("    [pipeline] rotating in the current process...")
                        try:
                            rotate_info = pose_adjust_handler(
                                args=args, obj=obj, hand=hand,
                                planner=planner, executor=executor, rcc=rcc,
                                scene_cfg=scene_cfg,
                                target_x=_ros_x, target_y=_target_y,
                                target_yaw_deg=_ros_yaw,
                                grasp_version=_eff_grasp_version,
                                tabletop_pose_stem=_plan_tabletop_stem,
                                candidate_order=_plan_candidate_order,
                                excluded_candidates=session_excluded,
                                priority_map=_plan_priority_map,
                                scene_type_filter=_plan_scene_type_filter,
                                scene_id=_plan_scene_id,
                                success_only=args.success_only,
                                skip_done=_skip_done_eff,
                                skip_scenes_with_success=_skip_scenes_eff,
                                cyl_axis_local=_cyl_axis,
                                cyl_yaw_grid=_cyl_grid,
                                tabletop_geometry=tabletop_geometry,
                                pipeline_trace=trial_scope,
                            )
                        except Exception as rotate_exc:
                            rotate_info = {
                                "success": False,
                                "reason": "inprocess_rotate_exception",
                                "exception": repr(rotate_exc),
                            }
                            print(f"    [pipeline] rotate exception: {rotate_exc!r}")
                    fail_record["rotation"] = rotate_info
                    rotate_scene_info = rotate_info.get("scene_info")
                    if rotate_scene_info is None:
                        rotate_scene_info = getattr(
                            rotate_info.get("result"), "scene_info", None)
                    if (session_attempted_candidates is not None
                            and isinstance(rotate_scene_info, (list, tuple))
                            and len(rotate_scene_info) == 3):
                        rotate_key = tuple(
                            str(part) for part in rotate_scene_info)
                        session_attempted_candidates.add(rotate_key)
                        print("    [session] rotation candidate attempted; "
                              f"excluding from later trials: {rotate_key}")
                    if rotate_info.get("success"):
                        _rotate_count = _record_successful_rotate(
                            rotate_recovery_state, tabletop_stem=pose_stem)
                        rotate_info["consecutive_rotate_count"] = _rotate_count
                        rotate_info["consecutive_rotate_limit"] = _rotate_limit
                        fail_record["rotate_recovery"] = {
                            "tabletop_stem": pose_stem,
                            "count": _rotate_count,
                            "limit": _rotate_limit,
                            "sequence_start_tabletop_stem": (
                                rotate_recovery_state.get(
                                    "sequence_start_tabletop_stem")
                                if rotate_recovery_state is not None else pose_stem),
                        }
                        print(f"    [rotate] consecutive recovery "
                              f"{_rotate_count}/{_rotate_limit} "
                              f"(current tabletop {pose_stem})")
                        fail_record["reason"] = "pose_adjusted"
                        fail_record["retry_current_trial"] = True
                return _save_result(fail_record)
        # Rotate is only the first recovery option.  Reorient must be
        # considered independently: it is the required next step when the
        # rotate streak reaches its cap, and it remains a fallback when the
        # pose sweep has no useful IK solution or a rotate action fails.
        if (_is_coverage_pool(args.grasp_version)
                and not args.ignore_coverage):
            from autodex.utils.coverage import pick_reorient_target
            target = pick_reorient_target(
                obj, pose_stem, hand=hand, version=args.grasp_version,
                obj_root=obj_root, success_root=candidate_state_root)
            if target is not None:
                j_int, stem, n_rem = target
                print(f"\n    [reorient] all candidates at tabletop {pose_stem} "
                      f"failed (IK/collision).")
                print(f"    target_j={j_int} (pose {stem}) has {n_rem} "
                      f"uncovered scenes.")
                print(f"    Suggested:")
                print(f"      python src/experiment/reset/reorient.py "
                      f"--obj {obj} --hand {hand} --arm {args.arm} "
                      f"--target_j {j_int} --auto "
                      f"--version {args.grasp_version}")
                _prompt = ("    Press Enter to RUN integrated reorient "
                           "(q to quit): " if reorient_handler is not None
                           else "    Run reorient then press Enter (q to quit): ")
                _cmd = _prompt_or_auto(args, _prompt)
                fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                        "success": False, "reason": "reorient_needed_ik_fail",
                        "reorient_target_j": j_int,
                        "reorient_target_stem": stem,
                        "reorient_uncovered_n": n_rem,
                        "timing": timing}
                if not _rotate_allowed:
                    fail["rotate_recovery"] = _rotate_cap_info
                if _cmd == "q":
                    fail["all_done"] = True
                    fail["reason"] = "user_quit_reorient"
                elif reorient_handler is not None:
                    print("    [pipeline] reorienting in the current process...")
                    reorient_info = _run_reorient_handler(j_int)
                    fail["reorientation"] = reorient_info
                    if reorient_info.get("fatal_cuda_planning_fault"):
                        fail["reason"] = "cuda_planning_fault"
                        fail["fatal_cuda_planning_fault"] = True
                    elif reorient_info.get("success"):
                        fail["reason"] = "reoriented"
                        fail["retry_current_trial"] = True
                elif _cmd == "" and not args.auto:
                    # As above, the no-handler prompt is an operator-confirmed
                    # external reorientation, which ends the rotate streak.
                    fail["reason"] = "reoriented_manual"
                    fail["manual_reorient_confirmed"] = True
                    fail["retry_current_trial"] = True
                return _save_result(fail)
            # The failed tabletop still has no strictly mapped legacy reset
            # transition to any remaining uncovered tabletop.  Report the
            # missing reset data rather than labelling the object complete.
            absent = _reorient_target_absent_result(
                obj=obj, hand=hand, version=args.grasp_version,
                obj_root=obj_root, dir_idx=dir_idx,
                scene_type=args.scene, timing=timing,
                success_root=candidate_state_root,
            )
            if not _rotate_allowed:
                absent["rotate_recovery"] = _rotate_cap_info
            return _save_result(absent)
        # No recovery route remains. Stop the cycle.
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": "planning_failed_all_candidates",
                "all_done": True, "timing": timing}
        if not _rotate_allowed:
            fail["rotate_recovery"] = _rotate_cap_info
        return _save_result(fail)

    if trial_scope is not None:
        trial_scope.event(
            "grasp.selected", phase="planning", kind="decision",
            parent_id=episode_span, scene_info=result.scene_info,
            candidate_index=(result.timing or {}).get("candidate_idx"),
            lift_preflight="passed",
        )
    t_plan_persistence = time.perf_counter()
    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    # Keep the exact post-symmetry hand configuration used by the robot.  The
    # source candidate alone is insufficient for reconstructing a run because
    # cylindrical symmetry expansion can change the selected wrist transform.
    np.save(os.path.join(plan_dir, "grasp_pose.npy"), result.grasp_pose)
    np.save(os.path.join(plan_dir, "pregrasp_pose.npy"), result.pregrasp_pose)
    from autodex.utils.conversion import cart2se3 as _plan_cart2se3
    _selected_obj_world = _plan_cart2se3(
        scene_cfg["mesh"]["target"]["pose"])
    np.save(
        os.path.join(plan_dir, "wrist_obj_local.npy"),
        np.linalg.inv(_selected_obj_world) @ result.wrist_se3,
    )
    if getattr(result, "lift_preflight", None) is not None:
        _lift_pf = result.lift_preflight
        np.savez(
            os.path.join(plan_dir, "lift_preflight.npz"),
            traj=np.asarray(_lift_pf.traj),
            start_full_qpos=np.asarray(_lift_pf.start_full_qpos),
            start_wrist_se3=np.asarray(_lift_pf.start_wrist_se3),
            target_wrist_se3=np.asarray(_lift_pf.target_wrist_se3),
            height_m=np.asarray(_lift_pf.height_m),
        )
    # Planner timing and candidate decisions live only in the run-level
    # timeline.  Do not create a second episode-local timing file.
    timing["plan_persistence_s"] = round(
        time.perf_counter() - t_plan_persistence, 3)
    print(f"    Scene info: {result.scene_info}")

    # Precomputed trajectories (shared with viz so what user sees == what
    # robot executes). Set inside the viz block below if --viz.
    _precomputed_lift_traj = None
    _precomputed_repo_traj = None

    if args.viz and not args.auto:
        print("    Launching visualizer (http://localhost:8080)...")
        sv = ScenePlanVisualizer(scene_cfg, result, port=8080,
                                 hand=_planner_robot(args.arm, hand))
        # Pre-compute lift / repose / place trajectories so viz shows the
        # entire planned motion before execution starts. Obj follows the
        # wrist rigidly through these phases.
        try:
            from autodex.utils.conversion import cart2se3 as _cart2se3
            import torch as _torch
            from scipy.spatial.transform import Rotation as _R
            T_obj_grasp_world = _cart2se3(scene_cfg["mesh"]["target"]["pose"])

            def _fk_wrist(qpos: np.ndarray) -> np.ndarray:
                """Compute WRIST 4×4 pose (= cuRobo ee_link = base_link)."""
                kin = planner._motion_gen.kinematics.get_state(
                    _torch.tensor(qpos, dtype=_torch.float32,
                                  device=planner._tensor_args.device).unsqueeze(0)
                )
                pos = kin.ee_position[0].detach().cpu().numpy()
                quat = kin.ee_quaternion[0].detach().cpu().numpy()
                Rmat = _R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = Rmat
                T[:3, 3] = pos
                return T

            grasp_end_qpos = np.asarray(result.traj[-1], dtype=np.float32)
            grasp_end_arm = grasp_end_qpos[:adof]
            T_wrist_grasp_end = _fk_wrist(grasp_end_qpos)
            # Use FK-derived wrist so obj viz at grasp end exactly matches
            # scene_cfg obj pose (no jump at lift start).
            T_obj_in_wrist = np.linalg.inv(T_wrist_grasp_end) @ T_obj_grasp_world

            def _obj_traj_along(robot_traj: np.ndarray) -> np.ndarray:
                """Compute (N, 4, 4) obj poses along trajectory (rigid grasp)."""
                out = np.zeros((len(robot_traj), 4, 4))
                for i, q in enumerate(robot_traj):
                    T_wrist = _fk_wrist(np.asarray(q, dtype=np.float32))
                    out[i] = T_wrist @ T_obj_in_wrist
                return out

            # 1. Lift trajectory — use the candidate preflight selected by
            # planner.plan(), so the visualized path is exactly the one that
            # passed candidate selection and will be replayed if live q agrees.
            preflight = getattr(result, "lift_preflight", None)
            if preflight is not None:
                lift_traj = np.asarray(preflight.traj)
            else:
                # Compatibility for callers that constructed PlanResult
                # manually. Main pipeline results always carry a preflight.
                lift_wrist = T_wrist_grasp_end.copy()
                lift_wrist[2, 3] += 0.10
                grasp_full = np.concatenate([
                    grasp_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                ])
                fallback_preflight = planner.plan_lift_preflight(
                    grasp_full, scene_cfg, lift_h=0.10)
                lift_traj = (None if fallback_preflight is None
                             else fallback_preflight.traj)
            if lift_traj is not None:
                _precomputed_lift_traj = lift_traj
                lift_obj_traj = _obj_traj_along(lift_traj)
                sv.add_traj("lift", {"traj_robot": lift_traj},
                            obj_traj={"mesh_target": lift_obj_traj})

                # 2. Repose trajectory for the v8 recovery visualisation.
                if _is_coverage_pool(args.grasp_version) and result.scene_info is not None:
                    lift_end_qpos = np.asarray(lift_traj[-1], dtype=np.float32)
                    lift_end_arm = lift_end_qpos[:adof]
                    T_wrist_lift_end = _fk_wrist(lift_end_qpos)
                    T_obj_lift_end = T_wrist_lift_end @ T_obj_in_wrist
                    R_PLACE_VIZ = 0.55
                    obj_z_now = float(T_obj_lift_end[2, 3])
                    R_obj_canonical = T_obj_grasp_world[:3, :3]
                    T_obj_repo = np.eye(4)
                    T_obj_repo[:3, :3] = R_obj_canonical
                    T_obj_repo[:3, 3] = [R_PLACE_VIZ, 0.0, obj_z_now]
                    T_wrist_repo = T_obj_repo @ np.linalg.inv(T_obj_in_wrist)
                    T_wrist_repo[2, 3] = T_wrist_lift_end[2, 3]  # force wrist z match
                    lift_full = np.concatenate([
                        lift_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                    ])
                    repo_traj = planner.plan_pose_constrained(
                        lift_full, T_wrist_repo,
                        hold_vec_weight=[0, 0, 0, 0, 0, 1],
                        scene_cfg=scene_cfg, include_obj_obstacle=False,
                    )
                    if repo_traj is not None:
                        _precomputed_repo_traj = repo_traj
                        repo_obj_traj = _obj_traj_along(repo_traj)
                        sv.add_traj("repose", {"traj_robot": repo_traj},
                                    obj_traj={"mesh_target": repo_obj_traj})

                        # 3. Place traj — wrist z descend
                        repo_end_qpos = np.asarray(repo_traj[-1], dtype=np.float32)
                        repo_end_arm = repo_end_qpos[:adof]
                        T_wrist_repo_end = _fk_wrist(repo_end_qpos)
                        place_wrist = T_wrist_repo_end.copy()
                        place_wrist[2, 3] -= 0.10
                        repo_full = np.concatenate([
                            repo_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                        ])
                        place_traj = planner.plan_vertical_stroke(
                            repo_full, T_wrist_repo_end, place_wrist,
                            expected_travel_m=0.10,
                            travel_tolerance_m=1.0e-4,
                            scene_cfg=scene_cfg, include_obj_obstacle=False,
                            label="visualized place descent",
                            attached_object_pose_at_start=(
                                T_wrist_repo_end @ T_obj_in_wrist),
                        )
                        if place_traj is not None:
                            place_obj_traj = _obj_traj_along(place_traj)
                            sv.add_traj("place", {"traj_robot": place_traj},
                                        obj_traj={"mesh_target": place_obj_traj})
        except CudaPlanningFault:
            # A failed CUDA kernel poisons the process; never continue from a
            # visualization-only planning branch into physical execution.
            raise
        except Exception as _viz_e:
            print(f"    [viz] phase precompute failed: {_viz_e!r}")
        sv.start_viewer(use_thread=True)
        _active_vis = sv
    elif args.viz:
        print("    [viz] --auto: visualizer disabled")

    # ── 4. Execute (stream off, video on) ───────────────────────────────────
    if (session_attempted_candidates is not None
            and isinstance(result.scene_info, (list, tuple))
            and len(result.scene_info) == 3):
        selected_key = tuple(str(part) for part in result.scene_info)
        session_attempted_candidates.add(selected_key)
        timing["session_candidate_key"] = list(selected_key)
        print("    [session] candidate marked attempted; excluding from later "
              f"trials: {selected_key}")
    print(f"[4/6] Executing on robot...")
    timing["execution_start"] = _ts()
    t_capture_setup = time.perf_counter()
    # Pause the always-on stream so video can take its place.
    try:
        rcc.stop()
    except Exception:
        pass

    raw_rel = os.path.join("AutoDex", "experiment", args.exp_name, sub, obj, dir_idx, "raw")
    exec_rel = os.path.join(raw_rel, "exec")
    place_rel = os.path.join(raw_rel, "place")
    raw_dir = os.path.join(img_dir, "raw")
    _rcc_start(rcc, "video", True, exec_rel)
    _safe_timestamp_start(timestamp_monitor, os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    _start_execution_trigger(fps=30)
    timing["execution_capture_setup_s"] = round(
        time.perf_counter() - t_capture_setup, 3)

    t0 = time.perf_counter()
    DEBUG_DUMP_DIR = "/tmp/pose_constrained_debug"
    t_grasp_lift = time.perf_counter()
    try:
        s_hand = executor.execute(
            result, planner=planner, scene_cfg=scene_cfg,
            debug_dump_dir=DEBUG_DUMP_DIR,
            lift_traj_override=_precomputed_lift_traj,
        )   # grasp + lift; lift uses precomputed traj if viz on
    except Exception as _exec_e:
        # A lift failure after squeeze is special: the object is still held.
        # Do not open the hand or issue a lateral reset from this state.
        # Ordinary approach/SDK failures retain the established recovery path.
        _cuda_planning_fault = isinstance(_exec_e, CudaPlanningFault)
        _held_lift_failure = (isinstance(_exec_e, LiftExecutionError)
                              or _cuda_planning_fault)
        print(f"    [execute FAIL] {type(_exec_e).__name__}: {_exec_e!r}")
        timing["execute_s"] = round(time.perf_counter() - t0, 3)
        timing["execute_grasp_lift_s"] = timing["execute_s"]
        _executor_exec_timing = dict(
            getattr(executor, "last_execute_timing", {}) or {})
        if _executor_exec_timing:
            _executor_exec_timing.setdefault(
                "total_s", timing["execute_grasp_lift_s"])
            timing["execute_grasp_lift_detail_s"] = _executor_exec_timing
        if _cuda_planning_fault:
            print("    [recovery] CUDA planner context is invalid; issuing no "
                  "further planned robot motion. Manual state check required")
            timing["fatal_cuda_planning_fault"] = True
            timing["held_object_safety_stop"] = True
        elif _held_lift_failure:
            print("    [recovery] lift could not be validated from live q — "
                  "keeping squeeze and arm position; manual recovery required")
            timing["held_object_safety_stop"] = True
        else:
            try:
                print(f"    [recovery] reset_fallback (verified lift + retract) ...")
                executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
            except Exception as _re:
                print(f"    [recovery] reset_fallback FAILED: {_re!r}")
        _stop_with_timeout("executor.recording", executor.stop_recording)
        # timestamp_monitor FIRST: its stop() waits on event["stop"], which the
        # capture loop only sets after camera.get_timestamp() returns — and that
        # blocks for the next frame. Kill the trigger first and no frame ever
        # arrives, so stop() waits forever. (Every other call site in this file
        # already stops it in this order.)
        _safe_timestamp_stop(timestamp_monitor)
        _stop_execution_trigger("execute_exception")
        # NOTE: rcc.stop() pauses the current capture (record / stream).
        # Do NOT call rcc.end() here — that tears down the remote camera
        # controller, which the next trial still needs.
        _stop_with_timeout("rcc", rcc.stop)
        # Persist the outcome for audit/resume. The process-local attempted set
        # above, not this failure record, prevents reuse in this session.
        if result.scene_info is not None:
            try:
                _write_candidate_outcome(
                    args, hand, obj, result.scene_info,
                    {"success": False, "dir_idx": dir_idx,
                     "arm": args.arm,
                     "reason": f"execute_{type(_exec_e).__name__}"})
            except Exception:
                pass
        fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                "success": False,
                "reason": ("cuda_planning_fault" if _cuda_planning_fault
                           else ("lift_live_replan_failed_hold"
                                 if _held_lift_failure else "execute_exception")),
                "exception": repr(_exec_e), "scene_info": result.scene_info,
                "manual_recovery_required": _held_lift_failure,
                "fatal_cuda_planning_fault": _cuda_planning_fault,
                "candidate_result_scope": (
                    "experiment" if args.isolate_experiment else "shared_v8"),
                "timing": timing}
        if not reposition_mode:
            fail = _with_task_outcome(
                fail,
                False,
                grasp_evidence={"source": "execute_exception"},
            )
        try:
            return _save_result(fail)
        except Exception:
            return _stamp_end(fail)

    # Keep this separate from the later label/transfer/place work.  ``execute_s``
    # remains the historical end-to-end field for compatibility.
    timing["execute_grasp_lift_s"] = round(
        time.perf_counter() - t_grasp_lift, 3)
    _executor_exec_timing = dict(getattr(executor, "last_execute_timing", {}) or {})
    if _executor_exec_timing:
        timing["execute_grasp_lift_detail_s"] = _executor_exec_timing

    # --auto: lift-time charuco snapshot — object is up, marker on the table
    # should be uncovered. Pause video, capture image set, resume video.
    auto_succ_lift = None
    auto_label_info = {}
    if args.auto:
        t_auto_label = time.perf_counter()
        _stop_with_timeout("rcc", rcc.stop)
        label_lift_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                       args.exp_name, sub, obj, dir_idx,
                                       "label_at_lift", "raw")
        label_lift_abs = os.path.join(img_dir, "label_at_lift", "raw", "images")
        rcc.start("image", False, label_lift_rel)
        # The image request is the first actual grasp-validation operation:
        # execute() has completed grasp + lift and capture is now underway.
        # Reset even if the capture is later unjudgeable or judged as a fail;
        # this recovery sequence did reach validation.
        _mark_grasp_validation_started("auto_lift_validation_started")
        rcc.stop()
        time.sleep(0.3)
        timing["auto_label_lift_capture_s"] = round(
            time.perf_counter() - t_auto_label, 3)
        t_auto_label_infer = time.perf_counter()
        auto_succ_lift, auto_label_info = auto_label_charuco(
            label_lift_abs, required_board=CHARUCO_BOARD)
        timing["auto_label_lift_infer_s"] = round(
            time.perf_counter() - t_auto_label_infer, 3)
        timing["auto_label_lift_s"] = round(
            time.perf_counter() - t_auto_label, 3)
        if trial_scope is not None:
            _validation_outcome = (
                "success" if auto_succ_lift is True else
                "failure" if auto_succ_lift is False else "unjudgeable"
            )
            trial_scope.event(
                "grasp.validation_result", phase="validation", kind="result",
                parent_id=episode_span, outcome=_validation_outcome,
                scene_info=result.scene_info, success=auto_succ_lift,
                classification=_validation_outcome,
                reason=auto_label_info.get("reason"),
                covered=auto_label_info.get("covered"),
                expected=auto_label_info.get("expected"),
                source="auto_label_charuco",
            )
        if auto_label_info.get("reason"):
            print(f"    [auto-label] FAILED ({auto_label_info['reason']})")
        else:
            print(f"    [auto-label] success={auto_succ_lift}  "
                  f"covered {auto_label_info.get('covered')}/"
                  f"{auto_label_info.get('expected')}")

        # Charuco fail → don't place, recover via reset_hybrid (self-collision
        # + placed-obj collision aware). This session already excludes the
        # physically attempted grasp; the result file records the outcome.
        # None = the label could not be JUDGED (no images captured), which is
        # not the robot failing. Recording it as a candidate failure would
        # permanently label a grasp that may well have worked, so the trial is
        # voided instead: recover the arm, write nothing to the candidate dir.
        # It still stays excluded for the remainder of this live session.
        _label_unjudgeable = auto_succ_lift is None
        if not auto_succ_lift:
            if _label_unjudgeable:
                print(f"    [auto-label] UNJUDGEABLE "
                      f"({auto_label_info.get('reason')}) — voiding this trial, "
                      f"candidate NOT marked failed")
            else:
                print("    [auto-label] charuco FAIL — recovering (reset_hybrid)")
            _stop_with_timeout("executor.recording", executor.stop_recording)
            _safe_timestamp_stop(timestamp_monitor)
            _stop_execution_trigger("lift_label_failed")
            # Release (squeeze→grasp→pregrasp gradient) so reset_hybrid's
            # pregrasp→openpose slow interp starts from the right state.
            t_release_and_reset = time.perf_counter()
            try:
                executor.release(result)
            except Exception as re_e:
                print(f"    release FAILED (continuing to retract): {re_e!r}")
            try:
                fb_log = executor.reset(result, planner, scene_cfg)
                timing["retract"] = fb_log
            except Exception as re_e2:
                print(f"    reset FAILED ({re_e2!r}), trying reset_hybrid")
                try:
                    fb_log = executor.reset_hybrid(result, planner, scene_cfg)
                    timing["retract"] = fb_log
                except Exception as fe:
                    timing["retract_error"] = repr(fe)
                    print(f"    reset_hybrid FAILED: {fe!r}")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            timing["release_and_reset_s"] = round(
                time.perf_counter() - t_release_and_reset, 3)
            # Persist a judged failure; skip this write when the label was
            # unjudgeable. Same-session exclusion is independent of this file.
            if result.scene_info is not None and not _label_unjudgeable:
                _write_candidate_outcome(
                    args, hand, obj, result.scene_info,
                    {"success": False, "dir_idx": dir_idx,
                     "arm": args.arm, "reason": "charuco_fail"})
            fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None if _label_unjudgeable else False,
                    "reason": ("label_unjudgeable" if _label_unjudgeable
                               else "charuco_fail"),
                    "scene_info": result.scene_info,
                    "candidate_result_scope": (
                        "experiment" if args.isolate_experiment else "shared_v8"),
                    "auto_label": auto_label_info, "timing": timing}
            if not reposition_mode:
                fail = _with_task_outcome(
                    fail,
                    auto_succ_lift,
                    grasp_evidence={
                        "source": "auto_label_charuco",
                        **auto_label_info,
                    },
                )
            return _save_result(fail)

        # Resume video for place phase.
        _rcc_start(rcc, "video", True, place_rel)

    # Reposition obj on board 11 before place. The physical board centre is
    # the default target; all fallbacks use the shared symmetric +/-5 cm and
    # +/-10 cm on-board grid.
    # ``obj_z`` intentionally remains the grasp-time object Z: reposition is
    # a carried-object motion, not a table-height release.
    # pick yaw that makes the NEXT cov-greedy grasp IK-reachable. Hold z so
    # obj doesn't dip / rise during reposition (plan_pose_constrained).
    t_transfer_selection = time.perf_counter()
    from autodex.utils.conversion import cart2se3
    from autodex.utils.coverage import next_grasp_after_success
    from autodex.utils.path import get_candidate_path

    _reposition_x = float(CHARUCO_BOARD_11_CENTER_XY[0])
    _reposition_y = float(CHARUCO_BOARD_11_CENTER_XY[1])
    if tabletop_geometry is not None:
        _board_center = np.asarray(
            tabletop_geometry["center_robot_m"], dtype=np.float64).reshape(3)
        _reposition_x = float(_board_center[0])
        _reposition_y = float(_board_center[1])
        print("    [reposition] board-11 centre from preflight: "
              f"x={_reposition_x:.3f}, y={_reposition_y:.3f}")

    R_PLACE_DEFAULT = _reposition_x
    R_PLACE = R_PLACE_DEFAULT   # overridden below if yaw_search picks better r
    T_wrist_now = executor.get_wrist_pose()
    T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
    T_obj_now = T_wrist_now @ T_obj_in_wrist
    obj_z = float(T_obj_now[2, 3])
    # IMPORTANT: use the perception-time object orientation as the placement
    # reference.  The Jacobian lift now verifies orientation tightly, but this
    # still prevents live tracking error from becoming the next tabletop
    # target.  A rigid grasp makes the object follow the requested wrist
    # correction back to the classified tabletop orientation.
    R_obj_now = T_obj_grasp[:3, :3]

    chosen_yaw = 0.0
    yaw_feasible_n = 0
    _repos = None
    if (_is_coverage_pool(args.grasp_version) and not args.ignore_coverage
            and pose_stem is not None):
        # Coverage-driven placement: score every (r, yaw) by the UNION of
        # scenes the grasps it makes reachable would newly cover, instead of
        # only chasing the single next set-cover pick below.
        from autodex.utils.reposition import pick_reposition_target, describe
        try:
            # All on-board placement paths share the exact same centre-first
            # X grid. ``pick_reposition_target`` ranks coverage first and then
            # distance to x_preferred, making the centre deterministic when
            # it is equally useful.
            _repo_search_kw = {
                "x_grid": (_reposition_x
                           + CHARUCO_BOARD_CENTER_X_OFFSETS_M),
                "y": _reposition_y,
            }
            _repos = pick_reposition_target(
                obj, pose_stem, hand, args.grasp_version,
                planner=planner, R_obj_robot=R_obj_now, obj_z=obj_z,
                x_preferred=R_PLACE_DEFAULT,
                success_root=candidate_state_root, **_repo_search_kw,
            )
        except Exception as _re:
            print(f"    [place_yaw] reposition search failed: {_re!r}")
            _repos = None
        print(f"    [place_yaw] coverage: {describe(_repos)}")
        if _repos is not None:
            R_PLACE = float(_repos["x"])
            chosen_yaw = _repos["yaw_rad"]
            yaw_feasible_n = _repos["n_feasible_grasps"]

    # Fallback: no coverage json (or nothing left to open) — aim the placement
    # at whichever grasp the set-cover would pick next, as before.
    if (_repos is None
            and _is_coverage_pool(args.grasp_version)
            and not args.ignore_coverage
            and result.scene_info is not None):
        cur_key = tuple(str(x) for x in result.scene_info)
        next_key = next_grasp_after_success(
            obj, cur_key, tabletop_pose_stem=pose_stem,
            hand=hand, version=args.grasp_version,
            success_root=candidate_state_root,
        )
        if next_key is not None:
            next_path = os.path.join(
                get_candidate_path(hand), args.grasp_version, obj,
                next_key[0], next_key[1], next_key[2], "wrist_se3.npy",
            )
            if os.path.exists(next_path):
                next_wrist_obj = np.load(next_path)
                # Sweep the same centre-first X grid used by coverage, rotate,
                # and reorient. Prefer the board centre when multiple choices
                # are otherwise equivalent.
                yaws = np.deg2rad(np.arange(0, 360, 10))
                rs = (_reposition_x + CHARUCO_BOARD_CENTER_X_OFFSETS_M)
                combos = [(r, y) for r in rs for y in yaws]
                wrist_targets = np.zeros((len(combos), 4, 4))
                for i, (r, yaw) in enumerate(combos):
                    c, s = np.cos(yaw), np.sin(yaw)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    T_obj_target = np.eye(4)
                    T_obj_target[:3, :3] = Rz @ R_obj_now
                    T_obj_target[:3, 3] = [float(r), _reposition_y, obj_z]
                    wrist_targets[i] = T_obj_target @ next_wrist_obj
                succ = planner.ik_pose_batch(wrist_targets)
                ok_idx = np.where(succ)[0]
                yaw_feasible_n = int(succ.sum())
                if len(ok_idx) > 0:
                    # rank by |r - R_PLACE_DEFAULT|, tie-break first yaw
                    best = min(ok_idx,
                                key=lambda k: (abs(combos[k][0] - R_PLACE_DEFAULT),
                                                combos[k][1]))
                    R_PLACE = float(combos[best][0])
                    chosen_yaw = float(combos[best][1])
                    print(f"    [place_yaw] next={next_key} → "
                          f"x={R_PLACE:.2f}  yaw={np.degrees(chosen_yaw):.0f}°  "
                          f"({yaw_feasible_n}/{len(combos)} feasible)")
                else:
                    print(f"    [place_yaw] next={next_key} 0 (r,yaw) feasible "
                          f"— using x={R_PLACE_DEFAULT:.2f}, yaw=0°")
            else:
                print(f"    [place_yaw] next grasp wrist_se3.npy missing: "
                      f"{next_path}")
        else:
            print(f"    [place_yaw] no next grasp in cov order")

    # Move arm so obj ends at the selected board-relative XY and the original
    # grasp object Z.  Do not replace this Z by the measured table height.
    c, s = np.cos(chosen_yaw), np.sin(chosen_yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T_obj_target = np.eye(4)
    T_obj_target[:3, :3] = Rz @ R_obj_now
    T_obj_target[:3, 3] = [R_PLACE, _reposition_y, obj_z]
    T_wrist_target = T_obj_target @ np.linalg.inv(T_obj_in_wrist)
    # Force goal wrist z = current wrist z exactly so cuRobo's held-z check
    # passes. Mathematically z is preserved under world-z rotation already,
    # but floating-point chain ops may drift by 1e-7 which trips the check.
    T_wrist_now_world = executor.get_wrist_pose()
    T_wrist_target[2, 3] = T_wrist_now_world[2, 3]
    start_full = np.concatenate([
        np.asarray(executor.get_arm_qpos()[:adof], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    timing["transfer_selection_s"] = round(
        time.perf_counter() - t_transfer_selection, 3)
    if _precomputed_repo_traj is not None:
        print(f"    [reposition] using precomputed repo_traj "
              f"shape={_precomputed_repo_traj.shape}")
        traj_repose = _precomputed_repo_traj
        timing["transfer_plan_s"] = 0.0
        timing["transfer_plan_source"] = "precomputed"
    else:
        t_transfer_plan = time.perf_counter()
        traj_repose = planner.plan_pose_constrained(
            start_full, T_wrist_target,
            hold_vec_weight=[0, 0, 0, 0, 0, 1],   # hold z only
            scene_cfg=scene_cfg,
            include_obj_obstacle=False,
            debug_dump_dir=DEBUG_DUMP_DIR,
            timing_phase="execution",
        )
        timing["transfer_plan_s"] = round(
            time.perf_counter() - t_transfer_plan, 3)
        timing["transfer_plan_source"] = "planned"
    timing["place_yaw_deg"] = round(np.degrees(chosen_yaw), 1)
    timing["place_yaw_feasible_n"] = yaw_feasible_n
    if traj_repose is not None:
        arm_repose = traj_repose[:, :adof]
        # Hold hand at squeeze pose during repose (planner traj's hand
        # portion = grasp_pose which is less closed than s_hand and would
        # open fingers mid-motion → drop obj).
        hand_repose = np.tile(s_hand, (len(traj_repose), 1))
        t_transfer_motion = time.perf_counter()
        executor.follow_joint_trajectory(arm_repose, hand_repose)
        timing["transfer_motion_s"] = round(
            time.perf_counter() - t_transfer_motion, 3)
        print(f"    [reposition] obj → (x={R_PLACE:.3f}, y={_reposition_y:.3f}, "
              f"yaw={np.degrees(chosen_yaw):.0f}°)")
    else:
        timing["transfer_motion_s"] = 0.0
        print(f"    [reposition] plan_pose_constrained failed — placing here")

    # The common placement request carries an explicit table-height wrist when
    # a transfer target was selected.  Each arm adapter owns its contact and
    # descent mechanics, but run_auto no longer branches on a robot class.
    _place_kw = {}
    # A successful post-grasp reposition is the actual placement transfer for
    # both ordinary coverage trials and explicit Charuco reposition trials.
    if reposition_mode or traj_repose is not None:
        T_obj_place_low = T_obj_target.copy()
        T_obj_place_low[2, 3] = T_obj_grasp[2, 3]
        _place_kw["placement_wrist"] = (
            T_obj_place_low @ np.linalg.inv(T_obj_in_wrist))
        if traj_repose is not None:
            # The transfer has just moved the held object to this +10cm
            # pre-place pose. The arm adapter validates live state before it
            # may reuse it; otherwise it plans a correction in its own world.
            _place_kw["preplace_traj"] = traj_repose
            _place_kw["preplace_wrist_target"] = T_wrist_target
    t_place = time.perf_counter()
    place_info = executor.place(result, planner=planner, scene_cfg=scene_cfg,
                                debug_dump_dir=DEBUG_DUMP_DIR, **_place_kw)
    timing["place_total_s"] = round(time.perf_counter() - t_place, 3)
    timing["execute_s"] = round(time.perf_counter() - t0, 3)
    timing["execution_states"] = executor.state_timestamps
    timing["place"] = place_info
    place_timing = dict((place_info or {}).get("timing_s") or {})
    # Flat fields are kept for simple CSV/table extraction.  The structured
    # taxonomy below is the canonical interpretation: its top-level buckets
    # are mutually exclusive wall-clock categories whose sum is ``execute_s``.
    execution_breakdown = {
        "total_s": timing["execute_s"],
        "grasp_lift_s": timing["execute_grasp_lift_s"],
        # This is a contingency after the object is already grasped. The
        # candidate preflight belongs to planner timing, not this field.
        "grasp_lift_runtime_replan_s": float(
            _executor_exec_timing.get("lift_runtime_replan_s",
                                      _executor_exec_timing.get("lift_plan_s", 0.0))),
        "auto_label_lift_s": float(timing.get("auto_label_lift_s", 0.0)),
        "transfer_selection_s": float(timing.get("transfer_selection_s", 0.0)),
        "transfer_plan_s": float(timing.get("transfer_plan_s", 0.0)),
        "transfer_motion_s": float(timing.get("transfer_motion_s", 0.0)),
        "place_planning_s": float(place_timing.get("planning_s", 0.0)),
        "place_motion_s": float(place_timing.get("motion_s", 0.0)),
    }
    # v1 CSV compatibility: lift_plan means runtime replan only.
    execution_breakdown["grasp_lift_plan_s"] = (
        execution_breakdown["grasp_lift_runtime_replan_s"])
    execution_breakdown["planning_s"] = round(sum(
        execution_breakdown[key] for key in (
            "grasp_lift_plan_s", "transfer_plan_s", "place_planning_s")), 3)
    execution_breakdown["auto_label_s"] = round(
        execution_breakdown["auto_label_lift_s"], 3)
    grasp_lift_motion_s = sum(float(_executor_exec_timing.get(key, 0.0))
                              for key in (
                                  "init_motion_s", "approach_motion_s",
                                  "pregrasp_motion_s", "grasp_motion_s",
                                  "squeeze_motion_s", "lift_motion_s"))
    planning_bucket = {
        "lift_s": execution_breakdown["grasp_lift_plan_s"],
        "transfer_s": execution_breakdown["transfer_plan_s"],
        "place_s": execution_breakdown["place_planning_s"],
    }
    planning_bucket["total_s"] = round(sum(planning_bucket.values()), 3)
    validation_bucket = {
        "auto_label_lift_capture_s": float(
            timing.get("auto_label_lift_capture_s", 0.0)),
        "auto_label_lift_infer_s": float(
            timing.get("auto_label_lift_infer_s", 0.0)),
        "total_s": execution_breakdown["auto_label_lift_s"],
    }
    motion_bucket = {
        # Explicit physical command spans. Do not use the old residual here:
        # it mixed monitor warmup, state checks and Python/SDK overhead.
        "grasp_lift_s": round(grasp_lift_motion_s, 3),
        "transfer_s": execution_breakdown["transfer_motion_s"],
        "place_s": execution_breakdown["place_motion_s"],
    }
    motion_bucket["total_s"] = round(sum(motion_bucket.values()), 3)
    decision_bucket = {
        "transfer_target_selection_s": execution_breakdown["transfer_selection_s"],
        "total_s": execution_breakdown["transfer_selection_s"],
    }
    executor_service_bucket = {
        "approach_monitor_warmup_s": float(
            _executor_exec_timing.get("approach_monitor_warmup_s", 0.0)),
        "lift_start_check_s": float(
            _executor_exec_timing.get("lift_start_check_s", 0.0)),
        "adapter_overhead_s": float(
            _executor_exec_timing.get("overhead_s", 0.0)),
    }
    executor_service_bucket["total_s"] = round(
        sum(executor_service_bucket.values()), 3)
    classified_s = (planning_bucket["total_s"] + validation_bucket["total_s"]
                    + motion_bucket["total_s"] + decision_bucket["total_s"]
                    + executor_service_bucket["total_s"])
    execution_breakdown["unattributed_s"] = round(
        max(0.0, timing["execute_s"] - classified_s), 3)
    execution_breakdown["taxonomy"] = {
        "planning": planning_bucket,
        "validation": validation_bucket,
        "motion": motion_bucket,
        "decision": decision_bucket,
        "executor_service": executor_service_bucket,
        # Camera/recorder boundaries and small Python/SDK overhead not covered
        # by the explicit spans above. This makes the categories exhaustive.
        "overhead_s": execution_breakdown["unattributed_s"],
    }
    timing["execution_breakdown_s"] = execution_breakdown

    # Release the obj BEFORE stopping cameras so the hand-open moment is
    # captured in the place video. Only release on normal full-descent
    # path; early-contact branch keeps the grasp closed for reset chain.
    _descended_pre = place_info.get("descended", 0.0)
    _target_d_pre = place_info.get("target", 0.0)
    # FrankaExecutor.place() performs the validated release itself between the
    # 10cm down/up stages, while the legacy executor releases here.
    _released_in_video = bool(place_info.get("released", False))
    if not _released_in_video and not (place_info.get("stopped_on_contact")
            and (_target_d_pre - _descended_pre) > 0.005):
        print(f"[6/6] Releasing (in-video)...")
        t_external_release = time.perf_counter()
        executor.release(result)
        timing["external_release_s"] = round(
            time.perf_counter() - t_external_release, 3)
        _released_in_video = True

    # STOP order: rcc (cameras) first WHILE sync_generator still pulsing
    # — cameras need pulses to flush buffers during stop. Then timestamp
    # and sync last.
    t_execution_teardown = time.perf_counter()
    _stop_with_timeout("rcc", rcc.stop)
    _safe_timestamp_stop(timestamp_monitor)
    _stop_execution_trigger("execution_complete")
    timing["execution_teardown_s"] = round(
        time.perf_counter() - t_execution_teardown, 3)

    # Place hit contact mid-descent → release at that supported pose, rise
    # vertically from the exact live wrist, then reset.  The reset adapters
    # reconstruct the placed-object pose and preflight both +Z clearance and
    # the following retract before moving the arm.
    # Treat as failure only when contact stopped descent BEFORE the target
    # depth — that means the obj hit something mid-descent (table itself at
    # full depth is the EXPECTED stop, not a failure). Threshold 5mm guards
    # against floating-point noise at exactly target.
    _descended = place_info.get("descended", 0.0)
    _target_d = place_info.get("target", 0.0)
    _early_contact = (place_info.get("stopped_on_contact")
                       and (_target_d - _descended) > 0.005)
    if _early_contact:
        print(f"    [place] EARLY contact stop "
              f"({_descended*1000:.1f}mm of "
              f"{_target_d*1000:.1f}mm) — releasing here, then +Z clearance "
              "and reset")
        _stop_with_timeout("executor.recording", executor.stop_recording)
        t_release_and_reset = time.perf_counter()
        recovery_error = None
        try:
            t_release = time.perf_counter()
            executor.release(result)
            timing["external_release_s"] = round(
                time.perf_counter() - t_release, 3)
            fb_log = executor.reset(result, planner, scene_cfg)
            timing["retract"] = fb_log
        except Exception as recovery_exc:
            # The object is already released.  Do not bypass the required +Z
            # clearance with reset_hybrid/reset_fallback; leave the open hand
            # in place for manual recovery if either preflight fails.
            recovery_error = repr(recovery_exc)
            timing["retract_error"] = recovery_error
            timing["post_release_safety_stop"] = True
            print(f"    early-contact release/clearance/reset FAILED: "
                  f"{recovery_exc!r}")
        timing["release_and_reset_s"] = round(
            time.perf_counter() - t_release_and_reset, 3)
        if recovery_error is None:
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if reposition_mode:
            # Reposition fail (contact stop) → update stats with fail.
            if result.scene_info is not None:
                from autodex.utils.coverage import update_grasp_stats
                sei = result.scene_info
                gd = _candidate_state_grasp_dir(args, hand, obj, sei)
                if gd is not None:
                    a, s = update_grasp_stats(gd, False)
                    print(f"    [stats] {sei} now {s}/{a} (rate={s/a if a else 0:.2f})")
                    timing["repo_stats"] = {"attempts": a, "successes": s}
        elif result.scene_info is not None:
            _write_candidate_outcome(
                args, hand, obj, result.scene_info,
                {"success": bool(auto_succ_lift),
                 "dir_idx": dir_idx,
                 "arm": args.arm,
                 "reason": "place_early_contact"})
        # Grasp success criterion = charuco at LIFT (auto_succ_lift). Place
        # quality is a separate metric — early contact during descent does
        # not invalidate a successful grasp. Keep trial.success = grasp
        # success, attach place_info as diagnostic.
        trial_success = bool(auto_succ_lift)
        record = {
            "dir_idx": dir_idx, "scene_type": args.scene,
            "success": trial_success,
            "reason": ("place_early_contact" if trial_success
                       else "charuco_fail_then_place_early_contact"),
            "scene_info": result.scene_info,
            "candidate_result_scope": (
                "experiment" if args.isolate_experiment else "shared_v8"),
            "auto_label_lift": auto_label_info,
            "place": place_info,
            "early_contact_recovery_error": recovery_error,
            "manual_recovery_required": recovery_error is not None,
            "timing": timing,
        }
        if not reposition_mode:
            record = _with_task_outcome(
                record,
                trial_success,
                grasp_evidence={
                    "source": "auto_label_charuco",
                    **auto_label_info,
                },
            )
        return _save_result(record)

    # ── 5. Label ─────────────────────────────────────────────────────────────
    timing["label_start"] = _ts()
    t_final_label = time.perf_counter()
    print(f"[5/6] Label the result")
    if args.auto:
        if reposition_mode:
            # Reposition success = obj covers the charuco board after place
            # (= board NOT fully visible).
            _stop_with_timeout("rcc", rcc.stop)
            repo_post_rel = os.path.join(
                "shared_data", "AutoDex", "experiment", args.exp_name, sub,
                obj, dir_idx, "_repo_check_post", "raw"
            )
            repo_post_abs = os.path.join(
                img_dir, "_repo_check_post", "raw", "images"
            )
            rcc.start("image", False, repo_post_rel)
            rcc.stop()
            time.sleep(0.3)
            post_vis, post_info = auto_label_charuco(
                repo_post_abs, required_board=CHARUCO_BOARD
            )
            timing["repo_charuco_after"] = post_info
            succ = (post_vis is False)  # covered = success
            note = (f"repo post-charuco "
                    f"{post_info.get('covered')}/{post_info.get('expected')}")
            print(f"    [reposition] post={post_info.get('covered')}/"
                  f"{post_info.get('expected')} → success={succ}")
            # Update stats for the chosen v8 grasp.
            if result.scene_info is not None:
                from autodex.utils.coverage import update_grasp_stats
                sei = result.scene_info
                grasp_dir = _candidate_state_grasp_dir(args, hand, obj, sei)
                if grasp_dir is not None:
                    a, s = update_grasp_stats(grasp_dir, succ)
                    print(f"    [stats] {sei} now {s}/{a} "
                          f"(rate={s/a if a else 0:.2f})")
                    timing["repo_stats"] = {"attempts": a, "successes": s}
        else:
            succ = bool(auto_succ_lift)
            note = (auto_label_info.get("reason")
                    or f"charuco covered "
                       f"{auto_label_info.get('covered')}/"
                       f"{auto_label_info.get('expected')}")
            timing["auto_label"] = auto_label_info
            print(f"    auto_label: success={succ}  note={note}")
    else:
        label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                 args.exp_name, sub, obj, dir_idx, "label", "raw")
        rcc.start("image", False, label_rel)
        rcc.stop()
        try:
            # Manual mode has no lift-time Charuco check. ``get_label`` is its
            # first explicit validation operation, so it is the equivalent
            # streak boundary.
            _mark_grasp_validation_started("manual_label_started")
            succ, note = get_label()
            if trial_scope is not None:
                trial_scope.event(
                    "grasp.validation_result", phase="validation", kind="result",
                    parent_id=episode_span,
                    outcome=("success" if succ is True else
                             "failure" if succ is False else "unjudgeable"),
                    scene_info=result.scene_info, success=succ,
                    classification=("success" if succ is True else
                                    "failure" if succ is False else
                                    "unjudgeable"),
                    reason=note, source="manual_label",
                )
        except KeyboardInterrupt:
            print("\n[interrupted] Releasing and cleaning up...")
            executor.release(result)
            executor.stop_recording()
            raise

    timing["final_label_s"] = round(time.perf_counter() - t_final_label, 3)

    # ── 6. Release & save ────────────────────────────────────────────────────
    if not _released_in_video:
        print(f"[6/6] Releasing...")
        t_external_release = time.perf_counter()
        executor.release(result)
        timing["external_release_s"] = round(
            time.perf_counter() - t_external_release, 3)
    else:
        print(f"[6/6] Release already done in-video — skipping")

    # reset_hybrid now does the slow pregrasp→openpose interp internally,
    # then keeps hand at openpose during sequential [1,2,0] + cuRobo wrist.
    #     replan around placed object, back to XARM_INIT.
    t_release_and_reset = time.perf_counter()
    try:
        fb_log = executor.reset(result, planner, scene_cfg)
        timing["retract"] = fb_log
        print(f"    retract OK  final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
    except Exception as re_e:
        print(f"    reset FAILED ({re_e!r}), trying reset_hybrid")
        try:
            fb_log = executor.reset_hybrid(result, planner, scene_cfg)
            timing["retract"] = fb_log
            print(f"    retract OK (hybrid)  final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
        except Exception as fb_e:
            print(f"    reset_hybrid FAILED: {fb_e!r}, falling back")
            try:
                executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
            except Exception as ff_e:
                print(f"    reset_fallback FAILED: {ff_e!r}")
    timing["release_and_reset_s"] = round(
        time.perf_counter() - t_release_and_reset, 3)

    executor.stop_recording()

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

    trial_result = {
        "dir_idx": dir_idx,
        "scene_type": args.scene,
        "arm": args.arm,
        "success": succ,
        "scene_info": result.scene_info,
        "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
        "candidate_result_scope": (
            "experiment" if args.isolate_experiment else "shared_v8"),
        "tabletop_before": tb_before,
        "timing": timing,
    }
    if note is not None:
        trial_result["note"] = note
    if not reposition_mode:
        trial_result = _with_task_outcome(
            trial_result,
            succ,
            grasp_evidence={
                "source": "auto_label_charuco" if args.auto else "manual_label",
                "note": note,
                **(auto_label_info if args.auto else {}),
            },
        )
    _save_result(trial_result)

    # Persist result back to the candidate dir for ALL scenes (table, wall,
    # shelf, etc.) — both success AND fail. Success records drive coverage and
    # completed-scene filtering; failures remain available after a fresh
    # process. The in-memory set prevents reuse during this session.
    # Reposition trials don't write here; their stats.json is updated above.
    if (succ is not None and result.scene_info is not None
            and not reposition_mode):
        _write_candidate_outcome(
            args, hand, obj, result.scene_info,
            {"success": succ, "dir_idx": dir_idx, "arm": args.arm})

    status = "SUCCESS" if succ else ("ISSUE" if succ is None else "FAIL")
    print(f"    Result: {status}  saved to {img_dir}/result.json")

    # Resume the stream so the next trial's init has live SHM frames.
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)

    return _stamp_end(trial_result)


# ── main ─────────────────────────────────────────────────────────────────────

def main(pose_adjust_handler=None, reorient_handler=None, startup_handler=None,
         pipeline_trace: Optional[PipelineTrace] = None,
         task: Optional[TaskInterface] = None):
    owns_pipeline_trace = pipeline_trace is None
    if pipeline_trace is None:
        pipeline_trace = PipelineTrace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="v8",
                        help="v8 candidate/tabletop asset contract")
    parser.add_argument("--exp_name", type=str, default=None, help="Defaults to grasp_version")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--arm", type=str, default="xarm",
                        choices=["xarm", "franka"],
                        help="franka = FR3 + inspire (planner robot fr3_inspire, "
                             "7-DOF arm, FrankaExecutor)")
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--max_trials", type=int, default=0,
                        help="0=unlimited. With --ignore_coverage: cap demo run.")
    parser.add_argument(
        "--max_consecutive_rotates", type=int, default=2,
        help="Maximum successful physical rotate recoveries in a row for "
             "one recovery streak before grasp validation starts (default: 2). "
             "0 disables rotate recovery and "
             "goes directly to reorient after a planning failure.",
    )
    parser.add_argument("--ignore_coverage", action="store_true",
                        help="Run regardless of scene coverage / past success: "
                             "no coverage filter, no skip_done, no reorient "
                             "suggestion.")
    parser.add_argument("--isolate_experiment", action="store_true",
                        help="Keep candidate labels in experiment/<exp_name> only. "
                             "Coverage ordering, progress, and reorient use that "
                             "private state; shared v8 labels remain untouched.")
    parser.add_argument("--candidate-scene-type", default=None,
                        choices=["table", "wall", "shelf", "box"],
                        help="Use only this candidate scene type while retaining normal skip-done semantics. "
                             "For new Franka collection use 'table' before coverage is computed.")

    # scene-specific args (pass-through to autodex.planner.obstacles.add_obstacles)
    parser.add_argument("--wall_gap", type=float, default=0.04)
    parser.add_argument("--wall_angle", type=float, default=0.0)
    parser.add_argument("--clutter_seed", type=int, default=42)
    parser.add_argument("--clutter_min_dist", type=float, default=0.12)
    parser.add_argument("--clutter_max_dist", type=float, default=0.20)
    parser.add_argument("--clutter_n", type=int, default=4)
    parser.add_argument("--shelf_width", type=float, default=0.30)
    parser.add_argument("--shelf_depth", type=float, default=0.30)
    parser.add_argument("--shelf_height", type=float, default=0.30)
    parser.add_argument("--shelf_gap", type=float, default=0.02)
    parser.add_argument("--no_shelf_back", action="store_true")
    parser.add_argument("--no_shelf_sides", action="store_true")
    parser.add_argument("--no_shelf_top", action="store_true")

    # init pipeline args
    parser.add_argument("--pc_list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port_mask", type=int, default=5006)
    parser.add_argument("--port_pose", type=int, default=5007)
    parser.add_argument("--port_cmd", type=int, default=6893)
    parser.add_argument("--port_snap", type=int, default=5009)
    parser.add_argument("--port_snap_cmd", type=int, default=6894)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-label via charuco snapshot at lift-time. "
                             "Also auto-approves pipeline rotate/reorient recovery; "
                             "default off falls back to manual get_label() prompt.")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument(
        "--perception_mode", choices=["iou", "ignore_sil_loss"], default="iou",
        help="Pose selection mode. 'iou' (default) rejects a pose when its "
             "silhouette loss exceeds 0.003; 'ignore_sil_loss' still runs "
             "cross-view matching and silhouette refinement, but never "
             "rejects solely on silhouette loss.",
    )
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=60.0,
                        help="Max wait for masks+poses. Ctrl-C during the "
                             "wait cuts it short and continues with "
                             "whatever arrived.")
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="Camera calib dir. Default: latest under ~/shared_data/cam_param/.")
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)
    parser.add_argument(
        "--external-sync-cue", action="store_true",
        help="Emit start/end TTL bursts for an LED visible in the separately "
             "recorded external-camera video.",
    )
    parser.add_argument(
        "--external-sync-cue-duration-s", type=float, default=1.2,
        help="Duration of each external-video LED burst (default: 1.2 s).",
    )
    parser.add_argument(
        "--external-sync-cue-fps", type=int, default=5,
        help="TTL pulse rate for the external-video LED cue (default: 5 Hz).",
    )
    parser.add_argument(
        "--charuco-preflight", choices=["prompt", "measure", "skip"],
        default="prompt",
        help="run_pipeline only: empty-board Charuco measurement before the "
             "first object is placed. prompt=choose at runtime, "
             "measure=require it, skip=use fixed default tabletop geometry.",
    )
    parser.add_argument(
        "--socket-preflight", choices=["auto", "prompt", "measure", "skip"],
        default="auto",
        help="run_pipeline only: measure and freeze the unified socket pose "
             "before precision-key trials. auto=measure for precision_key_* "
             "and precision_key_cylinder_* objects and skip otherwise.",
    )
    parser.add_argument("--socket-object", default="precision_socket_unified")
    parser.add_argument(
        "--socket-prompt", default="fixed red socket fixture with keyed opening",
        help="FoundPose segmentation prompt for the fixed socket.",
    )
    parser.add_argument(
        "--socket-measurements", type=int, default=3,
        help="Independent socket pose estimates used by the startup medoid gate.",
    )
    parser.add_argument("--socket-sil-iters", type=int, default=100)
    parser.add_argument(
        "--socket-sil-loss-max", type=float, default=0.01,
        help="Socket silhouette-loss rejection threshold. The open cavity often "
             "scores worse than a convex key; this is only a selection gate, "
             "not an absolute pose-accuracy guarantee.",
    )
    parser.add_argument(
        "--socket-repeat-translation-max-mm", type=float, default=2.0,
        help="Maximum residual from the selected socket-pose medoid.",
    )
    parser.add_argument(
        "--socket-repeat-rotation-max-deg", type=float, default=2.0,
        help="Maximum angular residual from the selected socket-pose medoid.",
    )

    args = parser.parse_args()
    from autodex.tasks.precision_insertion import validate_runtime_socket_pair

    try:
        validate_runtime_socket_pair(args.obj, args.socket_object)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.obj == "precision_key_cylinder_r15_h80" and
            args.socket_prompt == "fixed red socket fixture with keyed opening"):
        args.socket_prompt = "fixed cylindrical socket fixture with round opening"
    if args.grasp_version != "v8":
        parser.error("run_auto supports only --grasp_version v8; legacy asset pools are disabled")
    if args.max_consecutive_rotates < 0:
        parser.error("--max_consecutive_rotates must be >= 0")
    if args.external_sync_cue_duration_s <= 0:
        parser.error("--external-sync-cue-duration-s must be > 0")
    if args.external_sync_cue_fps <= 0:
        parser.error("--external-sync-cue-fps must be > 0")
    if args.socket_measurements < 2:
        parser.error("--socket-measurements must be >= 2")
    if args.socket_sil_iters < 0:
        parser.error("--socket-sil-iters must be >= 0")
    if args.socket_sil_loss_max <= 0:
        parser.error("--socket-sil-loss-max must be > 0")
    if args.socket_repeat_translation_max_mm <= 0:
        parser.error("--socket-repeat-translation-max-mm must be > 0")
    if args.socket_repeat_rotation_max_deg <= 0:
        parser.error("--socket-repeat-rotation-max-deg must be > 0")
    if args.exp_name is None:
        args.exp_name = args.grasp_version
    if args.isolate_experiment:
        if args.success_only:
            parser.error("--isolate_experiment cannot be combined with --success_only "
                         "because success-only reads shared v8 candidate labels")
        state_root = experiment_candidate_state_root(
            args.exp_name, args.hand, args.grasp_version, args.obj)
        print("[experiment] isolated mode: episode labels and candidate state "
              f"-> experiment/{args.exp_name}; shared v8 state is untouched")
        print("[experiment] private candidate state -> "
              f"{os.path.relpath(state_root, project_dir)}")

    # scene_prefix: '' (table), 'wall', 'shelf', 'cluttered', plus '_success_only' suffix.
    scene_prefix = args.scene if args.scene != "table" else ""
    if args.success_only:
        scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"

    sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
    trace_dir = (
        Path(project_dir) / "experiment" / args.exp_name / sub / args.obj /
        "_pipeline_runs" / pipeline_trace.run_id
    )
    pipeline_trace.bind(
        trace_dir,
        command="run_pipeline" if pose_adjust_handler is not None else "run_auto",
        arguments=vars(args),
        object=args.obj, hand=args.hand, arm=args.arm,
        scene=args.scene, experiment=args.exp_name,
    )
    startup_span = pipeline_trace.begin(
        phase="startup", kind="setup", name="session_initialization")

    # Mesh / FoundPose assets sanity check.
    # v8 candidates are expressed against object_processing. FoundPose must
    # initialize against that exact mesh frame; otherwise physical Franka
    # collection can mark grasps that are offset when the continuous runner
    # later consumes them.
    mesh_root = Path(get_obj_root(args.grasp_version))
    mesh_path = mesh_root / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj} (expected under {assets_root})")

    # A precision run must discover a missing socket representation before it
    # claims cameras, connects the robot, or performs the clear-view home.
    # Prompted manual runs may explicitly skip later; automatic/pinned measure
    # modes are fail-closed here.
    socket_measure_required = (
        startup_handler is not None
        and (
            args.socket_preflight == "measure"
            or (args.socket_preflight == "auto"
                and args.obj.startswith(
                    ("precision_key_", "precision_key_cylinder_")))
            or (args.socket_preflight == "prompt" and args.auto)
        )
    )
    if socket_measure_required:
        socket_root = mesh_root / args.socket_object
        socket_assets = ASSETS_BASE / args.socket_object
        socket_required = (
            socket_root / "raw_mesh" / f"{args.socket_object}.obj",
            socket_root / "processed_data" / "mesh" / "static_collision.obj",
            socket_assets / "object_repre" / "v1" / args.socket_object / "1" /
            "repre.pth",
        )
        socket_missing = [path for path in socket_required if not path.is_file()]
        if socket_missing:
            sys.exit("socket preflight assets missing before hardware startup: "
                     + ", ".join(str(path) for path in socket_missing))

    # FoundPose and the planner now share the version-resolved mesh. Keep this
    # check because a caller may still point a non-v8 pool at a mismatched
    # custom object tree.
    from src.execution.scene_cfg import check_mesh_frame_match
    _frame_ok, _frame_msg = check_mesh_frame_match(
        args.obj, str(mesh_path), get_obj_root(args.grasp_version))
    if not _frame_ok:
        sys.exit(f"[mesh_frame] {_frame_msg}")
    print(f"[mesh_frame] {_frame_msg}")

    # Calibration.
    if args.calib_dir:
        calib_dir = Path(args.calib_dir).expanduser()
    else:
        calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    missing_intrinsics = sorted(active_serials - set(intrinsics_full))
    missing_extrinsics = sorted(active_serials - set(extrinsics_full))
    if missing_intrinsics or missing_extrinsics:
        parser.error(
            f"camera calibration {calib_dir} does not cover the active AutoDex "
            f"camera set; missing intrinsics={missing_intrinsics}, "
            f"missing extrinsics={missing_extrinsics}. Pass an explicit "
            "--calib_dir matching paradex/system/current/pc.json."
        )
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs  ({H}x{W})")

    # Hardware init.
    # stall_timeout > the arm-to-trigger gap. Cameras are armed with
    # syncMode=True and produce nothing until sync_generator.start() runs a
    # few statements later, so the default 3 s logs a 20-camera "no new
    # frames" block on every single trial that means nothing.
    rcc = remote_camera_controller("run_auto", pc_list=args.pc_list,
                                   stall_timeout=15.0)
    _ensure_camera_lock(rcc)
    _clear_camera_errors(rcc)
    # Preserve the established AutoDex acquisition contract: all remote FLIR
    # cameras are hardware-triggered by the local UTG900 and accompanied by
    # the configured local timestamp camera.
    trigger_params, trigger_note = resolve_signal_generator_params(
        network_info["signal_generator"]["param"]
    )
    if trigger_note is not None:
        print(f"[video] {trigger_note}")
    sync_generator = UTGE900(**trigger_params)
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Init orchestrator (FoundPose distributed).
    print(f"[orch] initializing for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    # The planner's robot config keys on arm+hand; the CANDIDATE pool keys on
    # hand alone (fr3 and xarm share the inspire grasp pools), which is why
    # planner.plan() below still gets args.hand.
    planner_robot = _planner_robot(args.arm, args.hand)
    print(f"[planner] warming up ({planner_robot})...")
    planner = GraspPlanner(hand=planner_robot)
    quiet_curobo()   # GraspPlanner init re-runs curobo's own logger setup
    print(f"[executor] connecting to robot ({args.arm})...")
    if args.arm == "franka":
        from src.execution.franka_executor import FrankaExecutor
        executor = FrankaExecutor(hand_name=args.hand)
        # Bind before the initial clear-view move so free-hand recovery paths
        # can evaluate the shared live object-proximity speed profile.
        executor.set_speed_profile_planner(planner)
        # Park OUT of the cameras' view before the first perception so the arm
        # never occludes the object.
        print("[executor] homing to clear-view...")
        executor.home(clear_view=True)
    else:
        executor = RealExecutor(hand_name=args.hand)
        # Match the Franka startup contract: establish a known, calibrated
        # arm/hand pose before the empty-board preflight and first perception.
        print("[executor] homing to clear-view...")
        executor.home(clear_view=True)

    def _cleanup():
        print("\n[cleanup] Stopping hardware...")
        # Order: rcc first (cameras need pulses to flush during stop), then
        # timestamp, sync_generator last.
        for fn in (rcc.stop, timestamp_monitor.stop, sync_generator.stop,
                   executor.stop_recording):
            try:
                fn()
            except Exception:
                pass

    # ``run_pipeline`` installs an empty-board preflight here.  It runs after
    # clear-view but before any object perception, so the measurement can use
    # all Charuco corners without paying for a second FoundPose init.
    session_tabletop_geometry = None
    session_fixed_fixtures = None
    startup_cancelled = False
    if startup_handler is not None:
        try:
            startup_result = startup_handler(
                args=args, rcc=rcc, executor=executor, orch=orch,
                intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
                capture_ips=pc_ips, n_cameras=len(intrinsics_full),
                pc_serials=pc_serials, image_hw=(H, W),
                calib_dir=str(calib_dir),
                target_mesh_path=str(mesh_path),
                target_assets_root=str(assets_root),
                scene_prefix=scene_prefix,
                pipeline_trace=pipeline_trace.scoped(parent_id=startup_span),
            )
            if isinstance(startup_result, dict) and startup_result.get("cancel"):
                startup_cancelled = True
            elif (isinstance(startup_result, dict)
                  and startup_result.get("schema") == "autodex_session_startup_v1"):
                session_tabletop_geometry = startup_result.get("tabletop_geometry")
                session_fixed_fixtures = startup_result.get("fixed_fixtures")
            elif startup_result is not None:
                # Backward-compatible contract for existing startup hooks.
                session_tabletop_geometry = startup_result
        except Exception as startup_exc:
            # A startup hook must never silently leave us using stale geometry
            # after an operator selected measurement.  It can return None only
            # for an explicit skip; unexpected failures end this session.
            startup_cancelled = True
            print(f"[startup] preflight failed: {startup_exc!r}")

    # ``--auto`` removes per-trial and recovery confirmations, but an operator
    # must still be able to place the object deliberately before the very first
    # perception.  Keep this one session-start gate; recovery retries never
    # return here, so rotate/reorient remain fully automatic afterwards.
    if args.auto and not startup_cancelled:
        operator_span = pipeline_trace.begin(
            phase="operator", kind="wait", name="initial_object_placement",
            parent_id=startup_span)
        try:
            cmd = input("[auto] Place the object, then press Enter to start "
                        "the automatic session (q to quit): ").strip().lower()
        except KeyboardInterrupt:
            cmd = "q"
        pipeline_trace.end(
            operator_span,
            outcome=("aborted" if cmd == "q" else "success"),
            response=("quit" if cmd == "q" else "continue"),
        )
        if cmd == "q":
            startup_cancelled = True

    if not startup_cancelled:
        try:
            _emit_external_sync_cue(
                sync_generator, pipeline_trace, label="pipeline_start",
                enabled=args.external_sync_cue,
                duration_s=args.external_sync_cue_duration_s,
                fps=args.external_sync_cue_fps,
            )
        except Exception as cue_exc:
            # A missing LED/cable must not make the robot session unusable;
            # UTC/monotonic clock anchors remain available for manual sync.
            print(f"[sync] external start cue failed: {cue_exc!r}")

    pipeline_trace.end(
        startup_span,
        outcome=("aborted" if startup_cancelled else "success"),
        startup_cancelled=startup_cancelled,
    )

    results: List[dict] = []
    trial = 0
    resume_after_pose_adjust = False
    # One streak spans repeated successful physical rotations until grasp
    # validation starts. It is deliberately not keyed by the classified
    # tabletop stem: a yaw rotation can change that classification while still
    # being part of the same uninterrupted recovery sequence.
    rotate_recovery_state: Dict[str, object] = {"count": 0}
    # Once a candidate reaches real execution, keep it out of later trials in
    # this process. Persisted failures remain retryable after a fresh session.
    session_attempted_candidates: set[tuple[str, str, str]] = set()
    campaign_state_root = _candidate_state_root(args, args.hand, args.obj)
    try:
        while not startup_cancelled:
            trial += 1
            attempt_id = f"attempt_{trial:04d}"
            episode_id = (
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                f"_{attempt_id}")
            print(f"\n{'#'*60}\n# Trial {trial}\n{'#'*60}")
            chime.info()
            if not args.auto and not resume_after_pose_adjust:
                try:
                    cmd = input("Press Enter to start trial, 'q' to quit: ").strip().lower()
                except KeyboardInterrupt:
                    _cleanup()
                    break
                if cmd == "q":
                    break
            # The in-process rotation succeeded in the previous iteration;
            # immediately re-run the normal trial from perception, without an
            # extra manual prompt or treating the recovery itself as a trial.
            resume_after_pose_adjust = False
            # --auto: no pre-perception charuco check; perception runs first
            # and decides:
            #   pose_world None        → perception_failed prompt (case 1)
            #   pose_world OK + ...    → normal trial flow

            # Coverage snapshot BEFORE the trial — used to compute how
            # many new scenes the trial just covered.
            if (_is_coverage_pool(args.grasp_version)
                    and not args.ignore_coverage):
                from autodex.utils.coverage import (
                    uncovered_scenes, _tabletop_stems,
                )
                _stems_before = _tabletop_stems(
                    args.obj, get_obj_root(args.grasp_version))
                _rem_before = {}
                _sets_before = {}
                # First call walks the whole candidate tree on the NFS mount;
                # say so, or the trial looks hung before [1/6] prints.
                print(f"[coverage] snapshot over {len(_stems_before)} tabletop "
                      f"poses...", flush=True)
                for _s in _stems_before:
                    _u = uncovered_scenes(args.obj, _s, hand=args.hand,
                                          version=args.grasp_version,
                                          success_root=campaign_state_root)
                    _rem_before[_s] = (len(_u) if _u is not None else None)
                    _sets_before[_s] = (sorted(int(v) for v in _u)
                                        if _u is not None else None)
                coverage_ref = pipeline_trace.write_artifact_json(
                    f"artifacts/coverage/{attempt_id}_before.json",
                    {"attempt_id": attempt_id, "object": args.obj,
                     "uncovered_scene_ids_by_tabletop": _sets_before},
                )
                pipeline_trace.scoped(
                    episode_id=episode_id,
                    attempt_id=attempt_id,
                ).event(
                    "coverage.snapshot_before", phase="coverage",
                    kind="decision",
                    attributes={"artifact": coverage_ref,
                                "remaining_by_tabletop": _rem_before},
                )

            tr = run_single_trial(
                args, scene_prefix=scene_prefix,
                orch=orch, planner=planner, executor=executor,
                rcc=rcc, sync_generator=sync_generator,
                timestamp_monitor=timestamp_monitor,
                pose_adjust_handler=pose_adjust_handler,
                reorient_handler=reorient_handler,
                tabletop_geometry=session_tabletop_geometry,
                fixed_fixtures=session_fixed_fixtures,
                rotate_recovery_state=rotate_recovery_state,
                pipeline_trace=pipeline_trace,
                attempt_id=attempt_id,
                episode_id=episode_id,
                session_attempted_candidates=session_attempted_candidates,
                task=task,
            )
            if tr.get("retry_current_trial"):
                if tr.get("reason") in ("reoriented", "reoriented_manual"):
                    reset_event = _reset_rotate_streak(
                        rotate_recovery_state, reason="reorient_completed",
                        tabletop_stem=tr.get("reorient_target_stem"))
                    if reset_event is not None:
                        print("    [rotate] reorient completed; reset consecutive "
                              f"recovery count ({reset_event['count_before']} -> 0)")
                if tr.get("reason") == "planning_retry_feasible":
                    print("\n    [planner] retry found a feasible plan — restarting "
                          "from fresh perception without recovery motion")
                else:
                    print("\n    [pipeline] recovery complete — restarting normal "
                          "pipeline from perception using the existing session")
                resume_after_pose_adjust = True
                continue
            results.append(tr)
            n_succ = sum(1 for r in results if r.get("success"))
            print(f"\n    Running total: {n_succ}/{len(results)} success")

            if tr.get("fatal_cuda_planning_fault"):
                print("\n    CUDA planning fault: stopping this process after "
                      "hardware cleanup; restart before another trial.")
                break

            if tr.get("manual_recovery_required"):
                print("\n    Lift preflight failed after grasp; the object remains "
                      "held. Stopping the automatic loop for manual recovery.")
                break

            # After-trial coverage summary. Show per-tabletop
            # remaining uncovered count + how many scenes this trial just
            # covered (delta vs before).
            if (_is_coverage_pool(args.grasp_version)
                    and not args.ignore_coverage
                    and tr.get("grasp_success", tr.get("success"))):
                from autodex.utils.coverage import uncovered_scenes
                lines = []
                total_now = 0
                total_before = 0
                _sets_after = {}
                for _s, _b in _rem_before.items():
                    _u = uncovered_scenes(args.obj, _s, hand=args.hand,
                                          version=args.grasp_version,
                                          success_root=campaign_state_root)
                    _n = (len(_u) if _u is not None else None)
                    _sets_after[_s] = (sorted(int(v) for v in _u)
                                       if _u is not None else None)
                    if _b is None or _n is None:
                        lines.append(f"      pose={_s}: N/A")
                        continue
                    _delta = _b - _n
                    lines.append(f"      pose={_s}: {_n} uncovered "
                                 f"(was {_b}, -{_delta} this trial)")
                    total_now += _n
                    total_before += _b
                print(f"    [coverage] after trial (success):")
                for ln in lines:
                    print(ln)
                print(f"      TOTAL remaining: {total_now} "
                      f"(was {total_before}, "
                      f"-{total_before - total_now} this trial)")
                coverage_ref = pipeline_trace.write_artifact_json(
                    f"artifacts/coverage/{attempt_id}_after.json",
                    {"attempt_id": attempt_id, "object": args.obj,
                     "uncovered_scene_ids_by_tabletop": _sets_after,
                     "remaining_before": _rem_before,
                     "remaining_after": {
                         key: (len(value) if value is not None else None)
                         for key, value in _sets_after.items()},
                    },
                )
                pipeline_trace.scoped(
                    episode_id=tr.get("dir_idx"),
                    attempt_id=attempt_id,
                ).event(
                    "coverage.snapshot_after", phase="coverage",
                    kind="result",
                    attributes={"artifact": coverage_ref,
                                "newly_covered_total": total_before - total_now},
                )

            _write_experiment_coverage_progress(
                args, args.hand, args.obj, get_obj_root(args.grasp_version))

            if (tr.get("all_done") and
                    (not args.ignore_coverage
                     or tr.get("reason") == "no_candidates")):
                print(f"\n    No further trial is available for {args.obj} — "
                      "stopping loop.")
                break
            if args.max_trials and len(results) >= args.max_trials:
                print(f"\n    --max_trials {args.max_trials} reached — stopping loop.")
                break
    finally:
        cleanup_span = pipeline_trace.begin(
            phase="cleanup", kind="setup", name="hardware_cleanup")
        if not startup_cancelled:
            try:
                _emit_external_sync_cue(
                    sync_generator, pipeline_trace, label="pipeline_end",
                    enabled=args.external_sync_cue,
                    duration_s=args.external_sync_cue_duration_s,
                    fps=args.external_sync_cue_fps,
                )
            except Exception as cue_exc:
                print(f"[sync] external end cue failed: {cue_exc!r}")
        # Summary + cleanup.
        print(f"\n{'='*60}\nSUMMARY: {args.obj} x {len(results)} trials")
        n_succ = sum(1 for r in results if r.get("success"))
        if results:
            print(f"  Success: {n_succ}/{len(results)} ({100*n_succ/len(results):.0f}%)")
        for r in results:
            status = "OK" if r.get("success") else r.get("reason", "FAIL")
            print(f"  {r['dir_idx']}: {status}")

        sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
        summary_path = os.path.join(project_dir, "experiment", args.exp_name, sub,
                                    args.obj, "summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(_pipeline_result_value(results), f, indent=2)

        try:
            executor.shutdown()
        except Exception:
            pass
        # TimestampMonitor owns a Spinnaker camera reference independent of
        # FoundPose's orchestrator.  Release it before orch.close() clears the
        # camera system; reversing this order produces Spinnaker -1004 after
        # an otherwise handled planning failure.
        for fn in (timestamp_monitor.end, sync_generator.end, rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass
        try:
            orch.close()
        except Exception:
            pass
        pipeline_trace.end(cleanup_span, outcome="success")
        if owns_pipeline_trace:
            pipeline_trace.close(outcome="success")


if __name__ == "__main__":
    main()
