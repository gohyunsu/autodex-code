#!/usr/bin/env python3
"""Integrated run_auto pipeline with in-process pose-recovery actions.

This entry point intentionally reuses :mod:`src.execution.run_auto` for the
normal trial lifecycle: calibration, one camera controller, FoundPose
orchestrator, CUDA planner, executor, execution, labelling, coverage, and
cleanup are unchanged. Its only behavioural difference is that two recovery
branches reuse those already-live resources:

    exhausted/no-candidate/failed-plan tabletop -> in-process reset reorient
    -> fresh perception -> normal run_auto trial
    failed plan -> search (x, yaw) -> in-process rotate -> fresh perception
    -> normal run_auto trial

Both recoveries use the failed trial's perception instead of starting another
process or redoing hardware/FoundPose/planner initialisation. Perception is
deliberately repeated *after* the physical placement; that is a new scene
observation used by the normal trial, not duplicated setup work.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ``python src/execution/run_pipeline.py`` puts only this directory on
# sys.path. Match the existing execution scripts so direct CLI invocation can
# import the ``src`` package as well as project modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.execution import run_auto
from src.execution.rotate_obj_yaw import rotate_from_live_scene
from src.execution.scene_cfg import pose_world_to_scene_cfg
from autodex.planner.obstacles import add_obstacles
from autodex.utils.robot_config import CHARUCO_BOARD_11_CENTER_XY
from src.experiment.reset.reorient import reorient_from_live_scene


def _add_timing(info: dict | None, **values: float) -> dict | None:
    """Attach elapsed times to a recovery result without changing its contract."""
    if info is None:
        return None
    result = dict(info)
    timing = dict(result.get("timing_s") or {})
    timing.update({key: round(float(value), 3) for key, value in values.items()})
    result["timing_s"] = timing
    return result


def _rotation_recovery_priority(context: dict) -> dict:
    """Match standalone ``rotate_obj_yaw`` ordering without filtering.

    A completed candidate is excluded from normal collection because it adds
    no coverage.  It is still a valuable, previously verified way to acquire
    the object for a recovery move.  The standalone rotate CLI reloads the
    complete current-tabletop pool and merely boosts such candidates; keep the
    in-process path on that same contract.
    """
    from autodex.utils.coverage import (
        _disk_success_keys,
        experiment_candidate_state_root,
        load_coverage_map,
    )

    obj = context["obj"]
    hand = context["hand"]
    grasp_version = context["grasp_version"]
    arm = context["args"].arm
    tabletop_pose_stem = context["tabletop_pose_stem"]
    state_root = None
    if getattr(context["args"], "isolate_experiment", False):
        state_root = experiment_candidate_state_root(
            context["args"].exp_name, hand, grasp_version, obj)
    success_keys = _disk_success_keys(
        obj, hand, grasp_version, arm=arm, success_root=state_root)
    coverage = load_coverage_map(
        obj, tabletop_pose_stem=tabletop_pose_stem,
        hand=hand, version=grasp_version, arm=arm,
        success_root=state_root) or {}
    priority = {
        key: (1000 if key in success_keys else 0) + coverage.get(key, 0)
        for key in set(coverage) | set(success_keys)
    }
    if success_keys:
        scope = "experiment" if state_root is not None else "shared v8"
        print(f"[pipeline] rotation recovery: {len(success_keys)} {scope} "
              "success grasps restored and prioritized")
    return priority


def _charuco_preflight(**context):
    """Optionally measure board 11 before an object is placed on it.

    This callback is invoked by ``run_auto.main`` after clear-view and before
    the trial loop.  The returned dict is a session-only tabletop geometry;
    returning ``None`` is an explicit operator skip, while ``{"cancel": True}``
    stops before any object perception or robot interaction.
    """
    args = context["args"]
    mode = args.charuco_preflight
    if mode == "skip":
        print("[charuco-preflight] skipped — using fixed fallback tabletop geometry")
        return None

    if mode == "prompt" and args.auto:
        # ``run_auto.main`` presents its one-time "place object" gate after
        # this hook returns.  Therefore an automatic run can measure the
        # currently empty board here without any input, then wait exactly once
        # for the operator to put the object down.
        print("[charuco-preflight] --auto: measuring the empty board before "
              "the one-time object-placement gate")
    elif mode == "prompt":
        try:
            choice = input(
                "[charuco-preflight] Clear board 11. "
                "Press Enter to measure, 's' to skip, 'q' to quit: "
            ).strip().lower()
        except KeyboardInterrupt:
            choice = "q"
        if choice == "s":
            print("[charuco-preflight] skipped — using fixed fallback tabletop geometry")
            return None
        if choice == "q":
            return {"cancel": True}

    from paradex.calibration.utils import load_current_C2R
    from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator
    from src.execution.charuco_tabletop import (
        measure_tabletop_from_images,
        save_tabletop_measurement,
    )

    sub = (f"{context['scene_prefix']}/{args.hand}"
           if context["scene_prefix"] else args.hand)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preflight_dir = (
        Path(run_auto.project_dir) / "experiment" / args.exp_name / sub /
        args.obj / f"_charuco_preflight_{stamp}"
    )
    image_dir = preflight_dir / "images"
    c2r = np.asarray(load_current_C2R(), dtype=np.float64)
    active_preflight_s = 0.0

    while True:
        snap = None
        try:
            t_attempt = time.perf_counter()
            snap = SnapshotOrchestrator(
                pc_list=args.pc_list, capture_ips=context["capture_ips"],
                port_snap=args.port_snap, port_cmd=args.port_snap_cmd,
            )
            payloads, snap_timing = snap.snap(
                n_expected=context["n_cameras"], timeout_s=5.0,
                save_dir_local=str(image_dir), decode=True,
            )
            snapshot_s = time.perf_counter() - t_attempt
            images = {
                serial: item["image"] for serial, item in payloads.items()
                if item.get("image") is not None
            }
            print(f"[charuco-preflight] snapshot "
                  f"{len(images)}/{context['n_cameras']} cameras")
            if len(images) != context["n_cameras"]:
                raise RuntimeError(
                    "empty-board preflight requires one decodable snapshot from "
                    f"every active camera ({len(images)}/{context['n_cameras']} received)")
            t_measurement = time.perf_counter()
            geometry = measure_tabletop_from_images(
                images, context["intrinsics_full"], context["extrinsics_full"], c2r)
            measurement_s = time.perf_counter() - t_measurement
            attempt_s = time.perf_counter() - t_attempt
            geometry["snapshot_timing"] = snap_timing
            # Keep this in the session geometry: run_auto copies that geometry
            # into each trial result, while the saved measurement remains a
            # standalone audit record for this preflight.
            geometry["timing_s"] = {
                "snapshot_s": round(snapshot_s, 3),
                "measurement_s": round(measurement_s, 3),
                "attempt_s": round(attempt_s, 3),
                "active_total_s": round(active_preflight_s + attempt_s, 3),
            }
            geometry["measurement_dir"] = str(preflight_dir)
            np.save(preflight_dir / "C2R.npy", c2r)
            saved = save_tabletop_measurement(geometry, preflight_dir)
            active_preflight_s += attempt_s
            m = geometry["metrics"]
            c = geometry["center_robot_m"]
            print(f"[charuco-preflight] board=11 corners="
                  f"{m['corners_triangulated']}/{m['corners_expected']} "
                  f"cams={m['cameras_used']}")
            print(f"[charuco-preflight] center=({c[0]:.3f}, {c[1]:.3f}, "
                  f"{c[2]:.3f})m plane rms/max="
                  f"{m['plane_rms_mm']:.2f}/{m['plane_max_mm']:.2f}mm")
            print("[charuco-preflight] level table surface for this session: "
                  f"z={geometry['table_surface_z_m']:.4f}m")
            print(f"[charuco-preflight] accepted; geometry -> {saved}")
            if args.auto:
                print("[charuco-preflight] --auto: starting trials without "
                      "operator confirmation")
                return geometry
            try:
                after = input(
                    f"Place {args.obj} on the board, then press Enter to start trials "
                    "('q' to quit): "
                ).strip().lower()
            except KeyboardInterrupt:
                after = "q"
            if after == "q":
                return {"cancel": True}
            return geometry
        except Exception as exc:
            # The failed attempt has no geometry to persist, but include it in
            # the accepted retry's active total so the operator can see the
            # real sensing/measurement cost (excluding prompt wait time).
            if 't_attempt' in locals():
                active_preflight_s += time.perf_counter() - t_attempt
            print(f"[charuco-preflight] rejected: {exc}")
            if args.auto:
                print("[charuco-preflight] --auto: cannot request a manual "
                      "retry; ending before robot motion")
                return {"cancel": True}
            try:
                retry = input("Fix the empty board, then Enter=retry, "
                              "'s'=skip, 'q'=quit: ").strip().lower()
            except KeyboardInterrupt:
                retry = "q"
            if retry == "s":
                print("[charuco-preflight] skipped — using fixed fallback tabletop geometry")
                return None
            if retry == "q":
                return {"cancel": True}
        finally:
            if snap is not None:
                try:
                    snap.close()
                except Exception:
                    pass


def _capture_running_state(rcc) -> bool | None:
    """Return aggregate camera-acquisition state, or ``None`` if unknown.

    ``rcc.arm()`` is not idempotent at the remote daemons: calling it while
    their acquisition is already running makes every camera report
    ``Acquisition is already running`` and latches a camera error.  An
    in-process recovery therefore needs a real status check before deciding
    whether it is allowed to arm again.
    """
    try:
        pcs = rcc.get_status().get("pc") or {}
    except Exception as exc:
        print(f"[pipeline] camera state unavailable: {exc!r}")
        return None
    if not pcs:
        print("[pipeline] camera state unavailable: no PC status returned")
        return None
    states = [bool(status.get("running")) for status in pcs.values()]
    if all(states):
        return True
    if not any(states):
        return False
    # Re-arming a mixed state would duplicate acquisition only on the PCs
    # that are still live.  Treat it as unknown and leave an explicit failure
    # for the operator instead of turning a partial outage into 20 errors.
    print(f"[pipeline] camera state unavailable: mixed running states "
          f"({sum(states)}/{len(states)} PCs live)")
    return None


def _restart_stream_or_mark_failure(rcc, args, info: dict, recovery: str) -> dict:
    """Restore the one shared camera controller without duplicate acquire.

    Planning-only rotate/reorient failures keep the original stream running.
    In that case only re-enable its stream *sink*; a physical recovery has
    stopped acquisition and needs the usual arm + sink sequence.  For legacy
    or exception results without an explicit stop flag, daemon status is the
    conservative source of truth.
    """
    info = dict(info)
    capture_stopped = bool(info.get("camera_capture_stopped", False))
    try:
        if capture_stopped:
            print(f"[pipeline] {recovery}: restarting stopped camera acquisition")
            run_auto._rcc_start(rcc, "stream", False, fps=args.stream_fps)
            if args.stream_warmup_s > 0:
                time.sleep(args.stream_warmup_s)
            return info

        running = _capture_running_state(rcc)
        if running is True:
            # ``set_stream`` only changes the SHM/output sink.  Crucially it
            # does not send a second acquire command to already-running cams.
            print(f"[pipeline] {recovery}: acquisition already live; "
                  "restoring stream sink only")
            rcc.set_stream(True)
            return info
        if running is False:
            print(f"[pipeline] {recovery}: acquisition inactive; restarting it")
            run_auto._rcc_start(rcc, "stream", False, fps=args.stream_fps)
            if args.stream_warmup_s > 0:
                time.sleep(args.stream_warmup_s)
            return info

        # Do not guess and send ``arm`` without a daemon status: that is the
        # exact unsafe retry which caused the visible camera errors.
        raise RuntimeError("camera acquisition state is unknown; refusing to re-arm")
    except Exception as stream_exc:
        info["success"] = False
        info["reason"] = f"{recovery}_stream_restart_failed"
        info["stream_exception"] = repr(stream_exc)
        return info


def _rotate_in_process(**context) -> dict:
    """Run recovery with the standalone rotate candidate-pool semantics."""
    args = context["args"]
    rcc = context["rcc"]
    info = None
    t_total = time.perf_counter()
    t_priority = time.perf_counter()
    priority_s = 0.0
    recovery_s = 0.0
    stream_restore_s = 0.0
    try:
        # Do NOT inherit normal collection's coverage whitelist or completed-
        # scene filters.  They answer "what is left to collect?", whereas a
        # rotation answers "what can safely acquire this object now?".  The
        # latter must revisit every current-tabletop grasp, with the same
        # success-first ranking used by standalone rotate_obj_yaw.py.
        recovery_priority = _rotation_recovery_priority(context)
        priority_s = time.perf_counter() - t_priority
        t_recovery = time.perf_counter()
        info = rotate_from_live_scene(
            obj=context["obj"], hand=context["hand"], arm=args.arm,
            grasp_version=context["grasp_version"],
            planner=context["planner"], executor=context["executor"],
            scene_cfg=context["scene_cfg"],
            target_x=context["target_x"],
            target_y=context.get("target_y", float(CHARUCO_BOARD_11_CENTER_XY[1])),
            target_yaw_deg=context["target_yaw_deg"],
            tabletop_pose_stem=context["tabletop_pose_stem"],
            candidate_order=None,
            priority_map=recovery_priority,
            scene_type_filter=None,
            scene_id=None,
            success_only=False,
            skip_done=False,
            skip_scenes_with_success=False,
            cyl_axis_local=context["cyl_axis_local"],
            cyl_yaw_grid=context["cyl_yaw_grid"],
            rcc=rcc,
        )
        recovery_s = time.perf_counter() - t_recovery
    finally:
        # A planning-only rejection leaves capture live; an executed recovery
        # stops it.  The helper selects sink-only vs arm+stream accordingly.
        t_stream_restore = time.perf_counter()
        if info is not None:
            info = _restart_stream_or_mark_failure(rcc, args, info, "rotation")
        else:
            # Let run_auto's callback wrapper record the original exception,
            # but still make a best effort to leave cameras ready for an
            # operator-driven retry.
            _restart_stream_or_mark_failure(
                rcc, args, {"success": False}, "rotation")
        stream_restore_s = time.perf_counter() - t_stream_restore
        info = _add_timing(
            info,
            candidate_priority_s=priority_s,
            recovery_s=recovery_s,
            stream_restore_s=stream_restore_s,
            total_s=time.perf_counter() - t_total,
        )
    return info


def _reorient_in_process(**context) -> dict:
    """Adapter that reuses run_auto's perception, hardware and planner.

    The standalone reset script owns a full camera/FoundPose/planner/executor
    lifecycle.  Here only its reset-candidate planning and physical motion are
    invoked; the next ordinary run_auto trial supplies the one necessary fresh
    perception after the object has been placed.
    """
    args = context["args"]
    rcc = context["rcc"]
    t_total = time.perf_counter()
    sub = (f"{context['scene_prefix']}/{context['hand']}"
           if context["scene_prefix"] else context["hand"])
    lift_rel = os.path.join(
        "shared_data", "AutoDex", "experiment", args.exp_name, sub,
        context["obj"], os.path.basename(context["img_dir"]),
        "reorient_lift_check", "raw",
    )
    lift_abs = os.path.join(
        context["img_dir"], "reorient_lift_check", "raw", "images")
    # Reset candidates are table transitions.  Do not inherit wall/shelf/
    # clutter obstacles from the failed collection scene; that would make this
    # recovery differ from the retained standalone reorient policy.
    t_scene_build = time.perf_counter()
    table_scene = pose_world_to_scene_cfg(
        context["pose_world"], context["c2r"], context["obj"],
        context["obj_root"], tabletop_geometry=context.get("tabletop_geometry"),
    )
    table_scene = add_obstacles(
        table_scene, "table", tabletop_geometry=context.get("tabletop_geometry"))
    scene_build_s = time.perf_counter() - t_scene_build

    info = None
    recovery_s = 0.0
    try:
        t_recovery = time.perf_counter()
        info = reorient_from_live_scene(
            obj=context["obj"], hand=context["hand"], arm=args.arm,
            target_j=context["target_j"], planner=context["planner"],
            executor=context["executor"], rcc=rcc, scene_cfg=table_scene,
            obj_root=context["obj_root"], grasp_version=args.grasp_version,
            lift_label_rel=lift_rel,
            lift_label_abs=lift_abs,
            tabletop_geometry=context.get("tabletop_geometry"),
            debug_dump_dir=os.path.join(context["img_dir"], "planning_debug"),
        )
        recovery_s = time.perf_counter() - t_recovery
    finally:
        # As above, reorient can return before it has stopped capture while
        # evaluating reset candidates, so recovery must be state-aware.
        t_stream_restore = time.perf_counter()
        if info is not None:
            info = _restart_stream_or_mark_failure(rcc, args, info, "reorient")
        else:
            _restart_stream_or_mark_failure(
                rcc, args, {"success": False}, "reorient")
        info = _add_timing(
            info,
            scene_build_s=scene_build_s,
            recovery_s=recovery_s,
            stream_restore_s=time.perf_counter() - t_stream_restore,
            total_s=time.perf_counter() - t_total,
        )
    return info


def main() -> None:
    run_auto.main(
        pose_adjust_handler=_rotate_in_process,
        reorient_handler=_reorient_in_process,
        startup_handler=_charuco_preflight,
    )


if __name__ == "__main__":
    main()
