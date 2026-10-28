"""One explicit AutoDex-PC integration contract for the live demo CLI."""

from __future__ import annotations

from concurrent.futures import Future
import hashlib
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.machinery import PathFinder
import math
from pathlib import Path
from typing import Callable, Sequence

from .clear_view import ClearViewForPerception
from .live_orchestration import (
    LivePrecisionSession, compose_live_handlers, run_live_precision_session,
)
from .action_loop import ActionLoopResult
from .labeling import LABEL_MODES
from .session_launch import load_session_launch
from .live_session_runner import preflight_endpoint_catalog_file
from .timing_profile import timed


@dataclass(frozen=True)
class RunLiveOptions:
    launch_config_path: Path
    endpoint_catalog_path: Path
    run_output_dir: Path
    max_xy_retries: int | None = None
    max_steps: int | None = None
    enable_robot_motion: bool = False
    robot_config_path: Path | None = None
    label_mode: str = "vlm_only"
    xy_estimator: str = "vlm"
    annotator: str | None = None
    xy_update_weight: float = 1.0
    same_trial_xy_tolerance_m: float | None = None
    init_ports: tuple[int, int, int] | None = None
    stop_after_planning: bool = False
    reuse_session_evidence_dir: Path | None = None
    key_already_placed: bool = False
    initial_preferred_candidate: tuple[str, str, str] | None = None
    launch_snapshot: object | None = None


@dataclass(frozen=True)
class LiveRuntimeIntegration:
    """Robot-PC owned objects; the demo owns policy and session sequencing.

    ``build_components`` must return the ten live handler components bound to
    the supplied live session. ``close`` releases robot-side resources after
    success, review stop, or an exception. The same camera context is owned
    by ``run_live_precision_session`` and is closed there.
    """

    clear_view: ClearViewForPerception
    build_components: Callable[[LivePrecisionSession], Sequence[object]]
    close: Callable[[], None]
    begin_fixed_world_warmup: Callable[[LivePrecisionSession], Future] | None = None
    key_capture_overlaps_warmup: bool = False


def _wait_for_fixed_world_warmup(
    handlers: dict, future: Future, *, key_capture_overlaps_warmup: bool = False,
) -> dict:
    """Join planner warmup before use; optionally overlap fresh key capture."""
    result = dict(handlers)
    for action in result:
        if (key_capture_overlaps_warmup and action in
                {"capture_fresh_key", "reobserve_key_and_preflight"}):
            # These handlers join the future at their own capture→planning
            # boundary, allowing capture4 perception to overlap cuRobo warmup.
            continue
        original = result[action]

        def after_warmup(runner, decision, *, _original=original):
            with timed("startup.planner_fixed_world_warmup_wait",
                       phase="startup", kind="wait"):
                future.result()
            return _original(runner, decision)

        result[action] = after_warmup
    return result


def _confirm_initial_key_placement() -> bool:
    """Match stock AutoDex's one initial object-placement wait."""
    try:
        response = input(
            "[precision-insertion] Socket captured. Place the white key "
            "without moving the socket, then press Enter (q to quit): ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() not in {"q", "quit"}


def _stage_module_on_path(specification: str) -> bool:
    """Check a stage module's path without importing robot-PC code.

    Static preflight must never execute a module that could open FR3 or write
    Inspire registers. PathFinder resolves ordinary filesystem modules and
    packages without running their module bodies; custom importers can still
    be resolved by the actual runtime import.
    """
    module = specification.split(":", 1)[0]
    search_path = None
    for part in module.split("."):
        found = PathFinder.find_spec(part, search_path)
        if found is None:
            return False
        search_path = found.submodule_search_locations
    return True


def inspect_live_inputs(
    *, launch_config_path: Path, endpoint_catalog_path: Path,
    robot_config_path: Path | None = None,
    xy_estimator: str = "vlm",
) -> dict:
    """Check current v8 inputs without cameras, CUDA or robot construction."""
    if xy_estimator not in {"vlm", "tracked_tip", "sam_half_ellipse"}:
        raise ValueError("unknown failed-trial XY estimator")
    launch = load_session_launch(
        launch_config_path, allow_existing_evidence=True)
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    catalog_path = Path(endpoint_catalog_path).expanduser().resolve()
    catalog = preflight_endpoint_catalog_file(
        mode=launch.mode, shared_root=launch.shared_root,
        catalog_path=catalog_path)
    pose_counts: dict[str, dict[str, int]] = {}
    for row in catalog["candidates"]:
        stem = row.get("tabletop_pose_stem")
        if not isinstance(stem, str):
            continue
        counts = pose_counts.setdefault(stem, {"catalogued": 0, "eligible": 0})
        counts["catalogued"] += 1
        counts["eligible"] += row.get("eligible") is True
    result = {
        "schema": "precision_insertion_static_live_preflight_v2",
        "launch": launch.readiness(),
        "launch_config_path": str(Path(launch_config_path).expanduser().resolve()),
        "launch_config_sha256": hashlib.sha256(
            Path(launch_config_path).expanduser().resolve().read_bytes()).hexdigest(),
        "launch_evidence_dir_path": str(launch.evidence_dir),
        "launch_evidence_dir_exists": launch.evidence_dir.exists(),
        "endpoint_catalog_path": str(catalog_path),
        "endpoint_catalog_sha256": hashlib.sha256(
            catalog_path.read_bytes()).hexdigest(),
        "candidate_count": len(catalog["candidates"]),
        "eligible_candidate_count": sum(
            row.get("eligible") is True for row in catalog["candidates"]),
        "candidates_by_tabletop_pose": dict(sorted(pose_counts.items())),
        "direct_preflight_pose_stems": [
            stem for stem, counts in sorted(pose_counts.items())
            if launch.mode.family == "cylinder" and
            counts["eligible"] > 0 and (
                not getattr(launch, "upright_cylinder_only", True) or
                stem == "000")],
        "robot_config": None,
        "xy_estimator": xy_estimator,
        "sam_roi_inputs": None,
        "scope": "static_files_only_no_camera_cuda_or_robot_connection",
        "robot_ready": False,
    }
    if xy_estimator == "sam_half_ellipse":
        from .franka_integration import _sam_runtime_paths

        python, checkpoint = _sam_runtime_paths()
        result["sam_roi_inputs"] = {
            "python": str(python),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "scope": "file_presence_only_no_cuda_model_load",
        }
    if robot_config_path is not None:
        from .franka_integration import load_franka_integration_config

        parsed = load_franka_integration_config(
            robot_config_path, import_stages=False)
        source = parsed["source"]
        result["robot_config"] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "stages_factory": parsed["record"]["stages_factory"],
            "stage_factory_syntax_valid": True,
            "stage_factory_module_on_path": _stage_module_on_path(
                parsed["record"]["stages_factory"]),
            "stage_factory_imported": False,
            "exposure_sampler_configured": parsed["sampling"] is not None,
            "held_motion_monitor_configured": (
                parsed["held_motion_monitor"] is not None),
        }
    return result


def load_runtime_integration(
    specification: str, options: RunLiveOptions,
) -> LiveRuntimeIntegration:
    """Load only an explicitly selected ``module:factory`` from PYTHONPATH."""
    if (not isinstance(specification, str) or
            specification.count(":") != 1):
        raise ValueError("integration must be a Python module:factory")
    module_name, factory_name = specification.split(":", 1)
    if (not module_name or not factory_name or
            not all(part.isidentifier() for part in module_name.split(".")) or
            not factory_name.isidentifier()):
        raise ValueError("integration needs a valid module and factory name")
    with timed("startup.integration_module_import", phase="startup"):
        factory = getattr(import_module(module_name), factory_name)
    if not callable(factory):
        raise TypeError("integration factory is not callable")
    with timed("startup.integration_construct", phase="startup"):
        integration = factory(options)
    if (not isinstance(integration, LiveRuntimeIntegration) or
            not isinstance(integration.clear_view, ClearViewForPerception) or
            not callable(integration.build_components) or
            not callable(integration.close) or
            (integration.begin_fixed_world_warmup is not None and
             not callable(integration.begin_fixed_world_warmup))):
        raise TypeError("factory must return LiveRuntimeIntegration")
    return integration


def run_integrated_live_session(
    options: RunLiveOptions, *, integration_spec: str,
) -> ActionLoopResult:
    """Start the one-session AutoDex v8 loop with complete demo handlers."""
    if not isinstance(options, RunLiveOptions):
        raise TypeError("live run needs RunLiveOptions")
    if options.enable_robot_motion is not True:
        raise PermissionError("live session requires explicit robot-motion opt-in")
    if options.max_xy_retries is not None and (
            type(options.max_xy_retries) is not int or
            options.max_xy_retries < 0):
        raise ValueError("max_xy_retries must be a nonnegative integer or None")
    if (not math.isfinite(float(options.xy_update_weight)) or
            options.xy_update_weight <= 0):
        raise ValueError("XY update weight must be finite and positive")
    if options.max_steps is not None and (
            type(options.max_steps) is not int or options.max_steps < 1):
        raise ValueError("max_steps must be positive or None")
    if options.label_mode not in LABEL_MODES or (
            options.label_mode == "manual" and not options.annotator):
        raise ValueError("manual labeling needs a named annotator")
    if options.label_mode == "evidence_fused":
        raise ValueError("current cylinder live loop needs vlm_only or manual labels")
    if options.xy_estimator not in {"vlm", "tracked_tip", "sam_half_ellipse"}:
        raise ValueError("unknown next-trial XY estimator")
    if options.same_trial_xy_tolerance_m is not None and (
            not math.isfinite(options.same_trial_xy_tolerance_m) or
            options.same_trial_xy_tolerance_m <= 0):
        raise ValueError("same-trial XY tolerance must be positive metres")
    if (options.same_trial_xy_tolerance_m is not None and
            options.xy_estimator == "tracked_tip"):
        raise ValueError(
            "same-trial XY currently uses vlm or sam_half_ellipse")
    with timed("startup.launch_config", phase="startup", kind="io"):
        launch = load_session_launch(options.launch_config_path)
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    if getattr(launch, "execution_scope", "live") != "live":
        raise PermissionError("static preflight launch cannot start robot motion")
    with timed("startup.integration_factory", phase="startup"):
        integration = load_runtime_integration(
            integration_spec, replace(options, launch_snapshot=launch))
    try:
        def handlers(live: LivePrecisionSession):
            with timed("startup.build_handlers", phase="startup"):
                components = tuple(integration.build_components(live))
                joined = compose_live_handlers(live, *components)
            if integration.begin_fixed_world_warmup is not None:
                future = integration.begin_fixed_world_warmup(live)
                joined = _wait_for_fixed_world_warmup(
                    joined, future,
                    key_capture_overlaps_warmup=(
                        integration.key_capture_overlaps_warmup))
            return joined

        with timed("live.orchestration", phase="session"):
            return run_live_precision_session(
                launch_config_path=options.launch_config_path,
                endpoint_catalog_path=options.endpoint_catalog_path,
                run_output_dir=options.run_output_dir,
                max_xy_retries=options.max_xy_retries,
                xy_update_weight=options.xy_update_weight,
                clear_view=integration.clear_view,
                build_handlers=handlers, max_steps=options.max_steps,
                stop_after_planning=options.stop_after_planning,
                reuse_session_evidence_dir=options.reuse_session_evidence_dir,
                confirm_initial_key_placement=(
                    None if options.key_already_placed else
                    _confirm_initial_key_placement),
                init_ports=options.init_ports, launch_snapshot=launch,
                initial_preferred_candidate=options.initial_preferred_candidate)
    finally:
        with timed("cleanup.integration_close", phase="cleanup", kind="io"):
            integration.close()
