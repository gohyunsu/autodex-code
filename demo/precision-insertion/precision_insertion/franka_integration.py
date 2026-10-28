"""Built-in FR3 resource owner for the demo's live integration contract.

The robot PC contributes only stage-specific adapters and commissioned limits
through a Python factory. AutoDex's FR3 executor, planner, measured state and
clear-view controller come from :mod:`franka_core`; the existing stage classes
continue to own all task decisions and execution checkpoints.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import copy
from dataclasses import replace
from importlib import import_module
import json
import math
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from .checkpoint_camera import make_checkpoint_capture
from .curobo_compat import install_curobo_planner_compat
from .franka_core import (
    FrankaFeedbackSettings, FrankaSamplingSettings, open_franka_core,
)
from .franka_measured_lift import (
    FrankaMeasuredLiftAdapter, MeasuredLiftMonitorLimits,
)
from .stock_force_insertion import FrankaForceStopInsertionAdapter
from .stock_failed_return import StockFailedReturnRecorder
from .stock_repose_execution import StockReposeExecutor
from .stock_socket_release import StockSocketRelease
from .live_runtime import LiveRuntimeIntegration, RunLiveOptions
from .labeling import terminal_label_provider
from .distributed_qwen import DistributedQwenBackend
from .observer import load_vlm_backend
from .runtime_assembly import assemble_live_components
from .timing_profile import timed
from .raw_camera_capture import RawCameraCapture
from .sam_roi_backend import SamRoiBackend
from .tip_frame_window import fetch_tip_frame_window
from .visual_views import assessment_cameras


_SCHEMA = "precision_insertion_franka_integration_v1"
_FEEDBACK_KEYS = frozenset(FrankaFeedbackSettings.__dataclass_fields__)
_OBSOLETE_FEEDBACK_KEYS = frozenset({"max_arm_update_wait_s"})
_SAMPLING_KEYS = frozenset(FrankaSamplingSettings.__dataclass_fields__)
_VLM_KEYS = frozenset({"mode", "model_id", "max_new_tokens",
                       "require_native_pixels", "capture_timeout_s",
                       "parallel_views", "classification_image_size",
                       "worker_root", "workers"})
_LIFT_MONITOR_KEYS = frozenset({
    "daemon_binary_path", "max_state_gap_s", "minimum_samples",
})
_CHECKPOINT_CAPTURES = {
    "lift": {"capture_after_lift": "after_lift"},
    "task": {"capture_preinsert": "preinsert",
             "capture_final": "final_clear_view"},
}
_TIP_CAMERA_ID = "25322651"
_TIP_CAMERA_HOST = "capture2"
_TIP_CAMERA_PORT = 16951


def _sam_runtime_paths() -> tuple[Path, Path]:
    """Resolve optional SAM inputs before constructing a command-capable FR3 core."""
    python = Path(os.environ.get(
        "PRECISION_SAM_PYTHON",
        "/home/robot/anaconda3/envs/autodex/bin/python")).expanduser()
    checkpoint = Path(os.environ.get(
        "PRECISION_SAM_CHECKPOINT",
        "/home/robot/shared_data/AutoDex/weights/sam3/sam3.pt"
    )).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"SAM3 Python is missing or not executable: {python}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint is missing: {checkpoint}")
    return python.resolve(), checkpoint.resolve()


def _vlm_settings(value: object) -> dict:
    if value is None:
        return {"mode": "local", "model_id": None,
                "max_new_tokens": 128, "require_native_pixels": True,
                "capture_timeout_s": 10.0, "parallel_views": 1,
                "classification_image_size": [384, 288]}
    if not isinstance(value, dict) or set(value) - _VLM_KEYS:
        raise ValueError("VLM settings contain unknown fields")
    settings = {**_vlm_settings(None), **value}
    if (settings["mode"] not in {"local", "gemini", "distributed"} or
            settings["model_id"] is not None and
            (not isinstance(settings["model_id"], str) or
             not settings["model_id"].strip()) or
            type(settings["max_new_tokens"]) is not int or
            settings["max_new_tokens"] <= 0 or
            type(settings["require_native_pixels"]) is not bool or
            type(settings["capture_timeout_s"]) not in (int, float) or
            not math.isfinite(settings["capture_timeout_s"]) or
            settings["capture_timeout_s"] <= 0 or
            type(settings["parallel_views"]) is not int or
            not 1 <= settings["parallel_views"] <= 8 or
            settings["mode"] == "local" and settings["parallel_views"] != 1 or
            settings["mode"] == "distributed" and (
                not isinstance(settings.get("worker_root"), str) or
                not Path(settings["worker_root"]).is_absolute() or
                not isinstance(settings.get("workers"), dict) or
                len(settings["workers"]) != 5 or
                any(not isinstance(camera, str) or
                    not isinstance(spec, dict) or
                    set(spec) != {"host", "python", "model_dir"} or
                    not isinstance(spec["host"], str) or
                    not isinstance(spec["python"], str) or
                    not isinstance(spec["model_dir"], str) or
                    not Path(spec["model_dir"]).is_absolute()
                    for camera, spec in settings["workers"].items())) or
            settings["classification_image_size"] is not None and (
                not isinstance(settings["classification_image_size"], (list, tuple)) or
                len(settings["classification_image_size"]) != 2 or
                any(type(side) is not int or side <= 0
                    for side in settings["classification_image_size"])) or
            settings["mode"] == "gemini" and settings["model_id"] is None):
        raise ValueError("invalid VLM mode, model or checkpoint timeout")
    return settings


def _with_checkpoint_captures(
    stages: Mapping, timeout_s: float, *, family: str | None = None,
) -> dict:
    """Reuse one open camera session; every callback still takes fresh frames."""
    result = {name: dict(fields) for name, fields in stages.items()}
    for stage_name, configured in _CHECKPOINT_CAPTURES.items():
        if stage_name not in result:
            continue
        captures = configured
        for field, phase in captures.items():
            if field not in result[stage_name]:
                result[stage_name][field] = make_checkpoint_capture(
                    phase=phase, timeout_s=timeout_s)
    return result


def _validate_stage_factory_spec(specification: str) -> tuple[str, str]:
    if (not isinstance(specification, str) or
            specification.count(":") != 1):
        raise ValueError("stage factory must be an importable module:function")
    module_name, function_name = specification.split(":", 1)
    if (not module_name or not function_name or
            not all(part.isidentifier() for part in module_name.split(".")) or
            not function_name.isidentifier()):
        raise ValueError("stage factory has an invalid module or function name")
    return module_name, function_name


def _stage_factory(specification: str):
    module_name, function_name = _validate_stage_factory_spec(specification)
    result = getattr(import_module(module_name), function_name)
    if not callable(result):
        raise TypeError("stage factory must be callable")
    return result


def _lift_monitor_settings(value: object) -> tuple[Path, MeasuredLiftMonitorLimits] | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != _LIFT_MONITOR_KEYS or
            not isinstance(value.get("daemon_binary_path"), str) or
            not Path(value["daemon_binary_path"]).is_absolute()):
        raise ValueError("lift_monitor needs an absolute daemon path and all limits")
    limits = MeasuredLiftMonitorLimits(**{
        key: item for key, item in value.items()
        if key != "daemon_binary_path"})
    limits.validate()
    return Path(value["daemon_binary_path"]).resolve(), limits


def load_franka_integration_config(
    path: Path, *, import_stages: bool = True,
) -> dict:
    """Validate settings; optional static mode does not import stage code."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FR3 robot config is missing: {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
            not {"schema", "feedback", "stages_factory"} <= set(record) or
            set(record) - {"schema", "feedback", "sampling",
                           "stages_factory", "vlm", "lift_monitor",
                           "held_motion_monitor", "hand_force_set_raw"} or
            not isinstance(record["feedback"], dict) or
            not _FEEDBACK_KEYS <= set(record["feedback"]) <= (
                _FEEDBACK_KEYS | _OBSOLETE_FEEDBACK_KEYS)):
        raise ValueError("FR3 robot config needs schema, feedback and stages_factory")
    if any(type(value) not in (float, int) or
           not math.isfinite(value) or value <= 0
           for value in record["feedback"].values()):
        raise ValueError("FR3 feedback thresholds must be finite and positive")
    hand_force_set_raw = record.get("hand_force_set_raw")
    if hand_force_set_raw is not None:
        if (not isinstance(hand_force_set_raw, list) or
                len(hand_force_set_raw) != 6 or
                any(type(value) is not int or not 0 <= value <= 1000
                    for value in hand_force_set_raw)):
            raise ValueError("hand_force_set_raw needs six Inspire values 0..1000")
        hand_force_set_raw = tuple(hand_force_set_raw)
    sampling = None
    vlm = _vlm_settings(record.get("vlm"))
    lift_monitor = _lift_monitor_settings(record.get("lift_monitor"))
    held_motion_monitor = _lift_monitor_settings(
        record.get("held_motion_monitor"))
    if lift_monitor is not None and held_motion_monitor is not None:
        raise ValueError(
            "use held_motion_monitor or legacy lift_monitor, not both")
    if "sampling" in record:
        supplied = record["sampling"]
        if (not isinstance(supplied, dict) or
                set(supplied) != _SAMPLING_KEYS or
                type(supplied.get("max_samples")) is not int or
                supplied["max_samples"] < 2 or
                any(type(value) not in (float, int) or
                    not math.isfinite(value) or value <= 0
                    for key, value in supplied.items()
                    if key != "max_samples")):
            raise ValueError("FR3 sampling settings are incomplete or invalid")
        sampling = FrankaSamplingSettings(**supplied)
    _validate_stage_factory_spec(record["stages_factory"])
    return {
        "source": source,
        "record": record,
        "feedback": FrankaFeedbackSettings(**{
            key: record["feedback"][key] for key in _FEEDBACK_KEYS}),
        "sampling": sampling,
        "vlm": vlm,
        "lift_monitor": lift_monitor,
        "held_motion_monitor": held_motion_monitor,
        "hand_force_set_raw": hand_force_set_raw,
        "build_stages": (_stage_factory(record["stages_factory"])
                         if import_stages else None),
    }


def factory(options: RunLiveOptions) -> LiveRuntimeIntegration:
    """Create one complete AutoDex FR3 integration for ``run_live.py``.

    The JSON config holds feedback, optional sampler/VLM settings, and one
    robot-PC ``module:function``. That factory returns mappings named
    ``shared`` and ``stages`` (and optional ``held_relation_route``) for the
    existing ``assemble_live_components``. It receives ``(core, live,
    options)`` only after the frozen board/socket camera session opens. The
    core creates no guarded contact controller or camera timestamps.
    """
    if not isinstance(options, RunLiveOptions) or options.robot_config_path is None:
        raise ValueError("built-in FR3 integration needs --robot-config")
    if options.enable_robot_motion is not True:
        raise PermissionError("FR3 integration needs explicit robot-motion opt-in")
    with timed("startup.robot_config", phase="startup", kind="io"):
        parsed = load_franka_integration_config(options.robot_config_path)
    # Parse/import all static inputs before touching CUDA or a robot daemon.
    feedback = parsed["feedback"]
    sampling = parsed["sampling"]
    vlm = parsed["vlm"]
    lift_monitor = parsed["lift_monitor"]
    held_motion_monitor = parsed["held_motion_monitor"]
    hand_force_set_raw = parsed["hand_force_set_raw"]
    build_stages = parsed["build_stages"]
    sam_paths = (_sam_runtime_paths()
                 if options.xy_estimator == "sam_half_ellipse" else None)
    backend = None
    warmup_pool = None
    warmup_future = None
    vlm_load_pool = None
    vlm_load_future = None
    sam_segmentor = None
    sam_log = (options.run_output_dir.parent /
               f".{options.run_output_dir.name}.sam_roi_worker.{uuid4().hex}.stderr.log"
               if sam_paths is not None else None)
    if sam_paths is not None:
        python, model = sam_paths
        # Loading the CUDA model can fail even when its files exist. Finish
        # that work before opening the command-capable FR3 connection or
        # asking the robot to move to clear view.
        with timed("startup.sam3_roi_model_load", phase="startup"):
            sam_segmentor = SamRoiBackend(
                python_path=python, checkpoint=model,
                stderr_path=sam_log)
    core = None
    try:
        with timed("startup.franka_planner_core", phase="startup"):
            if hand_force_set_raw is None:
                core = open_franka_core(feedback)
            else:
                core = open_franka_core(
                    feedback, hand_force_set_raw=hand_force_set_raw)
        if sampling is not None:
            with timed("startup.feedback_sampler", phase="startup"):
                core.start_feedback_sampler(sampling)
    except BaseException:
        try:
            if sam_segmentor is not None:
                sam_segmentor.close()
        finally:
            if core is not None:
                core.close()
        raise

    def configured_backend(launch):
        height, width = launch.image_hw
        native_ceiling = (
            ((width + 31) // 32) * 32,
            ((height + 31) // 32) * 32)
        model_id = (vlm["model_id"] or str(
            launch.shared_root / "AutoDex/models/Qwen3-VL-2B-Instruct"))
        image_size = (tuple(vlm["classification_image_size"])
                      if vlm["classification_image_size"] is not None
                      else None)
        if vlm["mode"] == "distributed":
            if set(assessment_cameras(launch.camera_ids)) != set(vlm["workers"]):
                raise ValueError(
                    "selected VLM cameras differ from distributed workers")
            return DistributedQwenBackend(
                model_id=model_id,
                worker_root=Path(vlm["worker_root"]),
                workers=vlm["workers"],
                max_input_size=native_ceiling,
                max_new_tokens=vlm["max_new_tokens"],
                classification_image_size=image_size,
                log_path=options.run_output_dir / "distributed_qwen.jsonl")
        return load_vlm_backend(
            mode=vlm["mode"], model_id=model_id,
            max_input_size=native_ceiling,
            max_new_tokens=vlm["max_new_tokens"],
            require_native_pixels=vlm["require_native_pixels"],
            parallel_views=vlm["parallel_views"],
            **({"classification_image_size": image_size}
               if image_size is not None else {}))

    if (os.environ.get("PRECISION_PRELOAD_VLM") == "1" and
            vlm["mode"] == "local" and
            options.launch_snapshot is not None):
        launch = options.launch_snapshot
        vlm_load_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="precision-qwen-load")
        context = copy_context()

        def preload_local_backend():
            with timed("startup.vlm_model_load", phase="startup"):
                return configured_backend(launch)

        vlm_load_future = vlm_load_pool.submit(
            context.run, preload_local_backend)

    def components(live):
        with timed("startup.robot_pc_stage_factory", phase="startup"):
            parameters = build_stages(core, live, options)
        if (not isinstance(parameters, Mapping) or
                set(parameters) not in (
                    {"shared", "stages"},
                    {"shared", "stages", "held_relation_route"}) or
                not isinstance(parameters["shared"], Mapping) or
                not isinstance(parameters["stages"], Mapping)):
            raise TypeError("stage factory needs shared and stages mappings")
        shared = dict(parameters["shared"])
        owned = {"planner", "read_state"}
        if core.sampler is not None:
            owned.update({"state_for_observation", "state_for_capture"})
        if owned & shared.keys():
            raise ValueError("FR3 planner and measured readers belong to core")
        if "backend" not in shared:
            if vlm_load_future is not None:
                with timed("startup.vlm_model_load_join", phase="startup",
                           kind="wait"):
                    shared["backend"] = vlm_load_future.result()
            else:
                with timed("startup.vlm_model_load", phase="startup"):
                    shared["backend"] = configured_backend(live.cameras.launch)
        nonlocal backend
        backend = shared["backend"]
        grip_cameras = os.environ.get("PRECISION_VISIBLE_GRIP_CAMERA_IDS", "").strip()
        if vlm["mode"] == "local" and grip_cameras:
            selected = tuple(camera.strip() for camera in grip_cameras.split(","))
            if (len(selected) < 2 or len(set(selected)) != len(selected) or
                    not set(selected) <= set(live.cameras.launch.camera_ids)):
                raise ValueError("visible-grip cameras differ from the active rig")
            backend.visible_grip_camera_ids = selected
            backend.supports_visible_grip_lift = True
        shared.update(planner=core.planner, read_state=core.read_state)
        if core.sampler is not None:
            shared.update(
                state_for_observation=core.sampler.state_for_observation,
                state_for_capture=core.sampler.state_for_capture)
        family = getattr(getattr(getattr(
            getattr(live, "opened", None), "runner", None),
            "mode", None), "family", None)
        stages = _with_checkpoint_captures(
            parameters["stages"], vlm["capture_timeout_s"],
            family=family)
        if family == "cylinder" and "retry" in stages:
            stages["retry"]["xy_estimator"] = options.xy_estimator
            if sam_segmentor is not None:
                stages["retry"]["sam_segmentor"] = sam_segmentor
        if (family == "cylinder" and options.xy_estimator == "tracked_tip"
                and "task" in stages):
            if _TIP_CAMERA_ID not in live.cameras.launch.camera_ids:
                raise ValueError("tracked-tip camera is absent from this rig")
            ordinary_capture = stages["task"]["capture_preinsert"]

            def capture_with_tip_window(live_session, runner, decision):
                capture = ordinary_capture(live_session, runner, decision)
                if not isinstance(capture, RawCameraCapture):
                    return capture
                with timed("camera.tip_frame_window", phase="camera",
                           kind="network", camera_id=_TIP_CAMERA_ID):
                    window = fetch_tip_frame_window(
                        host=_TIP_CAMERA_HOST, port=_TIP_CAMERA_PORT,
                        camera_id=_TIP_CAMERA_ID,
                        target_frame_id=capture.frame_ids[_TIP_CAMERA_ID])
                return replace(capture, temporal_window=window)

            stages["task"]["capture_preinsert"] = capture_with_tip_window
        if family == "cylinder":
            for stage_name in ("key_planning", "held_return",
                               "transfer_recovery", "repose_execution"):
                if stage_name in stages:
                    stages[stage_name].setdefault(
                        "key_prompt", "white plastic cylindrical key")
        if (os.environ.get("PRECISION_PREWARM_FIXED_WORLD") == "1" and
                "key_planning" in stages):
            def join_world_before_planning():
                if warmup_future is not None:
                    with timed("startup.planner_fixed_world_warmup_wait",
                               phase="startup", kind="wait"):
                        warmup_future.result()

            stages["key_planning"]["before_preflight"] = (
                join_world_before_planning)
        if "pickup" in stages and "executor" not in stages["pickup"]:
            # Use the same FR3 connection as clear-view and measured state.
            # This is the demo's completion-checked stock subclass; the
            # robot-PC factory still supplies the motion interlock and limits.
            stages["pickup"]["executor"] = core.executor
        if family == "cylinder" and "task" in stages:
            # The cylinder stroke uses the same FR3 connection and follower
            # as AutoDex place, with the demo's configured object speed cap.
            stages["task"]["insertion_adapter"] = (
                FrankaForceStopInsertionAdapter(core.executor))
            if "release_and_clear" not in stages["task"]:
                release_limits = (stages.get("held_return", {}).get("limits")
                                  or shared.get("limits"))
                if release_limits is not None:
                    stages["task"]["release_and_clear"] = StockSocketRelease(
                        executor=core.executor, planner=core.planner,
                        read_state=core.read_state,
                        path_limits=release_limits)
        if (family == "cylinder" and "held_return" in stages and
                "record_execution" not in stages["held_return"]):
            stages["held_return"]["record_execution"] = (
                StockFailedReturnRecorder(core.executor))
        if (family == "cylinder" and
                {"held_return", "transfer_recovery"} <= stages.keys()):
            # A completed but visibly misaligned transfer returns the held
            # key through the same stock failed-return path as a force stop.
            # Keep explicit site overrides for unusual recovery hardware.
            for field in ("record_execution", "limits"):
                if (field not in stages["transfer_recovery"] and
                        field in stages["held_return"]):
                    stages["transfer_recovery"][field] = (
                        stages["held_return"][field])
            if ("return_parameters" not in stages["transfer_recovery"] and
                    "return_parameters" in stages["held_return"]):
                # The two stock return plans share their geometry, but the
                # feedback gap names describe different motion boundaries.
                transfer_return = dict(
                    stages["held_return"]["return_parameters"])
                transfer_return["max_transfer_state_gap_s"] = (
                    transfer_return.pop("max_withdrawal_state_gap_s"))
                stages["transfer_recovery"]["return_parameters"] = (
                    transfer_return)
        if family == "cylinder" and ({"held_return", "repose_execution"} <=
                                      stages.keys()):
            needs_restoration = "execute_restoration" not in stages["held_return"]
            needs_repose = "execute_repose" not in stages["repose_execution"]
            interlock = (stages.get("pickup", {}).get("motion_interlock") or
                         shared.get("motion_interlock"))
            if (needs_restoration or needs_repose) and not callable(interlock):
                raise ValueError(
                    "stock v8 reset needs pickup.motion_interlock or "
                    "shared.motion_interlock before any pickup")
            if needs_restoration or needs_repose:
                reset = StockReposeExecutor(
                    executor=core.executor, planner=core.planner,
                    clear_view=core.clear_view, read_state=core.read_state,
                    motion_interlock=interlock,
                    enable_robot_motion=options.enable_robot_motion)
                if needs_restoration:
                    stages["held_return"]["execute_restoration"] = (
                        reset.execute_restoration)
                if needs_repose:
                    stages["repose_execution"]["execute_repose"] = (
                        reset.execute_repose)
        assignments = (
            [("lift", "adapter"), ("task", "transfer_adapter")]
            if held_motion_monitor is not None else
            [("lift", "adapter")])
        monitor = held_motion_monitor or lift_monitor
        missing = [(stage, field) for stage, field in assignments
                   if stage in stages and field not in stages[stage]]
        if monitor is not None and missing:
            binary, limits = monitor
            adapter = FrankaMeasuredLiftAdapter(
                executor=core.executor, daemon_binary_path=binary,
                trace_root=options.run_output_dir / "controller_traces",
                limits=limits)
            for stage, field in missing:
                stages[stage][field] = adapter
        with timed("startup.assemble_components", phase="startup"):
            return assemble_live_components(
                live, shared=shared,
                stages=stages,
                enable_robot_motion=options.enable_robot_motion,
                held_relation_route=parameters.get("held_relation_route", "auto"),
                label_mode=options.label_mode,
                manual_label_provider=(
                    terminal_label_provider(options.annotator)
                    if options.label_mode == "manual" else None))

    def close() -> None:
        try:
            if warmup_pool is not None:
                warmup_pool.shutdown(wait=True)
        finally:
            try:
                if vlm_load_pool is not None:
                    vlm_load_pool.shutdown(wait=True)
            finally:
                close_backend_and_core()

    def close_backend_and_core() -> None:
        try:
            # A three-view XY quorum can leave remote Qwen requests draining.
            # Their view threads may still be entering the shared SAM worker.
            if callable(getattr(backend, "close", None)):
                backend.close()
        finally:
            try:
                if sam_segmentor is not None:
                    sam_segmentor.close()
            finally:
                try:
                    if (sam_log is not None and sam_log.is_file() and
                            options.run_output_dir.is_dir()):
                        sam_log.replace(
                            options.run_output_dir / "sam_roi_worker.stderr.log")
                finally:
                    core.close()

    def begin_fixed_world_warmup(live):
        nonlocal warmup_pool, warmup_future
        if warmup_pool is not None:
            raise RuntimeError("fixed-world planner warmup already started")
        scene = copy.deepcopy(live.started.calibration.collision_scene)
        warmup_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="precision-curobo-warmup")
        context = copy_context()

        def prewarm():
            with timed("startup.planner_fixed_world_warmup",
                       phase="startup", kind="planning"):
                install_curobo_planner_compat()
                core.planner._set_motion_world(scene)

        warmup_future = warmup_pool.submit(context.run, prewarm)
        return warmup_future

    return LiveRuntimeIntegration(
        core.clear_view, components, close,
        begin_fixed_world_warmup=(
            begin_fixed_world_warmup
            if os.environ.get("PRECISION_PREWARM_FIXED_WORLD") == "1"
            else None),
        key_capture_overlaps_warmup=(
            os.environ.get("PRECISION_PREWARM_FIXED_WORLD") == "1"))
