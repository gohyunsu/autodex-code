"""Built-in FR3 resource owner for the demo's live integration contract.

The robot PC contributes only stage-specific adapters and commissioned limits
through a Python factory. AutoDex's FR3 executor, planner, measured state and
clear-view controller come from :mod:`franka_core`; the existing stage classes
continue to own all task decisions and execution checkpoints.
"""

from __future__ import annotations

from importlib import import_module
import json
import math
from pathlib import Path
from typing import Mapping

from .checkpoint_camera import make_checkpoint_capture
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
from .observer import load_vlm_backend
from .runtime_assembly import assemble_live_components


_SCHEMA = "precision_insertion_franka_integration_v1"
_FEEDBACK_KEYS = frozenset(FrankaFeedbackSettings.__dataclass_fields__)
_SAMPLING_KEYS = frozenset(FrankaSamplingSettings.__dataclass_fields__)
_VLM_KEYS = frozenset({"mode", "model_id", "max_new_tokens",
                       "require_native_pixels", "capture_timeout_s"})
_LIFT_MONITOR_KEYS = frozenset({
    "daemon_binary_path", "max_force_norm_n", "max_torque_norm_nm",
    "max_hand_tracking_error_raw", "max_state_gap_s", "minimum_samples",
})
_CHECKPOINT_CAPTURES = {
    "lift": {"capture_after_lift": "after_lift"},
    "task": {"capture_preinsert": "preinsert",
             "capture_final": "final_clear_view"},
    "retry": {"capture_withdrawn": "preinsert"},
}


def _vlm_settings(value: object) -> dict:
    if value is None:
        return {"mode": "local", "model_id": None,
                "max_new_tokens": 512, "require_native_pixels": True,
                "capture_timeout_s": 10.0}
    if not isinstance(value, dict) or set(value) - _VLM_KEYS:
        raise ValueError("VLM settings contain unknown fields")
    settings = {**_vlm_settings(None), **value}
    if (settings["mode"] not in {"local", "gemini"} or
            settings["model_id"] is not None and
            (not isinstance(settings["model_id"], str) or
             not settings["model_id"].strip()) or
            type(settings["max_new_tokens"]) is not int or
            settings["max_new_tokens"] <= 0 or
            type(settings["require_native_pixels"]) is not bool or
            type(settings["capture_timeout_s"]) not in (int, float) or
            not math.isfinite(settings["capture_timeout_s"]) or
            settings["capture_timeout_s"] <= 0 or
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
                           "held_motion_monitor"} or
            not isinstance(record["feedback"], dict) or
            set(record["feedback"]) != _FEEDBACK_KEYS):
        raise ValueError("FR3 robot config needs schema, feedback and stages_factory")
    if any(type(value) not in (float, int) or
           not math.isfinite(value) or value <= 0
           for value in record["feedback"].values()):
        raise ValueError("FR3 feedback thresholds must be finite and positive")
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
        "feedback": FrankaFeedbackSettings(**record["feedback"]),
        "sampling": sampling,
        "vlm": vlm,
        "lift_monitor": lift_monitor,
        "held_motion_monitor": held_motion_monitor,
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
    parsed = load_franka_integration_config(options.robot_config_path)
    # Parse/import all static inputs before touching CUDA or a robot daemon.
    feedback = parsed["feedback"]
    sampling = parsed["sampling"]
    vlm = parsed["vlm"]
    lift_monitor = parsed["lift_monitor"]
    held_motion_monitor = parsed["held_motion_monitor"]
    build_stages = parsed["build_stages"]
    core = open_franka_core(feedback)
    if sampling is not None:
        try:
            core.start_feedback_sampler(sampling)
        except BaseException:
            core.close()
            raise

    def components(live):
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
            height, width = live.cameras.launch.image_hw
            # Qwen's processor rounds to a patch/merge factor. A slightly
            # larger ceiling lets the backend append lossless right/bottom
            # padding while keeping the original pixel coordinate origin.
            native_ceiling = (
                ((width + 31) // 32) * 32,
                ((height + 31) // 32) * 32)
            model_id = (vlm["model_id"] or str(
                live.cameras.launch.shared_root /
                "AutoDex/models/Qwen3-VL-2B-Instruct"))
            shared["backend"] = load_vlm_backend(
                mode=vlm["mode"], model_id=model_id,
                max_input_size=native_ceiling,
                max_new_tokens=vlm["max_new_tokens"],
                require_native_pixels=vlm["require_native_pixels"])
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
        if "pickup" in stages and "executor" not in stages["pickup"]:
            # Use the same FR3 connection as clear-view and measured state.
            # This is the demo's completion-checked stock subclass; the
            # robot-PC factory still supplies the motion interlock and limits.
            stages["pickup"]["executor"] = core.executor
        if family == "cylinder" and "task" in stages:
            # The cylinder stroke uses the same FR3 connection and stock
            # follower as AutoDex place, including its normal speed.
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
        return assemble_live_components(
            live, shared=shared,
            stages=stages,
            enable_robot_motion=options.enable_robot_motion,
            held_relation_route=parameters.get("held_relation_route", "auto"),
            label_mode=options.label_mode,
            manual_label_provider=(
                terminal_label_provider(options.annotator)
                if options.label_mode == "manual" else None))

    return LiveRuntimeIntegration(core.clear_view, components, core.close)
