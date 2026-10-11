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
from .live_runtime import LiveRuntimeIntegration, RunLiveOptions
from .observer import load_vlm_backend
from .runtime_assembly import assemble_live_components


_SCHEMA = "precision_insertion_franka_integration_v1"
_FEEDBACK_KEYS = frozenset(FrankaFeedbackSettings.__dataclass_fields__)
_SAMPLING_KEYS = frozenset(FrankaSamplingSettings.__dataclass_fields__)
_VLM_KEYS = frozenset({"mode", "model_id", "max_new_tokens",
                       "require_native_pixels", "capture_timeout_s"})
_CHECKPOINT_CAPTURES = {
    "lift": {"capture_after_lift": "after_lift"},
    "task": {"capture_preinsert": "preinsert",
             "capture_final": "final_or_abort"},
    "retry": {"capture_withdrawn": "preinsert",
              "capture_postshift": "post_lateral_hold",
              "capture_arrival": "preinsert",
              "capture_preinsert": "preinsert",
              "capture_final": "final_or_abort"},
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


def _with_checkpoint_captures(stages: Mapping, timeout_s: float) -> dict:
    """Reuse one open camera session; every callback still takes fresh frames."""
    result = {name: dict(fields) for name, fields in stages.items()}
    for stage_name, captures in _CHECKPOINT_CAPTURES.items():
        if stage_name not in result:
            continue
        for field, phase in captures.items():
            if field not in result[stage_name]:
                result[stage_name][field] = make_checkpoint_capture(
                    phase=phase, timeout_s=timeout_s)
    return result


def _stage_factory(specification: str):
    if (not isinstance(specification, str) or
            specification.count(":") != 1):
        raise ValueError("stage factory must be an importable module:function")
    module_name, function_name = specification.split(":", 1)
    if (not module_name or not function_name or
            not all(part.isidentifier() for part in module_name.split(".")) or
            not function_name.isidentifier()):
        raise ValueError("stage factory has an invalid module or function name")
    result = getattr(import_module(module_name), function_name)
    if not callable(result):
        raise TypeError("stage factory must be callable")
    return result


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
    source = Path(options.robot_config_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FR3 robot config is missing: {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
            not {"schema", "feedback", "stages_factory"} <= set(record) or
            set(record) - {"schema", "feedback", "sampling",
                           "stages_factory", "vlm"} or
            not isinstance(record["feedback"], dict) or
            set(record["feedback"]) != _FEEDBACK_KEYS):
        raise ValueError("FR3 robot config needs schema, feedback and stages_factory")
    if any(type(value) not in (float, int) or
           not math.isfinite(value) or value <= 0
           for value in record["feedback"].values()):
        raise ValueError("FR3 feedback thresholds must be finite and positive")
    sampling = None
    vlm = _vlm_settings(record.get("vlm"))
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
    # Parse/import all static inputs before touching CUDA or a robot daemon.
    feedback = FrankaFeedbackSettings(**record["feedback"])
    build_stages = _stage_factory(record["stages_factory"])
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
            shared["backend"] = load_vlm_backend(
                mode=vlm["mode"], model_id=vlm["model_id"],
                max_input_size=native_ceiling,
                max_new_tokens=vlm["max_new_tokens"],
                require_native_pixels=vlm["require_native_pixels"])
        shared.update(planner=core.planner, read_state=core.read_state)
        if core.sampler is not None:
            shared.update(
                state_for_observation=core.sampler.state_for_observation,
                state_for_capture=core.sampler.state_for_capture)
        return assemble_live_components(
            live, shared=shared,
            stages=_with_checkpoint_captures(
                parameters["stages"], vlm["capture_timeout_s"]),
            enable_robot_motion=options.enable_robot_motion,
            held_relation_route=parameters.get("held_relation_route", "physical"))

    return LiveRuntimeIntegration(core.clear_view, components, core.close)
