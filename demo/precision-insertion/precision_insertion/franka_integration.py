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

from .franka_core import (
    FrankaFeedbackSettings, FrankaSamplingSettings, open_franka_core,
)
from .live_runtime import LiveRuntimeIntegration, RunLiveOptions
from .runtime_assembly import assemble_live_components


_SCHEMA = "precision_insertion_franka_integration_v1"
_FEEDBACK_KEYS = frozenset(FrankaFeedbackSettings.__dataclass_fields__)
_SAMPLING_KEYS = frozenset(FrankaSamplingSettings.__dataclass_fields__)


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

    The JSON config is deliberately small: five feedback thresholds and one
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
            set(record) not in (
                {"schema", "feedback", "stages_factory"},
                {"schema", "feedback", "sampling", "stages_factory"}) or
            not isinstance(record["feedback"], dict) or
            set(record["feedback"]) != _FEEDBACK_KEYS):
        raise ValueError("FR3 robot config needs schema, feedback and stages_factory")
    if any(type(value) not in (float, int) or
           not math.isfinite(value) or value <= 0
           for value in record["feedback"].values()):
        raise ValueError("FR3 feedback thresholds must be finite and positive")
    sampling = None
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
        shared.update(planner=core.planner, read_state=core.read_state)
        if core.sampler is not None:
            shared.update(
                state_for_observation=core.sampler.state_for_observation,
                state_for_capture=core.sampler.state_for_capture)
        return assemble_live_components(
            live, shared=shared, stages=parameters["stages"],
            enable_robot_motion=options.enable_robot_motion,
            held_relation_route=parameters.get("held_relation_route", "physical"))

    return LiveRuntimeIntegration(core.clear_view, components, core.close)
