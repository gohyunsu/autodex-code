"""Assemble the existing live stage handlers from one robot-PC configuration.

This is wiring, not another candidate policy or motion controller. The robot
PC supplies hardware objects/callbacks and stage parameters; all stages share
the same opened camera session, planner, feedback sampler and VLM backend.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Mapping

from .insertion_grasp_calibration import calibration_path_for_candidate
from .labeling import LABEL_MODES
from .live_held_relation_handlers import LiveHeldRelationHandlers
from .live_lift_handlers import LiveLiftHandlers
from .live_nominal_relation_handlers import LiveNominalRelationHandlers
from .live_observed_relation_handlers import LiveObservedHeldRelationHandlers
from .live_pickup_handlers import LivePickupHandlers
from .live_planning_handlers import LiveKeyPlanningHandlers
from .live_recovery_handlers import LiveHeldReturnHandlers
from .live_repose_execution import LiveReposeExecutionHandlers
from .live_repose_handlers import LiveReposePlanningHandlers
from .live_retry_handlers import LiveGroundedRetryHandlers
from .live_task_handlers import LiveTaskHandlers
from .live_transfer_recovery import LiveTransferRecoveryHandlers


STAGE_CLASSES = {
    "key_planning": LiveKeyPlanningHandlers,
    "pickup": LivePickupHandlers,
    "lift": LiveLiftHandlers,
    "held_relation": LiveHeldRelationHandlers,
    "task": LiveTaskHandlers,
    "retry": LiveGroundedRetryHandlers,
    "held_return": LiveHeldReturnHandlers,
    "transfer_recovery": LiveTransferRecoveryHandlers,
    "repose_planning": LiveReposePlanningHandlers,
    "repose_execution": LiveReposeExecutionHandlers,
}


def _default_physical_calibration_for_candidate(live):
    """Use the insertion grasp commissioning layout under shared data."""
    runner = live.opened.runner
    output_root = (Path(runner.shared_root).expanduser().resolve() /
                   "AutoDex" / "precision_insertion" /
                   "physical_grasp_calibrations")
    mode = runner.mode
    return lambda candidate_id: calibration_path_for_candidate(
        output_root, mode, candidate_id)


def assemble_live_components(
    live: object, *, shared: Mapping[str, object],
    stages: Mapping[str, Mapping[str, object]],
    enable_robot_motion: bool,
    held_relation_route: str = "physical",
    label_mode: str = "evidence_fused",
    manual_label_provider=None,
) -> tuple[object, ...]:
    """Instantiate every existing handler without duplicating its behavior.

    ``shared`` holds identical cross-stage objects such as ``planner``,
    ``read_state`` and ``backend``. ``stages`` holds per-stage adapters,
    callbacks, and limits. An override of a shared value is rejected: a
    mismatched planner or feedback sampler would silently split a trial.
    ``live`` and ``enable_robot_motion`` are owned by the CLI and cannot be
    supplied by the integration configuration.
    """
    if not isinstance(shared, Mapping) or not isinstance(stages, Mapping):
        raise TypeError("runtime assembly needs shared and stage mappings")
    if type(enable_robot_motion) is not bool:
        raise TypeError("runtime assembly needs a boolean motion setting")
    if held_relation_route not in {"physical", "observed", "nominal"}:
        raise ValueError("held relation route must be physical, observed or nominal")
    family = getattr(getattr(getattr(live, "opened", None), "runner", None),
                     "mode", None)
    if (getattr(family, "family", None) == "cylinder" and
            held_relation_route == "observed"):
        raise ValueError(
            "cylinder live retries need physical or experimental nominal relation")
    if label_mode not in LABEL_MODES or (
            label_mode == "manual" and not callable(manual_label_provider)):
        raise ValueError("invalid label mode or missing manual label provider")
    classes = dict(STAGE_CLASSES)
    if held_relation_route == "observed":
        classes["held_relation"] = LiveObservedHeldRelationHandlers
    elif held_relation_route == "nominal":
        classes["held_relation"] = LiveNominalRelationHandlers
    missing = classes.keys() - stages.keys()
    unknown = stages.keys() - classes.keys()
    if missing or unknown:
        raise ValueError(
            f"runtime stages missing {sorted(missing)} or unknown {sorted(unknown)}")
    reserved = {"live", "enable_robot_motion", "key_planning"}
    if reserved & shared.keys():
        raise ValueError("live and robot-motion opt-in are CLI-owned")
    known = set()
    for cls in classes.values():
        known.update(field.name for field in fields(cls) if field.init)
    unknown_shared = shared.keys() - known
    if unknown_shared:
        raise ValueError(f"unknown shared runtime fields: {sorted(unknown_shared)}")

    components = []
    for name, cls in classes.items():
        stage = stages[name]
        if not isinstance(stage, Mapping):
            raise TypeError(f"{name} stage parameters must be a mapping")
        accepted = {field.name for field in fields(cls) if field.init}
        unknown_fields = stage.keys() - accepted
        collision = stage.keys() & shared.keys()
        if unknown_fields or collision or reserved & stage.keys():
            raise ValueError(
                f"{name} has unknown {sorted(unknown_fields)}, shared overrides "
                f"{sorted(collision)}, or CLI-owned fields "
                f"{sorted(reserved & stage.keys())}")
        arguments = {key: value for key, value in shared.items()
                     if key in accepted}
        arguments.update(stage)
        if name in {"lift", "task", "retry"}:
            if "label_mode" in accepted:
                arguments["label_mode"] = label_mode
            if "manual_label_provider" in accepted:
                arguments["manual_label_provider"] = manual_label_provider
        if (name == "held_relation" and cls is LiveHeldRelationHandlers and
                "calibration_for_candidate" not in arguments):
            arguments["calibration_for_candidate"] = (
                _default_physical_calibration_for_candidate(live))
        arguments["live"] = live
        if name == "repose_planning":
            arguments["key_planning"] = components[0]
        if "enable_robot_motion" in accepted:
            arguments["enable_robot_motion"] = enable_robot_motion
        components.append(cls(**arguments))
    retry = components[list(classes).index("retry")]
    held_return = components[list(classes).index("held_return")]
    if (isinstance(retry, LiveGroundedRetryHandlers) and
            isinstance(held_return, LiveHeldReturnHandlers) and
            held_return.reuse_withdrawal is None):
        held_return.reuse_withdrawal = retry.withdrawn_for_return
    return tuple(components)
