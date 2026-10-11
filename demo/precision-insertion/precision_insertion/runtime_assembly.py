"""Assemble the existing live stage handlers from one robot-PC configuration.

This is wiring, not another candidate policy or motion controller. The robot
PC supplies hardware objects/callbacks and stage parameters; all stages share
the same opened camera session, planner, feedback sampler and VLM backend.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping

from .live_held_relation_handlers import LiveHeldRelationHandlers
from .live_lift_handlers import LiveLiftHandlers
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


def assemble_live_components(
    live: object, *, shared: Mapping[str, object],
    stages: Mapping[str, Mapping[str, object]],
    enable_robot_motion: bool,
    held_relation_route: str = "physical",
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
    if held_relation_route not in {"physical", "observed"}:
        raise ValueError("held relation route must be physical or observed")
    classes = dict(STAGE_CLASSES)
    if held_relation_route == "observed":
        classes["held_relation"] = LiveObservedHeldRelationHandlers
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
        arguments["live"] = live
        if name == "repose_planning":
            arguments["key_planning"] = components[0]
        if "enable_robot_motion" in accepted:
            arguments["enable_robot_motion"] = enable_robot_motion
        components.append(cls(**arguments))
    return tuple(components)
