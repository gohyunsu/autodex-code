"""The robot-PC configuration is wired once, without stage duplication."""

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import runtime_assembly  # noqa: E402


@dataclass
class _Stage:
    live: object
    planner: object
    enable_robot_motion: bool = False


@dataclass
class _ReposeStage:
    live: object
    planner: object
    key_planning: object


@dataclass
class _ObservedStage:
    live: object
    planner: object


@dataclass
class _HeldReturnStage:
    live: object
    planner: object
    execute_restoration: object = None
    record_execution: object = None


@dataclass
class _ReposeExecutionStage:
    live: object
    planner: object
    execute_repose: object = None


def _fake_classes(monkeypatch):
    classes = {name: _Stage for name in runtime_assembly.STAGE_CLASSES}
    classes["repose_planning"] = _ReposeStage
    monkeypatch.setattr(runtime_assembly, "STAGE_CLASSES", classes)


def test_shared_planner_and_session_are_reused_for_every_stage(monkeypatch):
    _fake_classes(monkeypatch)
    live, planner = object(), object()
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    result = runtime_assembly.assemble_live_components(
        live, shared={"planner": planner}, stages=stages,
        enable_robot_motion=True)
    assert len(result) == 10
    assert all(stage.live is live and stage.planner is planner
               for stage in result)
    assert result[8].key_planning is result[0]
    assert result[1].enable_robot_motion is True


def test_assembly_rejects_split_session_or_missing_stage(monkeypatch):
    _fake_classes(monkeypatch)
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    with pytest.raises(ValueError, match="missing"):
        runtime_assembly.assemble_live_components(
            object(), shared={"planner": object()},
            stages={name: value for name, value in stages.items()
                    if name != "retry"}, enable_robot_motion=True)
    stages["lift"] = {"planner": object()}
    with pytest.raises(ValueError, match="shared overrides"):
        runtime_assembly.assemble_live_components(
            object(), shared={"planner": object()}, stages=stages,
            enable_robot_motion=True)
    stages["lift"] = {}
    stages["repose_planning"] = {"key_planning": object()}
    with pytest.raises(ValueError, match="CLI-owned"):
        runtime_assembly.assemble_live_components(
            object(), shared={"planner": object()}, stages=stages,
            enable_robot_motion=True)


def test_observed_relation_route_uses_distinct_handler(monkeypatch):
    _fake_classes(monkeypatch)
    monkeypatch.setattr(runtime_assembly,
                        "LiveObservedHeldRelationHandlers", _ObservedStage)
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    components = runtime_assembly.assemble_live_components(
        object(), shared={"planner": object()}, stages=stages,
        enable_robot_motion=True, held_relation_route="observed")
    assert isinstance(components[3], _ObservedStage)
    with pytest.raises(ValueError, match="auto, physical, observed or nominal"):
        runtime_assembly.assemble_live_components(
            object(), shared={"planner": object()}, stages=stages,
            enable_robot_motion=True, held_relation_route="unknown")


def test_cylinder_nominal_route_is_explicit_and_separate(monkeypatch):
    from precision_insertion.config import select_mode

    _fake_classes(monkeypatch)
    monkeypatch.setattr(runtime_assembly,
                        "LiveNominalRelationHandlers", _ObservedStage)
    live = SimpleNamespace(opened=SimpleNamespace(runner=SimpleNamespace(
        mode=select_mode("cylinder", 20))))
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    components = runtime_assembly.assemble_live_components(
        live, shared={"planner": object()}, stages=stages,
        enable_robot_motion=True, held_relation_route="nominal")
    assert isinstance(components[3], _ObservedStage)


def test_default_physical_calibration_lookup_is_grasp_and_mode_specific(tmp_path):
    from precision_insertion.config import select_mode

    mode = select_mode("cylinder", 5)
    live = SimpleNamespace(opened=SimpleNamespace(runner=SimpleNamespace(
        shared_root=tmp_path, mode=mode)))
    locate = runtime_assembly._default_physical_calibration_for_candidate(live)
    assert locate("table/0/184") == (
        tmp_path / "AutoDex/precision_insertion/physical_grasp_calibrations" /
        mode.family / mode.key_object / mode.socket_object /
        "table/0/184/physical_grasp_calibration.json")


def test_cylinder_live_rejects_postgrasp_observed_relation(monkeypatch):
    from precision_insertion.config import select_mode

    _fake_classes(monkeypatch)
    live = SimpleNamespace(opened=SimpleNamespace(runner=SimpleNamespace(
        mode=select_mode("cylinder", 20))))
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    with pytest.raises(ValueError, match="nominal v8 grasp relation only"):
        runtime_assembly.assemble_live_components(
            live, shared={"planner": object()}, stages=stages,
            enable_robot_motion=True, held_relation_route="observed")
    with pytest.raises(ValueError, match="nominal v8 grasp relation only"):
        runtime_assembly.assemble_live_components(
            live, shared={"planner": object()}, stages=stages,
            enable_robot_motion=True, held_relation_route="physical")


def test_cylinder_requires_restoration_executor_before_pickup(monkeypatch):
    from precision_insertion.config import select_mode

    _fake_classes(monkeypatch)
    monkeypatch.setattr(runtime_assembly,
                        "LiveNominalRelationHandlers", _ObservedStage)
    monkeypatch.setattr(runtime_assembly,
                        "LiveHeldReturnHandlers", _HeldReturnStage)
    runtime_assembly.STAGE_CLASSES["held_return"] = _HeldReturnStage
    monkeypatch.setattr(runtime_assembly,
                        "LiveReposeExecutionHandlers", _ReposeExecutionStage)
    runtime_assembly.STAGE_CLASSES["repose_execution"] = _ReposeExecutionStage
    live = SimpleNamespace(opened=SimpleNamespace(runner=SimpleNamespace(
        mode=select_mode("cylinder", 20))))
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    with pytest.raises(ValueError, match="execute_restoration.*before any pickup"):
        runtime_assembly.assemble_live_components(
            live, shared={"planner": object()}, stages=stages,
            enable_robot_motion=True)
    stages["held_return"]["execute_restoration"] = lambda *_args: None
    with pytest.raises(ValueError, match="record_execution.*before any pickup"):
        runtime_assembly.assemble_live_components(
            live, shared={"planner": object()}, stages=stages,
            enable_robot_motion=True)
    stages["held_return"]["record_execution"] = lambda *_args: None
    with pytest.raises(ValueError, match="execute_repose.*before any pickup"):
        runtime_assembly.assemble_live_components(
            live, shared={"planner": object()}, stages=stages,
            enable_robot_motion=True)
    stages["repose_execution"]["execute_repose"] = lambda *_args: None
    components = runtime_assembly.assemble_live_components(
        live, shared={"planner": object()}, stages=stages,
        enable_robot_motion=True)
    assert callable(components[6].execute_restoration)
    assert callable(components[6].record_execution)
    assert callable(components[9].execute_repose)


def test_square_session_uses_same_metric_retry_handler(monkeypatch):
    from precision_insertion.config import select_mode

    _fake_classes(monkeypatch)
    monkeypatch.setattr(runtime_assembly,
                        "LiveObservedHeldRelationHandlers", _ObservedStage)
    live = SimpleNamespace(opened=SimpleNamespace(runner=SimpleNamespace(
        mode=select_mode("square", 1.5))))
    stages = {name: {} for name in runtime_assembly.STAGE_CLASSES}
    physically_calibrated = runtime_assembly.assemble_live_components(
        live, shared={"planner": object()}, stages=stages,
        enable_robot_motion=True)
    assert isinstance(physically_calibrated[5], _Stage)
    assert isinstance(physically_calibrated[3], _Stage)
    components = runtime_assembly.assemble_live_components(
        live, shared={"planner": object()}, stages=stages,
        enable_robot_motion=True, held_relation_route="observed")
    assert isinstance(components[5], _Stage)
