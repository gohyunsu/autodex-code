"""One explicit AutoDex-PC integration contract for the live demo CLI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
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
    annotator: str | None = None
    xy_update_weight: float = 1.0


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


def inspect_live_inputs(
    *, launch_config_path: Path, endpoint_catalog_path: Path,
    robot_config_path: Path | None = None,
) -> dict:
    """Check current v8 inputs without cameras, CUDA or robot construction."""
    launch = load_session_launch(launch_config_path)
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    catalog_path = Path(endpoint_catalog_path).expanduser().resolve()
    catalog = preflight_endpoint_catalog_file(
        mode=launch.mode, shared_root=launch.shared_root,
        catalog_path=catalog_path)
    result = {
        "schema": "precision_insertion_static_live_preflight_v1",
        "launch": launch.readiness(),
        "launch_config_path": str(Path(launch_config_path).expanduser().resolve()),
        "launch_config_sha256": hashlib.sha256(
            Path(launch_config_path).expanduser().resolve().read_bytes()).hexdigest(),
        "endpoint_catalog_path": str(catalog_path),
        "endpoint_catalog_sha256": hashlib.sha256(
            catalog_path.read_bytes()).hexdigest(),
        "candidate_count": len(catalog["candidates"]),
        "eligible_candidate_count": sum(
            row.get("eligible") is True for row in catalog["candidates"]),
        "robot_config": None,
        "scope": "static_files_only_no_camera_cuda_or_robot_connection",
        "robot_ready": False,
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
    factory = getattr(import_module(module_name), factory_name)
    if not callable(factory):
        raise TypeError("integration factory is not callable")
    integration = factory(options)
    if (not isinstance(integration, LiveRuntimeIntegration) or
            not isinstance(integration.clear_view, ClearViewForPerception) or
            not callable(integration.build_components) or
            not callable(integration.close)):
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
            not 0 < options.xy_update_weight <= 1):
        raise ValueError("XY update weight must be in (0, 1]")
    if options.max_steps is not None and (
            type(options.max_steps) is not int or options.max_steps < 1):
        raise ValueError("max_steps must be positive or None")
    if options.label_mode not in LABEL_MODES or (
            options.label_mode == "manual" and not options.annotator):
        raise ValueError("manual labeling needs a named annotator")
    if options.label_mode == "evidence_fused":
        raise ValueError("current cylinder live loop needs vlm_only or manual labels")
    launch = load_session_launch(options.launch_config_path)
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    if getattr(launch, "execution_scope", "live") != "live":
        raise PermissionError("static preflight launch cannot start robot motion")
    integration = load_runtime_integration(integration_spec, options)
    try:
        def handlers(live: LivePrecisionSession):
            components = tuple(integration.build_components(live))
            return compose_live_handlers(live, *components)

        return run_live_precision_session(
            launch_config_path=options.launch_config_path,
            endpoint_catalog_path=options.endpoint_catalog_path,
            run_output_dir=options.run_output_dir,
            max_xy_retries=options.max_xy_retries,
            xy_update_weight=options.xy_update_weight,
            clear_view=integration.clear_view,
            build_handlers=handlers, max_steps=options.max_steps)
    finally:
        integration.close()
