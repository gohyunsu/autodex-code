"""One explicit AutoDex-PC integration contract for the live demo CLI."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunLiveOptions:
    launch_config_path: Path
    endpoint_catalog_path: Path
    run_output_dir: Path
    max_xy_retries: int | None = None
    max_steps: int = 100
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
    if type(options.max_steps) is not int or options.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if options.label_mode not in LABEL_MODES or (
            options.label_mode == "manual" and not options.annotator):
        raise ValueError("manual labeling needs a named annotator")
    if load_session_launch(options.launch_config_path).mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
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
