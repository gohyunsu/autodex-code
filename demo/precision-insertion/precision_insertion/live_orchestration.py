"""Own one camera session startup and its subsequent key-camera lifetime.

This joins existing demo boundaries without duplicating board/socket capture,
FoundPose, v8 candidate validation or trial planning. The returned runner can
be passed to ``prepare_next_live_key`` while the camera context remains open.
The supplied stock executor moves to clear view before perception; no new
robot controller or trial-motion policy is constructed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterator, Mapping

from .action_loop import ActionHandler, ActionLoopResult, run_session_actions
from .clear_view import ClearViewForPerception

from .live_session_runner import (
    OpenedPrecisionSession, open_verified_session,
    preflight_endpoint_catalog_file,
)
from .live_session_start import StartedPrecisionSession
from .session_launch import (
    OpenSessionCameras, load_session_launch, open_session_cameras,
)


@dataclass(frozen=True)
class LivePrecisionSession:
    started: StartedPrecisionSession
    opened: OpenedPrecisionSession
    cameras: OpenSessionCameras
    clear_view: ClearViewForPerception


_LIVE_ACTIONS = frozenset({
    "capture_fresh_key", "reobserve_key_and_preflight",
    "continue_candidate_preflight", "execution_gate_required",
    "await_lift_observation", "held_relation_evidence_required",
    "transfer_execution_gate_required",
    "await_guarded_insertion_and_observation",
    "guarded_withdrawal_then_xy_assessment",
    "await_retry_execution_and_observation",
    "recover_held_key_then_reobserve", "recover_key_then_reobserve",
    "preflight_repose", "continue_repose_seed_preflight",
    "preflight_repose_release", "preflight_alternative_repose_cell",
    "repose_execution_gate_required", "await_repose_observation",
})


def compose_live_handlers(
    live: LivePrecisionSession, *components: object,
) -> dict[str, ActionHandler]:
    """Join the demo's live stages without overriding or omitting an action.

    Construct each configured component with this same ``live`` instance in
    ``build_handlers``. Missing hardware producers remain explicit at startup,
    rather than emerging after a pickup or contact motion.
    """
    joined: dict[str, ActionHandler] = {}
    for component in components:
        if getattr(component, "live", None) is not live:
            raise ValueError("live handler belongs to another camera session")
        handlers = component.handlers()
        if not isinstance(handlers, Mapping):
            raise TypeError("live handler component needs a handler mapping")
        duplicates = joined.keys() & handlers.keys()
        if duplicates:
            raise ValueError(f"duplicate live actions: {sorted(duplicates)}")
        if any(name not in _LIVE_ACTIONS or not callable(handler)
               for name, handler in handlers.items()):
            raise ValueError("unknown or non-callable live action handler")
        joined.update(handlers)
    missing = _LIVE_ACTIONS - joined.keys()
    if missing:
        raise ValueError(f"live pipeline has unconfigured actions: {sorted(missing)}")
    return joined


@contextmanager
def open_live_precision_session(
    *, launch_config_path: Path, endpoint_catalog_path: Path,
    run_output_dir: Path, max_xy_retries: int,
    clear_view: ClearViewForPerception, xy_update_weight: float = 1.0,
) -> Iterator[LivePrecisionSession]:
    """Capture once, open the bound ledger, and retain camera connections.

    Static launch and obvious catalogue/output errors fail before network
    access. The stronger session-to-catalogue binding runs only after actual
    board/socket evidence exists. If that check fails, the captured evidence
    is retained and all camera connections are still closed.
    """
    launch = load_session_launch(launch_config_path)
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    catalog_input = Path(endpoint_catalog_path).expanduser()
    output_input = Path(run_output_dir).expanduser()
    if not catalog_input.is_absolute() or not output_input.is_absolute():
        raise ValueError("endpoint catalogue and run output need absolute paths")
    catalog = catalog_input.resolve()
    output = output_input.resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"endpoint catalogue is missing: {catalog}")
    if output.exists():
        raise FileExistsError(f"session run output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError("session run output parent is missing")
    if type(max_xy_retries) is not int or max_xy_retries < 0:
        raise ValueError("max_xy_retries must be a nonnegative integer")
    if not math.isfinite(float(xy_update_weight)) or not 0 < xy_update_weight <= 1:
        raise ValueError("XY update weight must be in (0, 1]")
    if not isinstance(clear_view, ClearViewForPerception):
        raise TypeError("live perception needs a measured clear-view controller")
    # Reject a stale/incomplete v8 pool before a slow board/socket capture.
    # ``open_verified_session`` repeats this check after capture and binds the
    # measured fixture to the catalogue's exact socket mesh.
    preflight_endpoint_catalog_file(
        mode=launch.mode, shared_root=launch.shared_root,
        catalog_path=catalog)
    # Stock AutoDex free-space home is appropriate only before the session's
    # first exposure, not after a failed insertion or tabletop release.
    clear_view.prepare_initial()
    with open_session_cameras(launch) as cameras:
        started = cameras.start()
        opened = open_verified_session(
            mode=launch.mode, shared_root=launch.shared_root,
            evidence_dir=started.evidence_dir, catalog_path=catalog,
            output_dir=output, max_xy_retries=max_xy_retries,
            xy_update_weight=xy_update_weight)
        yield LivePrecisionSession(started, opened, cameras, clear_view)


def run_live_precision_session(
    *, launch_config_path: Path, endpoint_catalog_path: Path,
    run_output_dir: Path, max_xy_retries: int,
    clear_view: ClearViewForPerception,
    build_handlers: Callable[[LivePrecisionSession], Mapping[str, ActionHandler]],
    max_steps: int = 100, xy_update_weight: float = 1.0,
) -> ActionLoopResult:
    """Keep the one ChArUco/socket camera session open across trial actions.

    ``build_handlers`` supplies AutoDex-PC integrations for the decisions it
    intends to execute. A missing handler returns ``needs_handler`` without
    skipping the stage. The supplied functions must use the existing
    ``SessionRunner`` and demo execution/observation boundaries, not a second
    candidate policy. Camera connections close when this function returns or
    a handler raises. No motion adapter is constructed here.
    """
    if not callable(build_handlers):
        raise TypeError("live action loop needs a handler factory")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    with open_live_precision_session(
            launch_config_path=launch_config_path,
            endpoint_catalog_path=endpoint_catalog_path,
            run_output_dir=run_output_dir,
            max_xy_retries=max_xy_retries,
            xy_update_weight=xy_update_weight,
            clear_view=clear_view) as live:
        return run_session_actions(
            live.opened.runner, handlers=build_handlers(live),
            max_steps=max_steps)
