"""Own one camera session startup and its subsequent key-camera lifetime.

This joins existing demo boundaries without duplicating board/socket capture,
FoundPose, v8 candidate validation or trial planning. The returned runner can
be passed to ``prepare_next_live_key`` while the camera context remains open.
The supplied stock executor moves to clear view before perception; no new
robot controller or trial-motion policy is constructed here.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable, Iterator, Mapping

from .action_loop import ActionHandler, ActionLoopResult, run_session_actions
from .assets import AssetPaths
from .calibration import (
    load_session_calibration, validate_session_camera_calibration,
)
from .clear_view import ClearViewForPerception
from .initial_visual import drain_pending_initial_visuals
from .camera_stream import (
    attach_running_autodex_camera_stream, open_autodex_camera_stream,
)
from .silhouette_compat import prepare_precision_silhouette

from .live_session_runner import (
    OpenedPrecisionSession, open_verified_session,
    preflight_endpoint_catalog_file,
)
from .runtime_evidence_cache import live_evidence_cache
from .live_session_start import StartedPrecisionSession
from .session_bootstrap import verify_session_evidence_bundle
from .session_launch import (
    OpenSessionCameras, load_session_launch, open_session_cameras,
)
from .timing_profile import timed


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
    "assess_saved_preinsert_xy",
    "recover_held_key_then_reobserve",
    "recover_released_key_then_reobserve", "recover_key_then_reobserve",
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
    run_output_dir: Path, max_xy_retries: int | None,
    clear_view: ClearViewForPerception, xy_update_weight: float = 1.0,
    init_ports: tuple[int, int, int] | None = None,
    launch_snapshot=None,
    reuse_session_evidence_dir: Path | None = None,
    initial_preferred_candidate: tuple[str, str, str] | None = None,
) -> Iterator[LivePrecisionSession]:
    """Capture once, open the bound ledger, and retain camera connections.

    Static launch and obvious catalogue/output errors fail before network
    access. The stronger session-to-catalogue binding runs only after actual
    board/socket evidence exists. If that check fails, the captured evidence
    is retained and all camera connections are still closed.
    """
    if launch_snapshot is None:
        with timed("startup.launch_config", phase="startup", kind="io"):
            launch = load_session_launch(launch_config_path)
    else:
        launch = launch_snapshot
    if launch.mode.family != "cylinder":
        raise ValueError("live insertion currently supports cylinder mode only")
    if getattr(launch, "execution_scope", "live") != "live":
        raise PermissionError("static preflight launch cannot start a live session")
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
    if max_xy_retries is not None and (
            type(max_xy_retries) is not int or max_xy_retries < 0):
        raise ValueError("max_xy_retries must be a nonnegative integer or None")
    if not math.isfinite(float(xy_update_weight)) or not 0 < xy_update_weight <= 1:
        raise ValueError("XY update weight must be in (0, 1]")
    if not isinstance(clear_view, ClearViewForPerception):
        raise TypeError("live perception needs a measured clear-view controller")
    if init_ports is not None and (
            not isinstance(init_ports, tuple) or len(init_ports) != 3 or
            any(type(port) is not int or not 0 < port < 65536
                for port in init_ports) or len(set(init_ports)) != 3):
        raise ValueError("init ports need three distinct TCP port numbers")
    # Offline-screened assets are immutable for the duration of this session.
    # Load the catalog once before capture, then reuse the same in-memory
    # snapshot in the ledger and every trial.
    with timed("startup.catalog_load", phase="startup", kind="io"):
        catalog_snapshot = preflight_endpoint_catalog_file(
            mode=launch.mode, shared_root=launch.shared_root,
            catalog_path=catalog)
    saved_session = None
    if reuse_session_evidence_dir is not None:
        saved_input = Path(reuse_session_evidence_dir).expanduser()
        if not saved_input.is_absolute():
            raise ValueError("reused session evidence path must be absolute")
        saved_path = saved_input.resolve()
        with timed("startup.saved_session_verify", phase="startup", kind="io"):
            manifest = verify_session_evidence_bundle(saved_path)
            calibration = load_session_calibration(
                saved_path / "session_calibration.json", mode=launch.mode,
                shared_root=launch.shared_root)
            validate_session_camera_calibration(
                calibration,
                intrinsics_undist={serial: row["K_undist"] for serial, row in
                                   launch.intrinsics_full.items()},
                extrinsics_full=launch.extrinsics_full,
                calibrated_camera_ids=set(launch.camera_ids),
                c2r=launch.c2r)
            launch_scene_mode = getattr(launch, "scene_mode", "charuco_board")
            if calibration.record.get("scene_mode", "charuco_board") != launch_scene_mode:
                raise ValueError("saved session scene mode differs from live launch")
            if launch_scene_mode == "charuco_board":
                if calibration.record["board"].get("board") != launch.board_id:
                    raise ValueError("saved session board differs from live launch")
            elif calibration.record["support_surface"]["table_surface_z_m"] != launch.table_surface_z_m:
                raise ValueError("saved table surface differs from live launch")
            saved_session = StartedPrecisionSession(
                calibration=calibration, evidence_dir=saved_path,
                board_request_id=manifest.get("board_request_id"),
                socket_request_ids=tuple(manifest["socket_request_ids"]),
                producer_manifest=manifest)
    # Stock AutoDex free-space home is appropriate only before the session's
    # first exposure, not after a failed insertion or tabletop release.
    with timed("startup.clear_view", phase="startup", kind="motion"):
        clear_view.prepare_initial()
    resources = ExitStack()
    try:
        with timed("startup.camera_transport_open", phase="startup", kind="io"):
            cameras = resources.enter_context(open_session_cameras(
                launch, **({} if init_ports is None else
                           {"init_ports": init_ports})))
        # CUDA renderer warmup can occupy the Python process longer than the
        # capture daemon's 5 s heartbeat lease. Prepare the object needed by
        # this session before acquisition starts.
        with timed(("startup.socket_renderer_warmup" if saved_session is None
                    else "startup.key_renderer_warmup"), phase="startup"):
            prepare_precision_silhouette(
                init_orchestrator=cameras.init_orchestrator,
                object_name=(launch.mode.socket_object if saved_session is None
                             else launch.mode.key_object),
                raw_mesh=AssetPaths(launch.shared_root, launch.mode).raw_mesh(
                    launch.mode.socket_object if saved_session is None
                    else launch.mode.key_object))
        with timed("startup.camera_stream_open", phase="startup", kind="io"):
            if os.environ.get("PRECISION_ATTACH_CAMERAS") == "1":
                resources.enter_context(attach_running_autodex_camera_stream(
                    launch.pc_list, launch.capture_ips))
            else:
                resources.enter_context(open_autodex_camera_stream(
                    launch.pc_list, name="precision_insertion_live"))
        if saved_session is None:
            with timed("startup.board_socket_session", phase="startup"):
                started = cameras.start()
        else:
            started = saved_session
        with timed("startup.session_ledger_open", phase="startup", kind="io"):
            opened = open_verified_session(
                mode=launch.mode, shared_root=launch.shared_root,
                evidence_dir=started.evidence_dir, catalog_path=catalog,
                output_dir=output, max_xy_retries=max_xy_retries,
                xy_update_weight=xy_update_weight,
                catalog_snapshot=catalog_snapshot,
                producer_manifest=getattr(started, "producer_manifest", None),
                calibration_snapshot=getattr(started, "calibration", None),
                initial_preferred_candidate=initial_preferred_candidate)
        yield LivePrecisionSession(started, opened, cameras, clear_view)
    finally:
        with timed("cleanup.camera_resources", phase="cleanup", kind="io"):
            resources.close()


def run_live_precision_session(
    *, launch_config_path: Path, endpoint_catalog_path: Path,
    run_output_dir: Path, max_xy_retries: int | None,
    clear_view: ClearViewForPerception,
    build_handlers: Callable[[LivePrecisionSession], Mapping[str, ActionHandler]],
    max_steps: int | None = None, xy_update_weight: float = 1.0,
    stop_after_planning: bool = False,
    reuse_session_evidence_dir: Path | None = None,
    confirm_initial_key_placement: Callable[[], bool] | None = None,
    init_ports: tuple[int, int, int] | None = None,
    launch_snapshot=None,
    initial_preferred_candidate: tuple[str, str, str] | None = None,
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
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    with live_evidence_cache(), open_live_precision_session(
            launch_config_path=launch_config_path,
            endpoint_catalog_path=endpoint_catalog_path,
            run_output_dir=run_output_dir,
            max_xy_retries=max_xy_retries,
            xy_update_weight=xy_update_weight,
            clear_view=clear_view, init_ports=init_ports,
            launch_snapshot=launch_snapshot,
            reuse_session_evidence_dir=reuse_session_evidence_dir,
            initial_preferred_candidate=initial_preferred_candidate) as live:
        # Stage adapters and load the VLM while the key is still absent.
        # The operator placement gate remains the last step before perception.
        handlers = build_handlers(live)
        # The fixed socket is measured with the key absent. Match stock
        # run_auto's one initial object-placement gate before key perception.
        # Retry captures never pause here or remeasure the socket.
        if (confirm_initial_key_placement is not None and
                live.opened.runner.current_decision().action ==
                "capture_fresh_key" and
            not _confirmed_key_placement(confirm_initial_key_placement)):
            raise RuntimeError("operator did not confirm initial key placement")
        with timed("live.action_loop", phase="trial"):
            try:
                return run_session_actions(
                    live.opened.runner, handlers=handlers,
                    max_steps=max_steps,
                    stop_before_actions=({
                        "execution_gate_required": "planning_ready",
                        "preflight_repose": "planning_no_candidate",
                    } if stop_after_planning else None))
            finally:
                drain_pending_initial_visuals()


def _confirmed_key_placement(confirm: Callable[[], bool]) -> bool:
    with timed("startup.key_placement_wait", phase="startup", kind="operator"):
        return confirm() is True
