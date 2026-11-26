"""Session-owned bridge to the isolated MV-GoTrack process on capture4."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import numpy as np

from .assets import AssetPaths
from .live_key_trial import ObservedLiveKey, PreparedLiveKeyTrial
from .mvgotrack import MVGoTrackSubscriber, alignment_from_tracker
from .timing_profile import timed


@dataclass(frozen=True)
class MVGoTrackRuntimeConfig:
    endpoint: str
    ssh_host: str
    ssh_port: int
    remote_manager: Path
    remote_session_root: Path
    timeout_s: float = 180.
    stop_on_close: bool = True

    @classmethod
    def from_environment(cls) -> "MVGoTrackRuntimeConfig":
        return cls(
            endpoint=os.environ.get(
                "PRECISION_MVGOTRACK_ENDPOINT",
                "tcp://147.46.219.136:5591"),
            ssh_host=os.environ.get(
                "PRECISION_MVGOTRACK_SSH_HOST",
                "capture4@147.46.219.136"),
            ssh_port=int(os.environ.get("PRECISION_MVGOTRACK_SSH_PORT", "77")),
            remote_manager=Path(os.environ.get(
                "PRECISION_MVGOTRACK_REMOTE_MANAGER",
                "/home/capture4/precision_insertion_mvgotrack/manage_mvgotrack.py")),
            remote_session_root=Path(os.environ.get(
                "PRECISION_MVGOTRACK_REMOTE_SESSION_ROOT",
                "/home/capture4/precision_insertion_mvgotrack/sessions")),
            timeout_s=float(os.environ.get(
                "PRECISION_MVGOTRACK_TIMEOUT_S", "180")),
            stop_on_close=(os.environ.get(
                "PRECISION_MVGOTRACK_STOP_ON_CLOSE", "1") == "1"),
        )

    def validate(self) -> None:
        if (not self.endpoint.startswith("tcp://") or
                not self.ssh_host or not 1 <= self.ssh_port <= 65535 or
                not self.remote_manager.is_absolute() or
                not self.remote_session_root.is_absolute() or
                not np.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("invalid MV-GoTrack capture4 runtime configuration")


@dataclass
class MVGoTrackRuntime:
    """Initialize once from FoundPose and serve fresh poses to same-trial XY.

    capture4 owns image acquisition and GPU tracking.  This object only sends
    the accepted FoundPose initializer and consumes metric poses.  It does not
    add confidence, jump, workspace, age, or task-success gates.
    """

    config: MVGoTrackRuntimeConfig
    runner: object
    _subscriber: MVGoTrackSubscriber | None = field(default=None, init=False)
    _last_seq: int = field(default=-1, init=False)
    _remote_session: Path | None = field(default=None, init=False)
    _initialized_capture_id: str | None = field(default=None, init=False)
    _initialization_pool: ThreadPoolExecutor | None = field(
        default=None, init=False)
    _initialization_future: Future | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.config.validate()

    def _ssh(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        command = ["ssh", "-p", str(self.config.ssh_port),
                   self.config.ssh_host, *map(str, arguments)]
        return subprocess.run(
            command, check=check, text=True, capture_output=True,
            timeout=max(10., self.config.timeout_s + 5.))

    def _scp(self, source: Path, destination: str) -> None:
        subprocess.run(
            ["scp", "-P", str(self.config.ssh_port), str(source),
             f"{self.config.ssh_host}:{destination}"],
            check=True, text=True, capture_output=True,
            timeout=max(20., self.config.timeout_s + 10.))

    def initialize(self, prepared: ObservedLiveKey | PreparedLiveKeyTrial) -> None:
        observation = prepared.observation
        if observation.family != "cylinder":
            raise ValueError("MV-GoTrack mode is defined for the cylindrical key")
        if self._subscriber is None:
            self._subscriber = MVGoTrackSubscriber(
                self.config.endpoint, expected_object=observation.key_object)
        local = (Path(self.runner.output_dir) / "mvgotrack" /
                 f"init_{observation.capture_id}")
        local.mkdir(parents=True, exist_ok=False)
        pose = local / "T_world_key.txt"
        np.savetxt(pose, observation.pose_world, fmt="%.12g")
        paths = AssetPaths(Path(self.runner.shared_root), self.runner.mode)
        mesh_source = paths.raw_mesh(observation.key_object)
        mesh = local / mesh_source.name
        shutil.copy2(mesh_source, mesh)
        metadata = {
            "schema": "precision_insertion_mvgotrack_initialization_v1",
            "capture_id": observation.capture_id,
            "key_object": observation.key_object,
            "pose_frame": "camera_calibration_world",
            "pose_source": "accepted_sam3_foundpose_tabletop",
            "pose_world": observation.pose_world.tolist(),
            "source_capture_dir": str(observation.source_capture_dir),
            "tracker_endpoint": self.config.endpoint,
        }
        (local / "initialization.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        remote = self.config.remote_session_root / (
            f"{observation.capture_id}_{uuid4().hex[:8]}")
        with timed("mvgotrack.remote_prepare", phase="tracker", kind="network"):
            self._ssh("mkdir", "-p", str(remote))
            self._scp(pose, str(remote / pose.name))
            self._scp(mesh, str(remote / mesh.name))
            self._scp(local / "initialization.json",
                      str(remote / "initialization.json"))
        before = self._subscriber.newest_seq()
        with timed("mvgotrack.start_or_reinit", phase="tracker", kind="gpu"):
            result = self._ssh(
                str(self.config.remote_manager), "start-or-reinit",
                "--session", str(remote),
                "--object", observation.key_object,
                "--mesh", str(remote / mesh.name),
                "--init-pose", str(remote / pose.name),
                "--endpoint", self.config.endpoint)
        (local / "manager_stdout.txt").write_text(
            result.stdout, encoding="utf-8")
        with timed("mvgotrack.first_pose", phase="tracker", kind="wait"):
            first = self._subscriber.wait_for_valid(
                after_seq=before, timeout_s=self.config.timeout_s)
        self._last_seq = first.seq
        self._remote_session = remote
        self._initialized_capture_id = observation.capture_id
        (local / "first_pose.json").write_text(
            json.dumps(first.to_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def begin_initialize(self, observed: ObservedLiveKey) -> None:
        """Start capture4 tracking as soon as the absolute key pose is accepted.

        Remote startup and the first tracker frame overlap cuRobo candidate
        planning.  ``await_initialized`` joins this exact future before the
        action loop can expose a pickup decision.
        """
        if not isinstance(observed, ObservedLiveKey):
            raise TypeError("MV-GoTrack initialization needs an observed live key")
        prior = self._initialization_future
        if prior is not None:
            if not prior.done():
                raise RuntimeError("MV-GoTrack initialization is already running")
            prior.result()
        if self._initialization_pool is None:
            self._initialization_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="precision-mvgotrack-init")
        context = copy_context()
        self._initialization_future = self._initialization_pool.submit(
            context.run, self.initialize, observed)

    def await_initialized(self, prepared: PreparedLiveKeyTrial) -> None:
        """Join the tracker initializer after planning and before pickup."""
        if not isinstance(prepared, PreparedLiveKeyTrial):
            raise TypeError("MV-GoTrack join needs a prepared live key trial")
        future = self._initialization_future
        if future is None:
            raise RuntimeError("MV-GoTrack initialization was not started")
        future.result()
        if self._initialized_capture_id != prepared.observation.capture_id:
            raise RuntimeError("MV-GoTrack initialized another key capture")

    def measure(self, *, runner) -> dict:
        if runner is not self.runner or self._subscriber is None:
            raise RuntimeError("MV-GoTrack has not been initialized for this session")
        # The caller invokes this only after the FR3 endpoint is measured.
        # Establish the sequence barrier now so a buffered in-motion frame
        # cannot be mistaken for the stopped pre-insertion observation.
        barrier = max(self._last_seq, self._subscriber.newest_seq())
        pose = self._subscriber.wait_for_valid(
            after_seq=barrier, timeout_s=self.config.timeout_s)
        self._last_seq = pose.seq
        return alignment_from_tracker(runner=runner, pose=pose)

    def close(self) -> None:
        try:
            future = self._initialization_future
            if future is not None:
                # Do not race a remote stop against an in-flight start. Any
                # initialization error has already surfaced at the pre-pickup
                # join on normal execution; cleanup still owns termination.
                try:
                    future.result()
                except BaseException:
                    pass
            if self.config.stop_on_close and self._remote_session is not None:
                self._ssh(str(self.config.remote_manager), "stop", check=False)
        finally:
            if self._subscriber is not None:
                self._subscriber.close()
                self._subscriber = None
            if self._initialization_pool is not None:
                self._initialization_pool.shutdown(wait=True)
                self._initialization_pool = None
            self._initialization_future = None
