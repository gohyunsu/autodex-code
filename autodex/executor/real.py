"""
Real-world grasp executor for xArm + Allegro hand.

Autonomous (no GUI) trajectory execution.

Execution sequence (matches RSS2026 reference: planner/inference/train/run_auto_v2.py):
    execute:  init(joint0) -> approach(traj) -> pregrasp -> grasp -> squeeze -> lift -> place
    release:  reverse_squeeze -> grasp -> pregrasp -> hand_init -> arm_return

Usage:
    executor = RealExecutor()
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
from __future__ import annotations

import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING, Optional
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.executor.lift_policy import LiftExecutionError, check_lift_start
from autodex.executor.timing import (
    finish_pickup_timing, finish_place_timing, new_pickup_timing,
    new_place_timing,
)
from autodex.utils.conversion import cart2se3
from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT, XARM_CLEAR_VIEW, XARM_INSPIRE_CLEAR_VIEW,
    ALLEGRO_INIT, ALLEGRO_LINK6_TO_WRIST,
    INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
)

if TYPE_CHECKING:
    from autodex.planner import PlanResult

# xArm speed-profile experiment controls.  ``XARM_BASE_SPEED_SCALE`` is the
# one primary knob: 1.0 reproduces the legacy trajectory rate and per-tick
# joint-step cap.  The active profile runs free/far motion at 2x and applies a
# 0.5 multiplier near the object or while holding it, restoring the legacy 1x
# rate in those two safety-critical states.
# These values are deliberately xArm-specific; Franka has independent limits
# and profile constants in src/execution/franka_executor.py.
XARM_BASE_SPEED_SCALE = 2.0
XARM_NEAR_SPEED_SCALE = 0.5
XARM_HELD_SPEED_SCALE = 0.5
XARM_APPROACH_SLOWDOWN_FAR_M = 0.30
XARM_APPROACH_SLOWDOWN_NEAR_M = 0.15
XARM_LEGACY_JOINT_STEP_RAD = 0.05
# EMA weight on the previous joint target.  This suppresses per-waypoint
# position kinks without changing nominal trajectory-index playback speed.
# 0 disables smoothing; values closer to 1 are smoother but add more lag.
XARM_COMMAND_SMOOTHING = 0.80
# Smooth changes in trajectory playback rate (for example 2x far -> 1x near)
# independently of the joint-target EMA above.  Tracking-error thresholds are
# L2 norms over the six arm joints: phase advancement tapers at SOFT and pauses
# at HARD until the physical arm catches up.
XARM_PLAYBACK_RATE_SMOOTHING = 0.85
XARM_TRACKING_ERROR_SOFT_RAD = 0.08
XARM_TRACKING_ERROR_HARD_RAD = 0.16
# FK is the synchronous part of proximity evaluation.  Limit it to 20 Hz and
# reuse the last asynchronous mesh result inside the 100 Hz servo loop.
XARM_PROXIMITY_QUERY_PERIOD_S = 0.05
# One-switch rollback for hardware A/B testing.  False routes every existing
# ``_move_joints`` caller through the preserved waypoint-wait implementation.
XARM_CONTINUOUS_PLAYBACK = True

# xArm placement contact-stop controls.  The learned torque residual can
# exceed 10 Nm during an otherwise clear planned descent, so use the intended
# 20 Nm threshold on the watched shoulder/elbow joints.  Keep this independent
# of the approach monitor (70 Nm) and of run_auto's 5 mm early-stop label.
XARM_PLACE_CONTACT_THRESHOLD_NM = 20.0
XARM_PLACE_CONTACT_SUSTAINED_TICKS = 8
XARM_PLACE_CONTACT_STARTUP_BLANK_S = 0.5

# A coverage/reposition transfer is allowed to finish with the same small
# physical tracking residual as the FR3 adapter.  This is deliberately
# separate from JacobianStrokeOptions.request_start_position_tolerance_m:
# the values below compare the *physical* controller pose with the requested
# pre-place pose, while the Jacobian option checks two planner-frame values.
XARM_PREPLACE_REUSE_POS_TOL_M = 0.005
XARM_PREPLACE_REUSE_ROT_TOL_RAD = np.deg2rad(3.0)
XARM_VERTICAL_STROKE_Z_TOL_M = 0.005

# Per-hand config: (init_joints, link6_to_wrist, convert_fn)
def _convert_allegro(hand_pose: np.ndarray) -> np.ndarray:
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    if hand_pose.ndim == 1:
        out = hand_pose.copy()
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out = hand_pose.copy()
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out

def _convert_inspire(hand_pose: np.ndarray) -> np.ndarray:
    """Convert inspire qpos (radians) to controller action (0-1000).

    qpos order:   [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    action order:  [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    """
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
    if hand_pose.ndim == 1:
        q = hand_pose[:6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros(6, dtype=np.float64)
        action[0] = np.clip(action_float[5], 0, 1000)  # pinky
        action[1] = np.clip(action_float[4], 0, 1000)  # ring
        action[2] = np.clip(action_float[3], 0, 1000)  # middle
        action[3] = np.clip(action_float[2], 0, 1000)  # index
        action[4] = np.clip(action_float[1], 0, 1000)  # thumb_pitch
        action[5] = np.clip(action_float[0], 0, 1000)  # thumb_yaw
    else:
        q = hand_pose[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros_like(hand_pose)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000)
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000)
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000)
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000)
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000)
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000)
    return action

HAND_CONFIG = {
    "allegro": {
        "init": ALLEGRO_INIT,
        "link6_to_wrist": ALLEGRO_LINK6_TO_WRIST,
        "convert": _convert_allegro,
        "xarm_init": XARM_INIT,
        "xarm_clear_view": XARM_CLEAR_VIEW,
    },
    "inspire": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
        "xarm_clear_view": XARM_INSPIRE_CLEAR_VIEW,
    },
    "inspire_left": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LEFT_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
        "xarm_clear_view": XARM_INSPIRE_CLEAR_VIEW,
    },
}


# ── Contact monitor (shared by place / execute / reset) ──────────────────────

# Tau conversion constants used by mcc_minimal's stream pipeline. Multiplies
# raw _joints_torque (current in xArm's reported units) to Nm.
KT = np.array([0.067, 0.067, 0.0573, 0.0573, 0.056, 0.056])
GEAR = np.full(6, 100.0)
# Per-joint baseline noise (Nm) — from mcc DEADBAND_J.
DEADBAND_J = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 0.5])
POST_RELEASE_CLEARANCE_M = 0.10


class ContactDetected(RuntimeError):
    """Raised by motion primitives when a ContactMonitor fires.
    Propagates out of execute() / reset() so the caller can abort the trial
    cleanly instead of continuing into pregrasp/grasp at the wrong pose."""
    def __init__(self, where: str, tau_dev, ratio):
        self.where = where
        self.tau_dev = tau_dev
        self.ratio = ratio
        super().__init__(f"contact during {where}: tau_dev={tau_dev.round(2)} "
                         f"ratio={ratio.round(2)}")


class ContactMonitor:
    """Torque-based contact detection using a learned tau_model.

    Usage:
        m = ContactMonitor(xarm_handle, model_path,
                           watch_joints=(1, 2), thresh_nm=10.0,
                           sustained_ticks=8, startup_blank_s=0.5)
        m.warmup(seconds=1.0)              # call when arm is static at start pose
        while moving:
            ...                            # send servo command
            if m.tick():                   # returns True on contact
                break
    """

    def __init__(self, xarm_handle, model_path,
                 watch_joints=(1, 2), thresh_nm=10.0,
                 sustained_ticks: int = 8, startup_blank_s: float = 0.5,
                 dt: float = 0.01,
                 filter_alpha: float = 0.1, qdot_alpha: float = 0.1):
        from pathlib import Path
        import torch  # noqa: F401  (deferred — only loaded if monitor used)
        from autodex.executor.tau_model import load_model
        if model_path is None:
            model_path = str(Path.home() / "shared_data" / "AutoDex"
                             / "weights" / "tau_model" / "inspire_left.pt")
        # Allow tighter / looser thresholds per joint by scaling the
        # deadband. thresh_nm applies to watch_joints (assumed shared scale).
        self._model = load_model(model_path)
        self._xarm = xarm_handle
        self._watch = list(watch_joints)
        # Per-joint threshold: scaled deadband so off-watch joints don't
        # accidentally count if caller widens the watch set later.
        scale_per_joint = np.maximum(DEADBAND_J, 1e-6)
        # On watched joints, set threshold to user-given thresh_nm. Others
        # default to DEADBAND_J * (thresh_nm / DEADBAND_J[watch_joints[0]]).
        ref_db = DEADBAND_J[self._watch[0]]
        self._thresh = scale_per_joint * (thresh_nm / ref_db)
        self._sustained_req = sustained_ticks
        self._blank = startup_blank_s
        self._dt = dt
        self._filter_alpha = filter_alpha
        self._qdot_alpha = qdot_alpha
        self._tau_filt = np.zeros(6)
        self._qdot_smooth = np.zeros(6)
        self._q_last = None
        self._t_last = None
        self._baseline = np.zeros(6)
        self._t0 = None
        self._sustained = 0
        self._last_dev = np.zeros(6)
        self._last_ratio = np.zeros(6)

    def _read(self):
        _, q_deg = self._xarm.get_servo_angle()
        q = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64))
        I = np.asarray(self._xarm._arm._joints_torque[:6], dtype=np.float64)
        tau = I * KT * GEAR
        return q, tau

    def _predict_tau_ext(self):
        import torch
        from autodex.executor.tau_model import build_input
        q, tau_motor = self._read()
        t_now = time.time()
        if self._q_last is not None and self._t_last is not None:
            dt = max(t_now - self._t_last, 1e-4)
            qdot = (q - self._q_last) / dt
        else:
            qdot = np.zeros(6)
        self._q_last, self._t_last = q.copy(), t_now
        self._qdot_smooth = (self._qdot_alpha * qdot
                             + (1 - self._qdot_alpha) * self._qdot_smooth)
        x = build_input(
            q[None, :], self._qdot_smooth[None, :],
            use_sincos=self._model.use_sincos,
            use_qdot=self._model.use_qdot,
            use_sign_qdot=getattr(self._model, "use_sign_qdot", False),
        )[0].astype(np.float32)
        with torch.no_grad():
            tau_hat = self._model.predict_full(torch.from_numpy(x)).numpy()
        tau_ext = tau_hat - tau_motor
        self._tau_filt = (self._filter_alpha * tau_ext
                          + (1 - self._filter_alpha) * self._tau_filt)
        return self._tau_filt

    def warmup(self, seconds: float = 1.0):
        """Hold the arm static at its current pose and capture baseline."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self._predict_tau_ext()
            time.sleep(self._dt)
        self._baseline = self._tau_filt.copy()
        self._t0 = time.time()
        self._sustained = 0

    def tick(self) -> bool:
        """Update reading once, return True iff contact sustained over watched
        joints AND the startup blank period has elapsed."""
        if self._t0 is None:
            return False
        self._predict_tau_ext()
        tau_dev = self._tau_filt - self._baseline
        ratio = np.abs(tau_dev) / np.maximum(self._thresh, 1e-6)
        self._last_dev = tau_dev
        self._last_ratio = ratio
        t = time.time() - self._t0
        watched = ratio[self._watch]
        crossed = bool(np.any(watched > 1.0))
        if crossed and t > self._blank:
            self._sustained += 1
        else:
            self._sustained = 0
        return self._sustained >= self._sustained_req

    @property
    def last_dev(self):
        return self._last_dev

    @property
    def last_ratio(self):
        return self._last_ratio


class RealExecutor:
    def __init__(
        self,
        arm_name: str = "xarm",
        hand_name: str = "allegro",
        dt: float = 0.01,
        squeeze_level: int = 2,
        base_speed_scale: float = XARM_BASE_SPEED_SCALE,
        near_speed_scale: float = XARM_NEAR_SPEED_SCALE,
        held_speed_scale: float = XARM_HELD_SPEED_SCALE,
        slowdown_far_m: float = XARM_APPROACH_SLOWDOWN_FAR_M,
        slowdown_near_m: float = XARM_APPROACH_SLOWDOWN_NEAR_M,
        command_smoothing: float = XARM_COMMAND_SMOOTHING,
        continuous_playback: bool = XARM_CONTINUOUS_PLAYBACK,
    ):
        if hand_name not in HAND_CONFIG:
            raise ValueError(f"Unknown hand: {hand_name}. Choose from {list(HAND_CONFIG)}")
        self.dt = dt
        self.squeeze_level = squeeze_level
        self.hand_name = hand_name
        if not np.isfinite(base_speed_scale) or base_speed_scale <= 0.0:
            raise ValueError("base_speed_scale must be finite and > 0")
        if not np.isfinite(near_speed_scale) or not 0.0 < near_speed_scale <= 1.0:
            raise ValueError("near_speed_scale must be in (0, 1]")
        if not np.isfinite(held_speed_scale) or not 0.0 < held_speed_scale <= 1.0:
            raise ValueError("held_speed_scale must be in (0, 1]")
        if (not np.isfinite(slowdown_near_m)
                or not np.isfinite(slowdown_far_m)
                or not 0.0 <= slowdown_near_m < slowdown_far_m):
            raise ValueError("slowdown distances require 0 <= near < far")
        if (not np.isfinite(command_smoothing)
                or not 0.0 <= command_smoothing < 1.0):
            raise ValueError("command_smoothing must be in [0, 1)")
        self.base_speed_scale = float(base_speed_scale)
        self.near_speed_scale = float(near_speed_scale)
        self.held_speed_scale = float(held_speed_scale)
        self.slowdown_far_m = float(slowdown_far_m)
        self.slowdown_near_m = float(slowdown_near_m)
        self.command_smoothing = float(command_smoothing)
        self.continuous_playback = bool(continuous_playback)

        hcfg = HAND_CONFIG[hand_name]
        self._convert = hcfg["convert"]
        self._hand_init = hcfg["init"]
        self._link6_to_wrist = hcfg["link6_to_wrist"]
        self._xarm_init = hcfg["xarm_init"]
        self._clear_view = np.asarray(
            hcfg["xarm_clear_view"], dtype=np.float64).copy()
        self._last_hand_qpos = np.asarray(self._hand_init, dtype=np.float64).copy()
        self._last_hand_action = self._convert(self._last_hand_qpos)
        self.state_timestamps = []
        self.last_execute_timing = {}
        self.last_place_timing = {}
        self._timing_recorder: Optional[Any] = None
        self._holding_object = False
        self._speed_profile_planner = None
        self._speed_profile_object_query = None
        self._speed_profile_object_signature = None
        self._speed_profile_hand_link_indices = None
        self._speed_profile_hand_link_names = None
        self._speed_profile_mesh_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xarm-mesh-proximity")
        self._speed_profile_mesh_future = None
        self._speed_profile_mesh_distance = None
        self._speed_profile_next_query_time = 0.0
        self._last_speed_profile_band = None
        # Public execution contract consumed by run_auto.  Keep private
        # low-level motion details inside this adapter.
        self.arm_dof = 6

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Disable xArm's internal collision sensitivity once for the whole
        # session. Otherwise the firmware can self-stop on small torque
        # spikes (esp. during sequential / approach motions) and silently
        # ignore subsequent servo commands — arm appears to "stall" mid
        # trajectory. Our ContactMonitor handles real collision detection.
        try:
            self.arm.arm.set_report_tau_or_i(1)
            self.arm.arm.set_collision_sensitivity(0)
        except Exception as _e:
            print(f"[executor] could not disable xarm collision sensitivity: {_e!r}")

        # Per-tick caps.  The joint cap follows the same single base-speed knob
        # as trajectory playback.  Cartesian values retain their legacy
        # semantics because the production pick/place path is joint planned.
        self.joint_vel_limit = XARM_LEGACY_JOINT_STEP_RAD * self.base_speed_scale
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    def set_timing_recorder(self, recorder: Optional[Any]) -> None:
        """Attach the current pipeline event trace."""
        self._timing_recorder = recorder

    def set_speed_profile_planner(self, planner) -> None:
        """Bind the planner whose xArm+hand FK supplies live link positions."""
        self._speed_profile_planner = planner

    def set_speed_profile_object(self, scene_cfg: Optional[dict]) -> None:
        """Cache the target mesh in robot-base coordinates for proximity speed.

        Mesh loading and transformation happen once per object pose.  The
        control loop only submits nearest-surface queries against this cache.
        A missing/broken mesh disables distance profiling, but the held-object
        cap remains active independently.
        """
        target = (scene_cfg or {}).get("mesh", {}).get("target")
        if not isinstance(target, dict):
            self._clear_speed_profile_object()
            return
        mesh_path = target.get("file_path")
        pose = target.get("pose")
        try:
            pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
            if (not mesh_path or pose_arr.shape != (7,)
                    or not np.isfinite(pose_arr).all()):
                raise ValueError("target requires a finite pose[7] and file_path")
            signature = (str(mesh_path), pose_arr.tobytes())
            if signature == self._speed_profile_object_signature:
                return

            import trimesh

            mesh = trimesh.load(str(mesh_path), process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("target mesh has no triangle surface")
            mesh = mesh.copy()
            mesh.apply_transform(cart2se3(pose_arr))
            self._speed_profile_object_query = trimesh.proximity.ProximityQuery(mesh)
            self._speed_profile_object_signature = signature
            self._speed_profile_mesh_distance = None
            self._speed_profile_next_query_time = 0.0
            self._last_speed_profile_band = None
            print(f"[xarm] speed proximity: {len(mesh.faces)} target triangles; "
                  "hand-link origins -> mesh surface", flush=True)
        except Exception as exc:
            self._clear_speed_profile_object()
            print(f"[xarm] object-mesh speed proximity unavailable; "
                  f"distance slowdown disabled: {exc!r}", flush=True)

    def _clear_speed_profile_object(self) -> None:
        self._speed_profile_object_query = None
        self._speed_profile_object_signature = None
        self._speed_profile_mesh_distance = None
        self._speed_profile_next_query_time = 0.0
        self._last_speed_profile_band = None

    def _speed_profile_hand_links(self, kinematics) -> tuple[list[int], list[str]]:
        """Select only the mounted palm and finger links, excluding arm links."""
        names = [str(name) for name in getattr(kinematics, "link_names", [])]
        if (self._speed_profile_hand_link_names == names
                and self._speed_profile_hand_link_indices is not None):
            indices = self._speed_profile_hand_link_indices
            return indices, [names[i] for i in indices]
        indices = [
            i for i, name in enumerate(names)
            if (name == "base_link"
                or name.startswith(("right_", "left_", "link_")))
        ]
        if not indices:
            raise RuntimeError("planner exposes no hand/palm link positions")
        self._speed_profile_hand_link_indices = indices
        self._speed_profile_hand_link_names = names
        return indices, [names[i] for i in indices]

    @staticmethod
    def _query_hand_mesh_distance(query, points: np.ndarray,
                                  link_names: tuple[str, ...], signature):
        _, distances, _ = query.on_surface(points)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        if distances.shape != (len(link_names),) or not np.isfinite(distances).all():
            raise RuntimeError("mesh proximity returned invalid link distances")
        nearest = int(np.argmin(distances))
        return signature, float(distances[nearest]), link_names[nearest]

    def _hand_mesh_distance(
            self, arm_qpos: np.ndarray,
            hand_qpos: Optional[np.ndarray] = None) -> Optional[tuple[float, str]]:
        """Return the latest hand-link-to-object-surface distance in metres."""
        query = self._speed_profile_object_query
        planner = self._speed_profile_planner
        motion_gen = getattr(planner, "_motion_gen", None)
        signature = self._speed_profile_object_signature
        if query is None or motion_gen is None or signature is None:
            return None
        try:
            future = self._speed_profile_mesh_future
            if future is not None and future.done():
                self._speed_profile_mesh_future = None
                result_signature, distance, link_name = future.result()
                if result_signature == self._speed_profile_object_signature:
                    self._speed_profile_mesh_distance = (distance, link_name)
            if self._speed_profile_mesh_future is not None:
                return self._speed_profile_mesh_distance

            now = time.perf_counter()
            next_query = getattr(self, "_speed_profile_next_query_time", 0.0)
            if now < next_query:
                return self._speed_profile_mesh_distance
            self._speed_profile_next_query_time = (
                now + XARM_PROXIMITY_QUERY_PERIOD_S)

            n_arm = int(getattr(planner, "_n_arm", self.arm_dof))
            state_q = np.asarray(planner._init_state, dtype=np.float32).copy()
            arm = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
            if state_q.ndim != 1 or len(state_q) < n_arm or len(arm) < n_arm:
                return self._speed_profile_mesh_distance
            state_q[:n_arm] = arm[:n_arm]
            hand = (self._last_hand_qpos if hand_qpos is None
                    else np.asarray(hand_qpos, dtype=np.float64).reshape(-1))
            hand = np.asarray(hand, dtype=np.float32).reshape(-1)
            n_hand = min(len(hand), len(state_q) - n_arm)
            if n_hand > 0 and np.isfinite(hand[:n_hand]).all():
                state_q[n_arm:n_arm + n_hand] = hand[:n_hand]

            import torch

            kin_model = motion_gen.kinematics
            state = kin_model.get_state(torch.tensor(
                state_q, dtype=torch.float32,
                device=planner._tensor_args.device).unsqueeze(0))
            indices, names = self._speed_profile_hand_links(kin_model)
            points = np.asarray(
                state.links_position[0, indices, :].detach().cpu().numpy(),
                dtype=np.float64)
            if points.shape != (len(indices), 3) or not np.isfinite(points).all():
                return self._speed_profile_mesh_distance
            self._speed_profile_mesh_future = self._speed_profile_mesh_executor.submit(
                self._query_hand_mesh_distance, query, points.copy(),
                tuple(names), signature)
            return self._speed_profile_mesh_distance
        except Exception as exc:
            self._speed_profile_object_query = None
            self._speed_profile_mesh_future = None
            print(f"[xarm] object-mesh speed query failed; "
                  f"distance slowdown disabled: {exc!r}", flush=True)
            return None

    def _approach_speed_scale(self, hand_mesh_distance_m: float) -> float:
        """Map distance to 1.0 far, near_speed_scale near, linear between."""
        distance = float(hand_mesh_distance_m)
        if not np.isfinite(distance) or distance >= self.slowdown_far_m:
            return 1.0
        if distance <= self.slowdown_near_m:
            return self.near_speed_scale
        progress = ((distance - self.slowdown_near_m)
                    / (self.slowdown_far_m - self.slowdown_near_m))
        return self.near_speed_scale + (1.0 - self.near_speed_scale) * progress

    def _motion_speed(self, arm_qpos: np.ndarray,
                      hand_qpos: Optional[np.ndarray] = None
                      ) -> tuple[float, float, str, Optional[float], Optional[str]]:
        """Return playback rate, joint-step cap, band, distance, nearest link."""
        distance = None
        nearest_link = None
        # A neutral profile cannot change the command.  Avoid synchronous GPU
        # FK and mesh-query scheduling entirely so enabling the feature with
        # 1/1/1 defaults adds no timing jitter to the 100 Hz servo producer.
        if self.near_speed_scale == 1.0 and self.held_speed_scale == 1.0:
            rate = self.base_speed_scale
            return (rate, XARM_LEGACY_JOINT_STEP_RAD * rate,
                    "neutral", None, None)
        if self._holding_object:
            safety_scale = self.held_speed_scale
            band = "held"
        else:
            mesh_distance = self._hand_mesh_distance(arm_qpos, hand_qpos)
            if mesh_distance is None:
                # If a mesh query is configured but its first async result is
                # pending, start conservatively instead of briefly racing at
                # far speed next to an object.
                if self._speed_profile_object_query is not None:
                    safety_scale = self.near_speed_scale
                    band = "proximity-pending"
                else:
                    safety_scale = 1.0
                    band = "unprofiled"
            else:
                distance, nearest_link = mesh_distance
                safety_scale = self._approach_speed_scale(distance)
                if distance <= self.slowdown_near_m:
                    band = "near"
                elif distance >= self.slowdown_far_m:
                    band = "far"
                else:
                    band = "transition"
        rate = self.base_speed_scale * safety_scale
        step_limit = XARM_LEGACY_JOINT_STEP_RAD * rate
        return rate, step_limit, band, distance, nearest_link

    def _log_speed_profile(self, rate: float, step_limit: float, band: str,
                           distance: Optional[float], nearest_link: Optional[str]) -> None:
        if band == self._last_speed_profile_band:
            return
        self._last_speed_profile_band = band
        proximity = ("" if distance is None else
                     f" distance={distance * 100:.1f}cm link={nearest_link}")
        print(f"[xarm] speed band={band} rate={rate:.2f}x "
              f"joint_step={step_limit:.3f}rad/tick{proximity}", flush=True)

    # ── low-level motion primitives ──────────────────────────────────────

    def get_arm_qpos(self) -> np.ndarray:
        """Read the physical arm configuration in planner joint order."""
        return np.asarray(self.arm.get_data()["qpos"][:self.arm_dof],
                          dtype=np.float64)

    def get_hand_qpos(self) -> tuple[np.ndarray, str]:
        """Return the planner-space hand state used for collision semantics.

        The deployed hand APIs do not expose a calibrated planner-qpos encoder
        for every hand, so this has explicit ``commanded_nominal`` provenance.
        """
        return (np.asarray(self._last_hand_qpos, dtype=np.float64).copy(),
                "commanded_nominal")

    def get_wrist_pose(self) -> np.ndarray:
        """Read the physical wrist pose in the planner frame."""
        link6 = np.asarray(self.arm.get_data()["position"], dtype=np.float64)
        wrist = link6 @ self._link6_to_wrist
        if wrist.shape != (4, 4) or not np.isfinite(wrist).all():
            raise RuntimeError("xarm returned an invalid wrist pose")
        return wrist

    @staticmethod
    def _pose_error(actual: np.ndarray,
                    target: np.ndarray) -> tuple[float, float]:
        """Return translation metres and rotation radians between two poses."""
        actual = np.asarray(actual, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if (actual.shape != (4, 4) or target.shape != (4, 4)
                or not np.isfinite(actual).all()
                or not np.isfinite(target).all()):
            return float("inf"), float("inf")
        position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation = float(Rotation.from_matrix(
            actual[:3, :3].T @ target[:3, :3]).magnitude())
        return position, rotation

    def follow_joint_trajectory(self, traj: np.ndarray,
                                hand_traj: Optional[np.ndarray] = None) -> None:
        """Execute an already planned full-DOF path through this arm adapter."""
        q = np.asarray(traj, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] < self.arm_dof:
            raise ValueError("joint trajectory has incompatible xarm shape")
        proximity_hand_traj = (
            q[:, self.arm_dof:] if q.shape[1] > self.arm_dof else None)
        self._move_joints(q[:, :self.arm_dof], hand_traj,
                          proximity_hand_traj=proximity_hand_traj)

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02,
                     monitor: "Optional[ContactMonitor]" = None,
                     proximity_hand_traj=None):
        """Continuously stream an interpolated path on the servo clock.

        Intermediate waypoints are references, not stop points.  Path phase
        advances every control tick, slows when physical tracking error grows,
        and pauses at a hard lag limit.  Only the final waypoint requires
        convergence.  ``_move_joints_waypoint_wait`` retains the old behavior
        as an explicit fallback.
        """
        if not getattr(self, "continuous_playback", True):
            return self._move_joints_waypoint_wait(
                arm_traj,
                hand_traj,
                threshold=threshold,
                monitor=monitor,
                proximity_hand_traj=proximity_hand_traj,
            )
        arm_path = np.atleast_2d(np.asarray(arm_traj, dtype=np.float64))
        if arm_path.ndim != 2 or len(arm_path) == 0:
            raise ValueError("arm_traj must be a non-empty 2D array")
        if not np.isfinite(arm_path).all():
            raise ValueError("arm_traj must contain only finite values")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and > 0")
        hands = (None if hand_traj is None else
                 np.atleast_2d(np.asarray(hand_traj, dtype=np.float64)))
        proximity_hands = (
            None if proximity_hand_traj is None else
            np.atleast_2d(np.asarray(proximity_hand_traj, dtype=np.float64)))
        if hands is not None and len(hands) != len(arm_path):
            raise ValueError("hand_traj length must match arm_traj")
        if proximity_hands is not None and len(proximity_hands) != len(arm_path):
            raise ValueError("proximity_hand_traj length must match arm_traj")

        def _at(path, index):
            if path is None:
                return None
            lo = min(int(index), len(path) - 1)
            hi = min(lo + 1, len(path) - 1)
            frac = index - int(index)
            return path[lo] * (1.0 - frac) + path[hi] * frac

        phase = 0.0
        last_phase = float(len(arm_path) - 1)
        filtered_target = np.asarray(
            self.arm.get_data()["qpos"], dtype=np.float64).copy()
        smoothed_rate = None
        prev_qpos = None
        stall_count = 0
        recovered = False
        next_tick = time.perf_counter()
        # A generous bound protects against a controller that moves just enough
        # to evade stall detection while never closing its tracking error.
        min_configured_rate = max(
            self.base_speed_scale
            * min(1.0, self.near_speed_scale, self.held_speed_scale),
            1e-3,
        )
        nominal_ticks = last_phase / min_configured_rate
        max_ticks = max(1000, int(np.ceil(10.0 * nominal_ticks)) + 500)

        for _ in range(max_ticks):
            cur = np.asarray(self.arm.get_data()["qpos"], dtype=np.float64)
            target_arm = _at(arm_path, phase)
            target_hand = _at(hands, phase)
            proximity_hand = _at(proximity_hands, phase)

            (desired_rate, desired_step_limit, band,
             distance, nearest_link) = self._motion_speed(cur, proximity_hand)
            if not np.isfinite(desired_rate) or desired_rate <= 0.0:
                raise RuntimeError(
                    f"invalid xArm trajectory playback rate: {desired_rate}")
            if smoothed_rate is None:
                smoothed_rate = float(desired_rate)
            else:
                smoothed_rate = (
                    XARM_PLAYBACK_RATE_SMOOTHING * smoothed_rate
                    + (1.0 - XARM_PLAYBACK_RATE_SMOOTHING) * desired_rate)
            step_limit = (float(desired_step_limit)
                          * smoothed_rate / desired_rate)
            self._log_speed_profile(
                smoothed_rate, step_limit, band, distance, nearest_link)

            filtered_target = (
                self.command_smoothing * filtered_target
                + (1.0 - self.command_smoothing) * target_arm)
            command_error = float(np.linalg.norm(filtered_target - cur))
            if (prev_qpos is not None
                    and np.linalg.norm(cur - prev_qpos) < 1e-4
                    and command_error > threshold):
                stall_count += 1
                if stall_count >= 50 and not recovered:
                    print("[executor] continuous trajectory stalled; "
                          "clearing error...")
                    self.arm.clear_error()
                    recovered = True
                    stall_count = 0
                elif stall_count >= 100:
                    raise RuntimeError(
                        "xArm continuous trajectory stalled after recovery")
            else:
                stall_count = 0
            prev_qpos = cur.copy()

            if target_hand is not None:
                self.hand.move(target_hand)
            nxt = self._safe_joint_step(
                cur, filtered_target, vel_limit=step_limit)
            self.arm.move(nxt, is_servo=True)

            if self.dt > 0.0:
                next_tick += self.dt
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    # Do not issue catch-up bursts after an FK or controller
                    # overrun; restart the clock from the actual tick time.
                    next_tick = time.perf_counter()
            if monitor is not None and monitor.tick():
                raise ContactDetected("_move_joints",
                                      monitor.last_dev, monitor.last_ratio)

            actual = np.asarray(
                self.arm.get_data()["qpos"], dtype=np.float64)
            at_endpoint = phase >= last_phase
            if (at_endpoint
                    and np.linalg.norm(actual - arm_path[-1]) < threshold):
                return

            if not at_endpoint:
                # Compare against the smoothed command reference, not the raw
                # future waypoint.  This measures controller lag without
                # treating the intentional target EMA as tracking failure.
                tracking_error = float(
                    np.linalg.norm(actual - filtered_target))
                if tracking_error <= XARM_TRACKING_ERROR_SOFT_RAD:
                    lag_scale = 1.0
                elif tracking_error >= XARM_TRACKING_ERROR_HARD_RAD:
                    lag_scale = 0.0
                else:
                    lag_scale = (
                        (XARM_TRACKING_ERROR_HARD_RAD - tracking_error)
                        / (XARM_TRACKING_ERROR_HARD_RAD
                           - XARM_TRACKING_ERROR_SOFT_RAD))
                phase = min(
                    phase + smoothed_rate * lag_scale,
                    last_phase,
                )

        raise RuntimeError(
            "xArm continuous trajectory exceeded its tracking timeout")

    def _move_joints_waypoint_wait(
            self, arm_traj, hand_traj=None, threshold=0.02,
            monitor: "Optional[ContactMonitor]" = None,
            proximity_hand_traj=None):
        """Follow a dense path with distance/held-object speed scaling.

        Preserved fallback implementation: command one path waypoint and wait
        until the measured joints are within ``threshold`` before advancing.
        ``base_speed_scale`` controls both the waypoint-index increment and
        joint-step cap.
        """
        arm_path = np.atleast_2d(np.asarray(arm_traj, dtype=np.float64))
        if arm_path.ndim != 2 or len(arm_path) == 0:
            raise ValueError("arm_traj must be a non-empty 2D array")
        hands = (None if hand_traj is None else
                 np.atleast_2d(np.asarray(hand_traj, dtype=np.float64)))
        proximity_hands = (
            None if proximity_hand_traj is None else
            np.atleast_2d(np.asarray(proximity_hand_traj, dtype=np.float64)))
        if hands is not None and len(hands) != len(arm_path):
            raise ValueError("hand_traj length must match arm_traj")
        if proximity_hands is not None and len(proximity_hands) != len(arm_path):
            raise ValueError("proximity_hand_traj length must match arm_traj")

        def _at(path, index):
            if path is None:
                return None
            lo = min(int(index), len(path) - 1)
            hi = min(lo + 1, len(path) - 1)
            frac = index - int(index)
            return path[lo] * (1.0 - frac) + path[hi] * frac

        idx = 0.0
        last_idx = float(len(arm_path) - 1)
        filtered_target = np.asarray(
            self.arm.get_data()["qpos"], dtype=np.float64).copy()
        while True:
            target_arm = _at(arm_path, idx)
            target_hand = _at(hands, idx)
            proximity_hand = _at(proximity_hands, idx)
            if target_hand is not None:
                self.hand.move(target_hand)
            stall_count = 0
            prev_qpos = None
            recovered = False
            playback_rate = self.base_speed_scale
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                (playback_rate, step_limit, band,
                 distance, nearest_link) = self._motion_speed(cur, proximity_hand)
                self._log_speed_profile(
                    playback_rate, step_limit, band, distance, nearest_link)
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print("[executor] stall detected, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print("[executor] stall after recovery, aborting")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                filtered_target = (
                    self.command_smoothing * filtered_target
                    + (1.0 - self.command_smoothing) * target_arm)
                nxt = self._safe_joint_step(
                    cur, filtered_target, vel_limit=step_limit)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if monitor is not None and monitor.tick():
                    raise ContactDetected("_move_joints_waypoint_wait",
                                          monitor.last_dev, monitor.last_ratio)
                # During playback, track the filtered reference so smoothing
                # does not reduce trajectory-index speed.  At the final sample,
                # keep iterating until the original endpoint is reached.
                arrival_target = target_arm if idx >= last_idx else filtered_target
                if (np.linalg.norm(self.arm.get_data()["qpos"] - arrival_target)
                        < threshold):
                    break
            if idx >= last_idx:
                break
            idx = min(idx + playback_rate, last_idx)

    def _move_hand(self, target):
        self.hand.move(target)
        time.sleep(self.dt)

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02,
                        vel_scale=1.0, stop_on_stall=False,
                        stall_window=30, stall_progress_ratio=0.3,
                        monitor: "Optional[ContactMonitor]" = None):
        """Stall detection is window-based + ratio to commanded velocity:
        over the last `stall_window` ticks the arm should advance roughly
        (cart_vel_limit * vel_scale * stall_window) meters in free motion.
        If actual progress < `stall_progress_ratio` of that expected, we count
        as stalled — robust to both reading latency and xarm yielding on contact.

        stop_on_stall=True breaks immediately on stall (placing mode — stop on
        contact, don't clear_error or retry).
        """
        from collections import deque

        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        pos_history = deque(maxlen=stall_window)
        stalled = False
        recovered = False
        recover_count = 0
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            cur_pos = cur[:3, 3].copy()
            pos_history.append(cur_pos)
            arm_qpos = self.arm.get_data()["qpos"]
            rate, step_limit, band, distance, nearest_link = \
                self._motion_speed(arm_qpos)
            self._log_speed_profile(
                rate, step_limit, band, distance, nearest_link)
            expected_progress = (
                self.cart_vel_limit * vel_scale * rate * stall_window)
            stall_thresh = expected_progress * stall_progress_ratio
            # Stall = full window collected and progress < expected*ratio.
            if len(pos_history) == stall_window:
                progress = np.linalg.norm(pos_history[-1] - pos_history[0])
                stalled = (progress < stall_thresh)
            if stalled:
                if stop_on_stall:
                    print(f"[executor] stall detected (window {stall_window} ticks, "
                          f"progress {progress*1000:.2f}mm) — stopping (placing mode)")
                    break
                if not recovered:
                    print("[executor] stall detected, clearing error...")
                    self.arm.clear_error()
                    recovered = True
                    pos_history.clear()
                    stalled = False
                else:
                    recover_count += 1
                    if recover_count >= stall_window:
                        print("[executor] stall after recovery, aborting")
                        break
            prev_pos = cur_pos
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            vel = self.cart_vel_limit * vel_scale * rate
            if t_dist > vel:
                t_delta = t_delta / t_dist * vel
            cur[:3, 3] += t_delta
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            rot_limit = self.rot_vel_limit * rate
            if r_dist > rot_limit:
                r_delta = r_delta / r_dist * rot_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            if monitor is not None and monitor.tick():
                raise ContactDetected("_move_cartesian",
                                      monitor.last_dev, monitor.last_ratio)
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.06,
                               vel_limit: "Optional[float]" = None,
                               first_vel_limit: "Optional[float]" = None,
                               monitor: "Optional[ContactMonitor]" = None):
        """Move joints in order while retaining the shared xArm speed policy.

        Explicit limits are absolute safety caps.  Otherwise the configured
        base/near/held joint-step limit is used dynamically.
        """
        current_target = self.arm.get_data()["qpos"].copy()
        filtered_target = np.asarray(current_target, dtype=np.float64).copy()
        for step_i, j in enumerate(joint_order):
            explicit_limit = (
                first_vel_limit if step_i == 0 and first_vel_limit is not None
                else vel_limit)
            current_target[j] = target_qpos[j]
            stall_count = 0
            prev_qpos = None
            recovered = False
            j_start = float(self.arm.get_data()["qpos"][j])
            iter_count = 0
            converged = False
            for _ in range(500):
                iter_count += 1
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        try:
                            err, warn = self.arm.arm.get_err_warn_code()
                        except Exception as _ee:
                            err, warn = ("?", repr(_ee))
                        print(f"[executor] joint {j} stall at qpos={cur.round(3)} "
                              f"target={current_target.round(3)}  "
                              f"xarm err={err} warn={warn} — clearing...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        try:
                            err, warn = self.arm.arm.get_err_warn_code()
                        except Exception as _ee:
                            err, warn = ("?", repr(_ee))
                        print(f"[executor] joint {j} stall after recovery at "
                              f"qpos={cur.round(3)} target={current_target.round(3)}  "
                              f"xarm err={err} warn={warn} — skipping")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                rate, profile_limit, band, distance, nearest_link = \
                    self._motion_speed(cur)
                self._log_speed_profile(
                    rate, profile_limit, band, distance, nearest_link)
                step_limit = (profile_limit if explicit_limit is None else
                              min(float(explicit_limit), profile_limit))
                filtered_target = (
                    self.command_smoothing * filtered_target
                    + (1.0 - self.command_smoothing) * current_target)
                nxt = self._safe_joint_step(
                    cur, filtered_target, vel_limit=step_limit)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if monitor is not None and monitor.tick():
                    raise ContactDetected(f"_move_joint_sequential (joint {j})",
                                          monitor.last_dev, monitor.last_ratio)
                if np.abs(self.arm.get_data()["qpos"][j] - target_qpos[j]) < threshold:
                    converged = True
                    break
            # Post-loop diagnostic: ALWAYS report if joint didn't converge,
            # even when stall_count never reached 50 (slow-motion / partial
            # progress case).
            j_end = float(self.arm.get_data()["qpos"][j])
            j_err = abs(j_end - target_qpos[j])
            if not converged:
                try:
                    err, warn = self.arm.arm.get_err_warn_code()
                except Exception as _ee:
                    err, warn = ("?", repr(_ee))
                print(f"[executor] joint {j} did NOT converge in {iter_count} iters: "
                      f"start={j_start:.3f} end={j_end:.3f} target={target_qpos[j]:.3f}  "
                      f"err={j_err:.3f} rad ({np.degrees(j_err):.1f}°)  "
                      f"xarm err={err} warn={warn}")

    # ── public API ────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        # Idempotent: each controller's .stop() crashes if its save-path attr is
        # None (recording never started / already stopped). Guard each.
        # xarm uses `save_path`; inspire/allegro use `capture_path` — without
        # the second check, hand.stop() never fired and only the last cycle's
        # data persisted (via hand.end() at process shutdown).
        for ctrl in (self.arm, self.hand):
            if (getattr(ctrl, "save_path", None) is not None
                    or getattr(ctrl, "capture_path", None) is not None):
                ctrl.stop()

    def _log_state(self, state):
        ts = datetime.datetime.now().isoformat()
        item = {"state": state, "time": ts}
        if self._timing_recorder is not None and hasattr(
                self._timing_recorder, "event"):
            event = self._timing_recorder.event(
                "robot.state", phase="execution", kind="state",
                state=state, arm="xarm", hand=self.hand_name)
            item.update({
                "pipeline_event_seq": event["seq"],
                "pipeline_time_s": event["pipeline_time_s"],
                "monotonic_ns": event["monotonic_ns"],
                "utc_ns": event["utc_ns"],
            })
        self.state_timestamps.append(item)

    def home(self, clear_view: bool = False) -> None:
        """Open the hand and move once to the calibrated xArm start pose.

        ``clear_view`` uses the same base-joint offset as the existing recovery
        path, keeping the arm outside the cameras before live perception.  It
        is intentionally a caller-controlled one-time action; local retries
        must use ``execute(..., start_from_current=True)`` instead.
        """
        self._log_state("clear_view" if clear_view else "init")
        self._move_hand(self._convert(self._hand_init))
        self._last_hand_qpos = np.asarray(self._hand_init, dtype=np.float64).copy()
        self._last_hand_action = self._convert(self._last_hand_qpos)
        time.sleep(0.5)
        target = (self._clear_view.copy() if clear_view
                  else np.asarray(self._xarm_init, dtype=np.float64).copy())
        order = [1, 2, 5, 0, 3, 4]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            order = [2, 1, 5, 0, 3, 4]
        self._move_joint_sequential(target[:6], order, threshold=0.06)
        err = float(np.linalg.norm(
            np.asarray(self.arm.get_data()["qpos"][:6], dtype=np.float64) - target[:6]
        ))
        if err > 0.1:
            raise RuntimeError(
                f"home(): final qpos err={err:.3f} > 0.1; target={target[:6].round(3)}"
            )

    def _make_monitor(self, thresh_nm: float = 15.0, model_path: str = None,
                      watch_joints=(1, 2),
                      sustained_ticks: int = 100,
                      startup_blank_s: float = 0.5) -> "ContactMonitor":
        """Construct a ContactMonitor for the current arm. Caller should call
        monitor.warmup(...) once the arm is static at the desired baseline pose.
        Defaults: 15 Nm threshold, 1s sustained — robust against motion-induced
        tau spikes; real collisions still fire (ratio >> 1 instantly)."""
        xarm_handle = self.arm.arm   # raw XArmAPI
        # Ensure the report mode is set so _joints_torque is populated.
        try:
            xarm_handle.set_report_tau_or_i(1)
            xarm_handle.set_collision_sensitivity(0)
        except Exception:
            pass
        return ContactMonitor(
            xarm_handle, model_path,
            watch_joints=watch_joints, thresh_nm=thresh_nm,
            sustained_ticks=sustained_ticks,
            startup_blank_s=startup_blank_s, dt=self.dt,
        )

    def execute(self, plan_result: PlanResult, lift_height: float = 0.10,
                skip_lift: bool = False, planner=None,
                scene_cfg=None, debug_dump_dir=None,
                lift_traj_override=None, start_from_current: bool = False):
        """
        Execute: init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        State timestamps stored in self.state_timestamps.
        Returns the squeezed hand pose.

        Place (descend) is now a separate `place(plan_result, ...)` call so
        callers can do work (e.g. capture label image) while the object is
        held up.

        ``skip_lift=True`` stops after the squeeze step (no lift). Use this
        when the caller wants to perform a joint-space lift via the planner
        (avoids ``_move_cartesian`` / ``set_servo_cartesian_aa`` kinematic-
        error spam at extreme wrist orientations).

        ``start_from_current`` preserves the measured raised state for a
        continuous retry whose planner was seeded with those joints.
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None
        if planner is not None:
            self.set_speed_profile_planner(planner)
        self.set_speed_profile_object(scene_cfg)
        self._holding_object = False
        print(f"[xarm] speed profile: base={self.base_speed_scale:.2f}x; "
              f">={self.slowdown_far_m * 100:.0f}cm 1.00x base, "
              f"<={self.slowdown_near_m * 100:.0f}cm "
              f"{self.near_speed_scale:.2f}x base, "
              f"held {self.held_speed_scale:.2f}x base", flush=True)

        execute_started = time.perf_counter()
        self.state_timestamps = []
        self.last_execute_timing = new_pickup_timing()
        pickup_trace_id = (self._timing_recorder.begin(
            phase="execution", kind="motion", name="pickup_and_lift")
            if self._timing_recorder is not None else None)
        traj = plan_result.traj
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)

        sl = self.squeeze_level

        # 1. Legacy trials return to init first. A continuous retry has a
        # trajectory from the physical raised state, so returning home here
        # would discard the local replanning benefit.
        self._log_state("current_start" if start_from_current else "init")
        if not start_from_current:
            t_init = time.perf_counter()
            order = [1, 2, 5, 0, 3, 4]
            if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
                order = [2, 1, 5, 0, 3, 4]
            self._move_joint_sequential(self._xarm_init[:6], order, threshold=0.06)
            init_err = float(np.linalg.norm(self.arm.get_data()["qpos"]
                                            - self._xarm_init[:6]))
            if init_err > 0.1:
                raise RuntimeError(
                    f"execute(): init step finished with err={init_err:.3f} > 0.1 "
                    f"— arm not at XARM_INIT, refusing to approach. "
                    f"final_qpos={self.arm.get_data()['qpos'].round(3)}"
                )
            self.last_execute_timing["init_motion_s"] = round(
                time.perf_counter() - t_init, 3)
        # Threshold raised 50→70 Nm because inertia spikes (joint 2
        # shoulder ~50-60Nm during free-space motion) were aborting valid
        # approaches.
        monitor = self._make_monitor(thresh_nm=70.0, sustained_ticks=50)
        print("[executor] warming up approach contact monitor (1s static)...")
        t_warmup = time.perf_counter()
        monitor.warmup(seconds=1.0)
        self.last_execute_timing["approach_monitor_warmup_s"] = round(
            time.perf_counter() - t_warmup, 3)
        print(f"[executor] approach baseline tau = {monitor._baseline.round(2)}  "
              f"(thresh=70Nm, sustained=0.5s)")

        # 2. Approach trajectory (contact-monitored).
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        t_approach = time.perf_counter()
        self._move_joints(
            traj[:, :6], hand_traj, monitor=monitor,
            proximity_hand_traj=traj[:, 6:])
        self.last_execute_timing["approach_motion_s"] = round(
            time.perf_counter() - t_approach, 3)

        # 3. Pregrasp
        self._log_state("pregrasp")
        t_pregrasp = time.perf_counter()
        self._move_hand(pg_hand)
        self.last_execute_timing["pregrasp_motion_s"] = round(
            time.perf_counter() - t_pregrasp, 3)
        self._last_hand_qpos = np.asarray(plan_result.pregrasp_pose, dtype=np.float64)
        self._last_hand_action = np.asarray(pg_hand, dtype=np.float64)

        # 4. Grasp — interpolated ramp pregrasp → grasp for slower close.
        self._log_state("grasp")
        t_grasp = time.perf_counter()
        n_grasp_steps = 50
        for i in range(1, n_grasp_steps + 1):
            t = i / n_grasp_steps
            self._move_hand(pg_hand * (1 - t) + g_hand * t)
            time.sleep(0.01)
        self.last_execute_timing["grasp_motion_s"] = round(
            time.perf_counter() - t_grasp, 3)

        # 5. Squeeze (2× slower than before: sleep 0.01 → 0.02)
        self._log_state("squeeze")
        t_squeeze = time.perf_counter()
        s_hand = g_hand
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.02)
        self.last_execute_timing["squeeze_motion_s"] = round(
            time.perf_counter() - t_squeeze, 3)
        self._last_hand_qpos = np.asarray(plan_result.grasp_pose, dtype=np.float64)
        self._last_hand_action = np.asarray(s_hand, dtype=np.float64)
        self._holding_object = True

        if skip_lift:
            self._log_state("squeeze_done")
            finish_pickup_timing(self.last_execute_timing, execute_started)
            if pickup_trace_id is not None:
                self._timing_recorder.end(pickup_trace_id, outcome="success", lift="skipped")
            return s_hand

        # 6. Lift — no contact monitor: arm is now carrying the object so the
        #    empty-arm baseline is invalid for tau_dev. place() does its own
        #    baseline at the lifted pose.
        self._log_state("lift")
        # Replay only when the arm and the planner-space hand state still
        # match the collision model used for the candidate preflight.
        preflight = getattr(plan_result, "lift_preflight", None)
        traj_lift = None
        if preflight is not None:
            t_lift_check = time.perf_counter()
            live_hand_qpos, hand_state_source = self.get_hand_qpos()
            start_check = check_lift_start(
                self.get_arm_qpos(), preflight.start_full_qpos,
                arm_dof=self.arm_dof, live_hand_qpos=live_hand_qpos)
            self.last_execute_timing.update({
                "lift_start_max_abs_rad": round(start_check.max_abs_rad, 4),
                "lift_start_l2_rad": round(start_check.l2_rad, 4),
                "lift_start_hand_checked": start_check.hand_checked,
                "lift_start_hand_accepted": start_check.hand_accepted,
                "lift_start_hand_max_abs": round(start_check.hand_max_abs, 4),
                "lift_start_hand_l2": round(start_check.hand_l2, 4),
                "lift_start_hand_source": hand_state_source,
            })
            self.last_execute_timing["lift_start_check_s"] = round(
                time.perf_counter() - t_lift_check, 3)
            if start_check.accepted:
                # An override is only a visualisation/legacy representation of
                # this same preflight.  The state gate always comes first.
                traj_lift = np.asarray(
                    lift_traj_override if lift_traj_override is not None
                    else preflight.traj)
                self.last_execute_timing["lift_plan_source"] = (
                    "precomputed" if lift_traj_override is not None
                    else "candidate_preflight")
            else:
                self.last_execute_timing["lift_plan_source"] = (
                    "live_replan_both_mismatch" if not start_check.hand_accepted
                    and not (start_check.max_abs_rad <= 0.12 and start_check.l2_rad <= 0.20)
                    else ("live_replan_hand_mismatch" if not start_check.hand_accepted
                          else "live_replan_arm_mismatch"))

        if traj_lift is None and preflight is None and lift_traj_override is not None:
            # Compatibility for standalone scripts which supply a trajectory
            # but construct PlanResult themselves.  The main pipeline always
            # has a LiftPreflight and therefore takes the guarded branch above.
            traj_lift = np.asarray(lift_traj_override)
            self.last_execute_timing["lift_plan_source"] = "legacy_override"

        if traj_lift is None:
            if planner is None or scene_cfg is None:
                raise LiftExecutionError(
                    "no validated lift trajectory and no planner/scene for a "
                    "live replan; holding the object")
            live_hand_qpos, hand_state_source = self.get_hand_qpos()
            self.last_execute_timing.setdefault("lift_start_hand_source", hand_state_source)
            start_full = np.concatenate([
                self.get_arm_qpos().astype(np.float32),
                np.asarray(live_hand_qpos, dtype=np.float32),
            ])
            t_lift_plan = time.perf_counter()
            try:
                live_preflight = planner.plan_lift_preflight(
                    start_full, scene_cfg, lift_h=lift_height,
                    timing_parent_id=pickup_trace_id, timing_phase="execution")
            except Exception as exc:
                raise LiftExecutionError(
                    f"live lift replan raised {exc!r}; holding the object") from exc
            self.last_execute_timing["lift_runtime_replan_s"] = round(
                time.perf_counter() - t_lift_plan, 3)
            if live_preflight is None:
                raise LiftExecutionError(
                    "live lift replan failed; refusing Cartesian or lateral "
                    "fallback while the object is held")
            traj_lift = live_preflight.traj

        # Hold the squeeze throughout: planner hand columns represent the
        # nominal grasp qpos and could partially open the fingers in flight.
        hand_lift = np.tile(s_hand, (len(traj_lift), 1))
        t_lift_motion = time.perf_counter()
        self.follow_joint_trajectory(traj_lift, hand_lift)
        self.last_execute_timing["lift_motion_s"] = round(
            time.perf_counter() - t_lift_motion, 3)

        self._log_state("lift_done")
        finish_pickup_timing(self.last_execute_timing, execute_started)
        if pickup_trace_id is not None:
            self._timing_recorder.end(
                pickup_trace_id, outcome="success",
                lift_plan_source=self.last_execute_timing.get("lift_plan_source"))
        return s_hand

    def execute_lift(self, lift_traj, hold_hand):
        """JOINT-SPACE lift: follow a pre-PLANNED qpos trajectory with
        ``_move_joints`` instead of ``_move_cartesian``. The cartesian servo
        (``set_servo_cartesian_aa``) throws kinematic errors at borderline wrist
        configs that are UNRECOVERABLE on xarm (force a full program/robot/camera
        restart). Feeding a planned joint trajectory sidesteps the cartesian IK
        entirely, and any infeasibility is caught at PLAN time (skip) rather than
        crashing the robot mid-execution.

        Args:
            lift_traj: (T, dof) planned qpos trajectory (e.g. from the planner's
                       grasp->lift segment). Only the arm joints (``[:, :6]``) are
                       commanded.
            hold_hand: hand command (controller units, e.g. the squeezed ``s_hand``
                       from ``execute(skip_lift=True)``) held constant throughout.
        """
        self._log_state("lift")
        self._holding_object = True
        arm_traj = np.asarray(lift_traj)[:, :self.arm_dof]
        hand_traj = np.tile(np.asarray(hold_hand, dtype=float), (len(arm_traj), 1))
        self.follow_joint_trajectory(arm_traj, hand_traj)  # no monitor: object held
        self._log_state("lift_done")

    def _place_planned(self, plan_result: PlanResult, planner, scene_cfg,
                       lift_height: float, overshoot: float,
                       mcc_model_path: str,
                       debug_dump_dir: str = None,
                       timing_s: Optional[dict] = None,
                       placement_wrist: Optional[np.ndarray] = None,
                       preplace_traj: Optional[np.ndarray] = None,
                       preplace_wrist_target: Optional[np.ndarray] = None,
                       ) -> "Optional[dict]":
        """Descend by replaying a Jacobian-continuation trajectory. Mirror of lift.

        ``lift`` follows a local differential-IK branch to ``wrist z + h``;
        the descent uses the same implementation with the sign flipped. Doing
        it this way instead of streaming Cartesian setpoints
        means the straight line is *checked before the arm moves* (a plan that
        cannot be made returns None here) rather than being left to the
        controller's internal IK, which is free to bend it.

        Contact stop is preserved: ``_move_joints`` ticks the ContactMonitor
        between waypoints and raises ``ContactDetected``, so the arm freezes
        where it touched instead of pushing through.

        Returns the same dict ``place()`` does, or ``None`` if no trajectory
        could be planned.  The production caller then keeps holding the object;
        only standalone diagnostics may choose a different recovery policy.
        """
        target_descend = float(lift_height) + float(overshoot)
        if target_descend <= 0.0:
            raise ValueError("place descent distance must be positive")

        # Read pose and joints from one controller snapshot.  Previously these
        # came from separate get_data() calls and the physical Cartesian pose
        # was then compared directly with FK of a slightly newer/older qpos.
        state = self.arm.get_data()
        link6_now = np.asarray(state["position"], dtype=np.float64).copy()
        live_arm = np.asarray(state["qpos"][:self.arm_dof],
                              dtype=np.float32).copy()
        if (link6_now.shape != (4, 4) or not np.isfinite(link6_now).all()
                or live_arm.shape != (self.arm_dof,)
                or not np.isfinite(live_arm).all()):
            raise RuntimeError("xArm returned an invalid pre-place state")
        live_wrist = link6_now @ self._link6_to_wrist
        hand = np.asarray(plan_result.grasp_pose, dtype=np.float32)
        live_start_full = np.concatenate([live_arm, hand])
        live_fk = planner.fk_wrist(live_start_full)

        model_pos_err, model_rot_err = self._pose_error(live_wrist, live_fk)
        if timing_s is not None:
            timing_s["preplace_model_pos_err_m"] = round(model_pos_err, 6)
            timing_s["preplace_model_rot_err_rad"] = round(model_rot_err, 6)

        chosen_preplace = None
        preplace_reused = False
        correction_required = False
        explicit_placement = placement_wrist is not None

        if not explicit_placement:
            if preplace_traj is not None or preplace_wrist_target is not None:
                raise ValueError(
                    "preplace trajectory/target requires placement_wrist")
            # Compatibility for standalone calls without run_auto's placement
            # contract.  Keep the old "descend here" policy, but define both
            # endpoints from the same planner FK instead of mixing physical
            # Cartesian FK with URDF FK.
            wrist_start_pose = np.asarray(live_fk, dtype=np.float64).copy()
            wrist_place_pose = wrist_start_pose.copy()
            wrist_place_pose[2, 3] -= target_descend
            descend_start = live_start_full
            preplace_source = "live_planner_fk"
        else:
            placement = np.asarray(placement_wrist, dtype=np.float64)
            if placement.shape != (4, 4) or not np.isfinite(placement).all():
                raise ValueError("placement_wrist must be a finite 4x4 pose")

            # placement_wrist is the nominal release pose.  Overshoot extends
            # below it; the ordinary lift-height segment remains exactly the
            # runner's table-to-pre-place clearance.
            wrist_place_pose = placement.copy()
            wrist_place_pose[2, 3] -= float(overshoot)
            if preplace_wrist_target is None:
                wrist_start_pose = placement.copy()
                wrist_start_pose[2, 3] += float(lift_height)
                supplied_target_valid = False
                target_pos_err = target_rot_err = float("inf")
            else:
                supplied_target = np.asarray(
                    preplace_wrist_target, dtype=np.float64)
                if (supplied_target.shape != (4, 4)
                        or not np.isfinite(supplied_target).all()):
                    raise ValueError(
                        "preplace_wrist_target must be a finite 4x4 pose")
                wrist_start_pose = placement.copy()
                wrist_start_pose[2, 3] += float(lift_height)
                target_pos_err, target_rot_err = self._pose_error(
                    supplied_target, wrist_start_pose)
                supplied_target_valid = (
                    target_pos_err <= XARM_PREPLACE_REUSE_POS_TOL_M
                    and target_rot_err <= XARM_PREPLACE_REUSE_ROT_TOL_RAD)

            live_pos_err, live_rot_err = self._pose_error(
                live_wrist, wrist_start_pose)
            candidate = (None if preplace_traj is None else
                         np.asarray(preplace_traj, dtype=np.float64))
            candidate_valid = (
                candidate is not None and candidate.ndim == 2
                and len(candidate) > 0
                and candidate.shape[1] >= self.arm_dof
                and np.isfinite(candidate[:, :self.arm_dof]).all())
            preplace_reused = bool(
                candidate_valid and supplied_target_valid
                and live_pos_err <= XARM_PREPLACE_REUSE_POS_TOL_M
                and live_rot_err <= XARM_PREPLACE_REUSE_ROT_TOL_RAD)

            if timing_s is not None:
                timing_s.update({
                    "preplace_target_pos_err_m": round(target_pos_err, 6),
                    "preplace_target_rot_err_rad": round(target_rot_err, 6),
                    "preplace_live_pos_err_m": round(live_pos_err, 6),
                    "preplace_live_rot_err_rad": round(live_rot_err, 6),
                    "preplace_reused": preplace_reused,
                })

            if preplace_reused:
                chosen_preplace = candidate
                preplace_source = "runner_reposition"
                print(
                    "[xarm] pre-place already reached by reposition; "
                    f"reusing endpoint (live={live_pos_err * 1000:.1f}mm/"
                    f"{np.degrees(live_rot_err):.1f}deg, "
                    f"model={model_pos_err * 1000:.1f}mm/"
                    f"{np.degrees(model_rot_err):.1f}deg)", flush=True)
            else:
                # Match the FR3 contract: a stale, absent, or inaccurate
                # reposition endpoint gets a collision-checked correction to
                # the explicit high pose instead of being fed into the strict
                # vertical-stroke start gate.
                print(
                    "[xarm] pre-place endpoint needs correction "
                    f"(target={target_pos_err * 1000:.1f}mm/"
                    f"{np.degrees(target_rot_err):.1f}deg, "
                    f"live={live_pos_err * 1000:.1f}mm/"
                    f"{np.degrees(live_rot_err):.1f}deg); planning correction",
                    flush=True)
                t_preplace_plan = time.perf_counter()
                chosen_preplace = planner.plan_cartesian_pose(
                    live_start_full, wrist_start_pose,
                    scene_cfg=scene_cfg, include_obj_obstacle=False,
                    debug_dump_dir=debug_dump_dir,
                    timing_phase="execution")
                if timing_s is not None:
                    timing_s["preplace_plan_s"] = round(
                        time.perf_counter() - t_preplace_plan, 3)
                if chosen_preplace is None:
                    raise RuntimeError(
                        "xArm pre-place correction failed; object remains held")
                chosen_preplace = np.asarray(chosen_preplace, dtype=np.float64)
                if (chosen_preplace.ndim != 2 or len(chosen_preplace) == 0
                        or chosen_preplace.shape[1] < self.arm_dof
                        or not np.isfinite(
                            chosen_preplace[:, :self.arm_dof]).all()):
                    raise RuntimeError(
                        "xArm pre-place correction returned an invalid trajectory; "
                        "object remains held")
                preplace_source = "live_correction"
                correction_required = True

            descend_start = np.concatenate([
                np.asarray(chosen_preplace[-1, :self.arm_dof],
                           dtype=np.float32),
                hand,
            ])

        if timing_s is not None:
            timing_s["preplace_plan_source"] = preplace_source

        t_plan = time.perf_counter()
        from autodex.utils.conversion import cart2se3

        object_at_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        object_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ object_at_grasp
        # Keep the attached payload in the exact same FK frame as the descent
        # start state.  The physical-vs-model residual has already been logged
        # and bounded separately above.
        object_at_descend_start = (
            planner.fk_wrist(descend_start) @ object_in_wrist)
        traj = planner.plan_vertical_stroke(
            descend_start, wrist_start_pose, wrist_place_pose,
            expected_travel_m=float(target_descend),
            travel_tolerance_m=XARM_VERTICAL_STROKE_Z_TOL_M,
            scene_cfg=scene_cfg,
            include_obj_obstacle=False,
            attached_object_pose_at_start=object_at_descend_start,
            label="xarm place descent",
            debug_dump_dir=debug_dump_dir,
            timing_phase="execution",
        )
        if timing_s is not None:
            timing_s["descend_plan_s"] = round(time.perf_counter() - t_plan, 3)
        if traj is None:
            failure = getattr(planner, "_last_vertical_stroke_result", None)
            failure_code = getattr(failure, "failure_code", None)
            failure_detail = getattr(failure, "failure_detail", None)
            if timing_s is not None:
                timing_s["failure_code"] = failure_code
                timing_s["failure_detail"] = failure_detail
            suffix = ("" if failure_code is None else
                      f" ({failure_code}: {failure_detail})")
            print(f"[place] Jacobian descent preflight failed{suffix}")
            return None
        print(f"[place] planned Jacobian straight descent, {len(traj)} samples")

        # Both the correction and the descent are fully preflighted before the
        # correction moves an already-held object.  The runner's reposition is
        # not replayed when it passed the live-state gate.
        if correction_required:
            t_preplace_motion = time.perf_counter()
            self.follow_joint_trajectory(chosen_preplace)
            if timing_s is not None:
                timing_s["preplace_motion_s"] = round(
                    time.perf_counter() - t_preplace_motion, 3)

        # Recheck the physical endpoint immediately before descending.  This
        # catches a stalled correction or drift during a long preflight while
        # retaining the object in the hand.
        pre_descent_state = self.arm.get_data()
        pre_descent_link6 = np.asarray(
            pre_descent_state["position"], dtype=np.float64)
        pre_descent_wrist = pre_descent_link6 @ self._link6_to_wrist
        if explicit_placement:
            final_pos_err, final_rot_err = self._pose_error(
                pre_descent_wrist, wrist_start_pose)
            if timing_s is not None:
                timing_s["preplace_final_pos_err_m"] = round(
                    final_pos_err, 6)
                timing_s["preplace_final_rot_err_rad"] = round(
                    final_rot_err, 6)
            if (final_pos_err > XARM_PREPLACE_REUSE_POS_TOL_M
                    or final_rot_err > XARM_PREPLACE_REUSE_ROT_TOL_RAD):
                raise RuntimeError(
                    "xArm pre-place alignment failed after planning "
                    f"({final_pos_err * 1000:.1f}mm/"
                    f"{np.degrees(final_rot_err):.1f}deg); object remains held")
        start_z = float(pre_descent_link6[2, 3])

        mon = ContactMonitor(
            self.arm.arm,
            mcc_model_path,
            watch_joints=(1, 2),
            thresh_nm=XARM_PLACE_CONTACT_THRESHOLD_NM,
            sustained_ticks=XARM_PLACE_CONTACT_SUSTAINED_TICKS,
            startup_blank_s=XARM_PLACE_CONTACT_STARTUP_BLANK_S,
        )
        print(
            "[place] contact monitor: "
            f"threshold={XARM_PLACE_CONTACT_THRESHOLD_NM:.1f}Nm, "
            f"sustained={XARM_PLACE_CONTACT_SUSTAINED_TICKS} ticks, "
            f"startup_blank={XARM_PLACE_CONTACT_STARTUP_BLANK_S:.1f}s"
        )
        t_warmup = time.perf_counter()
        mon.warmup(seconds=1.0)
        if timing_s is not None:
            timing_s["contact_monitor_warmup_s"] = round(
                time.perf_counter() - t_warmup, 3)

        arm_traj = traj[:, :6]

        contact = False
        t_motion = time.perf_counter()
        try:
            # No hand trajectory: the squeeze pose set during execute() stays
            # commanded (the hand controller re-sends its last action at 100 Hz),
            # so the grip is held to the end of the descent. Feeding the
            # planner's hand columns instead would command grasp_pose, which is
            # less closed and would open the fingers mid-descent.
            self._move_joints(arm_traj, None, monitor=mon)
        except ContactDetected as exc:
            contact = True
            print(f"[place] CONTACT — {exc}")

        final_z = float(self.arm.get_data()["position"][2, 3])
        if timing_s is not None:
            timing_s["descend_motion_s"] = round(
                time.perf_counter() - t_motion, 3)
        descended = start_z - final_z
        print(f"[place] descended {descended*1000:.1f}mm of target "
              f"{target_descend*1000:.0f}mm  (contact={contact})")
        return {"descended": float(descended),
                "stopped_on_contact": bool(contact),
                "target": float(target_descend),
                "released": False,
                "mode": "planned",
                "preplace_reused": bool(preplace_reused),
                "preplace_plan_source": preplace_source}

    def place(self, plan_result: PlanResult, lift_height: float = 0.10,
              overshoot: float = 0.0,
              mcc_model_path: str = None,
              descend_time_s: float = 4.0,
              total_time_s: float = 6.4,
              planner=None, scene_cfg=None, debug_dump_dir: str = None,
              log_path: str = None,
              placement_wrist: Optional[np.ndarray] = None,
              preplace_traj: Optional[np.ndarray] = None,
              preplace_wrist_target: Optional[np.ndarray] = None) -> dict:
        """Descend with mcc_minimal admittance control. Target z = lift_pose -
        (lift_height + overshoot) — i.e. with overshoot=0, the arm targets the
        original grasp z (where the object came from). Overshoot can be set >0
        to bias the motion downward past the original z if contact_stop is
        unreliable, but with the tau-model contact check this is usually 0.

        ``placement_wrist`` / ``preplace_*`` are the shared runner contract.
        Like the FR3 adapter, xArm reuses a just-executed reposition only when
        its live wrist is within 5 mm / 3 degrees of the requested high pose;
        otherwise it plans a correction while keeping the object held.  The
        strict Jacobian start check then compares planner-frame quantities,
        never physical Cartesian FK against URDF FK.

        paradex's XArmController control_loop stays alive (so its recording
        keeps running); mcc only computes q_ref and writes to xarm_ctrl.action,
        which paradex sends. On contact (tau_ext > threshold) we freeze and break."""
        from pathlib import Path

        place_started = time.perf_counter()
        place_timing = new_place_timing()
        # xArm's release is intentionally done by the runner while recording
        # remains active. Its duration is therefore a runner span, not a
        # hidden adapter-side zero.
        place_timing["release_execution"] = "external_runner"
        self.last_place_timing = place_timing

        if not plan_result.success:
            return {"descended": 0.0, "stopped_on_contact": False,
                    "target": 0.0, "released": False, "mode": "skipped",
                    "timing_s": finish_place_timing(place_timing, place_started)}

        if mcc_model_path is None:
            mcc_model_path = str(Path.home() / "shared_data" / "AutoDex"
                                 / "weights" / "tau_model" / "inspire_left.pt")

        self._log_state("place")
        target_descend = lift_height + overshoot

        # The main pipeline is strict: a failed Jacobian vertical continuation is
        # not silently replaced by a Cartesian servo whose internal IK may
        # swing laterally.  The legacy no-planner path remains only for direct
        # hardware diagnostics outside the pipeline.
        if planner is not None:
            planned = self._place_planned(
                plan_result, planner, scene_cfg, lift_height, overshoot,
                mcc_model_path, debug_dump_dir=debug_dump_dir,
                timing_s=place_timing,
                placement_wrist=placement_wrist,
                preplace_traj=preplace_traj,
                preplace_wrist_target=preplace_wrist_target)
            if planned is not None:
                planned["timing_s"] = finish_place_timing(
                    place_timing, place_started)
                return planned
            failure_code = place_timing.get("failure_code")
            failure_detail = place_timing.get("failure_detail")
            suffix = ("" if failure_code is None else
                      f" ({failure_code}: {failure_detail})")
            raise RuntimeError(
                "Jacobian place descent preflight failed"
                f"{suffix}; object remains held")

        start_pose = self.arm.get_data()["position"].copy()   # 4x4 homo, link6 in world
        current_pos = start_pose.copy()
        target_pose = start_pose.copy()
        target_pose[2, 3] -= target_descend                   # straight down in world z

        # Adapter: paradex's control thread keeps running (so its recording
        # continues), but mcc's writes are redirected to xarm_ctrl.action and
        # paradex sends them. Reads delegate to the raw XArmAPI handle.
        xarm_ctrl = self.arm
        xarm_handle = xarm_ctrl.arm   # raw XArmAPI

        # mcc-needed handle setup (one-shot, harmless to paradex).
        t_control_setup = time.perf_counter()
        xarm_handle.set_report_tau_or_i(1)
        xarm_handle.set_collision_sensitivity(0)

        # Tau model (copied from mcc_minimal/fit_tau_model.py; self-contained,
        # only depends on numpy + torch).
        import torch  # noqa: E402
        from autodex.executor.tau_model import load_model, build_input

        print(f"[place] loading mcc model: {mcc_model_path}")
        model = load_model(mcc_model_path)
        place_timing["control_setup_s"] = round(
            time.perf_counter() - t_control_setup, 3)

        # Contact-stop loop: use the learned tau model only to estimate tau_ext;
        # on contact (tau_ext > threshold), freeze q_des at current pose (paradex
        # holds it) and break. No yield, no bounce.
        DT = 0.01
        FILTER_ALPHA = 0.1
        QDOT_SMOOTH_ALPHA = 0.1
        WARMUP_SEC = 1.0
        # Baseline noise per joint (Nm) — from mcc DEADBAND_J. Kept for ref;
        # the actual place threshold is shared with the planned path above.
        DEADBAND_J = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 0.5])
        CONTACT_THRESH = np.full(6, XARM_PLACE_CONTACT_THRESHOLD_NM)
        # Same constants mcc uses to convert _joints_torque (raw current in
        # whatever units xarm reports) to Nm. tau_motor = I * KT * GEAR.
        KT = np.array([0.067, 0.067, 0.0573, 0.0573, 0.056, 0.056])
        GEAR = np.full(6, 100.0)

        def _read():
            _, q_deg = xarm_handle.get_servo_angle()
            q = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64))
            # tau_motor: raw _joints_torque × KT × GEAR → Nm. Matches what the
            # MLP was trained against in mcc's stream source path.
            I = np.asarray(xarm_handle._arm._joints_torque[:6], dtype=np.float64)
            tau = I * KT * GEAR
            return q, tau

        def _push_pose(pose4x4):
            """Push 4x4 link6 pose. paradex's control_loop sees non-(6,) shape
            and routes through set_servo_cartesian_aa — internal IK tracks
            current pose continuously, no arbitrary elbow flip."""
            with xarm_ctrl.lock:
                xarm_ctrl.action = pose4x4.astype(np.float64)
                xarm_ctrl.is_servo = True

        # Hold start_pose during warmup.
        _push_pose(start_pose)

        # Warmup: prime tau_filt at hold pose.
        tau_filt = np.zeros(6)
        qdot_smooth = np.zeros(6)
        q_last, t_last = None, None
        t_warmup = time.perf_counter()
        t_warm0 = time.time()
        while time.time() - t_warm0 < WARMUP_SEC:
            q, tau_motor = _read()
            t_now = time.time()
            if q_last is not None and t_last is not None:
                dt = max(t_now - t_last, 1e-4)
                qdot = (q - q_last) / dt
            else:
                qdot = np.zeros(6)
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth
            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt
            _push_pose(start_pose)
            time.sleep(DT)
        place_timing["contact_monitor_warmup_s"] = round(
            time.perf_counter() - t_warmup, 3)
        # Pose-dependent baseline (MLP residual + any reading offset). Subtract
        # this from later tau_filt so the contact check only sees DEVIATIONS.
        tau_baseline = tau_filt.copy()
        print(f"[place] warmup done. baseline tau_filt = {tau_baseline.round(2)}  (subtracted from now on)")
        print(f"[place] contact threshold per joint = {CONTACT_THRESH.round(2)}")

        # Descend loop with contact stop.
        log = [] if log_path else None
        contact = False
        contact_t = None
        sustained = 0
        last_print_t = -1.0
        t_descent_motion = time.perf_counter()
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < total_time_s:
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += DT
            t = time.time() - t0

            q, tau_motor = _read()
            t_now = time.time()
            dt = max(t_now - t_last, 1e-4)
            qdot = (q - q_last) / dt
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth

            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt

            # Contact check: only on joints 2 and 3 (shoulder/elbow — most
            # informative for downward contact). Skip first STARTUP_BLANK_S to
            # avoid warmup->descend dynamics spike. Require SUSTAINED_TICKS
            # consecutive ticks above threshold so single-tick noise doesn't fire.
            STARTUP_BLANK_S = XARM_PLACE_CONTACT_STARTUP_BLANK_S
            SUSTAINED_TICKS = XARM_PLACE_CONTACT_SUSTAINED_TICKS
            CONTACT_JOINTS = (1, 2)   # 0-indexed: joints 2 and 3
            tau_dev = tau_filt - tau_baseline
            ratio = np.abs(tau_dev) / np.maximum(CONTACT_THRESH, 1e-6)
            ratio_watch = ratio[list(CONTACT_JOINTS)]
            crossed = bool(np.any(ratio_watch > 1.0))
            if crossed and t > STARTUP_BLANK_S:
                sustained += 1
            else:
                sustained = 0
            # Periodic dump every 0.2s so torque evolution is visible. One
            # self-overwriting line: at 5 dumps/s a 6.4s descent is 30+ lines
            # of scrollback that bury the result of the trial.
            if t - last_print_t >= 0.2:
                last_print_t = t
                line = (f"[place] t={t:5.2f}s  tau_dev={tau_dev.round(2)}  "
                        f"ratio={ratio.round(2)}")
                sys.stdout.write("\r" + line.ljust(110))
                sys.stdout.flush()

            if (not contact) and (sustained >= SUSTAINED_TICKS):
                contact = True
                contact_t = t
                sys.stdout.write("\n")
                print(f"[place] CONTACT at t={t:.2f}s (watching joints {[i+1 for i in CONTACT_JOINTS]})")
                print(f"  tau_dev = {tau_dev.round(2)}")
                print(f"  ratio   = {ratio.round(2)}")
                print(f"  thresh  = {CONTACT_THRESH.round(2)}")
                # Freeze at current actual pose; break.
                cur_pose = self.arm.get_data()["position"].copy()
                _push_pose(cur_pose)
                if log is not None:
                    log.append((t, *q, *tau_dev, 1))
                break

            # Cartesian lerp: translate z toward target, rotation held constant.
            alpha = min(1.0, max(0.0, t / descend_time_s))
            pose_des = start_pose.copy()
            pose_des[2, 3] = (1 - alpha) * start_pose[2, 3] + alpha * target_pose[2, 3]
            _push_pose(pose_des)

            if log is not None:
                log.append((t, *q, *tau_dev, 0))

        if not contact:
            sys.stdout.write("\n")
            print(f"[place] no contact within {total_time_s}s — reached target. final pose held.")

        # Optional CSV log.
        if log_path and log:
            import csv as _csv
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["t"] + [f"q{i}" for i in range(6)] +
                           [f"tau_ext{i}" for i in range(6)] + ["contact"])
                w.writerows(log)
            print(f"[place] log -> {log_path}")

        # Read final pose for descended-distance reporting.
        try:
            _, final_pos_xarm = xarm_handle.get_position(is_radian=True)
        except Exception:
            final_pos_xarm = None

        # Compute descended distance from final pose.
        if final_pos_xarm is not None:
            final_z = final_pos_xarm[2] / 1000.0  # mm -> m
            descended = current_pos[2, 3] - final_z
        else:
            descended = float("nan")
        print(f"[place] descended {descended*1000:.1f}mm of target {target_descend*1000:.0f}mm "
              f"({'contact stop' if contact else 'reached target window'})")

        # paradex thread never stopped; it's been forwarding mcc's q_ref the
        # whole time, so no re-init needed. Just leave its action at last q_ref.

        self._log_state("place_done")
        place_timing["descend_motion_s"] = round(
            time.perf_counter() - t_descent_motion, 3)
        return {"descended": float(descended), "stopped_on_contact": bool(contact),
                "contact_t_s": float(contact_t) if contact_t is not None else None,
                "target": float(target_descend), "released": False,
                "mode": "cartesian_admittance",
                "timing_s": finish_place_timing(place_timing, place_started)}

    def release(self, plan_result: PlanResult, slow_factor: float = 1.0):
        """Release object while leaving the arm at its current raised pose.

        slow_factor > 1.0 stretches the open ramp (e.g. 4.0 → 2s instead of 0.5s).
        """
        if not plan_result.success:
            return

        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        self._release_auto(pg_hand, g_hand, slow_factor=slow_factor)
        self._last_hand_qpos = np.asarray(plan_result.pregrasp_pose, dtype=np.float64)
        self._last_hand_action = np.asarray(pg_hand, dtype=np.float64)
        self._holding_object = False

    def _release_auto(self, pg_hand, g_hand, slow_factor: float = 1.0):
        """Reverse squeeze -> grasp -> pregrasp, then STOP.
        Hand opening to hand_init and arm retract back to init are intentionally
        skipped — user resets those manually after inspecting the placed object."""
        sl = self.squeeze_level

        # Reverse squeeze (matches squeeze ramp speed: 0.02s per step).
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.02 * slow_factor)

        # Interpolated open ramp grasp → pregrasp (mirrors close ramp).
        n_open_steps = 50
        for i in range(1, n_open_steps + 1):
            t = i / n_open_steps
            self._move_hand(g_hand * (1 - t) + pg_hand * t)
            time.sleep(0.01 * slow_factor)

    def reset(self, plan_result: PlanResult,
              planner, scene_cfg: dict) -> dict:
        """Leave a released object via +Z clearance, then reset.

        The placed-object pose is reconstructed at the actual release wrist,
        which also covers a place descent stopped early by contact.  The hand
        opens first; then a +Z Jacobian clearance and the following joint-space
        retract are both planned before either arm segment moves.
        """
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log
        self.set_speed_profile_planner(planner)
        self._holding_object = False

        from autodex.utils.conversion import cart2se3, se32cart

        # 0. Snapshot released object pose (robot frame) under rigid-grasp.
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released_obj_pose = T_wrist_now @ T_obj_in_wrist
        log["T_wrist_grasp"] = plan_result.wrist_se3.tolist()
        log["T_wrist_at_reset_start"] = T_wrist_now.tolist()
        log["T_obj_in_wrist"] = T_obj_in_wrist.tolist()
        log["released_obj_pose_robot"] = released_obj_pose.tolist()

        # 1. Open hand pregrasp → openpose first (mirrors reset_hybrid step 0).
        #    With fingers open, plan_js_to_init's trajopt has more clearance
        #    around the placed obj, reducing TRAJOPT_FAIL rate.
        op_raw = getattr(plan_result, "openpose_pose", None)
        pg_raw = getattr(plan_result, "pregrasp_pose", None)
        self._log_state("hand_open")
        if op_raw is not None and pg_raw is not None:
            pg = np.asarray(pg_raw, dtype=np.float64)
            op = np.asarray(op_raw, dtype=np.float64)
            for i in range(11):
                a = i / 10.0
                qpos = (1.0 - a) * pg + a * op
                self._move_hand(self._convert(qpos))
                time.sleep(0.05)
            hold_hand_raw = op
        else:
            hold_hand_raw = (np.asarray(plan_result.pregrasp_pose, dtype=np.float64)
                              if plan_result.pregrasp_pose is not None
                              else self._hand_init)
        self._last_hand_qpos = np.asarray(hold_hand_raw, dtype=np.float64).copy()

        # 2. Preflight +Z clearance and the following reset with the object at
        #    its actual release pose.  Planning both before motion prevents a
        #    fallback lateral sweep through the newly placed object.
        self._log_state("reset_preflight")
        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released_obj_pose).tolist()
        self.set_speed_profile_object(new_scene)
        cur_qpos = np.asarray(
            self.arm.get_data()["qpos"][:self.arm_dof], dtype=np.float32)
        start_full = np.concatenate([
            cur_qpos,
            np.asarray(hold_hand_raw, dtype=np.float32),
        ])
        wrist_start = self.get_wrist_pose()
        wrist_clear = wrist_start.copy()
        wrist_clear[2, 3] += POST_RELEASE_CLEARANCE_M
        t_vertical_plan = time.perf_counter()
        vertical_traj = planner.plan_vertical_stroke(
            start_full, wrist_start, wrist_clear,
            expected_travel_m=POST_RELEASE_CLEARANCE_M,
            travel_tolerance_m=1.0e-4,
            scene_cfg=new_scene,
            include_obj_obstacle=True,
            label="xarm post-release reset clearance",
            timing_phase="post_execution",
        )
        log["steps"]["vertical_plan_s"] = round(
            time.perf_counter() - t_vertical_plan, 3)
        if vertical_traj is None:
            log["retract_mode"] = "vertical_clearance_failed"
            raise RuntimeError(
                "reset(): +Z clearance from the release pose failed; "
                "leaving the open hand and arm in place"
            )

        clear_view_arm = self._clear_view.copy()
        t_plan0 = time.perf_counter()
        retract_traj = planner.plan_js_to_init(
            new_scene, vertical_traj[-1, :self.arm_dof],
            start_hand_qpos=hold_hand_raw,
            goal_arm_qpos=clear_view_arm[:6],
        )
        log["steps"]["retract_plan_s"] = round(
            time.perf_counter() - t_plan0, 3)
        if retract_traj is None:
            log["retract_mode"] = "replan_failed"
            raise RuntimeError(
                "reset(): post-clearance plan_js_to_init failed; arm remains "
                "at the release pose because both segments preflight before motion"
            )
        log["retract_mode"] = "jacobian_clearance_then_replanned"

        # 3. Execute the vertical clearance first with the open hand fixed.
        self._log_state("post_release_clearance")
        t_vertical_motion = time.perf_counter()
        vertical_hand = np.tile(
            self._convert(np.asarray(hold_hand_raw, dtype=np.float64)),
            (len(vertical_traj), 1),
        )
        self.follow_joint_trajectory(vertical_traj, vertical_hand)
        log["steps"]["vertical_motion_s"] = round(
            time.perf_counter() - t_vertical_motion, 3)

        # Then execute the planner-generated retract. No contact monitor here:
        # its dynamics lie outside the tau model's training distribution.
        self._log_state("arm_retract")
        t1 = time.perf_counter()
        arm_traj = retract_traj[:, :6]
        hand_traj = np.array([self._convert(retract_traj[i, 6:])
                              for i in range(len(retract_traj))])
        self._move_joints(
            arm_traj, hand_traj,
            proximity_hand_traj=retract_traj[:, 6:])
        log["steps"]["arm_retract_s"] = round(
            time.perf_counter() - t1, 3)

        # 3. Verify final pose — RAISE if arm didn't actually reach
        #    clear_view (stall, partial traj, etc.) so the caller doesn't
        #    silently start the next cycle from a bad pose.
        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view_arm[:6]))
        log["final_qpos_err"] = err
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            raise RuntimeError(
                f"reset(): retract finished with final_qpos_err={err:.3f} > 0.1. "
                f"final_qpos={final_qpos.round(3)}  clear_view={clear_view_arm[:6].round(3)}"
            )
        return log

    def reset_hybrid(self, plan_result: PlanResult,
                      planner, scene_cfg: dict) -> dict:
        """Hybrid retract: sequential [1, 2, 0] first (moves arm base/shoulder/
        elbow away from the placed object — these are large, safe motions),
        then cuRobo plans the remaining wrist joints (3, 4, 5) to clear_view
        with collision-free self/world check.

        Steps:
          0. Open hand to hand_init.
          1. Snapshot placed object pose (from current wrist + rigid grasp).
          2. Sequential move on joints [1, 2, 0] to their clear_view values.
          3. cuRobo plan_js_to_init from current full qpos → clear_view arm
             goal. Hand stays at hand_init throughout. Collision world is
             scene_cfg with the placed object's snapshot pose as obstacle.
          4. Execute the planned trajectory.
        """
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log
        self.set_speed_profile_planner(planner)
        self._holding_object = False

        from autodex.utils.conversion import cart2se3, se32cart

        # 0. Slowly open fingers pregrasp → openpose (10 linear steps). Keep
        #    openpose for the rest of retract (skip hand_init). If openpose
        #    not given, fall back to single jump to hand_init (legacy path).
        op_raw = getattr(plan_result, "openpose_pose", None)
        pg_raw = getattr(plan_result, "pregrasp_pose", None)
        self._log_state("hand_open")
        t1 = time.time()
        if op_raw is not None and pg_raw is not None:
            pg = np.asarray(pg_raw, dtype=np.float64)
            op = np.asarray(op_raw, dtype=np.float64)
            for i in range(11):
                a = i / 10.0
                qpos = (1.0 - a) * pg + a * op
                self._move_hand(self._convert(qpos))
                time.sleep(0.05)
            hold_hand_raw = op
        else:
            hold_hand_raw = self._hand_init
            self._move_hand(self._convert(hold_hand_raw))
            time.sleep(0.3)
        self._last_hand_qpos = np.asarray(hold_hand_raw, dtype=np.float64).copy()
        log["steps"]["hand_open_s"] = round(time.time() - t1, 2)

        # 1. Snapshot placed object pose under rigid grasp assumption.
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released_obj_pose = T_wrist_now @ T_obj_in_wrist
        log["released_obj_pose_robot"] = released_obj_pose.tolist()

        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released_obj_pose).tolist()
        self.set_speed_profile_object(new_scene)

        # 2. Sequential on joints [1, 2, 0] only — coarse arm motion away from
        #    the just-placed object, before wrist replanning.
        clear_view = self._clear_view.copy()
        self._log_state("seq_base_shoulder")
        t1 = time.time()
        coarse_order = [1, 2, 0]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            coarse_order = [2, 1, 0]
        self._move_joint_sequential(clear_view[:6], coarse_order,
                                     threshold=0.06,
                                     first_vel_limit=0.02)
        log["steps"]["seq_arm_s"] = round(time.time() - t1, 2)

        # 3. cuRobo plan_js for remaining wrist joints (3, 4, 5). plan_js_to_init
        #    plans the full 22-DOF trajectory but joints 0/1/2 are already at
        #    clear_view so only 3/4/5 actually change. Self/world collision
        #    handled by cuRobo trajopt.
        self._log_state("plan_wrist")
        t1 = time.time()
        cur_qpos = self.arm.get_data()["qpos"]
        wrist_traj = planner.plan_js_to_init(
            new_scene, cur_qpos,
            start_hand_qpos=hold_hand_raw,
            goal_arm_qpos=clear_view[:6],
        )
        log["steps"]["wrist_plan_s"] = round(time.time() - t1, 2)
        if wrist_traj is None:
            log["wrist_plan_mode"] = "failed"
            log["retract_mode"] = "hybrid_wrist_failed"
            self._log_state("reset_done")
            log["total_s"] = round(time.time() - t_start, 2)
            raise RuntimeError(
                "reset_hybrid(): plan_js_to_init returned None — wrist retract "
                "not safe. Inspect placed object pose / scene_cfg."
            )
        log["wrist_plan_mode"] = "planned"
        t1 = time.time()
        arm_traj = wrist_traj[:, :6]
        hand_traj = np.array([self._convert(wrist_traj[i, 6:])
                              for i in range(len(wrist_traj))])
        self._move_joints(
            arm_traj, hand_traj,
            proximity_hand_traj=wrist_traj[:, 6:])
        log["steps"]["wrist_exec_s"] = round(time.time() - t1, 2)

        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view[:6]))
        log["final_qpos_err"] = err
        log["retract_mode"] = "hybrid"
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            print(f"[reset_hybrid] WARNING: final_qpos_err={err:.3f} > 0.1  "
                  f"final={final_qpos.round(3)}  target={clear_view[:6].round(3)}")
        return log

    def reset_fallback(self, plan_result: PlanResult, planner=None,
                       scene_cfg: Optional[dict] = None) -> dict:
        """Reset path for failed grasps (approach contact or charuco fail).
        Open hand to hand_init, then sequentially move arm to clear_view
        (joint 0 -40° from XARM_INIT) via [1, 2, 5, 0, 3, 4] (mirror if
        joint 1 below init). No planner involvement."""
        # Kept for the common executor contract; this legacy fallback is a
        # hardware-local sequential motion and does not use planner/scene.
        del planner, scene_cfg
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log

        init_hand = self._convert(self._hand_init)

        # 1. Open hand to hand_init.
        self._log_state("hand_init")
        t1 = time.time()
        self._move_hand(init_hand)
        time.sleep(0.5)
        log["steps"]["hand_open_s"] = round(time.time() - t1, 2)

        # 2. Sequential arm retract to clear-view pose (joint 0 -40° from
        #    XARM_INIT). No contact monitor — sequential motion has high
        #    per-joint acceleration that breaks the tau_model baseline.
        self._log_state("clear_view")
        t1 = time.time()
        clear_view = self._clear_view.copy()
        execute_order = [1, 2, 5, 0, 3, 4]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            execute_order = [2, 1, 5, 0, 3, 4]
        self._move_joint_sequential(clear_view[:6], execute_order,
                                     threshold=0.06,
                                     first_vel_limit=0.02)
        log["steps"]["arm_retract_s"] = round(time.time() - t1, 2)

        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view[:6]))
        log["final_qpos_err"] = err
        log["retract_mode"] = "fallback_sequential"
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            # Don't abort — fixed-trajectory sequential retract. Caller decides
            # what to do with a partial result via log["final_qpos_err"].
            print(f"[reset_fallback] WARNING: final_qpos_err={err:.3f} > 0.1  "
                  f"final={final_qpos.round(3)}  target={clear_view[:6].round(3)}")
        return log

    def shutdown(self):
        self._speed_profile_mesh_executor.shutdown(
            wait=False, cancel_futures=True)
        self.arm.end()
        self.hand.end()
