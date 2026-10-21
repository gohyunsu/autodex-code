"""Plan the open-hand socket-axis exit before a cylinder insertion stroke.

This reuses the saved, audited axial arm path in reverse and stock cuRobo
collision checking. The key is treated as staying at the nominal inserted
pose after release. Planning does not claim that release or insertion worked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from autodex.utils.conversion import se32cart
from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT

from .geometry import validate_se3
from .guarded_axial_handoff import verify_guarded_axial_handoff
from .path_audit import PathAuditLimits
from .world import validated_frozen_socket_pose


SCHEMA = "precision_insertion_socket_release_plan_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_path(planner, scene: dict, path: np.ndarray) -> dict:
    """Use the same cuRobo rollout used by the socket-axis planner."""
    from autodex.planner.jacobian_stroke import _check_states_batch
    from autodex.planner.planner import _to_curobo_world

    planner._set_motion_world(_to_curobo_world(scene))
    feasible, status, detail = _check_states_batch(planner, path)
    return {"all_feasible": bool(np.all(feasible)),
            "first_invalid_index": detail["first_invalid_index"],
            "status": status, "backend": detail["backend"]}


def _path(value, name: str, *, hand: np.ndarray,
          limits: PathAuditLimits) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 13 or len(path) < 2 or
            not np.all(np.isfinite(path)) or
            not np.allclose(path[:, 7:], hand, atol=1e-5, rtol=0) or
            np.max(np.abs(np.diff(path, axis=0))) >
            limits.max_joint_step_rad + 1e-6):
        raise ValueError(f"{name} lacks a dense open-hand FR3/Inspire path")
    return path


def plan_socket_release_exit(
    *, runner, planner, handoff_report_path: Path,
    limits: PathAuditLimits, output_dir: Path,
) -> Path:
    """Save a stock-speed release exit bound to the live axial handoff.

    The planned order is open the hand at nominal depth, withdraw along the
    saved socket axis, then stock joint-space retract to FR3 clear view.
    During opening the gripped key is excluded from robot collision checks;
    after opening it is included as a fixed inserted-key hypothesis.
    """
    if not isinstance(limits, PathAuditLimits):
        raise TypeError("socket release needs the existing path-audit limits")
    limits.validate()
    attempt = runner.active_attempt
    if (attempt is None or runner.mode.family != "cylinder" or
            runner._attempt_dir is None or runner._preflight is None):
        raise ValueError("socket release needs an active cylinder attempt")
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    handoff = verify_guarded_axial_handoff(handoff_path)
    if (handoff["attempt_id"] != attempt.attempt_id or
            handoff["candidate_id"] != attempt.candidate_id or
            handoff["session_calibration_sha256"] != runner.session_sha256 or
            handoff_path.parent.parent !=
            (Path(runner._attempt_dir) / "guarded_axial_handoffs").resolve()):
        raise ValueError("release path uses another insertion handoff")
    validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    postlift_path = Path(handoff["postlift_report_path"])
    postlift = json.loads(postlift_path.read_text(encoding="utf-8"))
    pose = validate_se3(postlift["targets"]["T_robot_key_verification"],
                        name="nominal inserted key pose")
    archive_path = Path(handoff["trajectory_archive_path"])
    if _sha(archive_path) != handoff["trajectory_archive_sha256"]:
        raise ValueError("release source axial path changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        axial = np.asarray(archive["axial"], dtype=np.float64)
    if (axial.ndim != 2 or axial.shape[1] != 13 or len(axial) < 2 or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(axial[0], handoff["axial_start_q"]) or
            not np.array_equal(axial[-1], handoff["axial_end_q"])):
        raise ValueError("release source differs from verified axial handoff")
    release_hand = np.asarray(INSPIRE_INIT, dtype=np.float64)
    axial_exit = axial[::-1].copy()
    axial_exit[:, 7:] = release_hand
    axial_exit = _path(axial_exit, "socket-axis exit", hand=release_hand,
                       limits=limits)
    held = axial[-1].copy()
    opening_count = max(10, int(math.ceil(float(np.max(np.abs(
        held[7:] - release_hand))) / limits.max_joint_step_rad)))
    if opening_count > 1000:
        raise ValueError("socket opening ramp is too long")
    opening = np.repeat(held[None], opening_count + 1, axis=0)
    for index, fraction in enumerate(np.linspace(0., 1., len(opening))):
        opening[index, 7:] = (1. - fraction) * held[7:] + fraction * release_hand
    fixed_scene = copy.deepcopy(runner.calibration.collision_scene)
    opening_check = _check_path(planner, fixed_scene, opening)
    if not opening_check["all_feasible"]:
        raise ValueError("opening fingers intersect the fixed socket/table world")
    key_spec = runner._preflight.trial_scene["mesh"]["target"]
    key_path = Path(key_spec["file_path"]).expanduser().resolve()
    if not key_path.is_file():
        raise FileNotFoundError("inserted key collision mesh is missing")
    release_scene = copy.deepcopy(fixed_scene)
    release_scene["mesh"]["target"] = {
        "file_path": str(key_path), "pose": se32cart(pose).tolist()}
    exit_check = _check_path(planner, release_scene, axial_exit)
    if not exit_check["all_feasible"]:
        raise ValueError("opened hand cannot retrace the socket-axis path")
    retract_raw = planner.plan_js_to_init(
        release_scene, axial_exit[-1, :7],
        start_hand_qpos=release_hand,
        goal_arm_qpos=np.asarray(FR3_CLEAR_VIEW, dtype=np.float64))
    if retract_raw is None:
        raise ValueError("stock clear-view retract has no collision-free path")
    # cuRobo's full-DOF joint solve may bend fingers between identical open
    # endpoints. The hand stays open during this arm-only retract; replay that
    # exact commanded path against the inserted-key world below.
    retract = np.asarray(retract_raw, dtype=np.float64).copy()
    if retract.ndim != 2 or retract.shape[1] != 13:
        raise ValueError("clear-view retract has no FR3/Inspire joint path")
    retract[:, 7:] = release_hand
    retract = _path(retract, "clear-view retract", hand=release_hand,
                    limits=limits)
    if (not np.allclose(retract[0], axial_exit[-1], atol=1e-4, rtol=0) or
            not np.allclose(retract[-1, :7], FR3_CLEAR_VIEW,
                            atol=1e-3, rtol=0)):
        raise ValueError("stock retract does not join socket exit and clear view")
    retract_check = _check_path(planner, release_scene, retract)
    if not retract_check["all_feasible"]:
        raise ValueError("stock retract fails sampled collision replay")
    folder = Path(output_dir).expanduser().resolve()
    if folder.parent != Path(runner._attempt_dir).resolve():
        raise ValueError("release plan must be saved under this attempt")
    folder.mkdir(exist_ok=False)
    archive_out = folder / "paths.npz"
    np.savez_compressed(archive_out, opening=opening,
                        axial_exit=axial_exit, retract=retract)
    report = {
        "schema": SCHEMA, "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "handoff": {"path": str(handoff_path), "sha256": _sha(handoff_path)},
        "inserted_key_mesh": {"path": str(key_path), "sha256": _sha(key_path)},
        "inserted_key_pose_robot": pose.tolist(),
        "fixed_scene_sha256": hashlib.sha256(json.dumps(
            fixed_scene, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "paths": {"path": str(archive_out), "sha256": _sha(archive_out)},
        "sample_counts": {"opening": len(opening),
                          "axial_exit": len(axial_exit),
                          "retract": len(retract)},
        "checks": {"opening_fixed_world": opening_check,
                   "axial_exit_with_key": exit_check,
                   "retract_with_key": retract_check},
        "scope": "stock_curobo_sampled_plan_not_measured_release_or_key_depth",
        "robot_ready": False,
    }
    report_path = folder / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report_path


def verify_socket_release_plan(report_path: Path) -> tuple[dict, dict]:
    """Replay immutable bindings and path joins before a release command."""
    source = Path(report_path).expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    if (report.get("schema") != SCHEMA or report.get("robot_ready") is not False or
            report.get("scope") !=
            "stock_curobo_sampled_plan_not_measured_release_or_key_depth"):
        raise ValueError("invalid socket release plan")
    handoff_ref = report["handoff"]
    handoff_path = Path(handoff_ref["path"])
    if _sha(handoff_path) != handoff_ref["sha256"]:
        raise ValueError("release axial handoff changed")
    handoff = verify_guarded_axial_handoff(handoff_path)
    key_ref = report["inserted_key_mesh"]
    if _sha(Path(key_ref["path"])) != key_ref["sha256"]:
        raise ValueError("release key mesh changed")
    postlift = json.loads(Path(handoff["postlift_report_path"]).read_text(
        encoding="utf-8"))
    if (report["attempt_id"] != handoff["attempt_id"] or
            report["candidate_id"] != handoff["candidate_id"] or
            report["session_calibration_sha256"] !=
            handoff["session_calibration_sha256"] or
            report["inserted_key_pose_robot"] !=
            postlift["targets"]["T_robot_key_verification"] or
            any(check.get("all_feasible") is not True
                for check in report["checks"].values())):
        raise ValueError("release plan differs from its bound insertion")
    archive_ref = report["paths"]
    archive_path = Path(archive_ref["path"])
    if (_sha(archive_path) != archive_ref["sha256"] or
            archive_path.parent != source.parent):
        raise ValueError("release path archive changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != {"opening", "axial_exit", "retract"}:
            raise ValueError("release path archive is incomplete")
        paths = {name: np.asarray(archive[name], dtype=np.float64)
                 for name in archive.files}
    opened = np.asarray(INSPIRE_INIT, dtype=np.float64)
    for name, path in paths.items():
        if (path.shape != (report["sample_counts"][name], 13) or
                len(path) < 2 or not np.all(np.isfinite(path))):
            raise ValueError("release path samples changed")
    with np.load(Path(handoff["trajectory_archive_path"]),
                 allow_pickle=False) as original:
        axial = np.asarray(original["axial"], dtype=np.float64)
    if (not np.allclose(paths["opening"][0], axial[-1], atol=1e-8, rtol=0) or
            not np.allclose(paths["opening"][:, :7], axial[-1, :7],
                            atol=1e-8, rtol=0) or
            not np.allclose(paths["opening"][-1], paths["axial_exit"][0],
                            atol=1e-8, rtol=0) or
            not np.allclose(paths["axial_exit"][:, :7], axial[::-1, :7],
                            atol=1e-8, rtol=0) or
            not np.allclose(paths["axial_exit"][:, 7:], opened,
                            atol=1e-8, rtol=0) or
            not np.allclose(paths["retract"][0], paths["axial_exit"][-1],
                            atol=1e-4, rtol=0) or
            not np.allclose(paths["retract"][-1, :7], FR3_CLEAR_VIEW,
                            atol=1e-3, rtol=0)):
        raise ValueError("release path no longer joins the axial stroke")
    return report, paths
