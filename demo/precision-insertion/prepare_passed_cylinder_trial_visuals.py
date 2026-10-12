#!/usr/bin/env python3
"""Export presentation trajectories from one passed cylinder trial preflight.

The input must be an auditable ``diagnose_synthetic_full_chain.py`` result
whose pickup, locked-hand lift, transfer, axial path, and sampled held-object
collision audit all passed.  This script never plans or commands a robot.  It
only time-compresses saved joint paths and derives the rigid key pose from
the exact FR3/Inspire URDF forward kinematics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


DEMO = Path(__file__).resolve().parent
REPO = DEMO.parents[1]
for path in (DEMO, REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.geometry import validate_se3  # noqa: E402


SCHEMA = "precision_insertion_passed_cylinder_trial_visuals_v1"
JOINT_NAMES = (
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint", "right_little_1_joint",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(array: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 13 or len(source) < 2:
        raise ValueError("saved robot path must have shape (N>=2, 13)")
    indices = np.linspace(0, len(source) - 1, count).round().astype(int)
    return source[indices]


def _joint_transition(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, count)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return ((1.0 - alpha[:, None]) * np.asarray(start, dtype=np.float64) +
            alpha[:, None] * np.asarray(end, dtype=np.float64))


def _repeat(value: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(np.asarray(value, dtype=np.float64)[None], count, axis=0)


def _robot_key_poses(urdf: URDF, qpos: np.ndarray,
                     T_key_hand: np.ndarray) -> np.ndarray:
    poses = []
    for q in np.asarray(qpos, dtype=np.float64):
        urdf.update_cfg(dict(zip(JOINT_NAMES, q)))
        T_robot_hand = validate_se3(
            urdf.get_transform("base_link", urdf.base_link),
            name="FR3/Inspire hand FK",
        )
        poses.append(T_robot_hand @ np.linalg.inv(T_key_hand))
    return np.asarray(poses)


def _write_trajectory(
    *, path: Path, qpos: np.ndarray, key_poses: np.ndarray,
    socket_pose: np.ndarray, paths: AssetPaths, stage_ids: np.ndarray,
    stage_labels: tuple[str, ...], source_indices: np.ndarray,
) -> None:
    q = np.asarray(qpos, dtype=np.float64)
    key = np.asarray(key_poses, dtype=np.float64)
    ids = np.asarray(stage_ids, dtype=np.int16)
    indices = np.asarray(source_indices, dtype=np.int64)
    if (q.ndim != 2 or q.shape[1] != 13 or key.shape != (len(q), 4, 4) or
            ids.shape != (len(q),) or indices.shape != (len(q),)):
        raise ValueError(f"malformed presentation stage: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=q.astype(np.float32),
        object_pose=key,
        socket_pose=np.asarray(socket_pose, dtype=np.float64),
        joint_names=np.asarray(JOINT_NAMES),
        object_mesh_path=np.asarray(str(paths.raw_mesh(paths.mode.key_object))),
        socket_mesh_path=np.asarray(str(paths.raw_mesh(paths.mode.socket_object))),
        robot_urdf_path=np.asarray(str(paths.robot_urdf)),
        table_surface_z_m=np.asarray(0.0),
        stage_ids=ids,
        stage_labels=np.asarray(stage_labels),
        source_indices=indices,
    )


def prepare(*, shared_root: Path, report_dir: Path, output_root: Path) -> dict:
    root = shared_root.expanduser().resolve()
    source = report_dir.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite presentation root: {output}")
    report_path = source / "report.json"
    context_path = source / "synthetic_context.json"
    trajectory_path = source / "planned_trajectories.npz"
    for path in (report_path, context_path, trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    artifact = report.get("artifacts", {})
    if (report.get("schema") != "precision_insertion_trial_preflight_v2" or
            report.get("status") != "sampled_planning_pass" or
            report.get("cartesian_planner_mode") != "native-locked-experimental" or
            context.get("schema") !=
            "precision_insertion_synthetic_full_chain_diagnostic_v1" or
            context.get("result_status") != "sampled_planning_pass" or
            context.get("synthetic") is not True or
            context.get("robot_ready") is not False or
            artifact.get("planned_trajectories") != trajectory_path.name or
            artifact.get("planned_trajectories_sha256") != _sha(trajectory_path)):
        raise ValueError("source is not an intact passed native-locked preflight")
    insertion = report.get("insertion_plan")
    audit = None if not isinstance(insertion, dict) else insertion.get(
        "sampled_held_path_audit")
    if (not isinstance(insertion, dict) or
            insertion.get("sampled_planning_pass") is not True or
            not isinstance(audit, dict) or audit.get("sampled_clear") is not True or
            audit.get("failures") != []):
        raise ValueError("source lacks the passed held-object sampled audit")
    candidate_key = report.get("selected_candidate_key")
    if (not isinstance(candidate_key, list) or len(candidate_key) != 3 or
            candidate_key[:2] != ["table", "0"]):
        raise ValueError("presentation expects a selected pose-000 cylinder grasp")

    mode_record = audit["mode"]
    mode = select_mode(mode_record["family"], mode_record["gap_mm"])
    if mode.family != "cylinder":
        raise ValueError("this exporter only accepts the cylinder mode")
    paths = AssetPaths(root, mode)
    candidate = paths.candidate_dir.joinpath(*candidate_key)
    T_key_hand = validate_se3(
        np.load(candidate / "wrist_se3.npy", allow_pickle=False),
        name="selected v8 T_key_hand",
    )
    T_robot_key_table = validate_se3(
        np.asarray(report["key_pose_world"], dtype=np.float64),
        name="synthetic T_robot_key",
    )
    T_robot_socket = validate_se3(
        np.asarray(context["T_robot_socket_hypothetical"], dtype=np.float64),
        name="synthetic T_robot_socket",
    )

    with np.load(trajectory_path, allow_pickle=False) as data:
        raw = {name: np.asarray(data[name], dtype=np.float64)
               for name in ("pickup_approach", "held_lift", "transfer", "axial")}
        pregrasp = np.asarray(data["pickup_pregrasp"], dtype=np.float64)
        grasp = np.asarray(data["pickup_grasp"], dtype=np.float64)
        held = np.asarray(data["held_hand_q"], dtype=np.float64)
    for name, path in raw.items():
        if path.ndim != 2 or path.shape[1] != 13 or not np.all(np.isfinite(path)):
            raise ValueError(f"invalid saved {name} path")
    if max(float(np.max(np.abs(raw[name][:, 7:] - held)))
           for name in ("held_lift", "transfer", "axial")) > 1e-6:
        raise ValueError("saved held path does not preserve the hand joint lock")

    urdf = URDF.load(str(paths.robot_urdf), build_scene_graph=True, load_meshes=False)
    if tuple(urdf.actuated_joint_names) != JOINT_NAMES:
        raise ValueError("FR3/Inspire URDF joint order changed")
    start_q = raw["pickup_approach"][0]
    approach = _rows(raw["pickup_approach"], 60)
    close_arm = approach[-1, :7]
    close_a = np.c_[
        _repeat(close_arm, 12), _joint_transition(pregrasp, grasp, 12)]
    close_b = np.c_[
        _repeat(close_arm, 12), _joint_transition(grasp, held, 12)]
    close = np.concatenate((close_a, close_b[1:]), axis=0)
    lift = _rows(raw["held_lift"], 72)
    transfer = _rows(raw["transfer"], 72)
    axial = _rows(raw["axial"], 72)
    lift_key = _robot_key_poses(urdf, lift, T_key_hand)
    transfer_key = _robot_key_poses(urdf, transfer, T_key_hand)
    axial_key = _robot_key_poses(urdf, axial, T_key_hand)

    task = json.loads(paths.task_geometry.read_text(encoding="utf-8"))
    expected_endpoint = T_robot_socket @ validate_se3(
        np.asarray(task["T_socket_key_verification"], dtype=np.float64),
        name="T_socket_key_verification",
    )
    endpoint_translation_error_m = float(np.linalg.norm(
        axial_key[-1, :3, 3] - expected_endpoint[:3, 3]))
    endpoint_rotation_error_deg = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(
            expected_endpoint[:3, :3].T @ axial_key[-1, :3, :3]
        ).as_rotvec())))
    limits = report["limits"]
    if (endpoint_translation_error_m > limits["goal_position_tolerance_m"] or
            endpoint_rotation_error_deg > limits["goal_rotation_tolerance_deg"]):
        raise ValueError("saved axial endpoint no longer satisfies source limits")

    hidden_key = np.eye(4)
    hidden_key[:3, 3] = (3.0, 3.0, 3.0)
    labels = (
        "Socket pose measurement", "Key perception", "Pickup and grasp",
        "Held lift", "Transfer", "Pre-insertion hold",
        "Axial insertion to 20 mm", "Success judgement",
    )
    stage_specs: dict[str, tuple[np.ndarray, np.ndarray, int, np.ndarray]] = {}

    def held_stage(q: np.ndarray, key: np.ndarray, hold_frames: int,
                   stage_id: int, raw_length: int):
        q_out = np.concatenate((q, _repeat(q[-1], hold_frames)))
        key_out = np.concatenate((key, _repeat(key[-1], hold_frames)))
        source_index = np.concatenate((
            np.linspace(0, raw_length - 1, len(q)).round().astype(int),
            np.full(hold_frames, raw_length - 1, dtype=int),
        ))
        return q_out, key_out, stage_id, source_index

    stage_specs["00_socket_pose"] = (
        _repeat(start_q, 36), _repeat(hidden_key, 36), 0,
        np.full(36, -1, dtype=int))
    stage_specs["01_key_perception"] = (
        _repeat(start_q, 36), _repeat(T_robot_key_table, 36), 1,
        np.full(36, -1, dtype=int))
    pickup_q = np.concatenate((approach, close, _repeat(
        np.r_[close_arm, held], 12)))
    stage_specs["02_pickup_grasp"] = (
        pickup_q, _repeat(T_robot_key_table, len(pickup_q)), 2,
        np.concatenate((
            np.linspace(0, len(raw["pickup_approach"]) - 1,
                        len(approach)).round().astype(int),
            np.full(len(close) + 12, len(raw["pickup_approach"]) - 1,
                    dtype=int),
        )))
    stage_specs["03_held_lift"] = held_stage(
        lift, lift_key, 12, 3, len(raw["held_lift"]))
    stage_specs["04_transfer"] = held_stage(
        transfer, transfer_key, 12, 4, len(raw["transfer"]))
    stage_specs["05_preinsert_hold"] = (
        _repeat(transfer[-1], 36), _repeat(transfer_key[-1], 36), 5,
        np.full(36, len(raw["transfer"]) - 1, dtype=int))
    stage_specs["06_axial_insertion"] = held_stage(
        axial, axial_key, 18, 6, len(raw["axial"]))
    stage_specs["07_success_judgement"] = (
        _repeat(axial[-1], 48), _repeat(axial_key[-1], 48), 7,
        np.full(48, len(raw["axial"]) - 1, dtype=int))

    trajectory_dir = output / "trajectories"
    stage_records = []
    for name, (q, key, stage_id, indices) in stage_specs.items():
        target = trajectory_dir / f"{name}.npz"
        _write_trajectory(
            path=target, qpos=q, key_poses=key, socket_pose=T_robot_socket,
            paths=paths, stage_ids=np.full(len(q), stage_id),
            stage_labels=labels, source_indices=indices)
        stage_records.append({
            "stage": name, "label": labels[stage_id], "frames": len(q),
            "trajectory": str(target), "sha256": _sha(target),
        })

    full_parts = []
    full_key_parts = []
    full_ids = []
    full_indices = []
    for name in stage_specs:
        q, key, stage_id, indices = stage_specs[name]
        full_parts.append(q)
        full_key_parts.append(key)
        full_ids.append(np.full(len(q), stage_id, dtype=np.int16))
        full_indices.append(indices)
    full_path = trajectory_dir / "08_full_pipeline.npz"
    _write_trajectory(
        path=full_path, qpos=np.concatenate(full_parts),
        key_poses=np.concatenate(full_key_parts), socket_pose=T_robot_socket,
        paths=paths, stage_ids=np.concatenate(full_ids),
        stage_labels=labels, source_indices=np.concatenate(full_indices))

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "source_scope": context["scope"],
        "source_synthetic": True,
        "robot_ready": False,
        "source_report": str(report_path),
        "source_report_sha256": _sha(report_path),
        "source_context": str(context_path),
        "source_context_sha256": _sha(context_path),
        "source_trajectories": str(trajectory_path),
        "source_trajectories_sha256": _sha(trajectory_path),
        "selected_candidate_key": candidate_key,
        "planner_mode": report["cartesian_planner_mode"],
        "sampled_held_path_clear": True,
        "source_sample_counts": {
            name: len(value) for name, value in raw.items()
        },
        "maximum_saved_held_hand_drift_rad": max(
            float(np.max(np.abs(raw[name][:, 7:] - held)))
            for name in ("held_lift", "transfer", "axial")),
        "endpoint_translation_error_mm": endpoint_translation_error_m * 1000.0,
        "endpoint_rotation_error_deg": endpoint_rotation_error_deg,
        "verification_depth_mm": mode.target_depth_m * 1000.0,
        "socket_radial_gap_mm": mode.gap_mm,
        "stage_labels": list(labels),
        "stages": stage_records,
        "full_pipeline": {
            "trajectory": str(full_path), "frames": sum(
                row["frames"] for row in stage_records),
            "sha256": _sha(full_path),
        },
        "render_contract": (
            "actual FR3/Inspire visual meshes at saved/subsampled planner joints; "
            "rigid key pose from exact URDF FK after grasp"
        ),
        "not_validated": [
            "measured camera/socket/key poses",
            "measured robot start joints or controller tracking",
            "physical post-squeeze key-hand relation",
            "force-controlled contact dynamics",
            "VLM success judgement accuracy",
            "real-robot execution or insertion success",
        ],
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(shared_root=args.shared_root, report_dir=args.report_dir,
                     output_root=args.output_root)
    print(json.dumps({
        "output_root": str(args.output_root.expanduser().resolve()),
        "candidate": result["selected_candidate_key"],
        "stages": len(result["stages"]),
        "full_pipeline_frames": result["full_pipeline"]["frames"],
        "robot_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
