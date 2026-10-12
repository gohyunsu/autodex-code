#!/usr/bin/env python3
"""Prepare source-audited cylinder presentation bundles for Blender.

The bundles use the exact cylinder key/socket CAD and the actual Inspire link
meshes at one current v8 grasp posture.  They visualize the task contract;
they are not continuous Franka plans or physical-success evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


DEMO = Path(__file__).resolve().parent
REPO = DEMO.parents[1]
for path in (DEMO, REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.candidates import _grasp_evidence  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.endpoint import (  # noqa: E402
    _hand_link_meshes,
    _load_mesh,
    nominal_inspire_hold_poses,
    screen_grasp_endpoint,
    validate_task_geometry,
)
from precision_insertion.geometry import validate_se3  # noqa: E402


SCHEMA = "precision_insertion_cylinder_presentation_bundles_v1"
PHASE_LABELS = (
    "1  Socket pose",
    "2  Key perception",
    "3  Grasp",
    "4  Lift + transfer",
    "5  Multi-view XY alignment",
    "6  Guarded 20 mm insertion",
    "7  Success judgement",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ease(count: int) -> np.ndarray:
    values = np.linspace(0.0, 1.0, count)
    return values * values * (3.0 - 2.0 * values)


def _interpolate(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    """Interpolate an SE(3) segment without averaging rotation matrices."""
    start = validate_se3(start, name="presentation segment start")
    end = validate_se3(end, name="presentation segment end")
    alpha = _ease(count)
    slerp = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3]))))
    rotations = slerp(alpha).as_matrix()
    translations = ((1.0 - alpha[:, None]) * start[:3, 3] +
                    alpha[:, None] * end[:3, 3])
    result = np.repeat(np.eye(4)[None], count, axis=0)
    result[:, :3, :3] = rotations
    result[:, :3, 3] = translations
    return result


def _hold(pose: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(validate_se3(pose)[None], count, axis=0)


def _offscreen() -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = (3.0, 3.0, 3.0)
    return pose


def _validated_candidate(catalog_path: Path, shared_root: Path, candidate_id: str):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    mode = select_mode("cylinder", 20)
    expected_mode = {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
        "target_depth_m": mode.target_depth_m,
    }
    if (catalog.get("schema") != "precision_insertion_endpoint_catalog_v1" or
            catalog.get("mode") != expected_mode or
            catalog.get("complete_scan") is not True or
            catalog.get("robot_ready") is not False):
        raise ValueError("presentation requires the complete cylinder/20 mm catalog")
    parts = candidate_id.split("/")
    if len(parts) != 3 or parts[:2] != ["table", "0"] or not parts[2].isdigit():
        raise ValueError("presentation candidate must be a pose-000 table ID")
    row = next((item for item in catalog["candidates"]
                if item.get("key") == parts), None)
    if row is None or row.get("eligible") is not True:
        raise ValueError(f"candidate is not eligible: {candidate_id}")
    paths = AssetPaths(shared_root, mode)
    candidate = paths.candidate_dir.joinpath(*parts).resolve()
    stable, reason = _grasp_evidence(candidate, paths.key_planning_mesh)
    if not stable:
        raise ValueError(f"current v8/MuJoCo grasp evidence failed: {reason}")
    screen = screen_grasp_endpoint(
        shared_root=shared_root, mode=mode, candidate_dir=candidate,
        minimum_hand_clearance_m=float(catalog["minimum_hand_clearance_m"]))
    expected_endpoint = validate_task_geometry(
        json.loads(paths.task_geometry.read_text(encoding="utf-8")), mode)
    if (screen.get("endpoint_pass") is not True or
            not np.allclose(screen["T_socket_key_tested"], expected_endpoint,
                            atol=1e-9)):
        raise ValueError("fresh candidate screen does not pass the exact endpoint")
    return catalog, mode, paths, candidate, screen


def _write_bundle(
    *, output: Path, mesh_dir: Path, geometry_names: list[str],
    hand: np.ndarray, key: np.ndarray, socket: np.ndarray,
    key_mesh: Path, socket_mesh: Path, phase_ids: np.ndarray,
) -> None:
    if len(hand) != len(key) or len(key) != len(phase_ids):
        raise ValueError("presentation animation tracks have different lengths")
    np.savez_compressed(
        output,
        robot_geometry_transforms=np.repeat(
            hand[:, None, :, :], len(geometry_names), axis=1).astype(np.float32),
        object_poses=key.astype(np.float32),
        socket_pose=socket.astype(np.float32),
        robot_mesh_dir=np.asarray(str(mesh_dir)),
        geometry_names=np.asarray(geometry_names),
        object_mesh_path=np.asarray(str(key_mesh)),
        socket_mesh_path=np.asarray(str(socket_mesh)),
        phase_ids=phase_ids.astype(np.int16),
        phase_labels=np.asarray(PHASE_LABELS),
    )


def prepare(*, shared_root: Path, catalog_path: Path, output_root: Path,
            candidate_id: str) -> dict:
    root = shared_root.expanduser().resolve()
    catalog_file = catalog_path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite presentation root: {output}")
    catalog, mode, paths, candidate, screen = _validated_candidate(
        catalog_file, root, candidate_id)
    output.mkdir(parents=True)
    mesh_dir = output / "mesh_cache/hand"
    mesh_dir.mkdir(parents=True)
    pre = np.load(candidate / "pregrasp_pose.npy", allow_pickle=False)
    grasp = np.load(candidate / "grasp_pose.npy", allow_pickle=False)
    hold_q = nominal_inspire_hold_poses(pre, grasp)[
        "autodex_default_controller_hold"]
    hand_meshes = _hand_link_meshes(paths.robot_urdf, hold_q)
    geometry_names = sorted(hand_meshes)
    for index, name in enumerate(geometry_names):
        hand_meshes[name].export(mesh_dir / f"geometry_{index:03d}.ply")
    key_mesh = output / "mesh_cache/key_exact.ply"
    socket_mesh = output / "mesh_cache/socket_exact.ply"
    _load_mesh(paths.raw_mesh(mode.key_object)).export(key_mesh)
    _load_mesh(paths.raw_mesh(mode.socket_object)).export(socket_mesh)

    T_world_socket = np.eye(4)
    T_world_socket[:3, 3] = (0.45, -0.10, 0.04)
    task = json.loads(paths.task_geometry.read_text(encoding="utf-8"))
    preinsert = T_world_socket @ validate_se3(task["T_socket_key_preinsert"])
    verification = T_world_socket @ validate_se3(
        task["T_socket_key_verification"])
    T_key_hand = validate_se3(screen["T_key_hand"], name="screened T_key_hand")

    # The endpoint clip moves the exact hand/key grasp as one rigid body.
    success_key = np.concatenate((
        _hold(preinsert, 28), _interpolate(preinsert, verification, 42),
        _hold(verification, 50)))
    success_hand = success_key @ T_key_hand
    _write_bundle(
        output=output / "success_20mm_bundle.npz", mesh_dir=mesh_dir,
        geometry_names=geometry_names, hand=success_hand, key=success_key,
        socket=T_world_socket, key_mesh=key_mesh, socket_mesh=socket_mesh,
        phase_ids=np.full(len(success_key), 5, dtype=int))

    # Pipeline visualization: socket-only initialization, tabletop key pose,
    # grasp, lift/transfer, a conceptual bounded XY correction, insertion,
    # and final outcome judgement. It is deliberately not a claimed arm plan.
    source = np.load(paths.key_tabletop_dir / "000.npy", allow_pickle=False)
    source = validate_se3(source, name="cylinder tabletop pose 000")
    source[:3, 3] += (0.34, 0.19, 0.04)
    lifted = source.copy()
    lifted[2, 3] += 0.12
    offset_preinsert = preinsert.copy()
    offset_preinsert[:2, 3] += (0.006, -0.004)
    hidden = _offscreen()
    final_source_hand = source @ T_key_hand
    approach_hand = final_source_hand.copy()
    approach_hand[2, 3] += 0.10

    key_tracks = []
    hand_tracks = []
    phase_tracks = []

    def append(key_track: np.ndarray, hand_track: np.ndarray, phase: int) -> None:
        key_tracks.append(key_track)
        hand_tracks.append(hand_track)
        phase_tracks.append(np.full(len(key_track), phase, dtype=int))

    append(_hold(hidden, 28), _hold(hidden, 28), 0)
    append(_hold(source, 32), _hold(hidden, 32), 1)
    append(_hold(source, 30), _interpolate(approach_hand, final_source_hand, 30), 2)
    lift_key = _interpolate(source, lifted, 25)
    append(lift_key, lift_key @ T_key_hand, 3)
    transfer_key = _interpolate(lifted, offset_preinsert, 42)
    append(transfer_key, transfer_key @ T_key_hand, 3)
    align_key = _interpolate(offset_preinsert, preinsert, 30)
    append(align_key, align_key @ T_key_hand, 4)
    insert_key = _interpolate(preinsert, verification, 36)
    append(insert_key, insert_key @ T_key_hand, 5)
    append(_hold(verification, 40), _hold(verification @ T_key_hand, 40), 6)
    pipeline_key = np.concatenate(key_tracks)
    pipeline_hand = np.concatenate(hand_tracks)
    phase_ids = np.concatenate(phase_tracks)
    _write_bundle(
        output=output / "pipeline_story_bundle.npz", mesh_dir=mesh_dir,
        geometry_names=geometry_names, hand=pipeline_hand, key=pipeline_key,
        socket=T_world_socket, key_mesh=key_mesh, socket_mesh=socket_mesh,
        phase_ids=phase_ids)

    manifest = {
        "schema": SCHEMA,
        "shared_root": str(root),
        "catalog": str(catalog_file),
        "catalog_sha256": _sha(catalog_file),
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate),
        "source_candidate_evidence": "current AutoDex v8/MuJoCo grasp and exact 20 mm endpoint pass",
        "socket_object": mode.socket_object,
        "key_object": mode.key_object,
        "socket_radial_gap_mm": mode.gap_mm,
        "verification_depth_mm": mode.target_depth_m * 1000.0,
        "success_bundle_frames": len(success_key),
        "pipeline_bundle_frames": len(pipeline_key),
        "phase_labels": list(PHASE_LABELS),
        "scope": "offline_presentation_visualization_not_continuous_franka_plan_or_physical_success",
        "key_hand_motion_after_grasp": "rigid_T_key_hand_from_screened_v8_hypothesis",
        "robot_ready": False,
        "not_validated": [
            "continuous Franka reachability or collision-free transfer",
            "physical post-squeeze key-hand relation",
            "guarded contact dynamics and force limits",
            "VLM or metric success-classification accuracy",
        ],
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", default="table/0/194")
    args = parser.parse_args()
    result = prepare(shared_root=args.shared_root, catalog_path=args.catalog,
                     output_root=args.output_root,
                     candidate_id=args.candidate_id)
    print(json.dumps({
        "output_root": str(args.output_root.expanduser().resolve()),
        "candidate_id": result["candidate_id"],
        "success_frames": result["success_bundle_frames"],
        "pipeline_frames": result["pipeline_bundle_frames"],
        "robot_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
