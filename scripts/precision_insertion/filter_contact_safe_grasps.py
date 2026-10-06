#!/usr/bin/env python3
"""Curate BODex grasps that obey the precision-key contact policy.

This is a declared-contact and numerical-quality screen, not a whole-hand
collision or physical-success label. A grasp is copied only when its BODex
errors meet explicit thresholds and every object-side declared fingertip
contact lies on a permitted handle face with the configured edge margin. The
final candidate still requires full-hand, reachability, and physical checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REQUIRED_FILES = (
    "wrist_se3.npy",
    "pregrasp_pose.npy",
    "grasp_pose.npy",
    "bodex_info.npy",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _point_region(
    point: np.ndarray,
    *,
    half_x: float,
    half_y: float,
    handle_top: float,
    margin: float,
    tolerance: float,
) -> str | None:
    """Return the permitted face name for an object-frame point, or None."""
    x, y, z = (float(value) for value in point)

    if (
        abs(z) <= tolerance
        and abs(x) <= half_x - margin
        and abs(y) <= half_y - margin
    ):
        return "handle_rear"

    if not margin <= z <= handle_top - margin:
        return None
    if abs(abs(x) - half_x) <= tolerance and abs(y) <= half_y - margin:
        return "handle_x_side"
    if abs(abs(y) - half_y) <= tolerance and abs(x) <= half_x - margin:
        return "handle_y_side"
    return None


def inspect_candidate(
    candidate_dir: Path,
    policy: dict[str, Any],
    *,
    max_grasp_error: float,
    max_contact_distance: float,
    object_pose_world: np.ndarray | None = None,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (candidate_dir / name).is_file()]
    if missing:
        return {"accepted": False, "reason": "missing_files", "missing": missing}

    data = np.load(candidate_dir / "bodex_info.npy", allow_pickle=True).item()
    grasp_error = np.asarray(data.get("grasp_error"), dtype=np.float64)
    distance_error = np.asarray(data.get("dist_error"), dtype=np.float64)
    if grasp_error.size == 0 or distance_error.size == 0:
        return {"accepted": False, "reason": "missing_bodex_quality_metrics"}
    quality = {
        "grasp_error_max": float(np.max(np.abs(grasp_error))),
        "contact_distance_mean_abs_m": float(np.mean(np.abs(distance_error))),
        "threshold_grasp_error_max": max_grasp_error,
        "threshold_contact_distance_mean_abs_m": max_contact_distance,
        "native_success": bool(np.asarray(data.get("success", False)).item()),
    }
    if (
        quality["grasp_error_max"] > max_grasp_error
        or quality["contact_distance_mean_abs_m"] > max_contact_distance
    ):
        return {"accepted": False, "reason": "bodex_quality_threshold", "quality": quality}

    contacts = np.asarray(data.get("contact_point"), dtype=np.float64)
    if contacts.size == 0 or contacts.shape[-1] < 3:
        return {"accepted": False, "reason": "invalid_contact_points"}
    source_contacts = contacts.reshape(-1, contacts.shape[-1])[:, :3]
    if object_pose_world is None:
        object_contacts = source_contacts
        contact_frame_conversion = "identity_assumed_for_legacy_caller"
    else:
        world_to_object = np.linalg.inv(object_pose_world)
        object_contacts = (
            (world_to_object[:3, :3] @ source_contacts.T).T
            + world_to_object[:3, 3]
        )
        contact_frame_conversion = "T_object_world @ p_world"

    handle_z = policy["handle_z_range"]
    margin = float(policy["allowed"]["edge_margin_m"])
    tolerance = float(policy["allowed"]["plane_tolerance_m"])
    half_extents = policy["handle_half_extents_xy_m"]
    regions = [
        _point_region(
            point,
            half_x=float(half_extents[0]),
            half_y=float(half_extents[1]),
            handle_top=float(handle_z[1]),
            margin=margin,
            tolerance=tolerance,
        )
        for point in object_contacts
    ]
    if any(region is None for region in regions):
        return {
            "accepted": False,
            "reason": "forbidden_or_edge_contact",
            "source_contacts_world_m": source_contacts.tolist(),
            "object_contacts_m": object_contacts.tolist(),
            "regions": regions,
            "contact_frame_conversion": contact_frame_conversion,
        }

    return {
        "accepted": True,
        "reason": "contact_and_quality_screened",
        "object_contacts_m": object_contacts.tolist(),
        "regions": regions,
        "source_contacts_world_m": source_contacts.tolist(),
        "contact_frame_conversion": contact_frame_conversion,
        "quality": quality,
        "solver_thresholds": data.get("solver_thresholds", {}),
    }


def _candidate_dirs(scene_root: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        return (int(path.name), path.name) if path.name.isdigit() else (10**12, path.name)

    return sorted(
        (
            path
            for path in scene_root.iterdir()
            if path.is_dir() and (path / "bodex_info.npy").is_file()
        ),
        key=key,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scene", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--contact-policy", type=Path, required=True)
    parser.add_argument(
        "--scene-json", type=Path, required=True,
        help=(
            "BODex scene that produced the candidates; saved contact points "
            "are in scene/world frame and are never safe to reinterpret directly"
        ),
    )
    parser.add_argument(
        "--max-grasp-error",
        type=float,
        default=0.2,
        help="maximum BODex grasp-error component (default: project convention 0.2)",
    )
    parser.add_argument(
        "--max-contact-distance",
        type=float,
        default=0.01,
        help="maximum mean absolute contact distance in metres (default: 0.01)",
    )
    parser.add_argument(
        "--replace-backup",
        type=Path,
        help="move an existing output scene to this explicit non-runtime path before curating",
    )
    args = parser.parse_args()

    raw_scene = args.raw_scene.expanduser().resolve()
    output_scene = args.output_scene.expanduser().resolve()
    policy_path = args.contact_policy.expanduser().resolve()
    scene_path = args.scene_json.expanduser().resolve()
    if not raw_scene.is_dir():
        parser.error(f"raw scene does not exist: {raw_scene}")
    if not policy_path.is_file():
        parser.error(f"contact policy does not exist: {policy_path}")
    if not scene_path.is_file():
        parser.error(f"scene JSON does not exist: {scene_path}")

    if output_scene.exists():
        if args.replace_backup is None:
            parser.error(f"output exists; pass --replace-backup PATH: {output_scene}")
        backup = args.replace_backup.expanduser().resolve()
        if backup.exists():
            parser.error(f"backup already exists; move it explicitly first: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        output_scene.rename(backup)

    policy = _load_json(policy_path)
    scene = _load_json(scene_path)
    mesh = scene["scene"]["mesh"]
    target = mesh.get("target")
    if target is None:
        if len(mesh) != 1:
            parser.error(
                f"cannot identify target mesh in scene: {scene_path}"
            )
        target = next(iter(mesh.values()))
    pose = np.asarray(target["pose"], dtype=np.float64)
    if pose.shape != (7,):
        parser.error(f"target pose must be xyz+wxyz: {scene_path}")
    object_pose_world = np.eye(4, dtype=np.float64)
    object_pose_world[:3, 3] = pose[:3]
    object_pose_world[:3, :3] = Rotation.from_quat(
        [pose[4], pose[5], pose[6], pose[3]]
    ).as_matrix()
    candidates = _candidate_dirs(raw_scene)
    output_scene.mkdir(parents=True)

    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for candidate in candidates:
        result = inspect_candidate(
            candidate,
            policy,
            max_grasp_error=args.max_grasp_error,
            max_contact_distance=args.max_contact_distance,
            object_pose_world=object_pose_world,
        )
        if result["accepted"]:
            destination = output_scene / candidate.name
            shutil.copytree(candidate, destination)
            with (destination / "contact_screen.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            accepted.append({"candidate": candidate.name, **result})
        else:
            reason = str(result["reason"])
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    report = {
        "schema_version": 1,
        "status": "contact_and_quality_screened_not_collision_or_physical_validated",
        "raw_scene": str(raw_scene),
        "output_scene": str(output_scene),
        "contact_policy": str(policy_path),
        "scene_json": str(scene_path),
        "bodex_contact_point_frame": "scene_world_transformed_to_object",
        "total_candidates": len(candidates),
        "accepted_count": len(accepted),
        "rejection_counts": rejection_counts,
        "accepted": accepted,
        "required_next_checks": [
            "render or inspect the full Inspire hand against contact_forbidden.obj",
            "run robot reachability and environment collision preflight",
            "physically validate grasp and lift on the 1.5 mm key",
            "physically validate insertion clearance before marking trusted",
        ],
    }
    with (output_scene / "contact_filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
