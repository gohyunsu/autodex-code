#!/usr/bin/env python3
"""Validate geometry assets and report remaining runtime blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_assets import (
    KEY_SPECS,
    FIXTURE_NAME,
    SOCKET_OBJECT_NAME,
    contact_face_partition,
    read_binary_stl,
)


def validate(shared_root: Path, source_dir: Path, require_runtime: bool = False) -> int:
    shared_root = shared_root.expanduser().resolve()
    source_dir = source_dir.resolve()
    project = shared_root / "AutoDex"
    failures: list[str] = []
    blockers: list[str] = []
    planned_sim_candidates: dict[str, dict[str, Path]] = {}

    for object_name, _gap, source_name in KEY_SPECS:
        source_mesh = read_binary_stl(source_dir / source_name)
        allowed, forbidden = contact_face_partition(source_mesh)
        object_dir = shared_root / "object_processing" / object_name
        required = [
            object_dir / "raw_mesh" / f"{object_name}.obj",
            object_dir / "processed_data" / "mesh" / "simplified.obj",
            object_dir / "processed_data" / "mesh" / "contact_allowed.obj",
            object_dir / "processed_data" / "mesh" / "contact_forbidden.obj",
            object_dir / "processed_data" / "info" / "simplified.json",
            object_dir / "processed_data" / "info" / "contact_regions.json",
            object_dir / "processed_data" / "info" / "tabletop" / "000.npy",
            project / "scene" / "inspire" / object_name / "table" / "0.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.extend(f"missing: {path}" for path in missing)
            continue

        pose = np.load(object_dir / "processed_data" / "info" / "tabletop" / "000.npy")
        if pose.shape != (4, 4) or not np.allclose(pose, np.eye(4), atol=1e-9):
            failures.append(f"{object_name}: baseline tabletop pose must be identity")
        policy = json.loads(
            (object_dir / "processed_data" / "info" / "contact_regions.json").read_text()
        )
        if policy["allowed"]["face_count"] != len(allowed):
            failures.append(f"{object_name}: allowed contact face count mismatch")
        if policy["forbidden"]["face_count"] != len(forbidden):
            failures.append(f"{object_name}: forbidden contact face count mismatch")

        repre = project / "foundpose_assets" / object_name / "object_repre" / "v1" / object_name / "1" / "repre.pth"
        if not repre.is_file():
            blockers.append(f"{object_name}: FoundPose repre.pth not generated")
        candidate_root = project / "candidates" / "inspire" / "v8" / object_name
        screened = []
        simulated = []
        franka_planned = []
        physically_validated = []
        for screen_path in candidate_root.rglob("contact_screen.json"):
            try:
                screen = json.loads(screen_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if screen.get("accepted") and screen.get("reason") in {
                "contact_and_quality_screened",
                # Compatibility with candidates produced before the screen's
                # provenance label was made precise.
                "simulation_screened",
            }:
                screened.append(screen_path)
                simulation_path = screen_path.parent / "simulation_validation.json"
                if simulation_path.is_file():
                    try:
                        simulation = json.loads(simulation_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        simulation = {}
                    if simulation.get("status") == "passed":
                        simulated.append(simulation_path)
                physical_path = screen_path.parent / "physical_validation.json"
                if physical_path.is_file():
                    try:
                        physical = json.loads(physical_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        physical = {}
                    if physical.get("status") == "passed":
                        physically_validated.append(physical_path)
                plan_path = screen_path.parent / "franka_plan_validation.json"
                if plan_path.is_file():
                    try:
                        plan = json.loads(plan_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        plan = {}
                    if plan.get("status") == "passed":
                        franka_planned.append(plan_path)
        simulated_dirs = {path.parent for path in simulated}
        planned_dirs = {path.parent for path in franka_planned}
        physical_dirs = {path.parent for path in physically_validated}
        planned_sim_dirs = simulated_dirs & planned_dirs
        planned_sim_candidates[object_name] = {
            path.name: path for path in planned_sim_dirs
        }
        if not screened:
            blockers.append(f"{object_name}: no contact-safe Inspire grasp candidate")
        elif not simulated:
            blockers.append(f"{object_name}: contact-safe grasp has not passed full-key simulation")
        elif not planned_sim_dirs:
            blockers.append(
                f"{object_name}: no single simulated grasp has also passed FR3 planning"
            )
        elif not (planned_sim_dirs & physical_dirs):
            blockers.append(f"{object_name}: simulation-validated grasp lacks physical validation")

        stage_path = project / "precision_insertion" / "stages" / f"{object_name}.json"
        if not stage_path.is_file():
            failures.append(f"{object_name}: stage profile is missing")
        else:
            stage = json.loads(stage_path.read_text())
            if stage.get("object") != object_name:
                failures.append(f"{object_name}: stage profile object mismatch")
            controller = stage.get("controller", {})
            controller_status = controller.get("implementation_status")
            if controller_status not in {
                "required", "implemented", "commissioned"
            }:
                failures.append(
                    f"{object_name}: unknown controller implementation status"
                )
            elif controller_status == "required":
                blockers.append(
                    f"{object_name}: {controller.get('mode', 'insertion')} controller "
                    "is not implemented or commissioned"
                )
            elif controller_status == "implemented":
                blockers.append(
                    f"{object_name}: {controller.get('mode', 'insertion')} controller "
                    "is implemented but not physically commissioned"
                )

    common_ids = set.intersection(
        *(set(candidates) for candidates in planned_sim_candidates.values())
    ) if planned_sim_candidates else set()
    if not common_ids:
        blockers.append(
            "no common Inspire grasp has passed full-key simulation and FR3 planning "
            "for all four keys"
        )
    else:
        # Equal IDs are not enough: assert that the actual wrist/finger arrays
        # match, so a comparison cannot silently use four unrelated seed 78s.
        for candidate_id in sorted(common_ids):
            reference: dict[str, np.ndarray] | None = None
            for object_name, _gap, _source_name in KEY_SPECS:
                candidate = planned_sim_candidates[object_name][candidate_id]
                arrays = {
                    filename: np.load(candidate / filename)
                    for filename in (
                        "wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy"
                    )
                }
                if reference is None:
                    reference = arrays
                    continue
                if any(
                    not np.array_equal(arrays[name], reference[name])
                    for name in arrays
                ):
                    failures.append(
                        f"common candidate {candidate_id}: wrist/finger arrays differ by key"
                    )
                    break

    fixture = project / "precision_insertion" / "fixtures" / FIXTURE_NAME
    if not (fixture / "socket_shared_bore_1p5.obj").is_file():
        failures.append("socket collision mesh is missing")
    if not (fixture / "task_geometry.json").is_file():
        failures.append("socket task geometry is missing")
    socket_object = shared_root / "object_processing" / SOCKET_OBJECT_NAME
    socket_required = [
        socket_object / "raw_mesh" / f"{SOCKET_OBJECT_NAME}.obj",
        socket_object / "processed_data" / "mesh" / "simplified.obj",
        socket_object / "processed_data" / "mesh" / "static_collision.obj",
        socket_object / "processed_data" / "info" / "simplified.json",
        socket_object / "processed_data" / "info" / "symmetry.json",
        socket_object / "processed_data" / "info" / "frame_contract.json",
        socket_object / "processed_data" / "urdf" / "socket_static_exact.urdf",
        fixture / "pose_measurement_asset.json",
    ]
    failures.extend(
        f"missing socket pose asset: {path}"
        for path in socket_required
        if not path.is_file()
    )
    frame_path = socket_object / "processed_data" / "info" / "frame_contract.json"
    task_geometry_path = fixture / "task_geometry.json"
    if frame_path.is_file():
        frame = json.loads(frame_path.read_text())
        transform = np.asarray(frame.get("T_socket_raw_mesh"), dtype=float)
        if transform.shape != (4, 4) or not np.allclose(transform, np.eye(4)):
            failures.append(
                f"{SOCKET_OBJECT_NAME}: raw mesh must preserve the socket frame"
            )
    if task_geometry_path.is_file():
        task_geometry = json.loads(task_geometry_path.read_text())
        if task_geometry.get("socket_pose_object") != SOCKET_OBJECT_NAME:
            failures.append("socket task geometry references the wrong pose object")
        transform = np.asarray(
            task_geometry.get("T_socket_pose_object"), dtype=float
        )
        if transform.shape != (4, 4) or not np.allclose(transform, np.eye(4)):
            failures.append("socket pose object must use the task socket frame exactly")
    socket_repre = (
        project / "foundpose_assets" / SOCKET_OBJECT_NAME / "object_repre" /
        "v1" / SOCKET_OBJECT_NAME / "1" / "repre.pth"
    )
    if not socket_repre.is_file():
        blockers.append(
            f"{SOCKET_OBJECT_NAME}: FoundPose repre.pth not generated"
        )
    measurement_contract = fixture / "pose_measurement_asset.json"
    if measurement_contract.is_file():
        payload = json.loads(measurement_contract.read_text())
        if payload.get("output_scope") != "session_only":
            failures.append("socket pose measurement must be session-scoped")
        if "fixture_pose.session.json" not in payload.get("output_pattern", ""):
            failures.append("socket session pose output pattern is missing")

    camera_profile = source_dir.parent / "autodex_camera_profile.json"
    if not camera_profile.is_file():
        failures.append("canonical AutoDex camera profile is missing")
    else:
        profile = json.loads(camera_profile.read_text())
        if profile.get("capture_sync") != "hardware":
            failures.append("precision insertion must preserve AutoDex hardware sync")
        if profile.get("pc_list") != [
            "capture1", "capture2", "capture3", "capture5", "capture6"
        ]:
            failures.append("precision insertion AutoDex capture-PC set changed")
    camera_audit = project / "precision_insertion" / "autodex_camera_runtime_audit.json"
    if not camera_audit.is_file():
        blockers.append("AutoDex camera/calibration/hand-eye runtime audit not recorded")
    else:
        audit = json.loads(camera_audit.read_text())
        if audit.get("status") != "PASS":
            blockers.append("recorded AutoDex camera runtime audit is not PASS")

    print("geometry validation:", "PASS" if not failures else "FAIL")
    for failure in failures:
        print("  ERROR:", failure)
    print("runtime blockers:", len(blockers))
    for blocker in blockers:
        print("  BLOCKED:", blocker)

    if failures or (require_runtime and blockers):
        return 1
    return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "assets" / "precision_insertion" / "source",
    )
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.shared_root, args.source_dir, args.require_runtime))


if __name__ == "__main__":
    main()
