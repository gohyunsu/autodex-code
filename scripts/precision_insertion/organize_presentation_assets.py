#!/usr/bin/env python3
"""Create portable manifests for the precision-insertion presentation set.

The script never invents missing grasp videos.  It derives the direct-success
set from the latest sampled screening report, inventories rendered files, and
records the difference between requested, generated, and physically validated
assets.  Pose folders are always created for all five exact tabletop poses.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SHARED = Path.home() / "shared_data"
DEFAULT_TASK_ROOT = SHARED / "AutoDex/precision_insertion"
DEFAULT_PRESENTATION = DEFAULT_TASK_ROOT / "presentation_assets"
REPO_ROOT = Path(__file__).resolve().parents[2]
def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_count(report: dict[str, Any]) -> int:
    return int(report.get("candidate_count", len(report.get("candidates", []))))


def _latest_report(task_root: Path, pose: str) -> Path:
    matches = list(task_root.glob(f"rigid_insertion_grasp_screen_table_{pose}*.json"))
    if not matches:
        raise FileNotFoundError(f"no screening report for pose {pose}")
    # A frame-fixed report is authoritative even when an older, larger report
    # contains more candidates.  BODex contact points are scene/world-frame;
    # choosing by candidate count alone previously resurrected stale labels.
    frame_fixed = [path for path in matches if "frame_fixed" in path.stem]
    pool = frame_fixed or matches
    return max(pool, key=lambda path: path.stat().st_mtime)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--presentation-root", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--requested-grasps-per-success-pose", type=int, default=20)
    args = parser.parse_args()
    task_root = args.task_root.expanduser().resolve()
    root = args.presentation_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pose_rows: list[dict[str, Any]] = []
    for index in range(5):
        pose = f"{index:03d}"
        planning = root / "04_planning" / f"pose_{pose}"
        for child in ("grasp_screen", "grasps", "reorientation"):
            (planning / child).mkdir(parents=True, exist_ok=True)
        report_path = _latest_report(task_root, pose)
        report = _load(report_path)
        contact_frame_verified = (
            "frame_fixed" in report_path.stem or pose == "000"
        )
        snapshot = planning / "grasp_screen" / "candidate_screen.json"
        shutil.copy2(report_path, snapshot)
        passed = (
            [str(value) for value in report.get("passed_candidates", [])]
            if contact_frame_verified else []
        )
        contact_policy_passed = (
            [
                str(item["candidate"])
                for item in report.get("candidates", [])
                if item.get("contact_policy", {}).get("status") == "sampled_pass"
            ]
            if contact_frame_verified else []
        )
        rendered: list[dict[str, str]] = []
        for video in sorted((planning / "grasps").glob("grasp_*/plan.mp4")):
            candidate = video.parent.name.removeprefix("grasp_")
            rendered.append({
                "candidate": candidate,
                "video": _relative(video, root),
            })
        pose_image = root / "02_tabletop_poses/key" / f"pose_{pose}" / "pose.png"
        direct = bool(rendered)
        row = {
            "pose_id": pose,
            "pose_image": _relative(pose_image, root) if pose_image.is_file() else None,
            "screen_report": _relative(snapshot, root),
            "contact_frame_verified": contact_frame_verified,
            "sparse_candidate_count": _candidate_count(report),
            "sampled_contact_policy_candidates": contact_policy_passed,
            "sampled_static_prefilter_candidates": passed,
            "sampled_full_trajectory_candidates": [
                item["candidate"] for item in rendered
            ],
            "rendered_direct_plans": rendered,
            "requested_direct_plan_videos": (
                args.requested_grasps_per_success_pose if direct else 0
            ),
            "direct_plan_video_shortfall": max(
                0,
                (args.requested_grasps_per_success_pose if direct else 0)
                - len(rendered),
            ),
            "runtime_policy": (
                "reperceive key pose and replay a validated object-relative grasp"
                if direct
                else (
                    "no replayable candidate; fail closed until a validated "
                    "reorientation or direct plan is available"
                )
            ),
            "reorientation_asset": None,
        }
        (planning / "manifest.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        pose_rows.append(row)

    reset_videos = sorted((root / "05_reset").glob("pose_*/grasp_*/*.mp4"))
    combined_poses = root / "02_tabletop_poses/key/all_poses.png"
    policy_grid = root / "03_contact_policy/grasp_policy_grid_5x5.png"
    policy_grid_manifest = root / "03_contact_policy/grid/manifest.json"
    reset_status = root / "05_reset/status.json"
    reorientation_status = root / "06_reorientation/status.json"
    manifest = {
        "schema_version": 1,
        "task_pose_symmetry": "identity",
        "tabletop_pose_count": 5,
        "grasp_animation_request_per_success_pose": args.requested_grasps_per_success_pose,
        "evidence_levels": {
            "screened": (
                "sampled full-hand/key contact policy plus initial and "
                "insertion-pose table/socket geometry"
            ),
            "preview": "sampled numerical IK and mesh collision preview",
            "not_yet_proven": [
                "continuous cuRobo trajectory",
                "MuJoCo grasp stability",
                "hardware execution",
                "physical insertion success",
            ],
        },
        "poses": pose_rows,
        "combined_tabletop_pose_image": (
            _relative(combined_poses, root) if combined_poses.is_file() else None
        ),
        "contact_policy_grid": (
            _relative(policy_grid, root) if policy_grid.is_file() else None
        ),
        "contact_policy_grid_manifest": (
            _relative(policy_grid_manifest, root)
            if policy_grid_manifest.is_file() else None
        ),
        "reset_videos": [_relative(path, root) for path in reset_videos],
        "reset_status": (
            _relative(reset_status, root) if reset_status.is_file() else None
        ),
        "reorientation_status": (
            _relative(reorientation_status, root)
            if reorientation_status.is_file() else None
        ),
        "deprecated_audit_assets": "audit/deprecated_pre_contact_frame_fix",
        "important_runtime_rule": (
            "replay object-relative T_key_hand and replan from current perception; "
            "never replay stale world-frame joint trajectories"
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# Precision insertion presentation assets

All assets use the exact key/socket CAD geometry.  The five tabletop poses are
kept independent (`task_pose_symmetry = identity`).

## Layout

- `00_geometry/`: socket and 0.1/0.3/0.5/1.0/1.5 mm key family
- `01_compatibility/`: key/socket compatibility animation
- `02_tabletop_poses/`: five separate key poses, one combined view, and the fixed socket pose
- `03_contact_policy/`: contact-policy animation and a 5x5 actual-mesh pass/fail grid
- `04_planning/pose_000` ... `pose_004`: pose-local screening and direct plans
- `05_reset/status.json`: reset is fail-closed until a corrected forward plan passes
- `06_reorientation/status.json`: BODex/reorientation generation audit
- `audit/`: preserved pre-frame-fix visual references; never use as success evidence

`manifest.json` is authoritative.  A video is called a *preview* only after
sampled IK and mesh checks; none of these files is physical-success evidence.
The request is {args.requested_grasps_per_success_pose} direct-plan videos per
successful pose, but the manifest reports only candidates that pass the
current gates.  Missing videos are deliberately not filled with camera
variants, modified hand configurations, or stale pre-frame-fix previews.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
