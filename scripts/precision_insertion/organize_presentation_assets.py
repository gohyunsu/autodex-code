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
DEFAULT_CANDIDATE_SET = (
    REPO_ROOT / "assets/precision_insertion/presentation_candidate_set.json"
)
DEFAULT_FAILURES = (
    REPO_ROOT / "assets/precision_insertion/presentation_trajectory_failures.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_count(report: dict[str, Any]) -> int:
    return int(report.get("candidate_count", len(report.get("candidates", []))))


def _latest_report(task_root: Path, pose: str) -> Path:
    matches = list(task_root.glob(f"rigid_insertion_grasp_screen_table_{pose}*.json"))
    if not matches:
        raise FileNotFoundError(f"no screening report for pose {pose}")
    return max(matches, key=lambda path: (_candidate_count(_load(path)), path.stat().st_mtime))


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--presentation-root", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--requested-grasps-per-success-pose", type=int, default=20)
    parser.add_argument("--candidate-set", type=Path, default=DEFAULT_CANDIDATE_SET)
    parser.add_argument("--trajectory-failures", type=Path, default=DEFAULT_FAILURES)
    args = parser.parse_args()
    task_root = args.task_root.expanduser().resolve()
    root = args.presentation_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_set = _load(args.candidate_set.expanduser().resolve())
    failures = _load(args.trajectory_failures.expanduser().resolve())
    selected_ids = set(candidate_set["selected_candidate_ids"])
    rejected_ids = [
        item["candidate"]
        for item in failures["rejected_after_static_prefilter"]
    ]

    pose_rows: list[dict[str, Any]] = []
    for index in range(5):
        pose = f"{index:03d}"
        planning = root / "04_planning" / f"pose_{pose}"
        for child in ("grasp_screen", "grasps", "reorientation"):
            (planning / child).mkdir(parents=True, exist_ok=True)
        report_path = _latest_report(task_root, pose)
        report = _load(report_path)
        snapshot = planning / "grasp_screen" / "candidate_screen.json"
        shutil.copy2(report_path, snapshot)
        passed = [str(value) for value in report.get("passed_candidates", [])]
        rendered: list[dict[str, str]] = []
        for video in sorted((planning / "grasps").glob("grasp_*/plan.mp4")):
            candidate = video.parent.name.removeprefix("grasp_")
            rendered.append({
                "candidate": candidate,
                "video": _relative(video, root),
            })
        pose_image = root / "02_tabletop_poses/key" / f"pose_{pose}" / "pose.png"
        reorient = (
            root / "06_reorientation" / f"pose_{pose}" /
            "reorient_to_pose_004_concept.mp4"
        )
        direct = bool(rendered)
        if pose == candidate_set["tabletop_pose_id"]:
            rendered_ids = {item["candidate"] for item in rendered}
            if rendered_ids != selected_ids:
                raise RuntimeError(
                    "rendered candidate set does not match the frozen selection: "
                    f"missing={sorted(selected_ids - rendered_ids)}, "
                    f"extra={sorted(rendered_ids - selected_ids)}"
                )
        row = {
            "pose_id": pose,
            "pose_image": _relative(pose_image, root) if pose_image.is_file() else None,
            "screen_report": _relative(snapshot, root),
            "sparse_candidate_count": _candidate_count(report),
            "sampled_static_prefilter_candidates": passed,
            "sampled_full_trajectory_candidates": [
                item["candidate"] for item in rendered
            ],
            "trajectory_rejected_candidates": (
                rejected_ids if pose == candidate_set["tabletop_pose_id"] else []
            ),
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
                else "invoke reorientation primitive, reperceive, then select a pose-004 grasp"
            ),
            "reorientation_asset": _relative(reorient, root) if reorient.is_file() else None,
            "reorientation_asset_is_concept_only": reorient.is_file(),
        }
        (planning / "manifest.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        pose_rows.append(row)

    reset_videos = sorted((root / "05_reset").glob("pose_*/grasp_*/reset.mp4"))
    manifest = {
        "schema_version": 1,
        "task_pose_symmetry": "identity",
        "tabletop_pose_count": 5,
        "grasp_animation_request_per_success_pose": args.requested_grasps_per_success_pose,
        "evidence_levels": {
            "screened": "sampled full-hand/key/table/socket geometry",
            "preview": "sampled numerical IK and mesh collision preview",
            "not_yet_proven": [
                "continuous cuRobo trajectory",
                "MuJoCo grasp stability",
                "hardware execution",
                "physical insertion success",
            ],
        },
        "poses": pose_rows,
        "frozen_candidate_set": candidate_set,
        "trajectory_failure_evidence": failures,
        "reset_videos": [_relative(path, root) for path in reset_videos],
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
- `02_tabletop_poses/`: five key poses and the operational fixed socket pose
- `03_contact_policy/`: allowed handle side/rear contact versus forbidden shaft/front contact
- `04_planning/pose_000` ... `pose_004`: pose-local screening and direct plans
- `05_reset/`: reverse trajectory after a successful insertion
- `06_reorientation/`: exact-mesh task concepts; these are not robot plans

`manifest.json` is authoritative.  A video is called a *preview* only after
sampled IK and mesh checks; none of these files is physical-success evidence.
The request is {args.requested_grasps_per_success_pose} direct-plan videos per
successful pose, but the manifest reports only distinct native BODex candidates
that actually pass the current gates.  Missing videos are deliberately not
filled with camera variants or modified hand configurations.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
