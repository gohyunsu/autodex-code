#!/usr/bin/env python3
"""Render the *offline-filtered* 1,000/scene cylinder grasps at 20 mm.

This consumes the isolated AutoDex sim-filter pool, not the empty live v8
candidate tree. It verifies the complete original filter result and freshly
re-screens every stable grasp for every socket before rendering anything.
Images are *nominal fixed-key/commanded-hand* diagnostics, not achieved
MuJoCo poses or robot/insertion successes. The separate grasp-fidelity audit
found deep hand/key overlap in all such fixed-pose cylinder illustrations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.endpoint import _load_mesh, screen_grasp_endpoint
from render_eligible_grasp_endpoints import RENDERER, _write_bundle
from screen_cylinder_tabletop_batch import KEY, SCENES, _scene_rows


REPO = Path(__file__).resolve().parents[2]
VIEWS = ("endpoint-oblique", "endpoint-side")
HOLDS = ("mujoco_squeeze", "autodex_default_controller_hold")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_rows(summary_path: Path, shared_root: Path) -> tuple[dict, list[dict]]:
    """Require a complete stage and exact agreement with fresh CAD screens."""
    source = Path(summary_path).expanduser().resolve()
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (summary.get("schema") != "precision_insertion_cylinder_1000_per_scene_screen_v1"
            or summary.get("status") != "offline_grasp_and_20mm_endpoint_only_not_robot_ready"
            or summary.get("physical_key_object") != KEY):
        raise ValueError("not the complete isolated cylinder screen summary")
    expected = summary["expected_proposals_per_scene"]
    if expected != 1000 or summary.get("total_raw_proposals") != 2 * expected:
        raise ValueError("expected exactly 1,000 staged seeds per tabletop scene")
    root = Path(shared_root).expanduser().resolve()
    stage = Path(summary["stage_root"]).resolve()
    candidates = Path(summary["original_autodex_sim_candidate_root"]).resolve()
    stage_manifest = json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))
    if (stage_manifest.get("seed_count_per_scene") != expected or
            stage_manifest.get("seed_count_total") != 2 * expected or
            stage_manifest.get("full_key_object") != KEY):
        raise ValueError("staging manifest does not describe the full-key 2,000-seed run")
    stable = {}
    for scene_id in SCENES:
        counts, passed = _scene_rows(stage, candidates, scene_id, expected)
        if counts != summary["scene_counts"][scene_id]:
            raise ValueError(f"original AutoDex filter count changed for scene {scene_id}")
        stable.update({f"table/{scene_id}/{seed}": candidate for seed, candidate in passed})
    if not stable:
        raise ValueError("no MuJoCo-stable grasp candidates")
    if set(summary.get("socket_gaps", {})) != {
            f"{int(gap):02d}mm" for gap in CYLINDER_RADIAL_GAPS_MM}:
        raise ValueError("socket gap list is incomplete")
    clearance = float(summary["minimum_hand_clearance_m"])
    if not 0 < clearance < 0.001:
        raise ValueError("invalid endpoint numerical clearance")

    eligible = []
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        label = f"{int(gap):02d}mm"
        declared = summary["socket_gaps"][label]
        if declared["socket_object"] != mode.socket_object or declared.get("errors"):
            raise ValueError(f"socket identity/error mismatch: {label}")
        current = []
        for identifier, candidate in stable.items():
            scene_id, seed = identifier.split("/")[1:]
            saved_path = source.parent / f"gap_{label}" / "table" / scene_id / f"{seed}.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            stage_eval = stage / KEY / "table" / scene_id / seed / "sim_eval.json"
            if (saved.get("candidate_dir") != str(candidate) or
                    saved.get("original_autodex_sim_filter_pass") is not True or
                    saved.get("original_autodex_sim_eval") != str(stage_eval) or
                    json.loads(stage_eval.read_text(encoding="utf-8")).get("success") is not True):
                raise ValueError(f"stored sim/endpoint provenance mismatch: {label}/{identifier}")
            fresh = screen_grasp_endpoint(
                shared_root=root, mode=mode, candidate_dir=candidate,
                minimum_hand_clearance_m=clearance)
            if (saved.get("input_sha256") != fresh["input_sha256"] or
                    saved.get("endpoint_pass") is not fresh["endpoint_pass"] or
                    not np.allclose(saved["T_socket_key_tested"],
                                    fresh["T_socket_key_tested"], atol=1e-9)):
                raise ValueError(f"stored 20 mm screen is stale: {label}/{identifier}")
            if fresh["endpoint_pass"]:
                if any(fresh["hold_pose_screens"][hold]["clear"] is not True
                       for hold in HOLDS):
                    raise ValueError(f"hold-pose screen mismatch: {label}/{identifier}")
                current.append(identifier)
                eligible.append({"mode": mode, "gap": label, "id": identifier,
                                 "candidate": candidate, "screen": fresh,
                                 "saved_screen": saved_path,
                                 "sim_eval": stage_eval})
        if current != declared["eligible_offline_grasp_ids"]:
            raise ValueError(f"eligible ID list changed: {label}")
        for scene_id in SCENES:
            recorded = declared["by_scene"][scene_id]
            if (recorded["tested"] != summary["scene_counts"][scene_id]["mujoco_stable"] or
                    recorded["endpoint_pass"] != sum(
                        identifier.startswith(f"table/{scene_id}/") for identifier in current)):
                raise ValueError(f"scene endpoint count changed: {label}/{scene_id}")
    return summary, eligible


def render(*, summary_path: Path, shared_root: Path, output_root: Path,
           blender: Path | None = None, width: int = 1600,
           height: int = 900, workers: int = 4) -> dict:
    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    if width < 320 or height < 180:
        raise ValueError("render dimensions are too small")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1, 8]")
    summary, rows = verified_rows(summary_path, shared_root)
    if not rows:
        raise ValueError("the complete screen has no eligible grasp to render")
    executable = Path(blender or shutil.which("blender") or "").expanduser()
    if not executable.is_file() or not RENDERER.is_file():
        raise FileNotFoundError("Blender actual-mesh renderer is missing")
    root = Path(shared_root).expanduser().resolve()
    target.mkdir(parents=True)
    manifest = {
        "schema": "precision_insertion_offline_cylinder_endpoint_renders_v1",
        "source_summary": str(Path(summary_path).resolve()),
        "source_summary_sha256": _sha256(Path(summary_path).resolve()),
        "source_pool": summary["original_autodex_sim_candidate_root"],
        "source_evidence": "full-key AutoDex tabletop filter + MuJoCo + exact 20 mm key/hand/socket endpoint",
        "pose_fidelity_warning": (
            "Nominal initial T_key_hand and commanded squeeze joints are combined. "
            "The resulting hand/key penetration does not depict achieved MuJoCo contact; "
            "consult the separate dynamic grasp-fidelity audit before use."),
        "verification_depth_mm": 20.0,
        "numerical_hand_clearance_m": summary["minimum_hand_clearance_m"],
        "eligible_per_socket": {label: len(info["eligible_offline_grasp_ids"])
                                for label, info in summary["socket_gaps"].items()},
        "expected_rendered_pairs": len(rows), "rendered_pairs": 0,
        "complete": False,
        "robot_ready": False,
        "not_validated": ["Franka reachability or continuous trajectory",
                          "rigid hand/key relation through squeeze and lift",
                          "achieved hand/key contact quality",
                          "guarded contact dynamics or physical insertion"],
        "grasps": [],
    }
    key_path = target / "mesh_cache" / "key.ply"
    key_path.parent.mkdir()
    _load_mesh(AssetPaths(root, rows[0]["mode"]).raw_mesh(KEY)).export(key_path)
    socket_paths = {}
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        path = target / "mesh_cache" / f"socket_gap_{int(gap):02d}mm.ply"
        _load_mesh(AssetPaths(root, mode).socket_collision_mesh).export(path)
        socket_paths[f"{int(gap):02d}mm"] = path
    for row in rows:
        scene_id, seed = row["id"].split("/")[1:]
        directory = target / f"gap_{row['gap']}" / f"pose_{scene_id}" / f"grasp_{seed}"
        directory.mkdir(parents=True)
        screen_path = directory / "fresh_endpoint_screen.json"
        screen_path.write_text(json.dumps(row["screen"], indent=2) + "\n",
                               encoding="utf-8")
        jobs = []
        for hold in HOLDS:
            hold_dir = directory / hold
            hold_dir.mkdir()
            bundle = _write_bundle(
                output=hold_dir, shared_root=root, mode=row["mode"],
                candidate=row["candidate"], endpoint=row["screen"],
                hold_name=hold, key_mesh_path=key_path,
                socket_mesh_path=socket_paths[row["gap"]])
            for view in VIEWS:
                image = hold_dir / f"{view}.png"
                command = [
                    str(executable), "--background", "--python", str(RENDERER),
                    "--", str(bundle), "--output", str(image),
                    "--width", str(width), "--height", str(height),
                    "--view", view, "--still-frame", "1",
                ]
                jobs.append((image, command))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(subprocess.run, command, check=True,
                                   stdout=subprocess.DEVNULL)
                       for _, command in jobs]
            for future in futures:
                future.result()
        images = []
        for image, _ in jobs:
            if not image.is_file():
                raise RuntimeError(f"Blender produced no image: {image}")
            images.append(str(image.relative_to(target)))
        manifest["grasps"].append({
            "id": row["id"], "gap_mm": row["mode"].gap_mm,
            "candidate_dir": str(row["candidate"]),
            "sim_eval": str(row["sim_eval"]),
            "saved_endpoint_screen": str(row["saved_screen"]),
            "fresh_endpoint_screen": str(screen_path.relative_to(target)),
            "minimum_observed_hand_clearance_m":
                row["screen"]["minimum_observed_hand_clearance_m"],
            "images": images,
        })
        manifest["rendered_pairs"] += 1
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["complete"] = True
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = render(summary_path=args.summary, shared_root=args.shared_root,
                    output_root=args.output_root, blender=args.blender,
                    width=args.width, height=args.height, workers=args.workers)
    print(json.dumps({"output_root": str(args.output_root.resolve()),
                      "eligible_per_socket": result["eligible_per_socket"],
                      "rendered_pairs": result["rendered_pairs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
