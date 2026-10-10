"""Persist a read-only v8 reset preflight with exact input provenance.

No artifact in this module authorizes a Franka/Inspire motion. Every output
directory is exclusive, so a later attempt cannot overwrite prior evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .repose_transition import ReposeTransitionPreflight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_repose_preflight_artifacts(
    *, result: ReposeTransitionPreflight, trial_scene: dict,
    output_dir: Path, source_files: Mapping[str, Path],
) -> Path:
    """Save the frozen trial world, outcome and selected dense trajectories."""
    required = {"session", "catalog", "key_pose_world", "live_start_q", "limits"}
    if set(source_files) != required:
        raise ValueError(f"reset preflight source files must be {sorted(required)}")
    sources = {name: Path(path).expanduser().resolve()
               for name, path in source_files.items()}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"reset preflight {name} missing: {path}")
    if (not isinstance(trial_scene, dict) or
            not isinstance(trial_scene.get("mesh"), dict) or
            "fixture_socket" not in trial_scene["mesh"] or
            "target" not in trial_scene["mesh"]):
        raise ValueError("reset trial scene has no frozen socket or fresh key")
    report = result.to_record()
    if report.get("robot_ready") is not False:
        raise ValueError("reset preflight report must not authorize the robot")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    scene_bytes = (json.dumps(trial_scene, indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
    scene_path = target / "trial_scene.json"
    scene_path.write_bytes(scene_bytes)
    report["artifacts"] = {
        "trial_scene": scene_path.name,
        "trial_scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(sources.items())
        },
    }
    if result.held_plan is not None:
        if result.pickup_plan is None:
            raise ValueError("held reset plan lacks AutoDex pickup plan")
        held = result.held_plan
        if any(path is None for path in (
                held.lift_trajectory, held.transfer_trajectory,
                held.descent_trajectory)):
            raise ValueError("selected held reset has missing trajectory stage")
        arrays = {
            "pickup_approach": np.asarray(result.pickup_plan.traj,
                                          dtype=np.float64),
            "held_lift": np.asarray(held.lift_trajectory, dtype=np.float64),
            "held_transfer": np.asarray(held.transfer_trajectory,
                                        dtype=np.float64),
            "held_descent": np.asarray(held.descent_trajectory,
                                       dtype=np.float64),
            "held_hand_q": np.asarray(held.held_hand_q, dtype=np.float64),
        }
        if result.release_plan is not None:
            release = result.release_plan
            if (release.post_release_lift_trajectory is None or
                    release.retract_trajectory is None):
                raise ValueError("selected release plan has missing trajectory stage")
            arrays.update({
                "release_hand_q": np.asarray(release.release_hand_q,
                                             dtype=np.float64),
                "post_release_lift": np.asarray(
                    release.post_release_lift_trajectory, dtype=np.float64),
                "post_release_retract": np.asarray(
                    release.retract_trajectory, dtype=np.float64),
            })
        plan_path = target / "planned_trajectories.npz"
        np.savez_compressed(plan_path, **arrays)
        report["artifacts"]["planned_trajectories"] = plan_path.name
        report["artifacts"]["planned_trajectories_sha256"] = _sha256(plan_path)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target
