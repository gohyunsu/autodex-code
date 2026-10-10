"""Promote a complete square-key MuJoCo pilot to an explicit v8 scene.

The original AutoDex sim-filter result is reused; this module adds an exact
nominal 20 mm hand/socket endpoint screen and immutable source provenance.
It does not certify the post-squeeze key-in-hand relation, Franka motion,
guarded contact, or physical success. An existing v8 scene is never replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3


SOURCE_FILES = (
    "wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
    "bodex_info.npy", "coll_valid.npy", "contact_screen.json",
    "sim_eval.json", "sim_traj.json",
)
SCHEMA = "precision_insertion_square_tabletop_simulation_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _seed_ids(scene: Path) -> set[str]:
    if not scene.is_dir():
        raise FileNotFoundError(f"missing candidate scene: {scene}")
    return {entry.name for entry in scene.iterdir()
            if entry.is_dir() and entry.name.isdigit() and
            entry.name == str(int(entry.name))}


def _source_seed_check(seed: Path, passed_seed: Path, *, version: str) -> None:
    if not all((seed / name).is_file() for name in SOURCE_FILES):
        raise FileNotFoundError(f"incomplete original filter result: {seed}")
    for name in ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
                 "bodex_info.npy"):
        if _sha(seed / name) != _sha(passed_seed / name):
            raise ValueError(f"sim-filter pass pool differs from source: {seed}/{name}")
    validate_se3(np.load(seed / "wrist_se3.npy", allow_pickle=False),
                 name=f"{seed.name} T_key_hand")
    for name in ("pregrasp_pose.npy", "grasp_pose.npy"):
        joints = np.load(seed / name, allow_pickle=False)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"invalid Inspire joints: {seed / name}")
    collision = np.load(seed / "coll_valid.npy", allow_pickle=False)
    if collision.shape != () or not bool(collision):
        raise ValueError(f"stock pregrasp scene collision failed: {seed}")
    contact = _json(seed / "contact_screen.json")
    quality = contact.get("quality")
    if (contact.get("accepted") is not True or
            contact.get("contact_policy_mode") != "report-only" or
            not isinstance(quality, dict)):
        raise ValueError(f"missing relaxed BODex quality evidence: {seed}")
    grasp_error = float(quality["grasp_error_max"])
    distance = float(quality["contact_distance_mean_abs_m"])
    if (not math.isfinite(grasp_error) or not math.isfinite(distance) or
            grasp_error > .2 or distance > .01 or
            grasp_error < 0 or distance < 0):
        raise ValueError(f"BODex quality threshold failed: {seed}")
    simulation = _json(seed / "sim_eval.json")
    if simulation != {"success": True, "hand": "inspire", "version": version}:
        raise ValueError(f"not a full original Inspire MuJoCo pass: {seed}")
    trajectory = _json(seed / "sim_traj.json")
    phases = trajectory.get("phase")
    if (not isinstance(phases, list) or not phases or
            "squeeze" not in phases or "force_gravity" not in phases or
            not isinstance(trajectory.get("object_pose"), list) or
            not isinstance(trajectory.get("robot_qpos"), list) or
            len(phases) != len(trajectory["object_pose"]) or
            len(phases) != len(trajectory["robot_qpos"])):
        raise ValueError(f"incomplete stock MuJoCo trajectory: {seed}")


def promote_square_scene(
    *, shared_root: Path, mode: TaskMode, source_scene: Path,
    pass_scene: Path, highres_report: Path, output_scene: Path,
    simulation_version: str, minimum_hand_clearance_m: float,
    screen: Callable | None = None,
) -> dict:
    """Validate a *whole* pass scene before one non-overwriting scene install."""
    if mode.family != "square":
        raise ValueError("this promotion is only for the matching square key")
    if (not isinstance(simulation_version, str) or
            not simulation_version.strip() or
            not math.isfinite(minimum_hand_clearance_m) or
            minimum_hand_clearance_m <= 0):
        raise ValueError("simulation version and clearance must be explicit")
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    source = Path(source_scene).expanduser().resolve()
    passed = Path(pass_scene).expanduser().resolve()
    highres = Path(highres_report).expanduser().resolve()
    output = Path(output_scene).expanduser().resolve()
    if (source.name != passed.name or source.parent.name != "table" or
            passed.parent.name != "table" or not source.name.isdigit() or
            source.name != str(int(source.name)) or
            source.parent.parent.name != mode.key_object or
            passed.parent.parent.name != mode.key_object or
            output != paths.candidate_dir / "table" / source.name):
        raise ValueError("source, pass and destination must name one v8 table scene")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace v8 candidate scene: {output}")
    scene = paths.scene_dir / f"{source.name}.json"
    scene_record = _json(scene)
    metadata = scene_record.get("meta")
    if not isinstance(metadata, dict) or not isinstance(
            metadata.get("pose_idx"), str):
        raise ValueError("source scene lacks a v8 tabletop pose")
    try:
        target = scene_record["scene"]["mesh"]["target"]
        scene_mesh = Path(target["file_path"]).resolve()
        scene_urdf = Path(target["urdf_path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("source scene lacks full-key geometry") from exc
    if scene_mesh != paths.key_planning_mesh or not scene_urdf.is_file():
        raise ValueError("source scene is not the full-key v8 scene")
    tabletop = paths.key_tabletop_dir / f"{metadata['pose_idx']}.npy"
    if not tabletop.is_file():
        raise FileNotFoundError(f"source v8 tabletop pose missing: {tabletop}")
    report = _json(highres)
    sampling = report.get("sampling")
    passed_list = report.get("passed_candidates")
    if (report.get("status") !=
            "sampled_prefilter_not_trajectory_or_physical_validation" or
            report.get("contact_policy_mode") != "disabled" or
            not isinstance(sampling, dict) or
            type(sampling.get("samples_per_link")) is not int or
            sampling["samples_per_link"] < 3000 or
            not isinstance(passed_list, list) or
            any(not isinstance(value, str) or not value.isdigit()
                for value in passed_list) or
            len(set(passed_list)) != len(passed_list) or
            Path(report.get("scene", "")).resolve().parts[-3:] !=
            (mode.key_object, "table", source.name) or
            Path(report.get("tabletop_pose", "")).resolve() != tabletop or
            Path(report.get("task_geometry", "")).resolve() !=
            paths.task_geometry or
            Path(report.get("socket_mesh", "")).resolve() !=
            paths.socket_collision_mesh):
        raise ValueError("high-resolution geometry screen is for another task")
    ids = _seed_ids(passed)
    if not ids or not ids.issubset(_seed_ids(source)):
        raise ValueError("pass pool is empty or not contained in original stage")
    highres_ids = set(passed_list)
    if not ids.issubset(highres_ids):
        raise ValueError("MuJoCo pass pool contains a high-resolution rejection")
    stable_ids = {name for name in _seed_ids(source)
                  if _json(source / name / "sim_eval.json").get("success") is True}
    if stable_ids != ids:
        raise ValueError("pass pool does not equal the complete stable source set")
    if screen is None:
        from .endpoint import screen_grasp_endpoint
        screen = screen_grasp_endpoint
    rows = []
    for name in sorted(ids, key=int):
        seed = source / name
        _source_seed_check(seed, passed / name, version=simulation_version)
        endpoint = screen(
            shared_root=root, mode=mode, candidate_dir=seed,
            minimum_hand_clearance_m=minimum_hand_clearance_m)
        if (not isinstance(endpoint, dict) or
                endpoint.get("endpoint_pass") is not True or
                not isinstance(endpoint.get("input_sha256"), dict)):
            raise ValueError(f"exact 20 mm endpoint failed: {seed}")
        rows.append((name, endpoint))
    if not paths.key_planning_mesh.is_file() or not paths.raw_mesh(
            mode.key_object).is_file():
        raise FileNotFoundError("full square-key CAD is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".precision-square-promotion-", dir=output.parent) as temp:
        stage = Path(temp) / "scene"
        stage.mkdir()
        shutil.copy2(highres, stage / "highres_report.json")
        for name, endpoint in rows:
            candidate = stage / name
            shutil.copytree(source / name, candidate)
            endpoint_file = candidate / "nominal_endpoint_screen.json"
            endpoint_file.write_text(json.dumps(endpoint, indent=2) + "\n",
                                     encoding="utf-8")
            validation = {
                "schema": SCHEMA,
                "status": "passed",
                "full_object_mesh_sha256": _sha(paths.key_planning_mesh),
                "raw_key_mesh_sha256": _sha(paths.raw_mesh(mode.key_object)),
                "source_candidate": str(source / name),
                "source_sim_filter_pass": str(passed / name),
                "source_highres_report": str(highres),
                "source_highres_report_sha256": _sha(highres),
                "highres_report_copy_sha256": _sha(
                    stage / "highres_report.json"),
                "source_scene_sha256": _sha(scene),
                "source_file_sha256": {
                    file: _sha(candidate / file) for file in SOURCE_FILES},
                "nominal_endpoint_screen_sha256": _sha(endpoint_file),
                "contact_policy_mode": "report-only",
                "physical_validation": False,
                "robot_ready": False,
                "warning": (
                    "Stock grasp stability and nominal 20 mm endpoint only; "
                    "key-in-hand relation after squeeze is unmeasured"),
            }
            (candidate / "simulation_validation.json").write_text(
                json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema": "precision_insertion_square_scene_promotion_v1",
            "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                     "key_object": mode.key_object,
                     "socket_object": mode.socket_object},
            "source_scene": str(source), "source_pass_scene": str(passed),
            "highres_report": str(highres),
            "highres_report_sha256": _sha(highres),
            "simulation_version": simulation_version,
            "minimum_hand_clearance_m": minimum_hand_clearance_m,
            "candidate_ids": [name for name, _ in rows],
            "scope": "v8_grasp_filter_plus_nominal_20mm_endpoint_only",
            "robot_ready": False,
        }
        (stage / "promotion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"candidate scene appeared during promotion: {output}")
        stage.rename(output)
    return manifest
