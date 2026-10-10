"""Load v8 reset grasps without AutoDex's legacy tabletop-index mapping.

This is a read-only seed loader, not a reset trajectory or execution policy.
Only explicitly provenance-bound, MuJoCo-stable full-key candidates enter it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from autodex.utils.coverage import grasp_priority_score, read_grasp_stats
from autodex.utils.path import RESET_RELEASE_HEIGHTS_CM

from .config import TaskMode
from .geometry import validate_se3
from .reorient_assets import (
    _paths, _same_scene_geometry, _scene_path, _sim_filter_scene_path,
    _tabletop_ids, _validate_scene,
)


GRASP_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
               "bodex_info.npy", "sim_eval.json")
EVIDENCE_SCHEMA = "precision_insertion_v8_reset_candidate_evidence_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stem(value: int | str, ids: tuple[int, ...], label: str) -> int:
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"{label} must be a numeric v8 tabletop stem")
    number = int(value)
    if number not in ids:
        raise ValueError(f"{label} has no v8 tabletop asset: {value}")
    return number


def _valid_scene_pair(root: Path, object_dir: Path, mode: TaskMode,
                      h_cm: int, i: int, j: int) -> tuple[Path, Path]:
    bodex = _scene_path(object_dir, h_cm, i, j)
    sim = _sim_filter_scene_path(root, mode, h_cm, i, j)
    a = json.loads(bodex.read_text(encoding="utf-8"))
    b = json.loads(sim.read_text(encoding="utf-8"))
    for scene in (a, b):
        _validate_scene(scene, mode=mode, object_dir=object_dir,
                        h_cm=h_cm, i=i, j=j)
    if not _same_scene_geometry(a, b):
        raise ValueError("BODex and sim-filter reset scenes differ")
    return bodex, sim


def _candidate_arrays(seed: Path, *, mode: TaskMode, cell: str,
                      h_cm: int, scenes: tuple[Path, Path]) -> tuple:
    evidence_path = seed / "source_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (evidence.get("schema") != EVIDENCE_SCHEMA or
            evidence.get("full_key_object") != mode.key_object or
            evidence.get("cell") != cell or
            evidence.get("height_cm") != h_cm or
            evidence.get("seed_id") != seed.name):
        raise ValueError(f"wrong full-key reset provenance: {seed}")
    hashes = evidence.get("candidate_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(GRASP_FILES):
        raise ValueError(f"incomplete reset file hashes: {seed}")
    for name in GRASP_FILES:
        if _sha256(seed / name) != hashes[name]:
            raise ValueError(f"changed reset candidate file: {seed / name}")
    if evidence.get("scene_sha256") != {
            "bodex": _sha256(scenes[0]), "sim_filter": _sha256(scenes[1])}:
        raise ValueError(f"reset proposal scene changed: {seed}")
    result = json.loads((seed / "sim_eval.json").read_text(encoding="utf-8"))
    if (result.get("success") is not True or
            result.get("hand") != "inspire" or
            result.get("version") != "v8"):
        raise ValueError(f"reset seed lacks original full-key MuJoCo pass: {seed}")
    wrist = validate_se3(np.load(seed / "wrist_se3.npy", allow_pickle=False),
                         name=f"reset {cell}/{seed.name} T_key_hand")
    hands = []
    for name in ("pregrasp_pose.npy", "grasp_pose.npy"):
        q = np.load(seed / name, allow_pickle=False)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid Inspire reset joints: {seed / name}")
        hands.append(q)
    openposes = []
    i, j = map(int, cell.split("_"))
    for pose in (i, j):
        path = seed / f"openpose_{pose:03d}.npy"
        if path.is_file():
            q = np.load(path, allow_pickle=False)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                raise ValueError(f"invalid Inspire reset openpose: {path}")
            openposes.append(q)
        else:
            openposes.append(None)
    return wrist, hands[0], hands[1], openposes[0], openposes[1]


def load_v8_reset_seeds(
    *, shared_root: Path, mode: TaskMode, height_cm: int,
    from_pose_stem: int | str, to_pose_stem: int | str,
    T_robot_key: np.ndarray, attempted_ids: tuple[str, ...] = (),
) -> dict | None:
    """Return stock-planner-shaped seeds for one exact v8 directed cell.

    The wrist transforms stored by BODex are key-frame transforms; this
    loader transforms them to robot frame from the *freshly observed* key.
    ``None`` means this cell has no unattempted stable seed, not that repose
    is impossible. The caller still owes socket-aware full-chain preflight.
    """
    root, object_dir = _paths(shared_root, mode)
    ids = _tabletop_ids(object_dir)
    i = _stem(from_pose_stem, ids, "from pose")
    j = _stem(to_pose_stem, ids, "to pose")
    if i == j:
        raise ValueError("reset transition needs two different tabletop poses")
    if type(height_cm) is not int or height_cm not in RESET_RELEASE_HEIGHTS_CM:
        raise ValueError("unsupported AutoDex reset release height")
    pose = validate_se3(T_robot_key, name="fresh T_robot_key")
    if any(not str(seed).isdigit() for seed in attempted_ids):
        raise ValueError("attempted reset seed IDs must be numeric")
    attempted = {str(int(seed)) for seed in attempted_ids}
    cell = f"{i}_{j}"
    folder = (root / "AutoDex" / "candidates" / "inspire" /
              f"reset_{height_cm}" / mode.key_object /
              f"reorient_{height_cm}" / cell)
    if not folder.is_dir():
        return None
    seeds = [entry for entry in folder.iterdir()
             if entry.is_dir() and entry.name.isdigit() and
             entry.name == str(int(entry.name)) and
             entry.name not in attempted]
    if not seeds:
        return None
    scenes = _valid_scene_pair(root, object_dir, mode, height_cm, i, j)
    seeds.sort(key=lambda seed: (
        -grasp_priority_score(*read_grasp_stats(str(seed))), int(seed.name)))
    rows = [_candidate_arrays(seed, mode=mode, cell=cell,
                              h_cm=height_cm, scenes=scenes) for seed in seeds]
    wrists = np.stack([pose @ row[0] for row in rows])
    return {
        "wrist_se3": wrists,
        "pregrasp": np.stack([row[1] for row in rows]),
        "grasp": np.stack([row[2] for row in rows]),
        "openpose_start": [row[3] for row in rows],
        "openpose_target": [row[4] for row in rows],
        "scene_info": [{
            "grasp_idx": seed.name, "cell": cell, "v8_cell": cell,
            "candidate_contract": "precision_insertion_v8_direct_reset",
            "h_cm": height_cm, "source": str(seed),
            "robot_ready": False,
        } for seed in seeds],
        "n_total": len(seeds), "v8_i": i, "v8_j": j,
        "robot_ready": False,
    }
