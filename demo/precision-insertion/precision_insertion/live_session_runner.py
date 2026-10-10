"""Open one saved live-calibration session for independent trial supervision.

The camera startup is performed by ``start_precision_session`` first.  This
boundary verifies its immutable evidence and the current v8 endpoint pool,
then creates a SessionRunner.  It does not connect to a robot or issue motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .assets import AssetPaths
from .calibration import load_session_calibration
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .session_bootstrap import verify_session_evidence_bundle
from .session_runner import SessionRunner
from .session_policy import SessionDecision


@dataclass(frozen=True)
class OpenedPrecisionSession:
    runner: SessionRunner
    evidence_dir: Path
    source_binding_path: Path
    next_decision: SessionDecision


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_verified_session(
    *, mode: TaskMode, shared_root: Path, evidence_dir: Path,
    catalog_path: Path, output_dir: Path, max_xy_retries: int,
) -> OpenedPrecisionSession:
    """Fail before creating a run if camera/fixture/candidate evidence is stale.

    This is an *opening*, not a fresh calibration.  A producer must first run
    the ChArUco-then-socket sequence and retain its image evidence.  The key
    and socket FoundPose files are checked only for canonical availability;
    their independent pose-accuracy QA is still a physical commissioning gate.
    """
    root_input = Path(shared_root).expanduser()
    session_input = Path(evidence_dir).expanduser()
    catalog_input = Path(catalog_path).expanduser()
    output_input = Path(output_dir).expanduser()
    if any(not path.is_absolute() for path in
           (root_input, session_input, catalog_input, output_input)):
        raise ValueError("session paths must be explicit absolute paths")
    root = root_input.resolve()
    source = session_input.resolve()
    catalog_file = catalog_input.resolve()
    target = output_input.resolve()
    if target.exists():
        raise FileExistsError(f"session output already exists: {target}")
    if type(max_xy_retries) is not int or max_xy_retries < 0:
        raise ValueError("max_xy_retries must be a nonnegative integer")
    if not root.is_dir() or not source.is_dir() or not catalog_file.is_file():
        raise FileNotFoundError("shared root, session evidence or catalog is missing")
    paths = AssetPaths(root, mode)
    required = {
        "key raw mesh": paths.raw_mesh(mode.key_object),
        "socket raw mesh": paths.raw_mesh(mode.socket_object),
        "key FoundPose representation": paths.foundpose_repre(mode.key_object),
        "socket FoundPose representation": paths.foundpose_repre(mode.socket_object),
        "exact socket collision mesh": paths.socket_collision_mesh,
        "Franka/Inspire URDF": paths.robot_urdf,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    manifest = verify_session_evidence_bundle(source)
    if (manifest.get("all_frames_bound_to_acquisition_evidence") is not True or
            len(manifest.get("socket_capture_ids", [])) < 2):
        raise ValueError("session lacks repeated acquisition-bound socket captures")
    calibration_path = source / "session_calibration.json"
    calibration = load_session_calibration(
        calibration_path, mode=mode, shared_root=root)
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("v8 endpoint catalog must be a JSON object")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if (Path(catalog.get("shared_root", "")).expanduser().resolve() != root or
            not catalog.get("complete_scan")):
        raise ValueError("v8 catalog is incomplete or belongs to another shared root")
    rows = catalog.get("candidates")
    if (not isinstance(rows, list) or not rows or
            any(not isinstance(row, dict) or
                not isinstance(row.get("tabletop_pose_stem"), str) or
                not row["tabletop_pose_stem"] for row in rows)):
        raise ValueError("complete v8 catalog needs pose-indexed candidate rows")
    # Reuse the existing source-hash and endpoint-implementation checks for
    # every indexed pose, including poses with no eligible grasp.  A stale
    # catalogue must fail before any fresh camera request is scheduled.
    for stem in sorted({row["tabletop_pose_stem"] for row in rows}):
        selection = select_pose_candidates(
            catalog, expected_mode=mode, tabletop_pose_stem=stem)
        if selection["status"] not in {
                "candidates_available", "no_eligible_in_screened_pool"}:
            raise ValueError(f"v8 catalog is stale for pose {stem}: "
                             f"{selection['status']}")
    source_hashes = {
        "session_manifest_sha256": _sha(source / "evidence_manifest.json"),
        "session_calibration_sha256": _sha(calibration_path),
        "catalog_source_sha256": _sha(catalog_file),
        "key_foundpose_sha256": _sha(required["key FoundPose representation"]),
        "socket_foundpose_sha256": _sha(
            required["socket FoundPose representation"]),
    }

    # All inputs are checked before the SessionRunner creates any output.
    runner = SessionRunner(
        mode=mode, calibration=calibration, catalog=catalog,
        shared_root=root, output_dir=target,
        max_xy_retries=max_xy_retries)
    binding_path = target / "source_evidence_binding.json"
    with binding_path.open("x", encoding="utf-8") as stream:
        json.dump({
            "schema": "precision_insertion_session_source_binding_v1",
            "session_evidence_dir": str(source),
            "catalog_source_path": str(catalog_file),
            **source_hashes,
            "scope": "evidence_binding_not_robot_motion_authorization",
            "robot_ready": False,
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return OpenedPrecisionSession(
        runner=runner, evidence_dir=source,
        source_binding_path=binding_path,
        next_decision=runner.current_decision())
