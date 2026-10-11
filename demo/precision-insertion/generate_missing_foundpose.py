#!/usr/bin/env python3
"""Generate mesh-specific FoundPose assets with the RTX GPU, then install.

Run this with the isolated ``precision-foundpose`` Python and GPU access.
An interrupted staging directory is intentionally left for inspection; this
command never overwrites an existing representation or treats file creation
as a physical pose-accuracy test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MISSING = (
    "precision_key_0p3mm", "precision_key_0p5mm",
    "precision_key_1p0mm",
    "precision_socket_cylinder_gap_01mm",
    "precision_socket_cylinder_gap_03mm",
    "precision_socket_cylinder_gap_05mm",
    "precision_socket_cylinder_gap_10mm",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_completed(
    *, shared_root: Path, staging_root: Path, object_name: str,
    generator_script: Path,
) -> dict:
    """Move one completed staged object into its empty canonical marker slot."""
    if object_name not in MISSING:
        raise ValueError("object is not in the missing representation list")
    staged = staging_root / object_name
    source_mesh = (shared_root / "object_processing" / object_name /
                   "raw_mesh" / f"{object_name}.obj")
    representation = (staged / "object_repre/v1" / object_name /
                      "1/repre.pth")
    summary_path = staged / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (not representation.is_file() or not source_mesh.is_file() or
            summary.get("dataset_name") != object_name or
            Path(summary.get("mesh_path", "")).resolve() != source_mesh.resolve() or
            summary.get("num_templates") != 798 or
            summary.get("mesh_scale") != 1000.0):
        raise ValueError("staged FoundPose source mesh/templates are incomplete")
    canonical = shared_root / "AutoDex/foundpose_assets" / object_name
    marker = canonical / "GENERATION_REQUIRED.json"
    if canonical.exists() and (not canonical.is_dir() or
                               set(canonical.iterdir()) != {marker}):
        raise FileExistsError(f"canonical FoundPose asset already has content: {canonical}")
    source = {
        "schema": "precision_insertion_foundpose_generation_v1",
        "object": object_name,
        "source_mesh": str(source_mesh),
        "source_mesh_sha256": _sha(source_mesh),
        "representation_sha256": _sha(representation),
        "generator_script_sha256": _sha(generator_script),
        "templates": 798,
        "status": "generated_and_reloaded_not_metric_pose_validated",
        "physical_pose_qualified": False,
    }
    with (staged / "ONBOARDING_PROVENANCE.json").open(
            "x", encoding="utf-8") as stream:
        json.dump(source, stream, indent=2, sort_keys=True)
        stream.write("\n")
    archive = (shared_root / "AutoDex/archive" /
               "precision_foundpose_generation_markers_20261011" /
               object_name / "GENERATION_REQUIRED.json")
    if marker.exists():
        if archive.exists():
            raise FileExistsError(f"archived marker exists: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        marker.rename(archive)
        canonical.rmdir()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(canonical)
    return {**source, "canonical_root": str(canonical)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--mv-gotrack-root", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", choices=MISSING, required=True)
    parser.add_argument("--install-completed-only", action="store_true",
                        help="do not generate; install already completed staging")
    args = parser.parse_args()
    shared = args.shared_root.expanduser().resolve()
    staging = args.staging_root.expanduser().resolve()
    gotrack = args.mv_gotrack_root.expanduser().resolve()
    generator = args.generator_script.expanduser().resolve()
    if not generator.is_file() or not gotrack.is_dir():
        parser.error("MV-GoTrack source and isolated generator are required")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(gotrack), str(gotrack / "external/bop_toolkit"),
        str(gotrack / "external/dinov2")))
    env["PYOPENGL_PLATFORM"] = "egl"
    env["EGL_PLATFORM"] = "surfaceless"
    for name in args.objects:
        canonical_repre = (shared / "AutoDex/foundpose_assets" / name /
                           "object_repre/v1" / name / "1/repre.pth")
        if canonical_repre.exists():
            print(json.dumps({"object": name, "status": "already_installed"}),
                  flush=True)
            continue
        staged = staging / name
        staged_repre = (staged / "object_repre/v1" / name / "1/repre.pth")
        if not staged_repre.is_file():
            if args.install_completed_only:
                raise FileNotFoundError(f"staged representation is missing: {name}")
            if staged.exists():
                raise FileExistsError(
                    f"incomplete staging needs inspection before rerun: {staged}")
            mesh = shared / "object_processing" / name / "raw_mesh" / f"{name}.obj"
            command = [
                sys.executable, str(generator),
                "--mesh-path", str(mesh), "--object-id", "1",
                "--dataset-name", name, "--output-root", str(staged),
                "--reference-intrinsics-json", str(
                    shared / "cam_param/20260921_213318/intrinsics.json"),
                "--reference-camera-id", "25322651",
                "--reference-image-scale", "1.0",
                "--mesh-scale", "1000", "--depth-min-mm", "600",
                "--depth-max-mm", "1200", "--min-num-viewpoints", "57",
                "--num-inplane-rotations", "14", "--ssaa-factor", "1.0",
                "--pca-components", "256", "--cluster-num", "2048",
            ]
            print(json.dumps({"object": name, "status": "generating"}),
                  flush=True)
            staging.mkdir(parents=True, exist_ok=True)
            log_path = staging / f"{name}.generation.log"
            with log_path.open("x", encoding="utf-8") as log:
                subprocess.run(command, env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        result = install_completed(
            shared_root=shared, staging_root=staging,
            object_name=name, generator_script=generator)
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
