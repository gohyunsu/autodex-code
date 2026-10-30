#!/usr/bin/env python3
"""Restore the NAS precision insertion bundle on a comparable ZeroDex PC.

The installer copies runtime data to local storage, then rewrites JSON paths
from the source machine to the chosen install root. It does not start cameras
or robot services and does not modify capture4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile


def _package_already_copied(
    package_path: Path, extract_to: str, asset_paths: set[str],
) -> bool:
    """The bundle may carry a directly copied socket and its restore TAR."""
    object_name = Path(extract_to).name
    prefix = f"assets/shared_data/{extract_to}/"
    with tarfile.open(package_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = member.name.removeprefix("./")
            if relative.startswith(object_name + "/"):
                relative = relative[len(object_name) + 1:]
            if prefix + relative not in asset_paths:
                return False
    return True


def _extract_socket_package(package_path: Path, target: Path) -> None:
    """Restore one handoff package into the FoundPose object layout."""
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package_path, "r:") as archive:
        # The source handoff wraps only models/ in an extra object-name
        # directory. Flatten it so models/, templates/, and object_repre/
        # share the same FoundPose object root.
        wrapped_models = f"./{target.name}/"
        members = []
        for member in archive:
            if member.name == f"./{target.name}":
                continue
            if member.name.startswith(wrapped_models):
                member.name = "./" + member.name[len(wrapped_models):]
            members.append(member)
        archive.extractall(target, members=members)


def install(bundle: Path, destination: Path,
            *, capture4_nas_root: str) -> dict:
    bundle = bundle.expanduser().resolve()
    destination = destination.expanduser().resolve()
    manifest = json.loads((bundle / "PROJECT_MANIFEST.json").read_text(
        encoding="utf-8"))
    if manifest.get("schema") != "precision_insertion_reproducible_project_v1":
        raise ValueError("not a precision insertion project bundle")
    destination.mkdir(parents=True, exist_ok=True)
    for record in manifest["code"]:
        target = destination / record["repository"]
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle / "code" / record["archive"], "r:gz") as archive:
            # The upstream AutoDex tree tracks unrelated absolute xArm mesh
            # links into another user's home. FR3/Inspire does not use them.
            archive.extractall(target, members=(
                member for member in archive
                if not ((member.issym() or member.islnk()) and
                        Path(member.linkname).is_absolute())))
    # ParaDex imports system/current at module import time. The original
    # workstation points that path outside the hyunsu NAS tree; restore the
    # captured configuration into this project's own checkout instead.
    paradex_system = destination / "paradex/system/current"
    paradex_system.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
                    str(bundle / "rigs/zerodex_capture4/main_pc_system/current") + "/",
                    str(paradex_system) + "/"], check=True)
    shared = destination / "shared_data"
    shared.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
                    str(bundle / "assets/shared_data") + "/",
                    str(shared) + "/"], check=True)
    asset_paths = {row["path"] for row in manifest["assets"]}
    for package in manifest.get("asset_packages", []):
        target = shared / package["extract_to"]
        package_path = bundle / package["path"]
        if not _package_already_copied(
                package_path, package["extract_to"], asset_paths):
            _extract_socket_package(package_path, target)
    original = manifest["source_shared_root"]
    for path in shared.rglob("*.json"):
        before = path.read_text(encoding="utf-8")
        after = before.replace(original, str(shared)).replace(
            "/home/robot/shared_data", str(shared))
        if after != before:
            path.write_text(after, encoding="utf-8")
    repo = destination / "autodex-code"
    stage = shared / "AutoDex/precision_insertion/zerodex_capture4"
    code_commit = next(row["commit"] for row in manifest["code"]
                       if row["repository"] == "autodex-code")
    profile = destination / "activate_zerodex_capture4.sh"
    profile.write_text(
        "#!/usr/bin/env bash\n"
        f"export PRECISION_REPO_ROOT={repo}\n"
        f"export PRECISION_CODE_COMMIT={code_commit}\n"
        f"export PRECISION_SHARED_ROOT={shared}\n"
        f"export PRECISION_STAGE_ROOT={stage}\n"
        f"export PRECISION_PARADEX_ROOT={destination / 'paradex'}\n"
        f"export PRECISION_SAM_CHECKPOINT={shared / 'AutoDex/weights/sam3/sam3.pt'}\n"
        f"export PRECISION_FR3_INSPIRE_URDF={repo / 'autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf'}\n"
        "export PRECISION_CAPTURE4_ASSET_ROOT=/home/capture4/precision_insertion_capture4_20261027/cache/shared_data\n"
        f"export PRECISION_CAPTURE4_PROJECT_ROOT={capture4_nas_root.rstrip('/')}\n"
        f"export PRECISION_NAS_PROJECT_ROOT={bundle}\n"
        "export PRECISION_GAP_MM=5\n",
        encoding="utf-8")
    profile.chmod(0o755)
    return {"installed_root": str(destination), "shared_root": str(shared),
            "source_profile": str(profile),
            "static_check": (f"cd {repo} && source {profile} && "
                             "bash demo/precision-insertion/run_zerodex_capture4.sh --check")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("/mnt/paradex2/hyunsu/precision_insertion"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--capture4-nas-root", default=(
        "/home/capture4/shared_data/hyunsu/precision_insertion"))
    args = parser.parse_args()
    print(json.dumps(install(args.bundle, args.destination,
                             capture4_nas_root=args.capture4_nas_root), indent=2))


if __name__ == "__main__":
    main()
