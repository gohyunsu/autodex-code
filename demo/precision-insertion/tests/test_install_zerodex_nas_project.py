"""Socket handoff TARs must restore into the FoundPose object root."""

from io import BytesIO
from pathlib import Path
import json
import subprocess
import sys
import tarfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from install_zerodex_nas_project import (  # noqa: E402
    _extract_socket_package, _package_already_copied, install,
)


def test_socket_handoff_models_are_not_nested_after_restore(tmp_path):
    name = "precision_socket_cylinder_gap_05mm"
    package = tmp_path / "socket.tar"
    files = {
        f"./{name}/models/obj_000001.ply": b"mesh",
        f"./object_repre/v1/{name}/1/repre.pth": b"representation",
        f"./templates/v1/{name}/1/rgb/template_0000.png": b"image",
    }
    with tarfile.open(package, "w") as archive:
        for path, data in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    extract_to = f"AutoDex/foundpose_assets/{name}"
    copied_paths = {
        f"assets/shared_data/{extract_to}/models/obj_000001.ply",
        f"assets/shared_data/{extract_to}/object_repre/v1/{name}/1/repre.pth",
        f"assets/shared_data/{extract_to}/templates/v1/{name}/1/rgb/template_0000.png",
    }
    assert _package_already_copied(package, extract_to, copied_paths)
    assert not _package_already_copied(
        package, extract_to, copied_paths - {
            f"assets/shared_data/{extract_to}/models/obj_000001.ply"})
    target = tmp_path / name
    _extract_socket_package(package, target)
    assert (target / "models/obj_000001.ply").read_bytes() == b"mesh"
    assert (target / f"object_repre/v1/{name}/1/repre.pth").is_file()
    assert (target / f"templates/v1/{name}/1/rgb/template_0000.png").is_file()
    assert not (target / name).exists()


def test_restored_profile_uses_restored_shared_data(tmp_path):
    bundle = tmp_path / "bundle"
    destination = tmp_path / "installed"
    (bundle / "code").mkdir(parents=True)
    (bundle / "rigs/zerodex_capture4/main_pc_system/current").mkdir(parents=True)
    (bundle / "assets/shared_data").mkdir(parents=True)
    records = []
    for repo in ("autodex-code", "paradex"):
        archive_name = f"{repo}.tar.gz"
        with tarfile.open(bundle / "code" / archive_name, "w:gz") as archive:
            member = tarfile.TarInfo("README")
            member.size = 0
            archive.addfile(member, BytesIO())
        records.append({"repository": repo, "archive": archive_name,
                        "commit": "test-commit"})
    (bundle / "PROJECT_MANIFEST.json").write_text(json.dumps({
        "schema": "precision_insertion_reproducible_project_v1",
        "code": records,
        "assets": [],
        "asset_packages": [],
        "source_shared_root": "/old/shared_data",
    }))
    result = install(bundle, destination, capture4_nas_root="/capture4/project")
    profile = result["source_profile"]
    subprocess.run([
        "bash", "-c", 'source "$1"; test "$PRECISION_SHARED_ROOT" = "$2"',
        "bash", profile, str(destination / "shared_data"),
    ], check=True)
