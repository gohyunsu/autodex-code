"""Socket handoff TARs must restore into the FoundPose object root."""

from io import BytesIO
from pathlib import Path
import sys
import tarfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from install_zerodex_nas_project import _extract_socket_package  # noqa: E402


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
    target = tmp_path / name
    _extract_socket_package(package, target)
    assert (target / "models/obj_000001.ply").read_bytes() == b"mesh"
    assert (target / f"object_repre/v1/{name}/1/repre.pth").is_file()
    assert (target / f"templates/v1/{name}/1/rgb/template_0000.png").is_file()
    assert not (target / name).exists()
