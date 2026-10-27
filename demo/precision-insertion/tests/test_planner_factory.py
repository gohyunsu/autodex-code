"""Offline planning must use the same mount as the selected live profile."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.planner_factory import (  # noqa: E402
    new_offline_fr3_planner,
)


def test_offline_planner_uses_selected_mount_for_both_profiles(
        tmp_path, monkeypatch):
    import autodex.planner as planner_module

    received = []

    def fake_planner(**options):
        received.append(options)
        return options

    # The module's lazy __getattr__ would import CUDA on an attribute probe.
    monkeypatch.setitem(planner_module.__dict__, "GraspPlanner", fake_planner)
    mode = select_mode("cylinder", 20)
    monkeypatch.delenv("PRECISION_FR3_INSPIRE_URDF", raising=False)
    default = AssetPaths(tmp_path, mode).robot_urdf
    assert new_offline_fr3_planner(
        shared_root=tmp_path, mode=mode) == {
            "hand": "fr3_inspire", "robot_urdf_path": str(default)}
    assert default.name == "fr3_inspire_oct12_mount.urdf"

    straight = tmp_path / "fr3_inspire.urdf"
    monkeypatch.setenv("PRECISION_FR3_INSPIRE_URDF", str(straight))
    assert new_offline_fr3_planner(
        shared_root=tmp_path, mode=mode,
        use_cuda_graph=False) == {
            "hand": "fr3_inspire", "robot_urdf_path": str(straight),
            "use_cuda_graph": False}
    assert len(received) == 2
