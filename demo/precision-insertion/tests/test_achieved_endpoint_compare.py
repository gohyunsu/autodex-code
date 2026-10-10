"""A nominal or single-phase endpoint pass cannot become an achieved pass."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import screen_cylinder_achieved_endpoints as achieved  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def test_both_achieved_states_required_and_output_is_immutable(tmp_path, monkeypatch):
    source = tmp_path / "audit.json"
    source.write_text("{}", encoding="utf-8")
    q = np.ones(6) * 0.25
    fidelity = {
        "closure": {
            "end_squeeze": {"trajectory_index": 1},
            "end_gravity": {"trajectory_index": 2},
        },
        "achieved_mujoco_visual_penetration": {
            state: {"achieved_hand_q": q.tolist()} for state in achieved.STATES
        },
    }
    paths = AssetPaths(tmp_path, select_mode("cylinder", 20.0))
    monkeypatch.setattr(achieved, "_load_audited_rows", lambda _path: (
        {"source_summary": "source.json", "source_summary_sha256": "sha",
         "key_mesh": str(paths.raw_mesh(achieved.KEY)),
         "robot_urdf": str(paths.robot_urdf)},
        {"minimum_hand_clearance_m": 1e-6,
         "socket_gaps": {"20mm": {"eligible_offline_grasp_ids": ["table/0/1"]}}},
        [{"id": "table/0/1", "candidate": tmp_path / "candidate",
          "trajectory": {}, "fidelity": fidelity}],
    ))
    monkeypatch.setattr(achieved, "achieved_hand_state", lambda _traj, state: (
        np.eye(4), q, 1 if state == "end_squeeze" else 2))
    seen = []

    def fake_screen(**kwargs):
        seen.append(kwargs["override_source"])
        return {"endpoint_pass": kwargs["override_source"] == "simulated_end_squeeze"}

    monkeypatch.setattr(achieved, "screen_grasp_endpoint", fake_screen)
    output = tmp_path / "output"
    result = achieved.run(
        fidelity_audit=source, shared_root=tmp_path, output_root=output,
        minimum_hand_clearance_m=1e-6, gaps_mm=(20.0,))
    assert seen == ["simulated_end_squeeze", "simulated_end_gravity"]
    assert result["socket_gaps"]["20mm"]["nominal_initial_pose_endpoint_pass_count"] == 1
    assert result["socket_gaps"]["20mm"]["both_achieved_states_clear_ids"] == []
    assert result["robot_ready"] is False
    assert (output / "gap_20mm/table/0/1.json").is_file()
    assert not (tmp_path / "output.incomplete").exists()
    with pytest.raises(FileExistsError):
        achieved.run(fidelity_audit=source, shared_root=tmp_path,
                     output_root=output, minimum_hand_clearance_m=1e-6,
                     gaps_mm=(20.0,))
