from pathlib import Path
import os
import subprocess


def test_machine_profile_defaults_to_saved_board_free_table_plane(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    profile = demo / "config/zerodex_capture4.env.sh"
    command = (
        f"source {profile}; "
        "printf '%s\\n' \"$PRECISION_TABLE_PLANE_CONFIG\"")
    completed = subprocess.run(
        ["bash", "-c", command], env={**os.environ, "HOME": str(tmp_path)},
        check=True, capture_output=True, text=True)

    assert completed.stdout.strip() == str(
        tmp_path / "precision_insertion_project/calibration/table_plane/"
        "current/table_plane.json")

