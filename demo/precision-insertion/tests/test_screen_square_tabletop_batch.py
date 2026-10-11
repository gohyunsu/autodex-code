"""A supplemental v8 cell scan must remain distinct from the default pool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
spec = importlib.util.spec_from_file_location(
    "screen_square_tabletop_batch", DEMO / "screen_square_tabletop_batch.py")
screen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(screen)


def test_single_scene_5000_supplement_does_not_claim_full_scan(
        tmp_path, monkeypatch):
    seen = []

    def audit(**kwargs):
        seen.append((kwargs["scene_id"], kwargs["expected_seed_count"]))
        return {"generated": 5000, "mujoco_stable": 1}, [
            ("7", tmp_path / "stock" / "table" / "0" / "7")]

    monkeypatch.setattr(screen, "audit_tabletop_cell", audit)
    monkeypatch.setattr(screen, "screen_grasp_endpoint", lambda **_: {
        "endpoint_pass": True})
    output = tmp_path / "screen"
    result = screen.run(
        shared_root=tmp_path, raw_root=tmp_path / "raw",
        pass_root=tmp_path / "stock", output_root=output,
        expected_per_scene=5000, minimum_hand_clearance_m=.0002,
        gap_mm=(.5,), scene_ids=("0",))
    assert seen == [("0", 5000)]
    assert result["schema"] == (
        "precision_insertion_square_tabletop_batch_screen_v2")
    assert result["scene_ids"] == ["0"]
    assert result["expected_tabletop_scenes_per_key"] == 1
    mode = result["modes"]["precision_key_0p5mm"]
    assert mode["eligible_offline_grasp_ids"] == ["table/0/7"]
    assert set(mode["scene_counts"]) == {"0"}
    with pytest.raises(FileExistsError):
        screen.run(
            shared_root=tmp_path, raw_root=tmp_path / "raw",
            pass_root=tmp_path / "stock", output_root=output,
            expected_per_scene=5000, minimum_hand_clearance_m=.0002,
            gap_mm=(.5,), scene_ids=("0",))


def test_reject_duplicate_or_invalid_supplement_scene(tmp_path):
    for scenes in (("0", "0"), ("5",), ()):
        with pytest.raises(ValueError):
            screen.run(
                shared_root=tmp_path, raw_root=tmp_path / "raw",
                pass_root=tmp_path / "stock",
                output_root=tmp_path / "not_created",
                expected_per_scene=5000,
                minimum_hand_clearance_m=.0002,
                gap_mm=(.5,), scene_ids=scenes)
