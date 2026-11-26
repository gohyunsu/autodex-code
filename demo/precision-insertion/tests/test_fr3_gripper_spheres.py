"""The ZeroDex stock-gripper model reuses cuRobo's stock hand geometry."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SPHERES = ROOT / "autodex/planner/src/curobo/content/configs/robot/spheres"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["collision_spheres"]


def test_fr3_gripper_hand_spheres_match_stock_franka_profile():
    stock = _load(SPHERES / "franka.yml")
    fr3 = _load(SPHERES / "fr3_gripper.yml")
    assert fr3["fr3_hand"] == stock["panda_hand"]
    assert fr3["fr3_leftfinger"] == stock["panda_leftfinger"]
    assert fr3["fr3_rightfinger"] == stock["panda_rightfinger"]
