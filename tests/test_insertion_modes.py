"""Pure/offline checks for insertion mode and retry evidence boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from autodex.tasks.insertion_vlm import parse_answer
from autodex.tasks.precision_insertion import (
    build_catalog, decide_retry, mode_config, require_robot_ready,
    select_scenario,
    validate_runtime_socket_pair,
)


class InsertionModesTest(unittest.TestCase):
    def test_modes_and_symmetry(self):
        square = mode_config("square")
        cylinder = mode_config("cylinder")
        self.assertTrue(square.yaw_relevant)
        self.assertFalse(cylinder.yaw_relevant)
        self.assertEqual(cylinder.socket, "precision_socket_cylinder_gap_20mm")
        with self.assertRaises(ValueError):
            mode_config("cylinder", gap_mm=1.5)

    def test_runtime_pair_rejects_wrong_socket_family(self):
        validate_runtime_socket_pair("precision_key_1p5mm", "precision_socket_unified")
        validate_runtime_socket_pair("precision_key_cylinder_r15_h80",
                                     "precision_socket_cylinder_gap_20mm")
        with self.assertRaises(ValueError):
            validate_runtime_socket_pair("precision_key_cylinder_r15_h80",
                                         "precision_socket_unified")
        with self.assertRaises(ValueError):
            validate_runtime_socket_pair("precision_key_1p5mm",
                                         "precision_socket_cylinder_gap_20mm")

    def test_catalog_does_not_promote_partial_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = (root / "AutoDex" / "sim_filter_pass" / "inspire" /
                         "precision_insertion_unconstrained_20mm_10k_top50" /
                         "precision_key_1p5mm" / "table" / "4" / "104")
            candidate.mkdir(parents=True)
            for name in ("grasp_pose.npy", "pregrasp_pose.npy", "wrist_se3.npy"):
                (candidate / name).write_bytes(b"placeholder")
            report = {"curobo_mujoco_pilot": {
                "mujoco_gravity_stability_pass_ids": ["104"],
                "successes_without_declared_contact_above_handle_front_ids": ["104"]}}
            catalog = build_catalog(root, report)
            chosen = select_scenario(catalog, "square")
            self.assertEqual(chosen["validation_level"], "grasp_sim_pass")
            self.assertFalse(chosen["full_task_sim_pass"])
            with self.assertRaises(RuntimeError):
                select_scenario(catalog, "square", tabletop_pose=0)
            with self.assertRaises(RuntimeError):
                select_scenario(catalog, "square", minimum_level="full_task_sim_pass")
            with self.assertRaises(RuntimeError):
                select_scenario(catalog, "cylinder")
            with self.assertRaises(RuntimeError):
                require_robot_ready(chosen)

    def test_retry_uses_metric_pose_not_vlm_numeric_guess(self):
        mode = mode_config("square")
        base = {"grasp_held": True, "vlm_class": "misaligned",
                "vlm_confidence": 0.9, "pose_uncertainty_mm": 0.2,
                "pose_error_xy_m": [0.001, -0.003], "measured_depth_mm": 0}
        decision = decide_retry(mode=mode, previous_offset_xy_m=(0, 0),
                                attempt=1, observation=base)
        self.assertEqual(decision.status, "propose_retry")
        self.assertEqual(decision.offset_xy_m, (-0.0005, 0.0005))
        base["vlm_class"] = "unknown"
        self.assertEqual(decide_retry(mode=mode, previous_offset_xy_m=(0, 0),
                                      attempt=1, observation=base).status, "inspect")
        base["vlm_class"] = "rim_jam"
        base["pose_error_xy_m"] = [0.0, 0.0]
        self.assertEqual(decide_retry(mode=mode, previous_offset_xy_m=(0, 0),
                                      attempt=1, observation=base).status, "inspect")
        base["abort_reason"] = "force_limit"
        self.assertEqual(decide_retry(mode=mode, previous_offset_xy_m=(0, 0),
                                      attempt=1, observation=base).status, "stop")

    def test_success_requires_multiple_independent_signals(self):
        mode = mode_config("cylinder")
        observed = {"measured_depth_mm": 20.1, "grasp_held": True,
                    "abort_reason": None, "force_within_limits": True,
                    "vlm_class": "partial_insertion", "vlm_confidence": 0.8}
        decide = lambda: decide_retry(mode=mode, previous_offset_xy_m=(0, 0),
                                      attempt=1, observation=observed)
        self.assertEqual(decide().status, "success")
        observed["vlm_class"] = "occluded"
        self.assertEqual(decide().status, "inspect")
        observed["vlm_class"] = "partial_insertion"
        observed["force_within_limits"] = False
        self.assertEqual(decide().status, "inspect")

    def test_vlm_schema_fails_closed(self):
        good = json.dumps({"checkpoint": "pre_insert", "class": "misaligned",
                           "confidence": 0.8, "visible_evidence": "offset",
                           "cameras_used": ["top"]})
        self.assertEqual(parse_answer(good, "pre_insert", {"top"})["class"], "misaligned")
        self.assertEqual(parse_answer(good, "pre_insert", {"side"})["class"], "unknown")
        self.assertEqual(parse_answer("not json", "pre_insert", {"top"})["class"], "unknown")


if __name__ == "__main__":
    unittest.main()
