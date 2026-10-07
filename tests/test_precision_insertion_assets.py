import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "precision_insertion" / "build_assets.py"
SPEC = importlib.util.spec_from_file_location("precision_asset_builder", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)

FILTER_PATH = REPO_ROOT / "scripts" / "precision_insertion" / "filter_contact_safe_grasps.py"
FILTER_SPEC = importlib.util.spec_from_file_location("precision_contact_filter", FILTER_PATH)
contact_filter = importlib.util.module_from_spec(FILTER_SPEC)
assert FILTER_SPEC.loader is not None
FILTER_SPEC.loader.exec_module(contact_filter)


class PrecisionInsertionAssetTest(unittest.TestCase):
    source_dir = REPO_ROOT / "assets" / "precision_insertion" / "source"

    def test_source_meshes_are_metric_watertight_after_loading(self):
        for _name, _gap, source in builder.KEY_SPECS:
            mesh = builder.read_binary_stl(self.source_dir / source)
            builder.validate_watertight(mesh)
            self.assertGreater(mesh.volume, 0.0)
            self.assertAlmostEqual(mesh.bounds[1, 2], 0.0855, places=7)

    def test_contact_partition_excludes_shaft_and_front_shoulder(self):
        mesh = builder.read_binary_stl(self.source_dir / "plug_gap_1p5.stl")
        allowed, forbidden = builder.contact_face_partition(mesh)
        self.assertTrue(allowed)
        self.assertTrue(forbidden)
        self.assertEqual(len(allowed), 15)
        triangles = mesh.vertices[mesh.faces[allowed]]
        normals = mesh.face_normals[allowed]
        self.assertLessEqual(float(triangles[:, :, 2].max()), builder.HANDLE_FRONT_Z_M + 1e-8)
        for triangle, normal in zip(triangles, normals):
            rear = normal[2] < -0.95 and triangle[:, 2].max() <= 1e-8
            side = abs(normal[2]) < 0.05
            self.assertTrue(rear or side)

    def test_contact_partition_includes_diagonal_handle_side(self):
        mesh = builder.read_binary_stl(self.source_dir / "plug_gap_1p5.stl")
        allowed, _forbidden = builder.contact_face_partition(mesh)
        normals = mesh.face_normals[allowed]
        diagonal = (
            (np.abs(normals[:, 2]) < 0.05)
            & np.isclose(np.abs(normals[:, 0]), np.sqrt(0.5), atol=1e-4)
            & np.isclose(np.abs(normals[:, 1]), np.sqrt(0.5), atol=1e-4)
        )
        self.assertEqual(int(np.count_nonzero(diagonal)), 2)

    def test_socket_alignment_flips_key_and_seats_shoulder(self):
        socket = builder.read_binary_stl(self.source_dir / builder.SOCKET_SOURCE)
        entry = socket.bounds[1, 2]
        transform = np.eye(4)
        transform[:3, :3] = builder._rotation_x(math.pi)
        transform[2, 3] = entry + builder.HANDLE_FRONT_Z_M
        shoulder = transform @ np.asarray([0.0, 0.0, builder.HANDLE_FRONT_Z_M, 1.0])
        tip = transform @ np.asarray([0.0, 0.0, 0.0855, 1.0])
        self.assertAlmostEqual(shoulder[2], entry, places=8)
        self.assertAlmostEqual(tip[2], 0.0180, places=8)

    def test_contact_filter_accepts_only_handle_sides_and_rear_interior(self):
        region = contact_filter._point_region
        kwargs = {
            "half_x": 0.0195,
            "half_y": 0.0165,
            "handle_top": 0.045,
            "margin": 0.002,
            "tolerance": 0.001,
        }
        self.assertEqual(region(np.array([0.0195, 0.0, 0.020]), **kwargs), "handle_x_side")
        self.assertEqual(region(np.array([0.0, -0.0165, 0.020]), **kwargs), "handle_y_side")
        self.assertEqual(region(np.array([0.0, 0.0, 0.0]), **kwargs), "handle_rear")
        self.assertIsNone(region(np.array([0.0, 0.0, 0.045]), **kwargs))
        self.assertIsNone(region(np.array([0.010, 0.0, 0.070]), **kwargs))
        self.assertIsNone(region(np.array([0.0195, 0.0155, 0.020]), **kwargs))

    def test_contact_filter_accepts_all_five_pentagonal_sides(self):
        region = contact_filter._point_region
        mesh = builder.read_binary_stl(self.source_dir / "plug_gap_1p5.stl")
        polygon = np.asarray(builder.handle_cross_section_xy(mesh))
        kwargs = {
            "half_x": 0.0195,
            "half_y": 0.0165,
            "handle_top": 0.045,
            "margin": 0.002,
            "tolerance": 0.001,
            "cross_section_xy": polygon,
        }
        for index, (start, end) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
            point = np.r_[0.5 * (start + end), 0.020]
            self.assertEqual(region(point, **kwargs), f"handle_lateral_{index}")
        # This point is inside the old bounding rectangle but outside the
        # chamfered pentagon and therefore cannot be a rear-face contact.
        self.assertIsNone(region(np.array([0.018, -0.015, 0.0]), **kwargs))

    def test_contact_filter_converts_bodex_world_contacts_to_object_frame(self):
        policy = {
            "handle_z_range": [0.0, 0.045],
            "handle_half_extents_xy_m": [0.0195, 0.0165],
            "allowed": {
                "edge_margin_m": 0.002,
                "plane_tolerance_m": 0.001,
            },
        }
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.4, -0.2, 0.1]
        canonical = np.asarray([
            [0.0195, 0.0, 0.020],
            [-0.0195, 0.0, 0.022],
        ])
        world = canonical + object_pose[:3, 3]
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            np.save(candidate / "wrist_se3.npy", np.eye(4))
            np.save(candidate / "pregrasp_pose.npy", np.zeros(12))
            np.save(candidate / "grasp_pose.npy", np.zeros(12))
            np.save(candidate / "bodex_info.npy", {
                "contact_point": world,
                "grasp_error": np.zeros(2),
                "dist_error": np.zeros(2),
                "success": True,
            })
            corrected = contact_filter.inspect_candidate(
                candidate,
                policy,
                max_grasp_error=0.2,
                max_contact_distance=0.01,
                object_pose_world=object_pose,
            )
            legacy = contact_filter.inspect_candidate(
                candidate,
                policy,
                max_grasp_error=0.2,
                max_contact_distance=0.01,
            )
        self.assertTrue(corrected["accepted"])
        self.assertEqual(
            corrected["contact_frame_conversion"],
            "T_object_world @ p_world",
        )
        self.assertTrue(np.allclose(corrected["object_contacts_m"], canonical))
        self.assertFalse(legacy["accepted"])

    def test_contact_filter_report_only_keeps_forbidden_contact_as_diagnostic(self):
        policy = {
            "handle_z_range": [0.0, 0.045],
            "handle_half_extents_xy_m": [0.0195, 0.0165],
            "allowed": {
                "edge_margin_m": 0.002,
                "plane_tolerance_m": 0.001,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            np.save(candidate / "wrist_se3.npy", np.eye(4))
            np.save(candidate / "pregrasp_pose.npy", np.zeros(12))
            np.save(candidate / "grasp_pose.npy", np.zeros(12))
            np.save(candidate / "bodex_info.npy", {
                # Shaft/front contact: deliberately outside the handle policy.
                "contact_point": np.asarray([[0.0, 0.0, 0.070]]),
                "grasp_error": np.zeros(1),
                "dist_error": np.zeros(1),
                "success": False,
            })
            enforced = contact_filter.inspect_candidate(
                candidate,
                policy,
                max_grasp_error=0.2,
                max_contact_distance=0.01,
            )
            diagnostic = contact_filter.inspect_candidate(
                candidate,
                policy,
                max_grasp_error=0.2,
                max_contact_distance=0.01,
                contact_policy_mode="report-only",
            )
        self.assertFalse(enforced["accepted"])
        self.assertTrue(diagnostic["accepted"])
        self.assertFalse(diagnostic["declared_contacts_obey_policy"])
        self.assertEqual(diagnostic["contact_policy_mode"], "report-only")

    def test_full_build_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = builder.build(self.source_dir, root)
            self.assertEqual(len(manifest["objects"]), 4)
            for object_name, _gap, _source in builder.KEY_SPECS:
                self.assertTrue(
                    (root / "object_processing" / object_name / "raw_mesh" / f"{object_name}.obj").is_file()
                )
                self.assertTrue(
                    (root / "AutoDex" / "scene" / "inspire" / object_name / "table" / "0.json").is_file()
                )
            proxy = root / "object_processing" / builder.HANDLE_PROXY_NAME
            self.assertTrue((proxy / "processed_data" / "mesh" / "simplified.obj").is_file())
            for pose_id in range(5):
                self.assertTrue(
                    (proxy / "processed_data" / "info" / "tabletop" /
                     f"{pose_id:03d}.npy").is_file()
                )
                self.assertTrue(
                    (root / "AutoDex" / "scene" / "inspire" /
                     builder.HANDLE_PROXY_NAME / "table" /
                     f"{pose_id}.json").is_file()
                )
            self.assertEqual(
                manifest["grasp_generation_proxy"]["runtime_object"],
                "precision_key_1p5mm",
            )
            self.assertEqual(len(manifest["grasp_generation_proxies"]), 4)
            for object_name, _gap, _source in builder.KEY_SPECS:
                proxy_name = builder.handle_proxy_name(object_name)
                proxy = root / "object_processing" / proxy_name
                self.assertTrue(
                    (proxy / "processed_data" / "mesh" / "simplified.obj").is_file()
                )
                info = json.loads(
                    (proxy / "processed_data" / "info" / "simplified.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(info["runtime_object"], object_name)
                for pose_id in range(5):
                    self.assertTrue(
                        (proxy / "processed_data" / "info" / "tabletop" /
                         f"{pose_id:03d}.npy").is_file()
                    )
                stage = json.loads(
                    (root / "AutoDex" / "precision_insertion" / "stages" /
                     f"{object_name}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(stage["object"], object_name)
                self.assertEqual(stage["controller"]["implementation_status"], "required")
                self.assertIsNone(stage["controller"]["force_torque_limits"])
                self.assertEqual(
                    Path(stage["assets"]["socket_pose_object"]).name,
                    f"{builder.SOCKET_OBJECT_NAME}.obj",
                )
            socket_object = (
                root / "object_processing" / builder.SOCKET_OBJECT_NAME
            )
            self.assertTrue(
                (socket_object / "raw_mesh" /
                 f"{builder.SOCKET_OBJECT_NAME}.obj").is_file()
            )
            self.assertTrue(
                (socket_object / "processed_data" / "mesh" /
                 "static_collision.obj").is_file()
            )
            frame = json.loads(
                (socket_object / "processed_data" / "info" /
                 "frame_contract.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(frame["T_socket_raw_mesh"]), np.eye(4)
                )
            )
            socket_urdf = (
                socket_object / "processed_data" / "urdf" /
                "socket_static_exact.urdf"
            ).read_text(encoding="utf-8")
            self.assertIn("../mesh/simplified.obj", socket_urdf)
            self.assertNotIn("convex", socket_urdf.lower())
            task_geometry = json.loads(
                (root / "AutoDex" / "precision_insertion" / "fixtures" /
                 builder.FIXTURE_NAME / "task_geometry.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                task_geometry["socket_pose_object"],
                builder.SOCKET_OBJECT_NAME,
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(task_geometry["T_socket_pose_object"]),
                    np.eye(4),
                )
            )
            entry = np.asarray(task_geometry["T_socket_key_entry"])
            verification = np.asarray(
                task_geometry["T_socket_key_verification"]
            )
            seated = np.asarray(task_geometry["T_socket_key_seated"])
            self.assertAlmostEqual(
                entry[2, 3] - verification[2, 3], 0.020, places=8
            )
            self.assertAlmostEqual(
                verification[2, 3] - seated[2, 3], 0.0205, places=8
            )
            self.assertEqual(
                task_geometry["task_success_target"],
                "T_socket_key_verification",
            )
            marker = (
                root / "AutoDex/foundpose_assets/precision_key_1p5mm/"
                "GENERATION_REQUIRED.json"
            )
            self.assertIn(
                "MV-GoTrack",
                marker.read_text(encoding="utf-8"),
            )
            socket_marker = (
                root / "AutoDex" / "foundpose_assets" /
                builder.SOCKET_OBJECT_NAME / "GENERATION_REQUIRED.json"
            )
            self.assertIn(
                "keyed bore",
                socket_marker.read_text(encoding="utf-8"),
            )

    def test_autodex_profile_preserves_existing_hardware_camera_contract(self):
        profile = json.loads(
            (REPO_ROOT / "assets/precision_insertion/autodex_camera_profile.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            profile["pc_list"],
            ["capture1", "capture2", "capture3", "capture5", "capture6"],
        )
        self.assertEqual(profile["capture_sync"], "hardware")
        self.assertEqual(profile["arm"], "franka")
        self.assertEqual(profile["hand"], "inspire")
        self.assertNotIn("camera_serials", profile)

    def test_task_symmetry_does_not_fold_the_keyed_insertion_pose(self):
        policy = json.loads(
            (REPO_ROOT / "assets/precision_insertion/task_symmetry.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(policy["exact_pose_symmetry"]["group"], "identity")
        self.assertEqual(policy["socket_pose_symmetry"]["group"], "identity")
        proposal = policy["grasp_proposal_symmetry"]
        self.assertEqual(proposal["group"], "identity")
        self.assertEqual(
            [item["representative"] for item in proposal["classes"]],
            ["000", "001", "002", "003", "004"],
        )
        self.assertTrue(
            policy["runtime_contract"]["retain_observed_exact_pose"]
        )
        self.assertFalse(
            policy["runtime_contract"]
            ["representative_class_may_replace_pose_for_planning"]
        )

    def test_presentation_candidate_set_uses_six_surface_20mm_gate(self):
        candidate_set = json.loads(
            (REPO_ROOT / "assets/precision_insertion/"
             "presentation_candidate_set.json").read_text(encoding="utf-8")
        )
        selected = candidate_set["selected_candidate_ids"]
        reserve = candidate_set["reserve_candidate_ids"]
        self.assertEqual(candidate_set["tabletop_pose_id"], "004")
        self.assertEqual(selected, ["40", "383", "6910", "31763", "46925"])
        self.assertEqual(reserve, [])
        evidence = candidate_set["screening_evidence"]
        self.assertEqual(evidence["raw_bodex_proposals"], 51000)
        self.assertEqual(evidence["contact_and_quality_candidates"], 51)
        self.assertEqual(
            evidence["sampled_whole_hand_contact_policy_passed"], 10
        )
        self.assertEqual(evidence["sampled_task_prefilter_passed"], 5)
        self.assertEqual(evidence["sampled_full_trajectory_passed"], 5)
        self.assertEqual(
            evidence["contact_point_frame"],
            "scene_world_transformed_to_object",
        )
        policy_passes = candidate_set["contact_policy_pass_candidate_ids"]
        self.assertEqual(len(policy_passes), 10)
        self.assertEqual(len(set(policy_passes)), 10)
        failures = json.loads(
            (REPO_ROOT / "assets/precision_insertion/"
             "presentation_trajectory_failures.json").read_text(
                 encoding="utf-8"
             )
        )
        rejected = {
            item["candidate"]
            for item in failures["rejected_after_contact_policy"]
        }
        passed = {
            item["candidate"]
            for item in failures["passed_at_verification_depth"]
        }
        self.assertEqual(set(policy_passes), rejected | passed)


if __name__ == "__main__":
    unittest.main()
