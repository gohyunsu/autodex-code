import importlib.util
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
        triangles = mesh.vertices[mesh.faces[allowed]]
        normals = mesh.face_normals[allowed]
        self.assertLessEqual(float(triangles[:, :, 2].max()), builder.HANDLE_FRONT_Z_M + 1e-8)
        for triangle, normal in zip(triangles, normals):
            rear = normal[2] < -0.95 and triangle[:, 2].max() <= 1e-8
            side = abs(normal[2]) < 0.05
            self.assertTrue(rear or side)

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


if __name__ == "__main__":
    unittest.main()
