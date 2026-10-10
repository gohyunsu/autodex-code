"""Regression checks for the cylinder geometry and AutoDex path contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "precision_insertion"))
import build_cylindrical_assets as builder  # noqa: E402
import validate_cylindrical_assets as validator  # noqa: E402


class CylindricalAssetsTest(unittest.TestCase):
    def test_generated_layout_and_contact_surfaces(self):
        source = REPO / "assets" / "precision_insertion" / "cylindrical_fixed_key" / "source"
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            manifest = builder.build(source, shared)
            self.assertEqual(validator.validate(shared), 0)
            self.assertEqual(manifest["key"]["object"], builder.KEY_OBJECT)
            self.assertEqual(len(manifest["sockets"]), 6)
            key_dir = shared / "object_processing" / builder.KEY_OBJECT
            allowed = trimesh.load(key_dir / "processed_data" / "mesh" /
                                   "contact_allowed.obj", force="mesh", process=False)
            side = np.abs(allowed.face_normals[:, 2]) < 0.05
            self.assertGreater(int(side.sum()), 0)
            self.assertAlmostEqual(allowed.bounds[1, 2], 0.025, places=8)
            self.assertEqual(len(allowed.faces), 768)
            proxy = shared / "object_processing" / builder.KEY_PROXY_OBJECT
            proxy_info = json.loads((proxy / "processed_data" / "info" /
                                     "simplified.json").read_text())
            proxy_symmetry = json.loads((proxy / "processed_data" / "info" /
                                         "symmetry.json").read_text())
            self.assertAlmostEqual(proxy_info["obb"][2], 0.08, places=8)
            self.assertEqual(proxy_symmetry["type"], "none")
            for socket in manifest["sockets"]:
                fixture = (shared / "AutoDex" / "precision_insertion" /
                           "fixtures" / socket["object"])
                self.assertEqual(Path(socket["task_geometry"]),
                                 fixture / "task_geometry.json")
                self.assertTrue((fixture / "pose_measurement_asset.json").is_file())
                self.assertFalse((shared / "AutoDex" / "precision_insertion" /
                                  "cylindrical" / "fixtures" / socket["object"]).exists())


if __name__ == "__main__":
    unittest.main()
