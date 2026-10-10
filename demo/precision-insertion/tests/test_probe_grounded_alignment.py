"""Saved-image local metric probe tests; these are not camera-accuracy tests."""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_grounded_alignment as probe  # noqa: E402


def _camera(center):
    center = np.asarray(center, dtype=float)
    forward = (np.array([0., 0., .10]) - center)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.vstack((right, down, forward))
    transform[:3, 3] = -transform[:3, :3] @ center
    intrinsics = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    return transform, intrinsics


def _pixel(transform, intrinsics, point):
    camera = transform[:3, :3] @ point + transform[:3, 3]
    projected = intrinsics @ camera
    return (projected[:2] / projected[2]).tolist()


def _manifest(tmp_path):
    tip = np.array([.0016, -.0002, .09])
    cameras = {
        "front": (.22, 0., .22),
        "side": (0., .22, .22),
        "back": (-.22, 0., .22),
    }
    rows, truth = [], {}
    for index, (name, center) in enumerate(cameras.items()):
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (640, 480), "gray").save(image)
        transform, intrinsics = _camera(center)
        truth[name] = {
            "tip_px": _pixel(transform, intrinsics, tip),
            "axis_line_px": [
                _pixel(transform, intrinsics, tip + [0, 0, .02]),
                _pixel(transform, intrinsics, tip + [0, 0, .05]),
            ],
            "evidence": "synthetic test landmark",
        }
        rows.append({
            "camera_id": name,
            "image": {
                "path": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "timestamp_s": 1791600000. + .002 * index,
            },
            "intrinsics": intrinsics.tolist(),
            "T_camera_socket": transform.tolist(),
        })
    manifest = {
        "schema": probe.SCHEMA,
        "max_camera_skew_s": .01,
        "socket_rim_z_m": .055,
        "verification_depth_m": .02,
        "alignment_limits": {
            "pixel_sigma_px": .20,
            "max_reprojection_px": 1.,
            "min_parallax_deg": 5.,
            "max_axis_tilt_deg": 4.,
            "max_20mm_axis_sweep_m": .0015,
            "max_lateral_uncertainty_95_m": .001,
            "systematic_lateral_sigma_m": .00005,
            "cad_spacing_sigma_m": .0003,
            "minimum_views": 2,
        },
        "views": rows,
    }
    path = tmp_path / "grounding_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, truth


def test_local_probe_reuses_metric_estimator_and_records_exact_sources(
        tmp_path, monkeypatch):
    manifest_path, _manifest_record, truth = _manifest(tmp_path)
    loads = []

    class Backend:
        native_pixel_coordinates = True

        def infer(self, _images, prompt):
            name = re.search(r"Camera (\w+)\.", prompt).group(1)
            return json.dumps(truth[name])

    def load(**kwargs):
        loads.append(kwargs)
        return Backend()

    monkeypatch.setattr(probe, "load_vlm_backend", load)
    output = tmp_path / "result.json"
    assert probe.main(["--manifest", str(manifest_path),
                       "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert loads[0]["mode"] == "local"
    assert loads[0]["require_native_pixels"] is True
    assert loads[0]["max_input_size"] == (640, 480)
    assert report["alignment"]["status"] == "diagnostic_metric_xy_correction"
    assert report["alignment"]["xy_correction_socket_m"] == pytest.approx(
        [-.0016, .0002], abs=1e-7)
    assert np.linalg.norm(
        report["alignment"]["bounded_xy_increment_socket_m"]) == pytest.approx(.001)
    assert len(report["observations"]) == 3
    assert report["source_images"][0]["sha256"] == hashlib.sha256(
        (tmp_path / "front.png").read_bytes()).hexdigest()
    assert report["robot_ready"] is False
    with pytest.raises(FileExistsError):
        probe.main(["--manifest", str(manifest_path),
                    "--output", str(output)])


def test_unobservable_local_landmarks_abstain_without_xy_command(
        tmp_path, monkeypatch):
    manifest_path, _manifest_record, _truth = _manifest(tmp_path)

    class Backend:
        native_pixel_coordinates = True

        def infer(self, _images, _prompt):
            return json.dumps({"tip_px": None, "axis_line_px": None,
                               "evidence": "key hidden by hand"})

    monkeypatch.setattr(probe, "load_vlm_backend", lambda **_kwargs: Backend())
    output = tmp_path / "abstain.json"
    assert probe.main(["--manifest", str(manifest_path),
                       "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["alignment"]["status"] == "abstain"
    assert report["alignment"]["bounded_xy_increment_socket_m"] is None
    assert report["robot_ready"] is False


@pytest.mark.parametrize("mutation,problem", [
    ("bad_hash", "saved image changed"),
    ("bad_skew", "exposures exceed skew"),
    ("duplicate_camera", "unique path-safe camera IDs"),
    ("bad_geometry", "rotation is not orthonormal"),
])
def test_bad_sources_fail_before_model_load(tmp_path, monkeypatch,
                                            mutation, problem):
    path, manifest, _truth = _manifest(tmp_path)
    if mutation == "bad_hash":
        manifest["views"][0]["image"]["sha256"] = "0" * 64
    elif mutation == "bad_skew":
        manifest["views"][2]["image"]["timestamp_s"] += .2
    elif mutation == "duplicate_camera":
        manifest["views"][2]["camera_id"] = "front"
    else:
        manifest["views"][0]["T_camera_socket"][0][0] = 2.
    path.write_text(json.dumps(manifest))

    def unexpected_load(**_kwargs):
        raise AssertionError("invalid input reached model loading")

    monkeypatch.setattr(probe, "load_vlm_backend", unexpected_load)
    with pytest.raises(ValueError, match=problem):
        probe.main(["--manifest", str(path),
                    "--output", str(tmp_path / "result.json")])
