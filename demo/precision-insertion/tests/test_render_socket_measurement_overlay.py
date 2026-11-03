"""Projection in socket diagnostics must match the saved pixel space."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "render_socket_measurement_overlay.py"
SPEC = importlib.util.spec_from_file_location("socket_measurement_overlay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record() -> dict:
    return {
        "c2r": np.eye(4).tolist(),
        "camera_calibration": {
            "extrinsics_full": {"camera": np.eye(4).tolist()},
            "intrinsics_full": {"camera": {
                "K_orig": [[200., 0., 20.], [0., 200., 30.], [0., 0., 1.]],
                "K_undist": [[100., 0., 10.], [0., 100., 15.], [0., 0., 1.]],
                "dist_params": [0., 0., 0., 0.],
            }},
        },
    }


def test_project_uses_undistorted_intrinsics_for_saved_pipeline_frames():
    point = np.array([[.5, .25, 1.]])
    pixel = MODULE._project(
        _record(), "camera", point, "autodex_undistorted_full_frame")
    assert pixel.tolist() == [[60, 40]]


def test_project_keeps_original_intrinsics_for_distorted_frames():
    point = np.array([[.5, .25, 1.]])
    pixel = MODULE._project(
        _record(), "camera", point, "camera_distorted_full_frame")
    assert pixel.tolist() == [[120, 80]]


def test_project_rejects_unidentified_pixel_space():
    with pytest.raises(ValueError, match="unsupported saved socket image space"):
        MODULE._project(_record(), "camera", np.array([[0., 0., 1.]]), "unknown")
