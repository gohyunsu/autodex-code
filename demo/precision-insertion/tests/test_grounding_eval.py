"""A metric VLM proposal needs an independent same-capture reference."""

import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.grounding_eval import (  # noqa: E402
    evaluate_grounding_manifest,
)
from evaluate_grounded_alignment import main as evaluate_cli  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _sample(tmp_path):
    directory = tmp_path / "attempt"
    frames = directory / "frames"
    frames.mkdir(parents=True)
    image = frames / "front.png"
    image.write_bytes(b"saved raw frame")
    diagnostic = directory / "report.json"
    _save(diagnostic, {
        "schema": "precision_insertion_grounded_xy_diagnostic_v2",
        "robot_ready": False,
        "attempt_id": "trial1", "candidate_id": "table/0/1",
        "session_calibration_sha256": "a" * 64,
        "task_geometry_sha256": "b" * 64,
        "frame_request_id": 42,
        "frame_binding": {"front": {"timestamp_s": 100.0,
                                    "max_error_s": .001}},
        "artifacts_sha256": {"frames/front.png": _sha(image)},
        "alignment": {
            "schema": "precision_insertion_grounded_alignment_v2",
            "status": "diagnostic_metric_xy_correction",
            "reason": "confident_lateral_reduction",
            "xy_correction_socket_m": [-.0015, 0.],
            "bounded_xy_increment_socket_m": [-.001, 0.],
            "socket_entry_plane_z_m": .055,
            "verification_depth_m": .02,
            "mean_error_xy_m": [.0015, 0.],
            "lateral_covariance_m2": [[1e-8, 0.], [0., 1e-8]],
            "increment_squared_error_improvement_lower_95_m2": (
                2e-6 - 2 * math.sqrt(-2 * math.log(.05)) * 1e-7),
            "insertion_axis_socket": [0., 0., -1.],
            "lateral_uncertainty_95_m": .0003,
        },
    })
    metrology = tmp_path / "external_measurement.bin"
    metrology.write_bytes(b"independent optical tracking sample")
    truth = tmp_path / "truth.json"
    _save(truth, {
        "schema": "precision_insertion_independent_key_pose_v1",
        "attempt_id": "trial1", "candidate_id": "table/0/1",
        "session_calibration_sha256": "a" * 64,
        "task_geometry_sha256": "b" * 64,
        "frame_request_id": 42,
        "measurement_method": "external_optical_metrology",
        "capture_time_s": 100.001,
        "translation_uncertainty_95_m": .00005,
        "axis_uncertainty_95_deg": .05,
        "tip_socket_m": [.0016, 0., .09],
        "insertion_axis_socket": [0., 0., -1.],
        "source_files": [{"path": str(metrology), "sha256": _sha(metrology)}],
    })
    manifest = tmp_path / "manifest.json"
    _save(manifest, {
        "schema": "precision_insertion_grounded_eval_manifest_v1",
        "max_reference_capture_skew_s": .01,
        "max_truth_translation_uncertainty_95_m": .0001,
        "max_truth_axis_uncertainty_95_deg": .1,
        "samples": [{"diagnostic_report": "attempt/report.json",
                     "independent_pose": "truth.json"}],
    })
    return manifest, diagnostic, truth, image


def test_evaluation_reports_true_continuous_increment_and_uncertainty(tmp_path):
    manifest, _diagnostic, _truth, _image = _sample(tmp_path)
    result = evaluate_grounding_manifest(manifest)
    assert result["sample_count"] == 1
    assert result["advice_count"] == 1
    assert result["false_advice_count"] == 0
    assert result["median_increment_regret_m2"] == pytest.approx(0.)
    assert result["median_lateral_error_m"] == pytest.approx(.0001)
    assert result["median_axis_error_deg"] == pytest.approx(0.)
    assert result["empirical_95_radius_coverage"] == 1.
    assert result["robot_ready"] is False


def test_evaluation_rejects_wrong_capture_or_modified_image(tmp_path):
    manifest, _report, truth, image = _sample(tmp_path)
    value = json.loads(truth.read_text())
    value["frame_request_id"] = 43
    _save(truth, value)
    with pytest.raises(ValueError, match="frame_request_id"):
        evaluate_grounding_manifest(manifest)
    value["frame_request_id"] = 42
    _save(truth, value)
    image.write_bytes(b"changed raw frame")
    with pytest.raises(ValueError, match="source image changed"):
        evaluate_grounding_manifest(manifest)


def test_evaluation_rejects_unbounded_or_nonindependent_reference(tmp_path):
    manifest, _report, truth, _image = _sample(tmp_path)
    value = json.loads(truth.read_text())
    value["measurement_method"] = "v8_nominal_wrist_pose"
    _save(truth, value)
    with pytest.raises(ValueError, match="independent measurement"):
        evaluate_grounding_manifest(manifest)
    value["measurement_method"] = "external_optical_metrology"
    value["translation_uncertainty_95_m"] = .002
    _save(truth, value)
    with pytest.raises(ValueError, match="uncertainty exceeds"):
        evaluate_grounding_manifest(manifest)


def test_evaluation_flags_wrong_direction_without_promoting(tmp_path):
    manifest, report, _truth, _image = _sample(tmp_path)
    value = json.loads(report.read_text())
    value["alignment"]["mean_error_xy_m"] = [-.0015, 0.]
    value["alignment"]["xy_correction_socket_m"] = [.0015, 0.]
    value["alignment"]["bounded_xy_increment_socket_m"] = [.001, 0.]
    _save(report, value)
    result = evaluate_grounding_manifest(manifest)
    assert result["false_advice_count"] == 1
    assert result["median_increment_regret_m2"] > 0
    assert result["robot_ready"] is False


def test_evaluation_rejects_legacy_cardinal_report(tmp_path):
    manifest, report, _truth, _image = _sample(tmp_path)
    value = json.loads(report.read_text())
    value["schema"] = "precision_insertion_grounded_xy_diagnostic_v1"
    _save(report, value)
    with pytest.raises(ValueError, match="read-only grounded XY diagnostic"):
        evaluate_grounding_manifest(manifest)


def test_evaluation_rejects_inconsistent_continuous_correction(tmp_path):
    manifest, report, _truth, _image = _sample(tmp_path)
    value = json.loads(report.read_text())
    value["alignment"]["xy_correction_socket_m"] = [.0015, 0.]
    _save(report, value)
    with pytest.raises(ValueError, match="disagrees with observed axis error"):
        evaluate_grounding_manifest(manifest)


def test_cli_writes_report_exclusively(tmp_path):
    manifest, _report, _truth, _image = _sample(tmp_path)
    output = tmp_path / "evaluation.json"
    assert evaluate_cli(["--manifest", str(manifest),
                         "--output", str(output)]) == 0
    assert json.loads(output.read_text())["robot_ready"] is False
    with pytest.raises(FileExistsError):
        evaluate_cli(["--manifest", str(manifest),
                      "--output", str(output)])
