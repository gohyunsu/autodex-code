"""Real-image NAS handoff is not silently promoted from numeric IoU."""

import hashlib
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_perception_handoff import audit_handoff, main  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    shot_dir = tmp_path / "shot"
    shot_dir.mkdir()
    images = {}
    for serial in ("cam_a", "cam_b"):
        image = shot_dir / f"{serial}.jpg"
        image.write_bytes(f"synthetic image {serial}".encode())
        images[serial] = {
            "file": image.name, "bytes": image.stat().st_size,
            "sha256": _sha(image), "width": 10, "height": 10,
        }
    condition = "square/socket_only"
    shot_json = shot_dir / "shot.json"
    shot_json.write_text(json.dumps({
        "status": "COMPLETE", "condition_id": condition,
        "calibration_session": "calib_1",
        "expected_camera_serials": ["cam_a", "cam_b"],
        "images": images,
    }), encoding="utf-8")
    capture_index = tmp_path / "captures.json"
    capture_index.write_text(json.dumps({
        "condition_count": 1, "missing_planned_conditions": ["square/key/pose_02"],
        "shots": [{"condition_id": condition, "shot_dir": str(shot_dir),
                   "camera_count": 2, "shot_json_sha256": _sha(shot_json)}],
    }), encoding="utf-8")
    repre = tmp_path / "repre.pth"
    repre.write_bytes(b"synthetic PTH candidate")
    candidate_index = tmp_path / "candidates.json"
    candidate_index.write_text(json.dumps({
        "candidates": [{"object": "precision_socket_unified",
                        "path": str(repre), "bytes": repre.stat().st_size,
                        "sha256": _sha(repre),
                        "status": "PENDING_REAL_IMAGE_VALIDATION"}],
    }), encoding="utf-8")
    eval_root = tmp_path / "evaluation"
    report_file = eval_root / "results/socket/report.json"
    report_file.parent.mkdir(parents=True)
    report_file.write_text(json.dumps({
        "condition_id": condition, "shot_dir": str(shot_dir),
        "n_valid_mask_pose": 2, "n_expected": 2,
        "passes_numeric_sil_threshold": True,
        "final_mean_iou_all_valid_masks": .97,
    }), encoding="utf-8")
    return capture_index, candidate_index, eval_root, shot_dir, report_file


def test_real_image_hash_and_numeric_fit_never_auto_promote(tmp_path):
    index, candidates, eval_root, _shot, _report = _fixture(tmp_path)
    result = audit_handoff(index, candidates, eval_root,
                           verify_candidate_hashes=True)
    assert result["capture_integrity_pass"] is True
    assert result["evaluations"][0]["numeric_silhouette_pass"] is True
    assert result["evaluation_variant_summary"]["socket"] == {
        "reports": 1, "numeric_pass": 1}
    assert result["candidate_representations"][0]["hash_verified"] is True
    assert result["conditions_without_per_camera_acquisition_time"] == [
        "square/socket_only"]
    assert result["perception_promotion_ready"] is False
    assert result["session_start_eligible"] is False


def test_tampered_image_and_unbound_evaluation_are_reported(tmp_path):
    index, candidates, eval_root, shot, report_file = _fixture(tmp_path)
    (shot / "cam_b.jpg").write_bytes(b"changed camera pixels")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    report["shot_dir"] = str(tmp_path / "different_shot")
    report_file.write_text(json.dumps(report), encoding="utf-8")
    result = audit_handoff(index, candidates, eval_root)
    assert result["capture_integrity_pass"] is False
    assert any("image(s) failed hash check" in row
               for row in result["integrity_errors"])
    assert any("evaluation is not bound" in row
               for row in result["integrity_errors"])


def test_cli_writes_exclusive_snapshot(tmp_path, capsys):
    index, candidates, eval_root, _shot, _report = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    args = ["--capture-index", str(index), "--candidate-index", str(candidates),
            "--evaluation-root", str(eval_root), "--output", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text())["selected_condition_count"] == 1
    capsys.readouterr()
    try:
        main(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit must not overwrite a previous snapshot")
