"""The optional same-frame SAM ROI path is independent of live hardware."""

import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import retry_session  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.observer import LabeledFrame  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.retry_session import (  # noqa: E402
    PreinsertXYDiagnostic, assess_saved_preinsert_xy_diagnostic,
    write_retry_session_artifacts,
    write_saved_preinsert_xy_artifacts,
)
from precision_insertion.sam_roi_backend import SamRoiBackend  # noqa: E402


def test_worker_protocol_keeps_rgb_roi_and_binary_mask(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import base64, io, json, sys
from PIL import Image
print(json.dumps({"type":"ready","model":sys.argv[2]}), flush=True)
for line in sys.stdin:
    row=json.loads(line)
    if row["type"]=="close": break
    image=Image.open(io.BytesIO(base64.b64decode(row["image_png_base64"])))
    mask=Image.new("L", image.size, 0)
    mask.putpixel((2,3),255)
    data=io.BytesIO(); mask.save(data,format="PNG")
    print(json.dumps({"type":"result","request_id":row["request_id"],
      "camera_id":row["camera_id"],"image_size_wh":list(image.size),
      "mask_png_base64":base64.b64encode(data.getvalue()).decode(),
      "score":0.8,"inference_s":0.01,"cuda_peak_bytes":123}),flush=True)
''')
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    backend = SamRoiBackend(
        python_path=Path(sys.executable), checkpoint=checkpoint,
        worker_path=worker, stderr_path=tmp_path / "worker.log")
    try:
        image = Image.new("RGB", (17, 19), "white")
        mask, metadata = backend.segment(
            image, camera_id="cam_a", prompt="white cylinder")
        assert mask.shape == (19, 17)
        assert mask[3, 2] == 255 and np.count_nonzero(mask) == 1
        assert metadata["score"] == .8
        assert metadata["model"] == str(checkpoint)
    finally:
        backend.close()


def test_timed_out_worker_cannot_reuse_stale_camera_reply(tmp_path):
    worker = tmp_path / "slow_worker.py"
    worker.write_text('''
import json, sys, time
print(json.dumps({"type":"ready","model":sys.argv[2]}), flush=True)
for line in sys.stdin:
    row=json.loads(line)
    if row["type"]=="close": break
    time.sleep(1.)
    print(json.dumps({"type":"result","request_id":row["request_id"],
      "camera_id":row["camera_id"],"image_size_wh":[8,8],
      "mask_png_base64":None,"score":0.8,"inference_s":1.,
      "cuda_peak_bytes":0}),flush=True)
''')
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    backend = SamRoiBackend(
        python_path=Path(sys.executable), checkpoint=checkpoint,
        worker_path=worker, stderr_path=tmp_path / "worker.log",
        response_timeout_s=.05)
    try:
        image = Image.new("RGB", (8, 8), "white")
        with pytest.raises(TimeoutError, match="response timed out"):
            backend.segment(image, camera_id="cam_a", prompt="white cylinder")
        assert backend._process.poll() is not None
        with pytest.raises(RuntimeError, match="not running"):
            backend.segment(image, camera_id="cam_b", prompt="white cylinder")
    finally:
        backend.close()


@pytest.mark.parametrize("failure", [
    None, "one_qwen", "one_sam", "all_qwen", "all_sam", "mixed_all"])
def test_same_saved_frame_qwen_roi_sam_center_maps_to_full_pixels(
    tmp_path, monkeypatch, failure,
):
    mode = select_mode("cylinder", 20.)
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({
        "key_frame": {"tip_z_m": .08},
        "socket_entry_plane_z_m": 0.0,
    }), encoding="utf-8")
    monkeypatch.setattr(retry_session, "AssetPaths",
                        lambda *_: SimpleNamespace(task_geometry=geometry))
    monkeypatch.setattr(retry_session, "validated_frozen_socket_pose",
                        lambda **_: np.eye(4))
    key = np.eye(4)
    key[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    key[:3, 3] = [0., 0., 1.]
    postlift = tmp_path / "postlift.json"
    postlift.write_text(json.dumps({
        "targets": {"T_robot_key_preinsert": key.tolist()}}),
        encoding="utf-8")
    K = np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]])
    calibration = SimpleNamespace(record={"c2r": np.eye(4).tolist()})
    frames, rows, intrinsics, extrinsics = [], {}, {}, {}
    for index, (camera, translation) in enumerate((
            ("cam_a", -.05), ("cam_b", 0.), ("cam_c", .05)), 1):
        E = np.eye(4)
        E[0, 3] = translation
        image = Image.new("RGB", (2048, 1536), "black")
        frames.append(LabeledFrame(camera, "preinsert_hold", 100., image))
        rows[camera] = {"frame_id": index, "timestamp_s": 100.,
                        "image_sha256": image_sha256(
                            np.asarray(image).copy())}
        intrinsics[camera] = {"K_undist": K.tolist()}
        extrinsics[camera] = E.tolist()

    class BoxBackend:
        max_parallel_views = 3
        model_id = "saved-box-stub"

        def infer_for_camera(self, camera, images, prompt):
            assert prompt.startswith(f"Camera {camera}.")
            assert images[0].size == (512, 512)
            if (failure == "all_qwen" or
                    failure == "one_qwen" and camera == "cam_c" or
                    failure == "mixed_all" and camera == "cam_a"):
                raise RuntimeError(f"capture-PC Qwen {camera} unavailable")
            return '{"bbox":[400,350,600,600]}'

    class MaskBackend:
        model = "saved-mask-stub"

        def segment(self, image, *, camera_id, prompt):
            assert camera_id in rows and prompt == "white cylinder"
            if (failure == "all_sam" or
                    failure == "one_sam" and camera_id == "cam_c" or
                    failure == "mixed_all" and camera_id != "cam_a"):
                raise TimeoutError(f"SAM3 ROI {camera_id} timed out")
            mask = np.zeros((image.height, image.width), np.uint8)
            mask[62:145, 55:100] = 255
            return mask, {"score": .9, "inference_s": .01,
                          "cuda_peak_bytes": 100, "model": self.model}

    if failure in {"all_qwen", "all_sam", "mixed_all"}:
        source = ("XY VLM backend" if failure == "all_qwen" else
                  "SAM3 ROI" if failure == "all_sam" else
                  "XY VLM/SAM backend")
        with pytest.raises(RuntimeError, match=source + " unavailable"):
            retry_session._compute_sam_half_ellipse_alignment(
                mode=mode, root=tmp_path, calibration=calibration,
                postlift_file=postlift, frames=frames, frame_rows=rows,
                backend=BoxBackend(), sam_segmentor=MaskBackend(),
                intrinsics_full=intrinsics, extrinsics_full=extrinsics,
                alignment_limits=SimpleNamespace(
                    max_reprojection_px=5., min_parallax_deg=1.),
                search_index=0, max_nominal_tip_height_error_m=.03)
        return
    _hash, alignment, masks = (
        retry_session._compute_sam_half_ellipse_alignment(
            mode=mode, root=tmp_path, calibration=calibration,
            postlift_file=postlift, frames=frames, frame_rows=rows,
            backend=BoxBackend(), sam_segmentor=MaskBackend(),
            intrinsics_full=intrinsics, extrinsics_full=extrinsics,
            alignment_limits=SimpleNamespace(
                max_reprojection_px=5., min_parallax_deg=1.),
            search_index=0, max_nominal_tip_height_error_m=.03))
    assert alignment["estimator"] == "sam3_half_ellipse_key_face_proxy"
    assert alignment["status"] == "diagnostic_metric_xy_correction"
    assert len(alignment["sam_half_ellipse_views"]) == 3
    expected = ({"cam_a", "cam_b", "cam_c"} if failure is None else
                {"cam_a", "cam_b"})
    assert set(masks) == expected
    for record in alignment["sam_half_ellipse_views"]:
        if record["camera_id"] not in expected:
            assert record["admitted"] is False
            assert record["reason"] == (
                "qwen_backend_unavailable" if failure == "one_qwen" else
                "sam_backend_unavailable")
            continue
        assert record["admitted"] is True
        assert record["mask_file_sha256"] == hashlib.sha256(
            masks[record["camera_id"]]).hexdigest()
        assert abs(record["sam_half_ellipse"]["center_full_px"][1]-848.) < 12.


def test_sam_roi_mask_bytes_are_saved_with_the_xy_diagnostic(tmp_path):
    frames = []
    frame_binding = {}
    mask_png = {}
    mask_records = []
    for index, camera in enumerate(("cam_a", "cam_b"), 1):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        frames.append(LabeledFrame(
            camera, "preinsert_hold", 100., Image.fromarray(image)))
        frame_binding[camera] = {"frame_id": index, "timestamp_s": 100.,
                                 "image_sha256": image_sha256(image)}
        stream = io.BytesIO()
        Image.fromarray(np.full((8, 10), 255, np.uint8)).save(
            stream, format="PNG")
        mask_png[camera] = stream.getvalue()
        mask_records.append({
            "camera_id": camera,
            "mask_file_sha256": hashlib.sha256(mask_png[camera]).hexdigest()})
    result = PreinsertXYDiagnostic(
        status="diagnostic_metric_xy_correction", attempt_id="attempt_a",
        candidate_id="table/0/1", session_calibration_sha256="session",
        camera_calibration_sha256="camera", task_geometry_sha256="geometry",
        frame_request_id=1, preinsert_checkpoint_path=tmp_path / "pre.json",
        preinsert_checkpoint_sha256="pre", preinsert_bundle_path=tmp_path,
        preinsert_manifest_sha256="manifest",
        postlift_preflight_path=tmp_path / "post.json",
        postlift_preflight_sha256="post", frame_binding=frame_binding,
        joint_sample=SimpleNamespace(to_record=lambda: {}),
        alignment={"sam_half_ellipse_views": mask_records},
        vlm_observations=(), sam_roi_masks_png=mask_png)
    target = write_retry_session_artifacts(result, frames, tmp_path / "xy")
    report = json.loads((target / "report.json").read_text())
    for camera in mask_png:
        name = f"sam_roi_masks/{camera}.png"
        assert (target / name).read_bytes() == mask_png[camera]
        assert report["artifacts_sha256"][name] == hashlib.sha256(
            mask_png[camera]).hexdigest()


def test_saved_preinsert_xy_reuses_source_pngs_and_checks_their_hashes(tmp_path):
    bundle = tmp_path / "source"
    image_dir = bundle / "images"
    image_dir.mkdir(parents=True)
    rows, png_hashes = {}, {}
    for camera in ("cam_a", "cam_b"):
        image = np.full((24, 32, 3), 127, np.uint8)
        path = image_dir / f"{camera}.png"
        Image.fromarray(image).save(path)
        rows[camera] = {"frame_id": 10, "timestamp_s": 100.,
                        "image_sha256": image_sha256(image)}
        png_hashes[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_file = bundle / "manifest.json"
    manifest_file.write_text(json.dumps({
        "frame_evidence": rows, "image_file_sha256": png_hashes,
    }))
    diagnostic = PreinsertXYDiagnostic(
        status="diagnostic_metric_xy_correction", attempt_id="attempt_a",
        candidate_id="table/0/1", session_calibration_sha256="session",
        camera_calibration_sha256="camera", task_geometry_sha256="geometry",
        frame_request_id=1, preinsert_checkpoint_path=tmp_path / "pre.json",
        preinsert_checkpoint_sha256="pre", preinsert_bundle_path=bundle,
        preinsert_manifest_sha256=hashlib.sha256(
            manifest_file.read_bytes()).hexdigest(),
        postlift_preflight_path=tmp_path / "post.json",
        postlift_preflight_sha256="post", frame_binding=rows,
        joint_sample=SimpleNamespace(to_record=lambda: {}),
        alignment={"status": "diagnostic_metric_xy_correction"},
        vlm_observations=())
    output = write_saved_preinsert_xy_artifacts(diagnostic, tmp_path / "xy")
    report = json.loads((output / "report.json").read_text())
    assert report["source_image_file_sha256"] == png_hashes
    assert report["frame_binding"] == rows
    assert report["artifacts_sha256"] == {}
    assert sorted(path.name for path in output.iterdir()) == ["report.json"]

    (image_dir / "cam_b.png").write_bytes(b"changed")
    with np.testing.assert_raises_regex(ValueError, "image changed"):
        write_saved_preinsert_xy_artifacts(diagnostic, tmp_path / "xy_bad")
    assert not (tmp_path / "xy_bad").exists()


@pytest.mark.parametrize("estimator", ["sam_half_ellipse", "tracked_tip"])
def test_public_saved_diagnostic_binds_only_selected_camera_images(
    tmp_path, monkeypatch, estimator,
):
    mode = select_mode("cylinder", 20.)
    bundle = tmp_path / "raw"
    images = bundle / "images"
    images.mkdir(parents=True)
    rows, png_hashes = {}, {}
    for camera in ("100", "200", "300", "400"):
        pixels = np.full((24, 32, 3), int(camera) // 4, np.uint8)
        path = images / f"{camera}.png"
        Image.fromarray(pixels).save(path)
        rows[camera] = {"frame_id": int(camera), "timestamp_s": 100.,
                        "max_error_s": .01,
                        "image_sha256": image_sha256(pixels)}
        png_hashes[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"frame_evidence": rows, "image_file_sha256": png_hashes,
                "request_id": 42}
    if estimator == "tracked_tip":
        manifest["temporal_window"] = {
            "status": "complete", "camera_id": "400", "target_frame_id": 400}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    postlift = tmp_path / "postlift.json"
    postlift.write_text("{}")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", "100,200,300")
    monkeypatch.setattr(retry_session, "_validated_retry_trial_context",
                        lambda **_: (None, None, postlift, tmp_path))
    monkeypatch.setattr(retry_session, "verify_preinsert_checkpoint",
                        lambda _: {"attempt_id": "attempt_a",
                                   "candidate_id": "table/0/1",
                                   "preinsert_reached": True,
                                   "postlift_report_path": str(postlift),
                                   "raw_bundle": str(bundle),
                                   "joint_feedback": {}})
    monkeypatch.setattr(retry_session, "verify_raw_camera_capture",
                        lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(retry_session, "_state_from_record",
                        lambda _: SimpleNamespace(validate=lambda **__: None,
                                                  to_record=lambda: {}))
    monkeypatch.setattr(retry_session,
                        "validate_session_camera_calibration", lambda *_a, **_k: None)

    def alignment(**kwargs):
        expected = (["100", "200", "300", "400"]
                    if estimator == "tracked_tip" else
                    ["100", "200", "300"])
        assert [frame.camera_id for frame in kwargs["frames"]] == expected
        record = {"status": "diagnostic_metric_xy_correction",
                  "sam_half_ellipse_views": []}
        return (("geometry", record) if estimator == "tracked_tip" else
                ("geometry", record, {}))

    if estimator == "tracked_tip":
        monkeypatch.setattr(retry_session, "_compute_tracked_alignment",
                            alignment)
    else:
        monkeypatch.setattr(retry_session, "_compute_sam_half_ellipse_alignment",
                            alignment)
    calibration = SimpleNamespace(record={
        "camera_calibration_sha256": "calibration",
        "camera_calibration": {
            "intrinsics_full": {camera: {"K_undist": np.eye(3).tolist()}
                                for camera in rows},
            "extrinsics_full": {camera: np.eye(4).tolist()
                                for camera in rows}}})
    limits = SimpleNamespace(
        validate=lambda: None, max_capture_skew_s=1.,
        max_arm_hand_skew_s=1., max_hand_command_error_raw=1.,
        max_arm_velocity_rad_s=1., max_grasp_translation_drift_m=.1)
    attempt = SimpleNamespace(attempt_id="attempt_a", candidate_id="table/0/1",
                              session_calibration_sha256="session")
    result = assess_saved_preinsert_xy_diagnostic(
        mode=mode, shared_root=tmp_path, calibration=calibration,
        catalog={}, trial=object(), attempt=attempt,
        preinsert_checkpoint_path=checkpoint,
        postlift_preflight_report_path=postlift,
        backend=object(), limits=limits,
        alignment_limits=SimpleNamespace(), xy_estimator=estimator,
        key_reference_dir=tmp_path, sam_segmentor=object())
    expected = ({"100", "200", "300", "400"}
                if estimator == "tracked_tip" else {"100", "200", "300"})
    assert set(result.frame_binding) == expected
    output = write_saved_preinsert_xy_artifacts(result, tmp_path / "xy")
    report = json.loads((output / "report.json").read_text())
    assert set(report["source_image_file_sha256"]) == expected
    assert sorted(path.name for path in output.iterdir()) == ["report.json"]
