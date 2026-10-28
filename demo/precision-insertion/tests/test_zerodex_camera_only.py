"""The capture4 diagnostic must follow the live perception path without motion."""

from contextlib import contextmanager, nullcontext
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_zerodex_camera_only as camera_only  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


@pytest.mark.parametrize("reuse", (False, True))
@pytest.mark.parametrize("stem", ("000", "001"))
@pytest.mark.parametrize("upright_only", (False, True))
@pytest.mark.parametrize("socket_only", (False, True))
def test_camera_only_reuses_socket_key_and_catalog_without_robot(
        tmp_path, monkeypatch, reuse, stem, upright_only, socket_only):
    events = []
    mode = select_mode("cylinder", 20)
    shared = tmp_path / "shared"
    shared.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    evidence = tmp_path / "run" / "socket_evidence"
    calibration = SimpleNamespace()
    launch = SimpleNamespace(
        mode=mode, socket_perception_mode="board_constrained_body",
        pc_list=("capture4",), capture_ips=("192.0.2.1",),
        shared_root=shared, capture_root=capture,
        camera_acquisition_mode="stream", camera_ids={"a", "b"},
        intrinsics_full={"a": {}, "b": {}},
        extrinsics_full={"a": {}, "b": {}},
        image_hw=(1536, 2048), view_limits=SimpleNamespace(),
        upright_cylinder_only=upright_only,
    )

    def load(_path, *, evidence_dir_override):
        assert evidence_dir_override == evidence
        events.append("launch")
        return launch

    def catalog(**_kwargs):
        events.append("catalog")
        return {"candidates": []}

    cameras = SimpleNamespace(
        init_orchestrator=object(),
        acquisition_metadata_for_request=lambda _id: {},
        start=lambda: (
            events.append("socket") or SimpleNamespace(
                evidence_dir=evidence, producer_manifest={},
                calibration=calibration)),
    )

    @contextmanager
    def open_cameras(_launch, *, init_ports):
        assert init_ports == (15006, 15007, 16893)
        events.append("camera_open")
        yield cameras
        events.append("camera_close")

    @contextmanager
    def open_stream(_pcs, **kwargs):
        assert kwargs["acquisition_mode"] == "stream"
        events.append("stream_open")
        yield object()
        events.append("stream_close")

    def capture_key(**kwargs):
        events.append("key")
        assert kwargs["silhouette_iterations"] == 100
        assert kwargs["silhouette_loss_threshold"] == 0.003
        assert "max_key_state_skew_s" not in kwargs
        assert kwargs["pose_only"] is True
        return SimpleNamespace(
            pose_world=np.eye(4), request_id=42, selected_camera_id="a")

    monkeypatch.setattr(camera_only, "load_session_launch", load)
    monkeypatch.setattr(camera_only, "preflight_endpoint_catalog_file", catalog)
    monkeypatch.setattr(camera_only, "open_session_cameras", open_cameras)
    monkeypatch.setattr(camera_only, "load_reused_socket_session",
                        lambda **_kwargs: (
                            events.append("saved_socket") or
                            SimpleNamespace(evidence_dir=evidence,
                                            producer_manifest={},
                                            calibration=calibration)))
    monkeypatch.setattr(camera_only, "open_autodex_camera_stream", open_stream)
    monkeypatch.setattr(camera_only, "prepare_precision_silhouette",
                        lambda **_kwargs: events.append("prewarm"))
    monkeypatch.setattr(camera_only, "open_verified_session",
                        lambda **_kwargs: (
                            events.append("direct_lie" if
                                          _kwargs["upright_cylinder_only"] is False
                                          else "legacy_policy") or
                            events.append("ledger") or SimpleNamespace(runner=object())))
    monkeypatch.setattr(camera_only, "_capture_admitted_key", capture_key)
    monkeypatch.setattr(camera_only, "build_trial_scene_from_session",
                        lambda **_kwargs: (
                            events.append("scene") or {
                                "mesh": {"target": {"pose": [0, 0, 0, 1, 0, 0, 0]}}}))
    monkeypatch.setattr(camera_only, "classify_key_tabletop_pose",
                        lambda **_kwargs: (
                            events.append("classify") or {"stem": stem}))
    monkeypatch.setattr(camera_only, "select_pose_candidates",
                        lambda *_args, **_kwargs: (
                            events.append("select") or {
                                "status": "candidates_available",
                                "candidates": [{"key": ["table", "0", "1"]}]}))
    monkeypatch.setattr(camera_only, "_key_capture", lambda: {
        "max_key_state_skew_s": 5., "capture_timeout_s": 120.,
        "image_write_timeout_s": 90.,
    })
    monkeypatch.setattr(camera_only, "profile_run", lambda _path: nullcontext())
    monkeypatch.setattr(camera_only, "timed",
                        lambda *_args, **_kwargs: nullcontext())

    report = camera_only.run_camera_only(
        launch_config=tmp_path / "launch.json",
        endpoint_catalog=tmp_path / "catalog.json",
        output_dir=tmp_path / "run",
        wait_for_key=lambda: events.append("key_placed"),
        reuse_session_evidence=(tmp_path / "saved" if reuse else None),
        socket_only=socket_only,
    )
    assert events == (
        ["launch", "catalog"] + (["saved_socket"] if reuse else []) +
        ["camera_open"] + ([] if socket_only else ["prewarm"]) +
        ["stream_open"] +
        ([] if reuse else ["socket"]) +
        ["legacy_policy" if upright_only else "direct_lie",
         "ledger"] + ([] if socket_only else ["key_placed", "key"]) +
        ["stream_close", "camera_close"] +
        ([] if socket_only else ["scene", "classify", "select"]))
    if socket_only:
        assert report["schema"] == "precision_insertion_zerodex_socket_only_session_v1"
        assert report["mode"]["gap_mm"] == 20
    else:
        assert report["endpoint_eligible_candidate_count"] == 1
        assert report["schema"] == "precision_insertion_zerodex_camera_only_session_v2"
        direct_preflight = not upright_only or stem == "000"
        assert report["direct_candidate_preflight_available"] is direct_preflight
        assert report["live_pose_route"] == (
            "direct_candidate_preflight" if direct_preflight else
            "reorientation_preflight_if_alternative_pose_available")
    assert report["socket_evidence_reused"] is reuse
    assert report["robot_ready"] is False
    assert (tmp_path / "run" / "report.json").is_file()


@pytest.mark.parametrize("explicit_output", (False, True))
def test_camera_only_launcher_keeps_default_output_with_other_options(
        tmp_path, explicit_output):
    launcher = Path(__file__).resolve().parents[1] / "run_zerodex_capture4.sh"
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_planner.chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    development = tmp_path / "development"
    command = ["bash", str(launcher), "--camera-only",
               "--reuse-session-evidence", str(tmp_path / "saved")]
    if explicit_output:
        command.extend(("--output-dir", str(tmp_path / "chosen")))
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True,
        env={**os.environ, "PRECISION_STAGE_ROOT": str(stage),
             "PRECISION_DEV_ROOT": str(development),
             "PRECISION_PLANNER_PY": str(fake_planner)})
    args = completed.stdout.splitlines()
    assert args.count("--output-dir") == 1
    output = args[args.index("--output-dir") + 1]
    assert output == (str(tmp_path / "chosen") if explicit_output else
                      str(development / "camera_only" / Path(output).name))
    assert args[args.index("--gap-mm") + 1] == "5"


def test_camera_only_gap_override_keeps_catalog_and_launch_consistent(
        tmp_path, monkeypatch):
    template = tmp_path / "stage_launch.json"
    template.write_text('{"mode":{"family":"cylinder","gap_mm":20}}\n',
                        encoding="utf-8")
    output = tmp_path / "run"

    def inspect_launch(path, *, evidence_dir_override):
        assert path == output / "launch.json"
        assert evidence_dir_override == output / "socket_evidence"
        assert camera_only.json.loads(path.read_text(encoding="utf-8"))["mode"] == {
            "family": "cylinder", "gap_mm": 5}
        raise RuntimeError("verified launch override")

    monkeypatch.setattr(camera_only, "load_session_launch", inspect_launch)
    monkeypatch.setattr(camera_only, "profile_run", lambda _path: nullcontext())
    monkeypatch.setattr(camera_only, "timed",
                        lambda *_args, **_kwargs: nullcontext())
    with pytest.raises(RuntimeError, match="verified launch override"):
        camera_only.run_camera_only(
            launch_config=template, endpoint_catalog=tmp_path / "gap5.json",
            output_dir=output, wait_for_key=lambda: None, gap_mm=5)
    assert camera_only.json.loads(template.read_text(encoding="utf-8"))["mode"][
        "gap_mm"] == 20


def test_camera_only_launcher_forwards_gap_mm(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "run_zerodex_capture4.sh"
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
                            encoding="utf-8")
    fake_planner.chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir()
    completed = subprocess.run(
        ["bash", str(launcher), "--camera-only"], capture_output=True,
        text=True, check=True,
        env={**os.environ, "PRECISION_STAGE_ROOT": str(stage),
             "PRECISION_PLANNER_PY": str(fake_planner),
             "PRECISION_GAP_MM": "5"})
    args = completed.stdout.splitlines()
    assert args[args.index("--gap-mm") + 1] == "5"
    assert args[args.index("--endpoint-catalog") + 1] == str(
        stage / "recipient_gap5_endpoint_catalog_straight.json")


def test_key_already_placed_reuses_socket_without_operator_wait(
        tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(camera_only, "run_camera_only", lambda **kwargs: (
        seen.append(kwargs) or {"robot_ready": False}))
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail(
        "already-placed key must not prompt for placement"))
    assert camera_only.main([
        "--launch-config", str(tmp_path / "launch.json"),
        "--endpoint-catalog", str(tmp_path / "catalog.json"),
        "--output-dir", str(tmp_path / "run"),
        "--reuse-session-evidence", str(tmp_path / "saved"),
        "--key-already-placed",
    ]) == 0
    assert len(seen) == 1
    assert seen[0]["reuse_session_evidence"] == tmp_path / "saved"
    assert seen[0]["wait_for_key"]() is None


def test_key_already_placed_fresh_capture_records_contaminated_baseline(
        tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(camera_only, "run_camera_only", lambda **kwargs: (
        seen.append(kwargs) or {"robot_ready": False}))
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail(
        "already-placed key must not prompt for placement"))
    assert camera_only.main([
        "--launch-config", str(tmp_path / "launch.json"),
        "--endpoint-catalog", str(tmp_path / "catalog.json"),
        "--output-dir", str(tmp_path / "run"),
        "--key-already-placed",
    ]) == 0
    assert seen[0]["key_present_during_socket_capture"] is True
    assert seen[0]["wait_for_key"]() is None
