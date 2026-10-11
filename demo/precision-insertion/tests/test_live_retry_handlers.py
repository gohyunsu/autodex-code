"""The live XY branch advances even when the camera declines correction."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_retry_handlers  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def test_live_grounded_retry_abstain_routes_to_held_return(tmp_path, monkeypatch):
    class State:
        pass

    monkeypatch.setattr(live_retry_handlers, "LiveRobotState", State)
    decision = SessionDecision(
        "guarded_withdrawal_then_xy_assessment", "failed", "table/0/1")
    events = []

    class Runner:
        active_attempt = SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/1")
        _postlift_report_path = tmp_path / "postlift.json"
        _attempt_dir = tmp_path
        _retry_assessment_index = 0

        def current_decision(self):
            return decision

        def prepare_grounded_xy_diagnostic(self, **kwargs):
            events.append(("diagnostic", kwargs["frame_request_id"]))
            report = tmp_path / "xy_retry_assessments/000/report.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"status":"abstain"}', encoding="utf-8")
            self._retry_assessment_index += 1
            return SimpleNamespace(status="abstain")

        def record_xy_retry_unavailable(self, **kwargs):
            events.append(("unavailable", kwargs["reason"]))

    runner = Runner()
    launch = SimpleNamespace(
        camera_ids=("front", "side"), intrinsics_full={},
        extrinsics_full={})
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(launch=launch))
    images = {camera: np.zeros((8, 8, 3), dtype=np.uint8)
              for camera in launch.camera_ids}
    metadata = {
        "request_id": 1, "source": "camera_acquisition",
        "frames": {
            camera: {
                "frame_id": index, "image_sha256": image_sha256(images[camera]),
                "timestamp_s": 5.0, "max_error_s": 0.001,
                "timestamp_method": "hardware_exposure",
                "clock_domain": "unix_utc",
            }
            for index, camera in enumerate(launch.camera_ids, start=1)
        },
    }
    capture = RawCameraCapture(
        "withdrawn_1", 1, images, {"front": 1, "side": 2}, metadata)
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text(json.dumps({"completed_at_s": 4.0}),
                          encoding="utf-8")
    params = dict(
        live=live, planner=object(), backend=object(),
        read_state=State,
        state_for_capture=lambda _: State(),
        withdraw_guarded=lambda *_: (State(), withdrawal),
        capture_withdrawn=lambda *_: capture,
        capture_postshift=lambda *_: None,
        capture_arrival=lambda *_: None,
        capture_preinsert=lambda *_: None,
        capture_final=lambda *_: None,
        lateral_adapter=object(), transfer_adapter=object(),
        insertion_adapter=object(), motion_interlock=lambda: True,
        retry_limits=object(), alignment_limits=object(),
        deviation_bounds=object(), lateral_limits=object(),
        transfer_limits=object(), insertion_limits=object(),
        contact_limits=object(), lateral_commissioning_path=tmp_path,
        transfer_commissioning_path=tmp_path,
        insertion_watchdog_path=tmp_path,
        insertion_contact_path=tmp_path,
        diagnostic_parameters={}, lateral_parameters={},
        lateral_execution_parameters={}, checkpoint_parameters={},
        insertion_preflight_parameters={}, path_handoff_parameters={},
        transfer_execution_parameters={}, arrival_parameters={},
        axial_replan_parameters={}, retry_handoff_parameters={},
        retry_execution_parameters={}, final_parameters={},
        enable_robot_motion=True, decision_time=lambda: 10.0)
    handler = live_retry_handlers.LiveGroundedRetryHandlers(**params)
    handler.assess_and_shift(runner, decision)
    assert events == [("diagnostic", 1), ("unavailable", "vlm_abstain")]
    assert handler._pending is None
    assert set(handler.handlers()) == {
        "guarded_withdrawal_then_xy_assessment",
        "await_retry_execution_and_observation"}
