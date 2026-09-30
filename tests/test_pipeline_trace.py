import csv
import json

import pytest

from autodex.pipeline_edit import build_edit_package
from autodex.pipeline_trace import PipelineTrace
from scripts.render_pipeline_video import (
    _cell_rect,
    build_captions,
    build_stage_intervals,
    build_sync_audit,
    build_time_mapping,
    layout_spec,
    trailing_interruption_trim,
)
from scripts.sync_external_video import fit_time_transform


class TickClock:
    def __init__(self, start, step):
        self.value = start - step
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_run_and_episode_views_share_identical_event_clock(tmp_path):
    monotonic = TickClock(10_000_000_000, 1_000_000)
    wall = TickClock(1_700_000_000_000_000_000, 2_000_000)
    trace = PipelineTrace(
        monotonic_ns=monotonic, wall_time_ns=wall, run_id="run-test")
    run_dir = tmp_path / "run-test"
    trace.bind(run_dir, object="mug")

    episode = trace.scoped(
        episode_id="episode-001", attempt_id="attempt_0001")
    span = episode.begin(
        phase="planning", kind="plan", name="grasp_search")
    episode.event(
        "grasp.selected", phase="planning", kind="decision",
        scene_info=["table", "1", "7"])
    episode.end(span, outcome="success")
    trace.add_episode({
        "episode_id": "episode-001", "attempt_id": "attempt_0001",
        "outcome": "success",
    })
    trace.close(outcome="success")

    run_events = _jsonl(run_dir / "events.jsonl")
    episode_dir = run_dir / "episodes" / "episode-001"
    episode_events = _jsonl(episode_dir / "events.jsonl")
    global_episode_events = [
        event for event in run_events
        if event.get("episode_id") == "episode-001"
    ]

    assert episode_events == global_episode_events
    assert all("pipeline_time_s" in event and "utc_ns" in event
               and "monotonic_ns" in event for event in episode_events)
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    assert manifest["canonical_run_timeline"] == "../../events.jsonl"
    assert manifest["event_count"] == len(episode_events)
    assert (run_dir / "timeline.json").exists()
    assert (run_dir / "edit" / "storyboard.json").exists()
    assert (run_dir / "edit" / "episodes" / "episode-001" /
            "storyboard.json").exists()


def test_edit_package_maps_all_markers_to_external_video(tmp_path):
    trace = PipelineTrace(tmp_path / "run", run_id="edit-test")
    scoped = trace.scoped(episode_id="ep", attempt_id="attempt_0001")
    span = scoped.begin(phase="recovery", kind="motion", name="rotation")
    scoped.end(span, outcome="success")
    trace.add_episode({"episode_id": "ep", "outcome": "success"})
    trace.close()
    sync = {
        "transform": {"slope": 1.0001, "offset_s": 3.25},
    }
    sync_path = tmp_path / "run" / "sync" / "external_video_sync.json"
    sync_path.write_text(json.dumps(sync))

    package = build_edit_package(tmp_path / "run")
    assert package["timebase"]["external_video_mapped"] is True
    assert package["segments"][0]["video_in_s"] is not None
    with (tmp_path / "run" / "edit" / "markers.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["video_time_s"] for row in rows)


def test_sync_transform_recovers_offset_and_drift():
    transform = fit_time_transform([1.0, 101.0], [4.0, 104.01])
    assert transform["slope"] == pytest.approx(1.0001)
    assert transform["offset_s"] == pytest.approx(2.9999)
    assert transform["anchor_residual_rms_s"] == pytest.approx(0.0)


def test_pipeline_video_uses_enter_anchor_and_explicit_speed_not_video_end():
    events = [
        {"name": "initial_object_placement", "edge": "end",
         "pipeline_time_s": 10.0},
        {"name": "pipeline.process_end", "edge": "instant",
         "pipeline_time_s": 110.0},
    ]
    short_video = build_time_mapping(events, speed=5.0)
    long_video = build_time_mapping(events, speed=5.0)

    assert short_video.mode == "fixed_speed"
    assert short_video.playback_speed == 5.0
    assert short_video.anchor_event == "initial_object_placement.end"
    assert short_video.anchor_video_s == 0.0
    assert short_video.video_time(10.0) == pytest.approx(0.0)
    assert short_video.video_time(60.0) == pytest.approx(10.0)
    assert asdict_time_mapping(short_video) == asdict_time_mapping(long_video)
    audit = build_sync_audit(events, short_video, [])
    assert audit["authority"] == "placement_enter_plus_explicit_speed"
    assert audit["anchor_residual_s"] == pytest.approx(0.0)


def test_pipeline_video_refuses_fixed_speed_sync_without_enter_anchor():
    with pytest.raises(ValueError, match="initial_object_placement.end"):
        build_time_mapping([
            {"name": "episode", "edge": "start", "pipeline_time_s": 10.0},
        ], speed=5.0)


def asdict_time_mapping(mapping):
    return (
        mapping.mode, mapping.slope, mapping.offset_s,
        mapping.origin_pipeline_s, mapping.playback_speed,
    )


def test_pipeline_video_trims_only_consecutive_interrupted_tail():
    events = [
        {"name": "initial_object_placement", "edge": "end",
         "pipeline_time_s": 10.0},
        {"name": "episode", "edge": "start", "episode_id": "charuco-failure",
         "pipeline_time_s": 20.0},
        {"name": "grasp.validation_result", "edge": "instant",
         "episode_id": "charuco-failure", "pipeline_time_s": 29.0,
         "outcome": "failure", "attributes": {"success": False,
                                                   "reason": "charuco_fail"}},
        {"name": "episode", "edge": "end", "episode_id": "charuco-failure",
         "pipeline_time_s": 30.0, "outcome": "failure", "attributes": {}},
        {"name": "episode", "edge": "start", "episode_id": "success",
         "pipeline_time_s": 40.0},
        {"name": "episode.result", "edge": "instant", "episode_id": "success",
         "pipeline_time_s": 49.0, "outcome": "success",
         "attributes": {"success": True}},
        {"name": "episode", "edge": "end", "episode_id": "success",
         "pipeline_time_s": 50.0, "outcome": "success", "attributes": {}},
        {"name": "episode", "edge": "start", "episode_id": "tail-1",
         "pipeline_time_s": 60.0},
        {"name": "episode", "edge": "end", "episode_id": "tail-1",
         "pipeline_time_s": 70.0, "outcome": "aborted",
         "attributes": {"reason": "pipeline_closed"}},
        {"name": "episode", "edge": "start", "episode_id": "tail-2",
         "pipeline_time_s": 80.0},
        {"name": "episode", "edge": "end", "episode_id": "tail-2",
         "pipeline_time_s": 90.0, "outcome": "aborted", "attributes": {}},
    ]
    mapping = build_time_mapping(events, speed=5.0)
    trim = trailing_interruption_trim(
        events, mapping, 100.0, min_interruptions=2)

    assert trim["applied"] is True
    assert trim["trimmed_episode_ids"] == ["tail-1", "tail-2"]
    assert trim["cut_pipeline_s"] == 60.0
    assert trim["output_duration_s"] == pytest.approx(10.0)


def test_pipeline_video_layout_is_flush_to_canvas_edges():
    layout = layout_spec(1280, 720, [])
    assert layout["edge_margin_px"] == 0
    assert layout["current"][:2] == [0, 0]
    history = layout["history_column"]
    assert history[1] == 0
    assert history[0] + history[2] == 1280
    assert layout["history_background"] == "white"
    assert layout["pipeline_diagram"][0] == 0
    assert layout["pipeline_diagram"][2] == 1280
    assert history[3] == 4 * history[2]
    assert history[1] + history[3] == layout["pipeline_diagram"][1]
    assert (layout["pipeline_diagram"][1]
            + layout["pipeline_diagram"][3]) == 720
    stack_card = _cell_rect(layout, 0.0)
    assert stack_card[2] == stack_card[3]
    assert _cell_rect(layout, 3.0)[1] + stack_card[3] == history[3]
    assert layout["stack_rows"] == 4


def test_pipeline_video_event_caption_has_minimum_hold():
    events = [
        {"name": "initial_object_placement", "edge": "end",
         "pipeline_time_s": 10.0},
        {"name": "reorientation_recovery", "edge": "start",
         "pipeline_time_s": 20.0, "episode_id": "ep", "attributes": {}},
    ]
    mapping = build_time_mapping(events, speed=5.0)
    captions = build_captions(
        events, mapping, 10.0, minimum_duration_s=1.5)
    recovery = next(item for item in captions
                    if item.text == "CHANGING REST POSE")
    assert recovery.end_s - recovery.start_s == pytest.approx(1.8)


def test_pipeline_video_builds_fixed_bottom_stage_intervals():
    events = [
        {"name": "initial_object_placement", "edge": "end",
         "pipeline_time_s": 10.0},
        {"name": "episode", "edge": "start", "episode_id": "ep",
         "pipeline_time_s": 15.0},
        {"name": "foundpose", "edge": "start", "episode_id": "ep",
         "pipeline_time_s": 20.0},
        {"name": "foundpose", "edge": "end", "episode_id": "ep",
         "pipeline_time_s": 30.0},
        {"name": "episode", "edge": "end", "episode_id": "ep",
         "pipeline_time_s": 60.0},
    ]
    mapping = build_time_mapping(events, speed=5.0)
    intervals = build_stage_intervals(events, [], mapping)

    pose = next(item for item in intervals if item.stage == "POSE EST.")
    rank = next(item for item in intervals if item.stage == "SELECT")
    assert (pose.start_s, pose.end_s) == pytest.approx((2.0, 4.0))
    assert (rank.start_s, rank.end_s) == pytest.approx((4.0, 10.0))


def test_charuco_failure_alone_is_not_trimmed():
    events = [
        {"name": "initial_object_placement", "edge": "end",
         "pipeline_time_s": 10.0},
        {"name": "episode", "edge": "start", "episode_id": "charuco-fail",
         "pipeline_time_s": 20.0},
        {"name": "grasp.validation_result", "edge": "instant",
         "episode_id": "charuco-fail", "pipeline_time_s": 29.0,
         "outcome": "failure", "attributes": {"success": False,
                                                   "reason": "charuco_fail"}},
        {"name": "episode", "edge": "end", "episode_id": "charuco-fail",
         "pipeline_time_s": 30.0, "outcome": "failure", "attributes": {}},
    ]
    mapping = build_time_mapping(events, speed=5.0)
    trim = trailing_interruption_trim(events, mapping, 25.0)

    assert trim["applied"] is False
    assert trim["output_duration_s"] == 25.0
