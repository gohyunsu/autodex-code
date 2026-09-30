#!/usr/bin/env python3
"""Render a run-scoped AutoDex experiment over an external-camera video.

The input video stays full-frame.  The selected grasp occupies the top-left
corner, while all judged executed grasps accumulate chronologically in one
opaque-white column on the right, above a separate full-width pipeline bar.
Every grasp is rendered as a fixed 3-D image from the episode's observed
world-frame object pose, exact wrist transform, and hand configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.pipeline_edit import load_events


SUCCESS_BGR = (76, 205, 116)
FAILURE_BGR = (86, 91, 245)
NEUTRAL_BGR = (220, 220, 220)
TRANSITION_S = 0.48
GRASP_CAMERA_SERIAL = "25322649"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int
    has_audio: bool


@dataclass
class TimeMapping:
    mode: str
    slope: float
    offset_s: float
    origin_pipeline_s: float
    playback_speed: Optional[float] = None
    anchor_event: str = ""
    anchor_video_s: float = 0.0

    def video_time(self, pipeline_time_s: float) -> float:
        return self.slope * float(pipeline_time_s) + self.offset_s


@dataclass
class GraspAttempt:
    episode_id: str
    attempt_id: Optional[str]
    scene_info: tuple[str, str, str]
    selected_pipeline_s: float
    selected_video_s: float
    result_pipeline_s: Optional[float]
    result_video_s: Optional[float]
    success: Optional[bool]
    reason: Optional[str]
    result_source: str
    episode_dir: Path
    render_path: Path
    object_world_pose: Optional[tuple[float, ...]]
    states: list[tuple[float, str]]

    def metadata(self, run_dir: Path) -> dict[str, Any]:
        value = asdict(self)
        value["scene_info"] = list(self.scene_info)
        for key in ("episode_dir", "render_path"):
            path = Path(value[key])
            try:
                value[key] = path.relative_to(run_dir).as_posix()
            except ValueError:
                value[key] = str(path)
        return value


@dataclass
class Caption:
    start_s: float
    end_s: float
    text: str
    priority: int
    color: tuple[int, int, int] = NEUTRAL_BGR


@dataclass
class StageInterval:
    stage: str
    start_s: float
    end_s: float
    episode_id: str


PIPELINE_STAGES = (
    "POSE EST.", "SELECT", "APPROACH", "GRASP",
    "LIFT & HOLD", "LABEL", "PLACE", "RESET",
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def resolve_run_dir(experiment: str | Path, run_id: Optional[str] = None) -> Path:
    """Resolve either a concrete run or an experiment directory to one run."""
    root = Path(experiment).expanduser().resolve()
    if ((root / "manifest.json").is_file()
            and ((root / "events.jsonl").is_file()
                 or (root / "timeline.json").is_file())):
        if run_id and root.name != run_id:
            raise FileNotFoundError(f"run {run_id!r} is not {root}")
        return root
    candidates = []
    for manifest_path in root.rglob("_pipeline_runs/*/manifest.json"):
        run = manifest_path.parent
        if run_id is None or run.name == run_id:
            manifest = _read_json(manifest_path, {}) or {}
            candidates.append((int(manifest.get("started_utc_ns", 0)), run))
    if not candidates:
        suffix = f" with id {run_id!r}" if run_id else ""
        raise FileNotFoundError(f"no pipeline run found under {root}{suffix}")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next(item for item in streams if item.get("codec_type") == "video")
    numerator, denominator = str(video.get("avg_frame_rate", "0/1")).split("/")
    fps = float(numerator) / max(float(denominator), 1.0)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    frame_count = int(video.get("nb_frames") or round(duration * fps))
    return VideoInfo(
        width=int(video["width"]), height=int(video["height"]), fps=fps,
        duration_s=duration, frame_count=frame_count,
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )


def _initial_placement_origin(events: list[dict[str, Any]]) -> float:
    placement_ends = [
        float(event["pipeline_time_s"]) for event in events
        if event.get("name") == "initial_object_placement"
        and event.get("edge") == "end"
    ]
    if placement_ends:
        return placement_ends[-1]
    raise ValueError(
        "fixed-speed synchronization requires initial_object_placement.end; "
        "provide an explicit --sync transform for traces without the Enter anchor")


def build_time_mapping(
    events: list[dict[str, Any]], *, sync: Optional[dict[str, Any]] = None,
    speed: float = 5.0,
    video_offset_s: float = 0.0,
) -> TimeMapping:
    """Map canonical pipeline timestamps to timestamps in the input video."""
    if sync and isinstance(sync.get("transform"), dict):
        transform = sync["transform"]
        return TimeMapping(
            mode="external_sync", slope=float(transform["slope"]),
            offset_s=float(transform["offset_s"]), origin_pipeline_s=0.0,
            anchor_event="external_sync.transform", anchor_video_s=0.0,
        )
    origin = _initial_placement_origin(events)
    # Deliberately do not infer speed from either endpoint: an external camera
    # often keeps recording after the pipeline stops.  Only the Enter anchor
    # and the explicitly supplied playback rate are timing authorities.
    playback_speed = float(speed)
    if playback_speed <= 0:
        raise ValueError("--speed must be positive")
    mode = "fixed_speed"
    slope = 1.0 / playback_speed
    return TimeMapping(
        mode=mode, slope=slope,
        offset_s=video_offset_s - slope * origin,
        origin_pipeline_s=origin, playback_speed=playback_speed,
        anchor_event="initial_object_placement.end",
        anchor_video_s=float(video_offset_s),
    )


def build_sync_audit(
    events: list[dict[str, Any]], mapping: TimeMapping,
    attempts: list[GraspAttempt],
) -> dict[str, Any]:
    """Persist the timing authority and mapped checkpoints for edit review."""
    placement = next((
        event for event in reversed(events)
        if event.get("name") == "initial_object_placement"
        and event.get("edge") == "end"
    ), None)
    checkpoints = []
    for attempt in attempts:
        checkpoints.append({
            "attempt_id": attempt.attempt_id,
            "event": "grasp.selected",
            "pipeline_time_s": attempt.selected_pipeline_s,
            "video_time_s": attempt.selected_video_s,
        })
        if attempt.result_pipeline_s is not None:
            checkpoints.append({
                "attempt_id": attempt.attempt_id,
                "event": attempt.result_source,
                "pipeline_time_s": attempt.result_pipeline_s,
                "video_time_s": attempt.result_video_s,
            })
    audit = {
        "authority": ("external_sync_transform" if mapping.mode == "external_sync"
                      else "placement_enter_plus_explicit_speed"),
        "anchor_event": mapping.anchor_event,
        "anchor_pipeline_s": mapping.origin_pipeline_s,
        "anchor_video_s": mapping.anchor_video_s,
        "anchor_utc": placement.get("utc") if placement else None,
        "playback_speed": mapping.playback_speed,
        "equation": "video_s = slope * pipeline_s + offset_s",
        "checkpoints": checkpoints,
    }
    if mapping.mode == "fixed_speed":
        anchor_error = abs(
            mapping.video_time(mapping.origin_pipeline_s) - mapping.anchor_video_s)
        if anchor_error > 1e-9:
            raise RuntimeError(f"Enter-anchor mapping residual is {anchor_error:.9f}s")
        audit["anchor_residual_s"] = anchor_error
    return audit


def trailing_interruption_trim(
    events: list[dict[str, Any]], mapping: TimeMapping, video_duration_s: float,
    *, min_interruptions: int = 1,
) -> dict[str, Any]:
    """Return a cut point that removes only trailing interrupted episodes.

    Grasp/Charuco failures are completed trials and deliberately remain in the
    video.  Only an aborted/pipeline-closed/exception-interrupted episode is
    eligible.  The reverse scan stops at the first normally completed episode,
    so an interruption in the middle of otherwise useful footage is retained.
    """
    episodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        if event.get("name") != "episode" or not event.get("episode_id"):
            continue
        episode_id = str(event["episode_id"])
        record = episodes.setdefault(episode_id, {
            "episode_id": episode_id, "start_pipeline_s": None,
            "end_pipeline_s": None, "interrupted": False, "reason": None,
        })
        if episode_id not in order:
            order.append(episode_id)
        if event.get("edge") == "start":
            record["start_pipeline_s"] = float(event["pipeline_time_s"])
        elif event.get("edge") == "end":
            record["end_pipeline_s"] = float(event["pipeline_time_s"])
            attributes = event.get("attributes") or {}
            reason = str(attributes.get("reason") or "")
            if (event.get("outcome") == "aborted"
                    or reason in {"pipeline_closed", "keyboard_interrupt"}
                    or "interrupt" in reason or "exception" in reason):
                record["interrupted"] = True
                record["reason"] = reason or event.get("outcome")
    for event in events:
        episode_id = event.get("episode_id")
        if episode_id is None or str(episode_id) not in episodes:
            continue
        record = episodes[str(episode_id)]
        if event.get("name") == "episode.result":
            attrs = event.get("attributes") or {}
            reason = str(attrs.get("reason") or "")
            if (event.get("outcome") == "aborted"
                    or reason in {"pipeline_closed", "keyboard_interrupt"}
                    or "interrupt" in reason or "exception" in reason):
                record["interrupted"] = True
                record["reason"] = reason or event.get("outcome")
        name = str(event.get("name") or "")
        if "unhandled_exception" in name or "keyboard_interrupt" in name:
            record["interrupted"] = True
            record["reason"] = name
    trailing = []
    for episode_id in reversed(order):
        record = episodes[episode_id]
        if not record["interrupted"]:
            break
        trailing.append(record)
    trailing.reverse()
    if len(trailing) < max(1, int(min_interruptions)):
        return {
            "applied": False, "output_duration_s": video_duration_s,
            "trimmed_episode_ids": [], "cut_pipeline_s": None,
            "cut_video_s": None,
        }
    first_start = trailing[0].get("start_pipeline_s")
    if first_start is None:
        return {
            "applied": False, "output_duration_s": video_duration_s,
            "trimmed_episode_ids": [], "cut_pipeline_s": None,
            "cut_video_s": None,
        }
    cut_video_s = min(video_duration_s, max(0.0, mapping.video_time(first_start)))
    return {
        "applied": True, "output_duration_s": cut_video_s,
        "trimmed_episode_ids": [item["episode_id"] for item in trailing],
        "cut_pipeline_s": first_start, "cut_video_s": cut_video_s,
    }


def _episode_directories(run_dir: Path) -> dict[str, Path]:
    records = (_read_json(run_dir / "episodes.json", {}) or {}).get("episodes", [])
    directories: dict[str, Path] = {}
    for record in records:
        episode_id = str(record.get("episode_id"))
        direct = run_dir.parent.parent / episode_id
        if direct.is_dir():
            directories[episode_id] = direct
            continue
        relative = record.get("directory")
        if relative:
            for ancestor in run_dir.parents:
                if ancestor.name == "experiment":
                    candidate = ancestor.parent / str(relative)
                    if candidate.is_dir():
                        directories[episode_id] = candidate
                    break
    return directories


def _event_success(event: Optional[dict[str, Any]]) -> Optional[bool]:
    if event is None:
        return None
    value = (event.get("attributes") or {}).get("success")
    return value if isinstance(value, bool) else None


def parse_grasp_attempts(
    run_dir: Path, events: list[dict[str, Any]], mapping: TimeMapping, *,
    camera_serial: str = GRASP_CAMERA_SERIAL,
) -> list[GraspAttempt]:
    """Build collection-grasp state transitions from the run timeline."""
    episode_dirs = _episode_directories(run_dir)
    cache_dir = run_dir / "edit" / "grasp_pose_renders"
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        episode_id = event.get("episode_id")
        if episode_id is not None:
            by_episode.setdefault(str(episode_id), []).append(event)

    attempts = []
    for selected in events:
        if selected.get("name") != "grasp.selected":
            continue
        episode_id = str(selected.get("episode_id"))
        attributes = selected.get("attributes") or {}
        scene = attributes.get("scene_info")
        if not isinstance(scene, list) or len(scene) != 3:
            continue
        episode_events = by_episode.get(episode_id, [])
        selected_time = float(selected["pipeline_time_s"])
        exact = next((
            event for event in episode_events
            if event.get("name") == "grasp.validation_result"
            and float(event.get("pipeline_time_s", -1)) >= selected_time
        ), None)
        execution = next((
            event for event in episode_events
            if event.get("name") == "grasp.execution_result"
            and float(event.get("pipeline_time_s", -1)) >= selected_time
        ), None)
        episode_result = next((
            event for event in episode_events
            if event.get("name") == "episode.result"
            and float(event.get("pipeline_time_s", -1)) >= selected_time
        ), None)
        result_event = exact or execution or episode_result
        success = _event_success(result_event)
        reason = ((result_event.get("attributes") or {}).get("reason")
                  if result_event else None)
        result_time: Optional[float] = None
        result_source = "incomplete"
        if exact is not None:
            result_time = float(exact["pipeline_time_s"])
            result_source = "grasp.validation_result"
        elif result_event is not None:
            result_time = float(result_event["pipeline_time_s"])
            result_source = str(result_event.get("name"))
            # Old traces ended the episode after reset.  A successful
            # lift-time Charuco result necessarily exists before placement
            # planning begins, so that first post-lift event is a tighter and
            # visually useful upper bound than episode.result.
            if success is True:
                place_plan = next((
                    event for event in episode_events
                    if event.get("name") == "endpoint_approximation"
                    and event.get("edge") == "start"
                    and selected_time < float(event.get("pipeline_time_s", -1))
                    < result_time
                ), None)
                if place_plan is not None:
                    result_time = float(place_plan["pipeline_time_s"])
                    result_source = "inferred_before_endpoint_approximation"

        states = [
            (mapping.video_time(float(event["pipeline_time_s"])),
             str((event.get("attributes") or {}).get("state")))
            for event in episode_events if event.get("name") == "robot.state"
        ]
        states.extend(
            (mapping.video_time(float(event["pipeline_time_s"])), "validation")
            for event in episode_events
            if (event.get("name") == "grasp.validation_started"
                or (event.get("name") == "pickup_and_lift"
                    and event.get("edge") == "end"))
        )
        states.sort(key=lambda item: item[0])
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id)
        safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera_serial)
        scene_payload = _read_json(
            run_dir / "artifacts" / "scene" / f"{selected.get('attempt_id')}.json",
            {}) or {}
        object_pose = (((scene_payload.get("mesh") or {}).get("target") or {})
                       .get("pose"))
        object_world_pose = (
            tuple(float(value) for value in object_pose)
            if isinstance(object_pose, list) and len(object_pose) == 7 else None)
        attempts.append(GraspAttempt(
            episode_id=episode_id,
            attempt_id=selected.get("attempt_id"),
            scene_info=(str(scene[0]), str(scene[1]), str(scene[2])),
            selected_pipeline_s=selected_time,
            selected_video_s=mapping.video_time(selected_time),
            result_pipeline_s=result_time,
            result_video_s=(mapping.video_time(result_time)
                            if result_time is not None else None),
            success=success, reason=reason, result_source=result_source,
            episode_dir=episode_dirs.get(
                episode_id, run_dir.parent.parent / episode_id),
            render_path=cache_dir / f"{safe_episode}_cam{safe_camera}.png",
            object_world_pose=object_world_pose,
            states=states,
        ))
    return sorted(attempts, key=lambda item: item.selected_pipeline_s)


def _render_python(explicit: Optional[str]) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    if importlib.util.find_spec("open3d") is not None:
        return sys.executable
    known = Path("/home/robot/anaconda3/envs/autodex/bin/python")
    if known.is_file():
        return str(known)
    raise RuntimeError(
        "Open3D is unavailable; pass --render-python for the AutoDex environment")


def render_grasp_pose_images(
    attempts: list[GraspAttempt], manifest: dict[str, Any], *,
    python_executable: Optional[str], size: int,
    camera_serial: str = GRASP_CAMERA_SERIAL,
    force: bool = False,
) -> None:
    renderer_python = _render_python(python_executable)
    hand = str(manifest.get("hand") or "allegro")
    version = str((manifest.get("arguments") or {}).get("grasp_version") or "v8")
    obj = str(manifest.get("object"))
    renderer = REPO_ROOT / "src" / "visualization" / "turntable_grasp.py"
    for index, attempt in enumerate(attempts, start=1):
        output = attempt.render_path
        if output.is_file() and output.stat().st_size > 0 and not force:
            print(f"[pose render {index}/{len(attempts)}] cached {output.name}")
            continue
        if attempt.object_world_pose is None:
            raise RuntimeError(
                f"missing planning object pose for {attempt.attempt_id}; "
                "refusing to substitute a catalog pose")
        cam_param_dir = attempt.episode_dir / "cam_param"
        intrinsics_path = cam_param_dir / "intrinsics.json"
        extrinsics_path = cam_param_dir / "extrinsics.json"
        c2r_path = attempt.episode_dir / "C2R.npy"
        for required in (intrinsics_path, extrinsics_path, c2r_path):
            if not required.is_file():
                raise RuntimeError(
                    f"camera {camera_serial} calibration input missing: {required}")
        intrinsics = _read_json(intrinsics_path, {}) or {}
        extrinsics = _read_json(extrinsics_path, {}) or {}
        if camera_serial not in intrinsics or camera_serial not in extrinsics:
            raise RuntimeError(
                f"camera {camera_serial} is absent from {cam_param_dir}")
        output.parent.mkdir(parents=True, exist_ok=True)
        scene = "/".join(attempt.scene_info)
        command = [
            renderer_python, str(renderer), "--hand", hand,
            "--version", version, "--obj", obj, "--scene", scene,
            "--width", str(size), "--height", str(size),
            "--padding", "1.18", "--still", "--no-object-texture",
            "--camera-serial", camera_serial,
            "--cam-param-dir", str(cam_param_dir), "--c2r", str(c2r_path),
            "--output", str(output),
            "--object-world-pose-json",
            json.dumps(list(attempt.object_world_pose), separators=(",", ":")),
        ]
        plan = attempt.episode_dir / "plan"
        exact_local = plan / "wrist_obj_local.npy"
        exact_world = plan / "wrist_se3.npy"
        grasp_pose = plan / "grasp_pose.npy"
        if exact_local.is_file():
            command += ["--wrist-se3", str(exact_local)]
        elif exact_world.is_file():
            command += ["--wrist-world-se3", str(exact_world)]
        if grasp_pose.is_file():
            command += ["--grasp-pose", str(grasp_pose)]
        print(f"[pose render {index}/{len(attempts)}] {scene}")
        environment = os.environ.copy()
        environment.setdefault("EGL_PLATFORM", "surfaceless")
        environment.setdefault("OPEN3D_CPU_RENDERING", "true")
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"fixed pose render failed for {scene}:\n{completed.stdout}")
        print(completed.stdout.strip())


def build_captions(
    events: list[dict[str, Any]], mapping: TimeMapping, duration_s: float,
    *, minimum_duration_s: float = 1.5,
) -> list[Caption]:
    """Build full-frame captions for recovery actions only."""
    captions: list[Caption] = []

    def add(pipeline_s: float, text: str, duration: float, priority: int,
            color: tuple[int, int, int] = NEUTRAL_BGR) -> None:
        start = mapping.video_time(pipeline_s)
        duration = max(float(duration), float(minimum_duration_s))
        if start < duration_s and start + duration > 0:
            captions.append(Caption(
                max(0.0, start), min(duration_s, start + duration),
                text, priority, color))

    for event in events:
        name = str(event.get("name"))
        edge = event.get("edge")
        pipeline_s = float(event.get("pipeline_time_s", 0.0))
        if name == "reorientation_recovery" and edge == "start":
            add(pipeline_s, "CHANGING REST POSE", 1.8, 90)
        elif name == "rotation_pick_and_lift" and edge == "start":
            add(pipeline_s, "ROTATING OBJECT", 1.8, 90)
    return captions


def build_stage_intervals(
    events: list[dict[str, Any]], attempts: list[GraspAttempt],
    mapping: TimeMapping,
) -> list[StageInterval]:
    """Build the neutral bottom-diagram state from major pipeline events."""
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("episode_id") is not None:
            by_episode.setdefault(str(event["episode_id"]), []).append(event)
    attempt_by_episode = {item.episode_id: item for item in attempts}
    intervals: list[StageInterval] = []

    def add(stage: str, start: Optional[float], end: Optional[float],
            episode_id: str) -> None:
        if start is None or end is None or end <= start:
            return
        intervals.append(StageInterval(stage, start, end, episode_id))

    for episode_id, episode_events in by_episode.items():
        episode_events.sort(key=lambda item: float(item.get("pipeline_time_s", 0.0)))

        def event_time(name: str, edge: Optional[str] = None) -> Optional[float]:
            match = next((
                event for event in episode_events
                if event.get("name") == name
                and (edge is None or event.get("edge") == edge)
            ), None)
            return (mapping.video_time(float(match["pipeline_time_s"]))
                    if match is not None else None)

        episode_end = event_time("episode", "end")
        pose_start = event_time("foundpose", "start")
        pose_end = event_time("foundpose", "end")
        add("POSE EST.", pose_start, pose_end, episode_id)

        attempt = attempt_by_episode.get(episode_id)
        recovery_start = event_time("reorientation_recovery", "start")
        if recovery_start is None:
            recovery_start = event_time("rotation_pick_and_lift", "start")
        rank_end = (attempt.selected_video_s if attempt is not None
                    else recovery_start or episode_end)
        add("SELECT", pose_end, rank_end, episode_id)
        if attempt is None:
            continue

        state_times: dict[str, float] = {}
        for timestamp, state in attempt.states:
            state_times.setdefault(state, timestamp)

        def first_state(*names: str) -> Optional[float]:
            values = [state_times[name] for name in names if name in state_times]
            return min(values) if values else None

        grasp_start = first_state("grasp", "squeeze")
        lift_start = first_state("lift")
        validation_start = first_state("validation")
        result_time = attempt.result_video_s
        reset_start = first_state("reset_preflight", "post_release_clearance",
                                  "arm_retract")

        add("APPROACH", attempt.selected_video_s,
            grasp_start or lift_start or validation_start or result_time or episode_end,
            episode_id)
        add("GRASP", grasp_start,
            lift_start or validation_start or result_time or episode_end,
            episode_id)
        add("LIFT & HOLD", lift_start,
            validation_start or result_time or episode_end, episode_id)
        add("LABEL", validation_start, result_time or episode_end, episode_id)
        add("PLACE", result_time,
            reset_start or episode_end, episode_id)
        add("RESET", reset_start, episode_end, episode_id)
    return sorted(intervals, key=lambda item: (item.start_s, item.end_s))


def layout_spec(
    width: int, height: int, attempts: list[GraspAttempt], *,
    show_pipeline_diagram: bool = True,
) -> dict[str, Any]:
    if show_pipeline_diagram:
        target_diagram_height = max(48, int(round(height * 0.09)))
        # Four square cards fill the white column exactly. Any rounding
        # remainder belongs to the separate pipeline bar.
        rail_width = max(2, min(
            int(round(width * 0.13)),
            max(2, (height - target_diagram_height) // 4)))
        stack_height = 4 * rail_width
        diagram_height = height - stack_height
    else:
        if height % 4:
            raise ValueError(
                "timeline-free layout requires a height divisible by four")
        rail_width = height // 4
        stack_height = height
        diagram_height = 0
    rail_x = width - rail_width
    current_size = min(int(round(width * 0.30)), int(round(height * 0.54)), rail_x)
    return {
        "canvas": [width, height],
        "current": [0, 0, current_size, current_size],
        "history_column": [rail_x, 0, rail_width, stack_height],
        "history_background": "white",
        "pipeline_diagram": [0, stack_height, width, diagram_height],
        "pipeline_diagram_enabled": show_pipeline_diagram,
        "stack_rows": 4,
        "edge_margin_px": 0,
    }


class PoseRender:
    def __init__(self, path: Path, cv2_module) -> None:
        self.cv2 = cv2_module
        frame = self.cv2.imread(str(path), self.cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot open fixed grasp pose render: {path}")
        self.image = _white_render_background(frame, self.cv2)

    def frame(self):
        return self.image


def _white_render_background(frame, cv2_module):
    """Map the renderer's border-connected near-white matte to true white."""
    channel_min = frame.min(axis=2)
    channel_max = frame.max(axis=2)
    candidate = ((channel_min >= 218) &
                 ((channel_max - channel_min) <= 10)).astype("uint8")
    _, labels = cv2_module.connectedComponents(candidate, connectivity=4)
    border_labels = set(labels[0, :]) | set(labels[-1, :])
    border_labels |= set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)
    if border_labels:
        mask = None
        for label in border_labels:
            component = labels == label
            mask = component if mask is None else (mask | component)
        frame[mask] = 255
    return frame


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _rect_lerp(a, b, progress: float):
    p = _ease(progress)
    return tuple(int(round(x + (y - x) * p)) for x, y in zip(a, b))


def _cell_rect(layout: dict[str, Any], row: float):
    column = layout["history_column"]
    x, _, width, _ = column
    y0 = int(round(row * width))
    return (x, y0, width, width)


def _cover(image, width: int, height: int, cv2_module):
    source_h, source_w = image.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, int(math.ceil(source_w * scale)))
    resized_h = max(height, int(math.ceil(source_h * scale)))
    resized = cv2_module.resize(
        image, (resized_w, resized_h), interpolation=cv2_module.INTER_AREA)
    x = (resized_w - width) // 2
    y = (resized_h - height) // 2
    return resized[y:y + height, x:x + width]


def _paste(frame, image, rect, cv2_module, alpha: float = 1.0) -> None:
    x, y, width, height = rect
    if width <= 0 or height <= 0 or alpha <= 0:
        return
    card = _cover(image, width, height, cv2_module)
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + width), min(frame_h, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    source = card[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha >= 0.999:
        frame[y0:y1, x0:x1] = source
    else:
        frame[y0:y1, x0:x1] = cv2_module.addWeighted(
            source, alpha, frame[y0:y1, x0:x1], 1.0 - alpha, 0)


def _blend_color(frame, rect, color, alpha: float, cv2_module) -> None:
    x, y, width, height = rect
    roi = frame[y:y + height, x:x + width]
    if roi.size == 0:
        return
    overlay = roi.copy()
    overlay[:] = color
    frame[y:y + height, x:x + width] = cv2_module.addWeighted(
        overlay, alpha, roi, 1.0 - alpha, 0)


def _draw_outcome_mark(frame, rect, success: Optional[bool], cv2_module) -> None:
    x, y, width, height = rect
    color = SUCCESS_BGR if success is True else FAILURE_BGR if success is False else NEUTRAL_BGR
    center = (x + width // 2, y + height // 2)
    radius = max(14, min(width, height) // 10)
    cv2_module.circle(frame, center, radius, (15, 15, 15), -1,
                      lineType=cv2_module.LINE_AA)
    thickness = max(3, radius // 5)
    if success is True:
        cv2_module.line(frame, (center[0] - radius // 2, center[1]),
                        (center[0] - radius // 8, center[1] + radius // 3),
                        color, thickness, cv2_module.LINE_AA)
        cv2_module.line(frame, (center[0] - radius // 8, center[1] + radius // 3),
                        (center[0] + radius // 2, center[1] - radius // 3),
                        color, thickness, cv2_module.LINE_AA)
    elif success is False:
        delta = radius // 3
        cv2_module.line(frame, (center[0] - delta, center[1] - delta),
                        (center[0] + delta, center[1] + delta),
                        color, thickness, cv2_module.LINE_AA)
        cv2_module.line(frame, (center[0] + delta, center[1] - delta),
                        (center[0] - delta, center[1] + delta),
                        color, thickness, cv2_module.LINE_AA)
    else:
        cv2_module.putText(frame, "?", (center[0] - radius // 3,
                            center[1] + radius // 2), cv2_module.FONT_HERSHEY_DUPLEX,
                           radius / 24.0, color, thickness, cv2_module.LINE_AA)


def _draw_outcome_badge(frame, rect, success: Optional[bool], cv2_module) -> None:
    """Draw a compact emoji-like result badge at the card's top-right."""
    if success is None:
        return
    x, y, width, height = rect
    radius = max(11, min(width, height) // 10)
    margin = max(5, radius // 3)
    center = (x + width - margin - radius, y + margin + radius)
    color = SUCCESS_BGR if success else FAILURE_BGR
    cv2_module.circle(frame, center, radius, color, -1,
                      lineType=cv2_module.LINE_AA)
    thickness = max(3, radius // 4)
    if success:
        cv2_module.line(frame, (center[0] - radius // 2, center[1]),
                        (center[0] - radius // 8, center[1] + radius // 3),
                        (255, 255, 255), thickness, cv2_module.LINE_AA)
        cv2_module.line(frame, (center[0] - radius // 8, center[1] + radius // 3),
                        (center[0] + radius // 2, center[1] - radius // 3),
                        (255, 255, 255), thickness, cv2_module.LINE_AA)
    else:
        delta = radius // 3
        cv2_module.line(frame, (center[0] - delta, center[1] - delta),
                        (center[0] + delta, center[1] + delta),
                        (255, 255, 255), thickness, cv2_module.LINE_AA)
        cv2_module.line(frame, (center[0] + delta, center[1] - delta),
                        (center[0] - delta, center[1] + delta),
                        (255, 255, 255), thickness, cv2_module.LINE_AA)


def _draw_outcome_indicator(
    frame, rect, success: Optional[bool], layout: dict[str, Any], cv2_module,
) -> None:
    if layout.get("outcome_style", "bar") == "emoji":
        _draw_outcome_badge(frame, rect, success, cv2_module)
        return
    edge = max(5, rect[2] // 28)
    _blend_color(frame, (rect[0], rect[1], edge, rect[3]),
                 SUCCESS_BGR if success else FAILURE_BGR, 0.92, cv2_module)


def _draw_stack(
    frame, video_time_s: float, attempts: list[GraspAttempt],
    renders: dict[str, PoseRender], layout: dict[str, Any], cv2_module,
) -> Optional[GraspAttempt]:
    arrivals = [
        item for item in attempts
        if item.success is not None and item.result_video_s is not None
        and item.result_video_s <= video_time_s
    ]
    arrivals.sort(key=lambda item: float(item.result_video_s or 0.0))
    transition = next((
        item for item in reversed(arrivals)
        if video_time_s < float(item.result_video_s) + TRANSITION_S
    ), None)
    if transition is not None:
        progress = ((video_time_s - float(transition.result_video_s))
                    / TRANSITION_S)
        previous = [item for item in arrivals if item is not transition]
        previous.reverse()
        for row, item in enumerate(previous):
            moving_row = row + _ease(progress)
            if moving_row >= layout["stack_rows"]:
                continue
            rect = _cell_rect(layout, moving_row)
            image = renders[item.episode_id].frame()
            _paste(frame, image, rect, cv2_module)
            _draw_outcome_indicator(
                frame, rect, item.success, layout, cv2_module)
        target = _cell_rect(layout, 0.0)
        rect = _rect_lerp(layout["current"], target, progress)
        image = renders[transition.episode_id].frame()
        _paste(frame, image, rect, cv2_module)
        _draw_outcome_indicator(
            frame, rect, transition.success, layout, cv2_module)
        if layout.get("outcome_style", "bar") == "bar" and progress < 0.55:
            _draw_outcome_mark(frame, rect, transition.success, cv2_module)
        return transition

    arrivals.reverse()
    rows = int(layout["stack_rows"])
    # The history is a bounded FIFO view: newest at the top, at most four
    # cards. Older cards simply move past the bottom edge and disappear.
    for row, item in enumerate(arrivals[:rows]):
        rect = _cell_rect(layout, float(row))
        image = renders[item.episode_id].frame()
        _paste(frame, image, rect, cv2_module)
        _draw_outcome_indicator(
            frame, rect, item.success, layout, cv2_module)
    return None


def _draw_current(
    frame, video_time_s: float, attempts: list[GraspAttempt],
    renders: dict[str, PoseRender], layout: dict[str, Any], transitioning: set[str],
    cv2_module,
) -> None:
    active = None
    for item in attempts:
        if item.selected_video_s > video_time_s:
            continue
        result = item.result_video_s
        if result is None or video_time_s < result:
            active = item
    if active is None or active.episode_id in transitioning:
        return
    elapsed = video_time_s - active.selected_video_s
    appear = min(1.0, max(0.0, elapsed / 0.18))
    base = layout["current"]
    scale = 0.92 + 0.08 * _ease(appear)
    rect = (base[0], base[1], int(base[2] * scale), int(base[3] * scale))
    image = renders[active.episode_id].frame()
    _paste(frame, image, rect, cv2_module, alpha=appear)


def _active_stage(
    intervals: list[StageInterval], video_time_s: float,
) -> Optional[str]:
    active = [item for item in intervals
              if item.start_s <= video_time_s < item.end_s]
    if not active:
        return None
    return max(active, key=lambda item: item.start_s).stage


def _draw_pipeline_diagram(
    frame, active_stage: Optional[str], layout: dict[str, Any], cv2_module,
) -> None:
    """Draw a fixed, neutral pipeline diagram along the bottom edge."""
    if not layout.get("pipeline_diagram_enabled", True):
        return
    height, _ = frame.shape[:2]
    x0, y0, diagram_width, diagram_height = layout["pipeline_diagram"]
    _blend_color(
        frame, (x0, y0, diagram_width, diagram_height),
        (4, 4, 4), 0.74, cv2_module)

    cell_width = diagram_width / len(PIPELINE_STAGES)
    node_y = height - max(12, diagram_height // 5)
    first_x = x0 + int(round(cell_width * 0.5))
    last_x = x0 + int(round(diagram_width - cell_width * 0.5))
    cv2_module.line(frame, (first_x, node_y), (last_x, node_y),
                    (88, 88, 88), 2, cv2_module.LINE_AA)
    font = cv2_module.FONT_HERSHEY_DUPLEX
    font_scale = max(0.32, min(0.48, diagram_width / 2500.0))
    for index, stage in enumerate(PIPELINE_STAGES):
        center_x = x0 + int(round((index + 0.5) * cell_width))
        selected = stage == active_stage
        node_color = (245, 245, 245) if selected else (105, 105, 105)
        text_color = (248, 248, 248) if selected else (130, 130, 130)
        cv2_module.circle(
            frame, (center_x, node_y), 6 if selected else 3,
            node_color, -1, lineType=cv2_module.LINE_AA)
        (text_width, text_height), _ = cv2_module.getTextSize(
            stage, font, font_scale, 1)
        text_y = y0 + max(text_height + 5, diagram_height // 2)
        cv2_module.putText(
            frame, stage, (center_x - text_width // 2, text_y),
            font, font_scale, text_color, 1, cv2_module.LINE_AA)


def _draw_caption(
    frame, caption: Caption, cv2_module, *,
    caption_font_path: Optional[Path] = None,
) -> None:
    height, width = frame.shape[:2]
    _blend_color(frame, (0, 0, width, height), (0, 0, 0), 0.62, cv2_module)
    text_color = (caption.color if caption.color in {SUCCESS_BGR, FAILURE_BGR}
                  else (248, 248, 248))
    if caption_font_path is not None:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        font_size = max(28, int(round(height * 0.075)))
        rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        while True:
            font = ImageFont.truetype(str(caption_font_path), font_size)
            bounds = draw.textbbox((0, 0), caption.text, font=font)
            text_w = bounds[2] - bounds[0]
            text_h = bounds[3] - bounds[1]
            if ((text_w <= width * 0.88 and text_h <= height * 0.16)
                    or font_size <= 24):
                break
            font_size -= 2
        x = (width - text_w) // 2 - bounds[0]
        y = (height - text_h) // 2 - bounds[1]
        draw.text(
            (x, y), caption.text, font=font,
            fill=tuple(reversed(text_color)))
        frame[:] = cv2_module.cvtColor(
            np.asarray(image), cv2_module.COLOR_RGB2BGR)
        return

    font = cv2_module.FONT_HERSHEY_DUPLEX
    scale = max(0.7, height / 720.0 * 1.65)
    thickness = max(2, int(round(height / 240)))
    while scale > 0.5:
        (text_w, text_h), _ = cv2_module.getTextSize(
            caption.text, font, scale, thickness)
        if text_w <= width * 0.88 and text_h <= height * 0.16:
            break
        scale -= 0.08
    x = (width - text_w) // 2
    y = (height + text_h) // 2
    cv2_module.putText(frame, caption.text, (x, y), font, scale,
                       text_color, thickness, cv2_module.LINE_AA)


def compose_video(
    video_path: Path, output_path: Path, video_info: VideoInfo,
    attempts: list[GraspAttempt], captions: list[Caption],
    stage_intervals: list[StageInterval], layout: dict[str, Any], *,
    output_duration_s: float, crf: int, preset: str,
    caption_font_path: Optional[Path] = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "composition requires opencv-python; run with the AutoDex environment") from exc

    renders = {item.episode_id: PoseRender(item.render_path, cv2)
               for item in attempts}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.rendering.mp4")
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{video_info.width}x{video_info.height}",
        "-r", f"{video_info.fps:.8f}", "-i", "pipe:0",
        "-i", str(video_path), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-shortest", str(temporary),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_index = 0
    report_every = max(1, video_info.frame_count // 20)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = frame_index / video_info.fps
            if t >= output_duration_s:
                break
            _blend_color(
                frame, layout["history_column"], (255, 255, 255), 1.0, cv2)
            transition = _draw_stack(
                frame, t, attempts, renders, layout, cv2)
            transitioning = ({transition.episode_id}
                             if transition is not None else set())
            _draw_current(frame, t, attempts, renders, layout, transitioning, cv2)
            _draw_pipeline_diagram(
                frame, _active_stage(stage_intervals, t), layout, cv2)
            active = [item for item in captions if item.start_s <= t < item.end_s]
            if active:
                chosen = max(active, key=lambda item: (item.priority, item.start_s))
                _draw_caption(
                    frame, chosen, cv2,
                    caption_font_path=caption_font_path)
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
            if frame_index % report_every == 0:
                print(f"[compose] {100 * frame_index / max(1, video_info.frame_count):5.1f}%")
    except BrokenPipeError as exc:
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        raise RuntimeError(f"ffmpeg encoder stopped: {stderr}") from exc
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg encode failed ({return_code}): {stderr}")
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an external-camera pipeline video from a run trace")
    parser.add_argument("--experiment", "--run", dest="experiment", required=True,
                        help="Experiment directory or concrete _pipeline_runs/<id>")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--video", default=None, help="External-camera input video")
    parser.add_argument("--sync", default=None,
                        help="external_video_sync.json (auto-detected under run/sync)")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Pipeline seconds per video second (default: 5)")
    parser.add_argument("--video-offset-s", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--render-python", "--turntable-python",
                        dest="render_python", default=None)
    parser.add_argument("--render-size", "--turntable-size",
                        dest="render_size", type=int, default=480)
    parser.add_argument("--force-grasp-renders", "--force-turntables",
                        dest="force_grasp_renders", action="store_true")
    parser.add_argument("--grasp-renders-only", "--turntables-only",
                        dest="grasp_renders_only", action="store_true")
    parser.add_argument("--keep-trailing-interruptions", action="store_true",
                        help="Do not cut consecutive interrupted final episodes")
    parser.add_argument("--no-timeline", action="store_true",
                        help="Hide the bottom pipeline diagram and use a full-height grasp stack")
    parser.add_argument(
        "--outcome-style", choices=("bar", "emoji"), default="bar",
        help="Executed-grasp outcome marker (default: bar)")
    parser.add_argument("--min-trailing-interruptions", type=int, default=1,
                        help="Minimum trailing interruption count required to cut")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--crf", type=int, default=25,
                        help="H.264 quality (lower is larger/better; default 25)")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--event-min-duration-s", type=float, default=1.5,
                        help="Minimum event text hold unless superseded (default: 1.5)")
    parser.add_argument(
        "--caption-font", default=None,
        help="Optional TrueType/OpenType font for full-frame event captions")
    args = parser.parse_args()

    caption_font_path = (
        Path(args.caption_font).expanduser().resolve()
        if args.caption_font else None)
    if caption_font_path is not None and not caption_font_path.is_file():
        parser.error(f"caption font not found: {caption_font_path}")

    run_dir = resolve_run_dir(args.experiment, args.run_id)
    manifest = _read_json(run_dir / "manifest.json", {}) or {}
    sync_path = Path(args.sync).expanduser().resolve() if args.sync else None
    sync = _read_json(sync_path, None) if sync_path and sync_path.is_file() else None
    video_arg = args.video or ((sync or {}).get("external_video") if sync else None)
    if not video_arg:
        parser.error("--video is required when no synced external video is recorded")
    video_path = Path(video_arg).expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"video not found: {video_path}")
    info = probe_video(video_path)
    events = load_events(run_dir)
    mapping = build_time_mapping(
        events, sync=sync, speed=args.speed,
        video_offset_s=args.video_offset_s)
    attempts = parse_grasp_attempts(
        run_dir, events, mapping, camera_serial=GRASP_CAMERA_SERIAL)
    if not attempts:
        raise SystemExit("run contains no grasp.selected events")
    layout = layout_spec(
        info.width, info.height, attempts,
        show_pipeline_diagram=not args.no_timeline)
    layout["outcome_style"] = args.outcome_style
    trim = ({
        "applied": False, "output_duration_s": info.duration_s,
        "trimmed_episode_ids": [], "cut_pipeline_s": None, "cut_video_s": None,
    } if args.keep_trailing_interruptions else trailing_interruption_trim(
        events, mapping, info.duration_s,
        min_interruptions=args.min_trailing_interruptions))
    output_duration_s = float(trim["output_duration_s"])
    captions = build_captions(
        events, mapping, output_duration_s,
        minimum_duration_s=args.event_min_duration_s)
    stage_intervals = build_stage_intervals(events, attempts, mapping)
    variant_name = "no_timeline" if args.no_timeline else "timeline"
    if args.outcome_style == "emoji":
        variant_name += "_emoji"
    variant_suffix = "" if variant_name == "timeline" else f"_{variant_name}"
    default_output_name = f"pipeline_video{variant_suffix}.mp4"
    output = (Path(args.output).expanduser().resolve() if args.output else
              run_dir / "edit" / default_output_name)

    metadata = {
        "schema_version": 1,
        "run_id": manifest.get("run_id") or run_dir.name,
        "source_video": str(video_path),
        "output_video": str(output),
        "video": asdict(info),
        "time_mapping": asdict(mapping),
        "sync_audit": build_sync_audit(events, mapping, attempts),
        "trailing_interruption_trim": trim,
        "output_duration_s": output_duration_s,
        "render_mode": variant_name,
        "layout": layout,
        "grasp_visualization": {
            "mode": "fixed_pose",
            "object_pose": "episode_planning_world_pose",
            "wrist_pose": "selected_grasp_relative_to_episode_object",
            "camera_frame": "episode_calibrated_robot_frame",
            "camera_serial": GRASP_CAMERA_SERIAL,
            "camera_transform": "inv(C2R) @ inv(world_to_camera)",
            "camera_axes": "OpenCV +z forward, -y up",
            "projection": "calibrated_vertical_fov_auto_fit_distance",
            "background": "native_rgb_255_white",
            "object_material": "uniform_untextured_gray",
            "history_overflow": "newest_four_fifo",
            "outcome_style": args.outcome_style,
            "outcome_position": "top_right" if args.outcome_style == "emoji" else "left_edge",
            "render_directory": "edit/grasp_pose_renders",
        },
        "event_style": {
            "minimum_duration_s": args.event_min_duration_s,
            "overlay": "full_frame_black",
            "overlay_alpha": 0.62,
            "front_events": ["rotation", "reorientation"],
            "front_text_color": "white",
            "current_grasp_border": "none",
            "font": (str(caption_font_path) if caption_font_path is not None
                     else "opencv_hershey_duplex"),
        },
        "pipeline_diagram": {
            "enabled": not args.no_timeline,
            "stages": list(PIPELINE_STAGES),
            "position": "bottom",
            "active_style": "neutral_brightness",
            "intervals": [asdict(item) for item in stage_intervals],
        },
        "attempts": [item.metadata(run_dir) for item in attempts],
        "captions": [asdict(item) for item in captions],
    }
    print(json.dumps({
        "run": str(run_dir), "video": str(video_path),
        "duration_s": info.duration_s,
        "playback_speed": mapping.playback_speed,
        "output_duration_s": output_duration_s,
        "trimmed_episode_ids": trim["trimmed_episode_ids"],
        "attempts": len(attempts), "output": str(output),
    }, indent=2))
    if args.dry_run:
        return

    render_grasp_pose_images(
        attempts, manifest, python_executable=args.render_python,
        size=args.render_size, camera_serial=GRASP_CAMERA_SERIAL,
        force=args.force_grasp_renders)
    metadata_name = f"pipeline_video{variant_suffix}.json"
    metadata_path = (output.with_suffix(".json") if args.output else
                     run_dir / "edit" / metadata_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.grasp_renders_only:
        print(f"grasp pose renders: {run_dir / 'edit' / 'grasp_pose_renders'}")
        return
    compose_video(
        video_path, output, info, attempts, captions, stage_intervals, layout,
        output_duration_s=output_duration_s, crf=args.crf, preset=args.preset,
        caption_font_path=caption_font_path)
    print(f"video: {output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
