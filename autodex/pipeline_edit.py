"""Generate a compact edit/visualization package from a pipeline timeline."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import re
from typing import Any, Optional


PHASE_COLORS = {
    "startup": "#64748b",
    "operator": "#94a3b8",
    "perception": "#38bdf8",
    "coverage": "#a78bfa",
    "planning": "#f59e0b",
    "capture": "#22d3ee",
    "execution": "#22c55e",
    "validation": "#14b8a6",
    "recovery": "#f43f5e",
    "episode": "#e2e8f0",
    "cleanup": "#64748b",
    "sync": "#f8fafc",
}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    timeline = _read_json(run_dir / "timeline.json", {})
    if timeline.get("events"):
        return list(timeline["events"])
    events = []
    try:
        with (run_dir / "events.jsonl").open() as stream:
            for line in stream:
                if line.strip():
                    events.append(json.loads(line))
    except OSError:
        pass
    return events


def _load_sync(run_dir: Path, sync_path: Optional[Path]) -> Optional[dict]:
    path = sync_path or run_dir / "sync" / "external_video_sync.json"
    sync = _read_json(path, None)
    return sync if isinstance(sync, dict) and "transform" in sync else None


def _video_time(pipeline_s: Optional[float], sync: Optional[dict]) -> Optional[float]:
    if pipeline_s is None or sync is None:
        return None
    transform = sync["transform"]
    return round(
        float(transform["slope"]) * float(pipeline_s)
        + float(transform["offset_s"]),
        6,
    )


def _layout_for(phase: str) -> dict[str, Any]:
    return {
        "id": "edge_flush_single_grasp_history",
        "external_video": {"x": 0, "y": 0, "w": 1920, "h": 1080},
        "current_grasp": {"x": 0, "y": 0, "w": 576, "h": 576},
        "grasp_history_column": {
            "x": 1675, "y": 0, "w": 245, "h": 980,
            "background": "#ffffff",
        },
        "pipeline_diagram": {"x": 0, "y": 980, "w": 1920, "h": 100},
        "event_caption": {"x": 0, "y": 0, "w": 1920, "h": 1080},
        "visuals": ["current_fixed_grasp_pose", "grasp_history_pose_renders",
                    "pipeline_stage_diagram",
                    "recovery_front_caption"],
    }


def _pair_spans(events: list[dict[str, Any]], sync: Optional[dict]) -> list[dict]:
    starts = {
        event["span_id"]: event for event in events
        if event.get("edge") == "start" and event.get("span_id")
    }
    spans = []
    for end in events:
        if end.get("edge") != "end":
            continue
        start = starts.get(end.get("span_id"))
        if start is None:
            continue
        begin_s = float(start["pipeline_time_s"])
        end_s = float(end["pipeline_time_s"])
        spans.append({
            "span_id": end["span_id"],
            "parent_id": start.get("parent_id"),
            "name": start.get("name"),
            "phase": start.get("phase"),
            "kind": start.get("kind"),
            "episode_id": start.get("episode_id"),
            "attempt_id": start.get("attempt_id"),
            "outcome": end.get("outcome"),
            "pipeline_in_s": begin_s,
            "pipeline_out_s": end_s,
            "video_in_s": _video_time(begin_s, sync),
            "video_out_s": _video_time(end_s, sync),
            "layout": _layout_for(str(start.get("phase"))),
            "attributes": {
                "start": start.get("attributes") or {},
                "end": end.get("attributes") or {},
            },
        })
    return spans


def _artifact_index(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    root = run_dir / "artifacts"
    if not root.exists():
        return records
    for path in sorted(root.rglob("*.json")):
        payload = _read_json(path, {})
        record = {
            "path": path.relative_to(run_dir).as_posix(),
            "category": path.parent.name,
        }
        if path.parent.name == "planner":
            record["summary"] = payload.get("summary", {})
            candidates = payload.get("candidates", [])
            record["candidate_stack"] = [
                {
                    "candidate_id": item.get("candidate_id"),
                    "rank": item.get("policy_rank"),
                    "status": item.get("status"),
                    "failure_code": item.get("failure_code"),
                }
                for item in candidates
            ]
        elif path.parent.name == "coverage" and "grasp_policy" in path.name:
            record["coverage_ranking"] = payload.get("candidates", [])[:24]
            record["dropped_zero_gain"] = payload.get("dropped_zero_gain")
        elif path.parent.name == "recovery" and "pose_grid" in path.name:
            grid = payload.get("grid", [])
            record["rotation_grid"] = {
                "selected": payload.get("selected"),
                "cell_count": len(grid),
                "max_ik_feasible": max(
                    (int(cell.get("ik_feasible_count", 0)) for cell in grid),
                    default=0),
            }
        elif path.parent.name == "recovery" and "reorient_heights" in path.name:
            record["reorient_heights"] = payload.get("height_attempts", [])
        records.append(record)
    return records


def _write_html(path: Path, package: dict[str, Any]) -> None:
    spans = package["segments"]
    origin = min((item["pipeline_in_s"] for item in spans), default=0.0)
    timeline_end = max((item["pipeline_out_s"] for item in spans), default=1.0)
    total = max(timeline_end - origin, 1e-9)
    rows = []
    for item in spans:
        left = 100 * (item["pipeline_in_s"] - origin) / total
        width = max(0.18, 100 * (item["pipeline_out_s"]
                                - item["pipeline_in_s"]) / total)
        color = PHASE_COLORS.get(item["phase"], "#cbd5e1")
        title = html.escape(
            f"{item['name']} | {item['pipeline_in_s']:.3f}–"
            f"{item['pipeline_out_s']:.3f}s | {item['outcome']}")
        rows.append(
            f'<div class="row"><span>{html.escape(str(item["phase"]))}</span>'
            f'<div class="track"><i title="{title}" style="left:{left:.5f}%;'
            f'width:{width:.5f}%;background:{color}"></i></div>'
            f'<b>{html.escape(str(item["name"]))}</b></div>')
    body = "\n".join(rows)
    cards = []

    def candidate_class(item: dict[str, Any]) -> str:
        status = str(item.get("status") or "pending")
        if status == "selected":
            return "selected"
        if ("fail" in status or "collision" in status
                or status == "backward"):
            return "failed"
        return "pending"

    for artifact in package.get("artifacts", []):
        if artifact.get("coverage_ranking"):
            ranking = artifact["coverage_ranking"]
            max_gain = max((int(item.get("uncovered_scene_gain", 0))
                            for item in ranking), default=1)
            bars = "".join(
                '<i class="gain" title="rank {rank}: {gain}" '
                'style="height:{height:.1f}%"></i>'.format(
                    rank=item.get("rank"),
                    gain=item.get("uncovered_scene_gain"),
                    height=100 * int(item.get("uncovered_scene_gain", 0))
                    / max(1, max_gain))
                for item in ranking)
            cards.append(
                f'<section><h2>coverage</h2><div class="gains">{bars}</div>'
                f'<small>{len(ranking)} ranked · '
                f'{artifact.get("dropped_zero_gain", 0)} dropped</small></section>')
        if artifact.get("candidate_stack"):
            stack = "".join(
                f'<i class="candidate {candidate_class(item)}" '
                f'title="{html.escape(str(item.get("candidate_id")))} · '
                f'{html.escape(str(item.get("status")))}"></i>'
                for item in artifact["candidate_stack"][:60])
            summary = artifact.get("summary", {})
            cards.append(
                f'<section><h2>grasp stack</h2><div class="stack">{stack}</div>'
                f'<small>{html.escape(str(summary.get("selected_candidate_id") or summary.get("reason") or "–"))}</small></section>')
        if artifact.get("rotation_grid"):
            grid = artifact["rotation_grid"]
            cards.append(
                '<section><h2>recovery grid</h2>'
                f'<strong>{grid["max_ik_feasible"]}</strong><small> max IK · '
                f'{grid["cell_count"]} cells · '
                f'{html.escape(str(grid.get("selected") or "no selection"))}</small></section>')
    card_body = "".join(cards) or "<section><small>No decision artifacts</small></section>"
    path.write_text(f"""<!doctype html>
<meta charset="utf-8"><title>AutoDex pipeline edit map</title>
<style>
body{{font:13px system-ui;background:#07111f;color:#e2e8f0;margin:28px}}
h1{{font-size:20px}} .note{{color:#94a3b8;margin-bottom:20px}}
.row{{display:grid;grid-template-columns:90px 1fr 270px;gap:12px;align-items:center;margin:7px 0}}
.track{{height:17px;background:#172033;position:relative;border-radius:4px;overflow:hidden}}
.track i{{position:absolute;height:100%;min-width:2px;border-radius:3px}}
.row b{{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:22px 0}}
section{{background:#111c2f;border:1px solid #26344d;border-radius:10px;padding:14px;min-height:72px}}
h2{{font-size:12px;text-transform:uppercase;color:#94a3b8;margin:0 0 10px}}
small{{display:block;color:#94a3b8;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.gains{{height:44px;display:flex;align-items:end;gap:3px}} .gain{{width:7px;background:#a78bfa;border-radius:2px}}
.stack{{display:flex;gap:4px;flex-wrap:wrap}} .candidate{{width:10px;height:18px;background:#64748b;border-radius:2px}}
.candidate.failed{{background:#f43f5e}} .candidate.selected{{background:#f59e0b}} section strong{{font-size:34px;color:#f59e0b}}
</style><h1>AutoDex pipeline timeline</h1>
<div class="note">run {html.escape(str(package['run_id']))} · {total:.3f}s ·
hover a bar for exact timing</div><div class="cards">{card_body}</div>{body}""",
                    encoding="utf-8")


def _write_markers(
    path: Path, events: list[dict[str, Any]], sync: Optional[dict],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "seq", "name", "phase", "kind", "edge", "outcome",
            "pipeline_time_s", "video_time_s", "episode_id", "attempt_id",
        ])
        writer.writeheader()
        for event in events:
            writer.writerow({
                "seq": event.get("seq"), "name": event.get("name"),
                "phase": event.get("phase"), "kind": event.get("kind"),
                "edge": event.get("edge"), "outcome": event.get("outcome"),
                "pipeline_time_s": event.get("pipeline_time_s"),
                "video_time_s": _video_time(event.get("pipeline_time_s"), sync),
                "episode_id": event.get("episode_id"),
                "attempt_id": event.get("attempt_id"),
            })


def build_edit_package(
    run_dir: str | Path, *, sync_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Write editor markers, cut segments, and a minimal-text layout spec."""
    run_dir = Path(run_dir).expanduser().resolve()
    edit_dir = run_dir / "edit"
    edit_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(run_dir)
    sync = _load_sync(
        run_dir, None if sync_path is None else Path(sync_path).expanduser())
    spans = _pair_spans(events, sync)
    manifest = _read_json(run_dir / "manifest.json", {})
    package = {
        "schema_version": 1,
        "scope": "run",
        "run_id": manifest.get("run_id") or (events[0].get("run_id") if events else None),
        "timebase": {
            "canonical": "pipeline_time_s",
            "external_video_mapped": sync is not None,
            "sync_file": "sync/external_video_sync.json" if sync is not None else None,
        },
        "canvas": {"width": 1920, "height": 1080, "safe_margin_px": 0},
        "style": {
            "text_policy": "icons, counts, and short status tokens only",
            "success_color": "#22c55e",
            "failure_color": "#f43f5e",
            "pending_color": "#64748b",
            "selected_grasp_color": "#f59e0b",
        },
        "segments": spans,
        "artifacts": _artifact_index(run_dir),
    }
    (edit_dir / "storyboard.json").write_text(
        json.dumps(package, indent=2) + "\n", encoding="utf-8")
    _write_markers(edit_dir / "markers.csv", events, sync)
    _write_html(edit_dir / "storyboard.html", package)

    episode_records = (_read_json(run_dir / "episodes.json", {})
                       .get("episodes", []))
    for episode_record in episode_records:
        episode_id = str(episode_record.get("episode_id"))
        episode_events = [
            event for event in events
            if str(event.get("episode_id")) == episode_id
        ]
        if not episode_events:
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._")
        episode_edit_dir = edit_dir / "episodes" / (safe_id or "episode")
        episode_edit_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = episode_record.get("attempt_id")
        episode_package = {
            **package,
            "scope": "episode",
            "episode": episode_record,
            "segments": _pair_spans(episode_events, sync),
            "artifacts": [
                artifact for artifact in package["artifacts"]
                if attempt_id is None or str(attempt_id) in artifact["path"]
            ],
        }
        (episode_edit_dir / "storyboard.json").write_text(
            json.dumps(episode_package, indent=2) + "\n", encoding="utf-8")
        _write_markers(episode_edit_dir / "markers.csv", episode_events, sync)
        _write_html(episode_edit_dir / "storyboard.html", episode_package)
    return package
