#!/usr/bin/env python3
"""Apply an audited manual outcome correction to one pipeline episode.

The pipeline materializes an episode outcome in the canonical run timeline,
the episode-filtered timeline, the episode index/manifest, and experiment
result summaries.  This command updates those copies together and stores the
original files below ``<run>/corrections/<correction-id>/originals`` so a
manual review never becomes an unaudited destructive edit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable


RESULT_EVENT_NAMES = {
    "grasp.validation_result",
    "grasp.execution_result",
    "episode.result",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def _correction(original: dict[str, Any], *, success: bool, reason: str,
                corrected_utc: str) -> dict[str, Any]:
    existing = original.get("outcome_correction")
    if isinstance(existing, dict):
        return existing
    return {
        "corrected_utc": corrected_utc,
        "source": "manual_review",
        "reason": reason,
        "original_success": original.get("success"),
        "original_outcome": original.get("outcome"),
        "original_reason": original.get("reason"),
        "corrected_success": success,
    }


def _update_event(event: dict[str, Any], episode_id: str, *, success: bool,
                  reason: str, corrected_utc: str) -> bool:
    if event.get("episode_id") != episode_id:
        return False
    is_result = event.get("name") in RESULT_EVENT_NAMES
    is_episode_end = event.get("name") == "episode" and event.get("edge") == "end"
    if not (is_result or is_episode_end):
        return False
    attrs = event.setdefault("attributes", {})
    original = {
        "success": attrs.get("success"),
        "outcome": event.get("outcome"),
        "reason": attrs.get("reason"),
        "source": attrs.get("source"),
        "classification": attrs.get("classification"),
        "outcome_correction": attrs.get("outcome_correction"),
    }
    attrs["outcome_correction"] = _correction(
        original, success=success, reason=reason, corrected_utc=corrected_utc)
    attrs["success"] = success
    attrs["reason"] = None if success else reason
    if "classification" in attrs:
        attrs["classification"] = "success" if success else "failure"
    if "source" in attrs:
        attrs["source"] = "manual_review"
    event["outcome"] = "success" if success else "failure"
    return True


def _update_record(record: dict[str, Any], *, success: bool, reason: str,
                   corrected_utc: str) -> None:
    original = {
        "success": record.get("success"),
        "outcome": record.get("outcome"),
        "reason": record.get("reason"),
        "outcome_correction": record.get("outcome_correction"),
    }
    record["outcome_correction"] = _correction(
        original, success=success, reason=reason, corrected_utc=corrected_utc)
    record["success"] = success
    if "outcome" in record:
        record["outcome"] = "success" if success else "failure"
    if "status" in record:
        record["status"] = "success" if success else "failure"
    record["reason"] = None if success else reason


def _jsonl_transform(path: Path, transform: Callable[[dict[str, Any]], bool]) -> int:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    changed = sum(bool(transform(event)) for event in events)
    _write_text_atomic(
        path, "".join(json.dumps(event, separators=(",", ":")) + "\n"
                      for event in events))
    return changed


def _timeline_transform(path: Path, transform: Callable[[dict[str, Any]], bool]) -> int:
    payload = _read_json(path)
    changed = sum(bool(transform(event)) for event in payload["events"])
    _write_json(path, payload)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--episode", required=True)
    outcome = parser.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--success", action="store_true")
    outcome.add_argument("--failure", action="store_true")
    parser.add_argument("--reason", default="manual_review")
    parser.add_argument("--apply", action="store_true",
                        help="Write the correction; otherwise only print targets")
    args = parser.parse_args()

    run = Path(args.run).expanduser().resolve()
    episode_id = args.episode
    success = bool(args.success)
    episode_index = _read_json(run / "episodes.json")
    record = next((item for item in episode_index.get("episodes", [])
                   if item.get("episode_id") == episode_id), None)
    if record is None:
        raise SystemExit(f"episode is absent from episodes.json: {episode_id}")

    manifest = _read_json(run / "manifest.json")
    object_dir = run.parent.parent
    source_result = object_dir / episode_id / "result.json"
    summary_path = object_dir / "summary.json"
    episode_root = run / "episodes" / episode_id
    scene_info = record.get("scene_info")
    version = str((manifest.get("arguments") or {}).get("grasp_version") or "v8")
    candidate_result = (
        run.parents[3] / "candidate_state" / str(manifest["hand"]) / version
        / str(manifest["object"]) / str(scene_info[0]) / str(scene_info[1])
        / str(scene_info[2]) / "result.json"
    )
    required_targets = [
        run / "events.jsonl",
        run / "episodes.json",
        source_result,
        summary_path,
    ]
    optional_targets = [
        candidate_result,
        run / "timeline.json",
        episode_root / "events.jsonl",
        episode_root / "timeline.json",
        episode_root / "manifest.json",
    ]
    missing = [path for path in required_targets if not path.is_file()]
    if missing:
        raise SystemExit("missing correction target(s):\n" +
                         "\n".join(str(path) for path in missing))
    targets = required_targets + [path for path in optional_targets if path.is_file()]
    print(json.dumps({
        "run": str(run), "episode": episode_id,
        "corrected_success": success, "reason": args.reason,
        "targets": [str(path) for path in targets], "apply": args.apply,
    }, indent=2))
    if not args.apply:
        return

    now = datetime.now(timezone.utc)
    corrected_utc = now.isoformat()
    correction_id = now.strftime("%Y%m%dT%H%M%S_%fZ")
    correction_root = run / "corrections" / correction_id
    originals = correction_root / "originals"
    originals.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(targets):
        shutil.copy2(path, originals / f"{index:02d}_{path.name}")

    transform = lambda event: _update_event(  # noqa: E731
        event, episode_id, success=success, reason=args.reason,
        corrected_utc=corrected_utc)
    changed_counts = {
        "run_events": _jsonl_transform(run / "events.jsonl", transform),
    }
    if (run / "timeline.json").is_file():
        changed_counts["run_timeline"] = _timeline_transform(
            run / "timeline.json", transform)
    if (episode_root / "events.jsonl").is_file():
        changed_counts["episode_events"] = _jsonl_transform(
            episode_root / "events.jsonl", transform)
    if (episode_root / "timeline.json").is_file():
        changed_counts["episode_timeline"] = _timeline_transform(
            episode_root / "timeline.json", transform)

    episode_index = _read_json(run / "episodes.json")
    record = next(item for item in episode_index["episodes"]
                  if item.get("episode_id") == episode_id)
    _update_record(record, success=success, reason=args.reason,
                   corrected_utc=corrected_utc)
    _write_json(run / "episodes.json", episode_index)

    if (episode_root / "manifest.json").is_file():
        episode_manifest = _read_json(episode_root / "manifest.json")
        _update_record(episode_manifest, success=success, reason=args.reason,
                       corrected_utc=corrected_utc)
        _write_json(episode_root / "manifest.json", episode_manifest)

    result = _read_json(source_result)
    _update_record(result, success=success, reason=args.reason,
                   corrected_utc=corrected_utc)
    _write_json(source_result, result)

    summary = _read_json(summary_path)
    summary_record = next(item for item in summary
                          if item.get("dir_idx") == episode_id)
    _update_record(summary_record, success=success, reason=args.reason,
                   corrected_utc=corrected_utc)
    _write_json(summary_path, summary)

    if candidate_result.is_file():
        candidate = _read_json(candidate_result)
        _update_record(candidate, success=success, reason=args.reason,
                       corrected_utc=corrected_utc)
        _write_json(candidate_result, candidate)

    audit = {
        "schema_version": 1,
        "correction_id": correction_id,
        "corrected_utc": corrected_utc,
        "episode_id": episode_id,
        "corrected_success": success,
        "reason": args.reason,
        "changed_event_counts": changed_counts,
        "targets": [str(path) for path in targets],
        "originals": str(originals),
    }
    _write_json(correction_root / "correction.json", audit)
    print(f"correction: {correction_root / 'correction.json'}")


if __name__ == "__main__":
    main()
