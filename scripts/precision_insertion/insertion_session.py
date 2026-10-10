#!/usr/bin/env python3
"""Offline scenario catalog and evidence-gated insertion retry session.

This CLI does not move the robot.  ``start`` selects ONE scenario, ``update``
records a checkpoint observation and proposes a bounded correction for a
future independently planned attempt.  Robot execution remains fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from autodex.tasks.insertion_vlm import observe_with_gemini, parse_answer  # noqa: E402
from autodex.tasks.precision_insertion import (  # noqa: E402
    build_catalog, decide_retry, mode_config, read_catalog, require_robot_ready,
    select_scenario,
)


REPORT = REPO / "assets" / "precision_insertion" / "unconstrained_contact_ablation_20mm_10k_results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog_cmd = sub.add_parser("catalog", help="index existing simulation evidence")
    catalog_cmd.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    catalog_cmd.add_argument("--out", type=Path, required=True)
    start = sub.add_parser("start", help="freeze one scenario for an offline trial session")
    start.add_argument("--catalog", type=Path, required=True)
    start.add_argument("--mode", choices=("square", "cylinder"), required=True)
    start.add_argument("--gap-mm", type=float)
    start.add_argument("--scenario-id")
    start.add_argument("--session", type=Path, required=True)
    start.add_argument("--fixture-pose", type=Path,
                       help="frozen socket calibration record for this session")
    start.add_argument("--purpose", choices=("inspect", "robot"), default="inspect")
    start.add_argument("--minimum-level", choices=("geometry_only", "sampled_insertion_geometry",
                       "grasp_sim_pass", "full_task_sim_pass", "hardware_validated"),
                       help="offline evidence gate (default: grasp_sim_pass)")
    update = sub.add_parser("update", help="append observation and compute safe next proposal")
    update.add_argument("--session", type=Path, required=True)
    update.add_argument("--observation", type=Path, required=True,
                        help="JSON with calibrated pose/force/depth measurements")
    update.add_argument("--vlm-response", type=Path,
                        help="saved JSON-only VLM response for offline replay")
    update.add_argument("--gemini-model", help="explicit live VLM call; needs GEMINI_API_KEY")
    update.add_argument("--image", nargs=2, action="append", metavar=("CAMERA", "PATH"),
                        help="saved AutoDex checkpoint image; repeat per camera")
    update.add_argument("--checkpoint", choices=("post_lift", "pre_insert",
                        "insertion_abort", "insertion_hold", "finish"), default="pre_insert")
    args = parser.parse_args()

    if args.command == "catalog":
        report = json.loads(REPORT.read_text())
        result = build_catalog(args.shared_root.expanduser(), report)
        write_json(args.out.expanduser(), result)
        print(json.dumps({"catalog": str(args.out), "scenarios": len(result["scenarios"]),
                          "missing": len(result["missing_candidate_directories"])}))
        return

    if args.command == "start":
        mode = mode_config(args.mode, gap_mm=args.gap_mm)
        catalog = read_catalog(args.catalog.expanduser())
        minimum = "hardware_validated" if args.purpose == "robot" else (
            args.minimum_level or "grasp_sim_pass")
        scenario = select_scenario(catalog, args.mode, minimum_level=minimum,
                                   scenario_id=args.scenario_id, key=mode.key,
                                   socket=mode.socket, gap_mm=mode.socket_gap_mm)
        # The recorded square pilot applies ONLY to the 1.5 mm key/socket.
        if scenario["key"] != mode.key or scenario["socket"] != mode.socket or scenario["gap_mm"] != mode.socket_gap_mm:
            raise RuntimeError("catalog scenario does not match requested geometry")
        if args.purpose == "robot":
            require_robot_ready(scenario)
            raise RuntimeError("robot insertion controller is not commissioned")
        if args.session.exists():
            raise FileExistsError(f"session already exists: {args.session}")
        fixture = None
        if args.fixture_pose:
            fixture_path = args.fixture_pose.expanduser().resolve()
            fixture = {"path": str(fixture_path), "sha256": sha256(fixture_path)}
        session = {"schema_version": 1, "mode": mode.name, "gap_mm": mode.socket_gap_mm,
                   "scenario": scenario, "purpose": args.purpose,
                   "offset_xy_m": [0.0, 0.0], "attempt": 1, "history": [],
                   "fixture_pose": fixture,
                   "note": "Offline proposal only; not a robot command."}
        write_json(args.session, session)
        print(json.dumps({"session": str(args.session), "scenario": scenario["id"],
                          "validation_level": scenario["validation_level"]}))
        return

    session = json.loads(args.session.read_text())
    if session["purpose"] != "inspect":
        raise RuntimeError("robot sessions are not supported by this CLI")
    fixture = session.get("fixture_pose")
    if fixture and sha256(Path(fixture["path"])) != fixture["sha256"]:
        raise RuntimeError("frozen fixture calibration changed; start a new session")
    observation = json.loads(args.observation.read_text())
    views = {label: Path(path) for label, path in (args.image or [])}
    if args.gemini_model and args.vlm_response:
        raise ValueError("choose either --gemini-model or --vlm-response")
    if args.gemini_model:
        if not views:
            raise ValueError("--gemini-model requires at least one --image")
        from google import genai  # type: ignore
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        vlm = observe_with_gemini(checkpoint=args.checkpoint, image_paths=views,
                                  measurements=observation,
                                  client=genai.Client(api_key=key), model=args.gemini_model)
    elif args.vlm_response:
        vlm = parse_answer(args.vlm_response.read_text(), args.checkpoint,
                           set(views) or set(observation.get("camera_labels", [])))
    else:
        vlm = {"checkpoint": args.checkpoint, "class": "unknown", "confidence": 0.0,
               "visible_evidence": "no_vlm_observation", "cameras_used": []}
    observation["vlm_class"] = vlm["class"]
    observation["vlm_confidence"] = vlm["confidence"]
    mode = mode_config(session["mode"], gap_mm=session["gap_mm"])
    decision = decide_retry(mode=mode,
                            previous_offset_xy_m=tuple(session["offset_xy_m"]),
                            attempt=session["attempt"], observation=observation)
    session["history"].append({"observation": observation, "vlm": vlm,
                               "decision": decision.to_record()})
    if decision.status == "propose_retry":
        session["offset_xy_m"] = list(decision.offset_xy_m)
        session["attempt"] = decision.attempt
    write_json(args.session, session)
    print(json.dumps({"decision": decision.to_record(), "vlm": vlm}))


if __name__ == "__main__":
    main()
