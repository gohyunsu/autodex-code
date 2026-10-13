#!/usr/bin/env python3
"""Camera-only AutoDex ChArUco/socket then white key/SAM/FoundPose QA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from precision_insertion.key_socket_camera_qa import (
    DEFAULT_KEY_PROMPT, run_camera_only_key_socket_qa,
)
from precision_insertion.session_launch import load_session_launch


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() == ""
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True,
                        help="validated AutoDex launch, including a static staged launch")
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="new absolute camera-only QA session directory")
    parser.add_argument("--key-prompt", default=DEFAULT_KEY_PROMPT)
    args = parser.parse_args(argv)
    launch = replace(load_session_launch(
        args.launch_config, evidence_dir_override=args.evidence_dir),
        execution_scope="camera_only_diagnostic")
    initial_scene = (
        "Keep the key absent; place the fixed white socket on board 9"
        if launch.board_capture_phase == "socket_present_partial" else
        "Keep the board empty and the key absent")
    if not _confirm(f"{initial_scene}, keep the robot clear, then press Enter (q to cancel): "):
        return 2
    report = run_camera_only_key_socket_qa(
        launch=launch, key_prompt=args.key_prompt,
        confirm_socket_placement=lambda: _confirm(
            "Empty board accepted. Place the fixed white socket, then press Enter (q to cancel): "),
        confirm_key_placement=lambda: _confirm(
            "Socket measurement saved. Place the white cylindrical key without moving the socket, then press Enter (q to cancel): "),
    )
    print(json.dumps({"report": str(report), "robot_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
