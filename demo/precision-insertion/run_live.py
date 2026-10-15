#!/usr/bin/env python3
"""Run one precision-insertion session with an AutoDex-PC integration module.

This CLI is intentionally opt-in for robot motion. It calls the demo-local
session loop, not stock run_pipeline.py or run_auto.py. The integration module
supplies commissioned robot/camera adapters and all ten action components.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for module_root in (str(REPO_ROOT), str(SRC_DIR), str(DEMO_DIR)):
    if module_root not in sys.path:
        sys.path.insert(0, module_root)


class _CuroboInfoFilter(logging.Filter):
    """Keep actionable cuRobo warnings without per-iteration INFO noise."""

    _REPEATED_WARNINGS = {
        "Deprecated: external_asset_path is deprecated, use ContentPath",
        "Deprecated: external_robot_configs_path is deprecated, use ContentPath",
        "Object already in warp cache, using existing instance for: fixture_socket",
    }

    def __init__(self) -> None:
        super().__init__()
        self._seen_warnings: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return False
        message = record.getMessage()
        if record.levelno == logging.WARNING and message in self._REPEATED_WARNINGS:
            if message in self._seen_warnings:
                return False
            self._seen_warnings.add(message)
        return True


# Install before importing the live runtime, whose cuRobo imports can log.
logging.getLogger("curobo").addFilter(_CuroboInfoFilter())

from precision_insertion.live_runtime import (  # noqa: E402
    RunLiveOptions, inspect_live_inputs, run_integrated_live_session,
)
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.timing_profile import (  # noqa: E402
    profile_run, timed, timing_directory,
)


_EXP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def default_run_output(launch_config: Path, exp_name: str) -> Path:
    """Derive an output path; full launch validation runs inside the trace."""
    if not _EXP_NAME.fullmatch(exp_name):
        raise ValueError("exp name must be a single path-safe directory name")
    record = json.loads(Path(launch_config).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("mode"), dict):
        raise ValueError("launch config needs an explicit socket/key mode")
    mode = select_mode(record["mode"].get("family"),
                       record["mode"].get("gap_mm"))
    object_root = (Path.home() / "shared_data" / "AutoDex" / "experiment" /
                   exp_name / "inspire" / mode.key_object)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return object_root / "_pipeline_runs" / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--exp-name", "--exp_name", default="precision-insertion",
                        help="AutoDex experiment name (default: precision-insertion)")
    parser.add_argument("--output-dir", type=Path,
                        help="new run directory; defaults to stock AutoDex experiment hierarchy")
    parser.add_argument("--integration",
                        help="importable AutoDex-PC module:factory")
    parser.add_argument("--preflight", action="store_true",
                        help="validate static v8 inputs without cameras or robot")
    parser.add_argument("--robot-config", type=Path,
                        help="FR3 config for precision_insertion.franka_integration:factory")
    parser.add_argument("--init-port-mask", type=int)
    parser.add_argument("--init-port-pose", type=int)
    parser.add_argument("--init-port-cmd", type=int)
    parser.add_argument(
        "--max-xy-retries", type=int,
        help="optional cap on failed-trial XY advisories; default has no arbitrary count cap")
    parser.add_argument(
        "--xy-update-weight", type=float, default=1.0,
        help="gain applied to each failed-trial XY residual before adding to the next trial (default: 1.0)")
    parser.add_argument("--max-steps", type=int,
                        help="optional action-loop cap; default runs until success or a real stop decision")
    parser.add_argument("--label-mode", choices=(
        "vlm_only", "manual"), default="vlm_only",
        help="task labels: VLM only or interactive human with independent VLM evidence")
    parser.add_argument("--annotator", help="required with --label-mode manual")
    parser.add_argument("--enable-robot-motion", action="store_true",
                        help="required: initial clear-view and trial motions")
    parser.add_argument("--stop-after-planning", action="store_true",
                        help="stop when a candidate plan is ready, before pickup motion")
    parser.add_argument("--reuse-session-evidence", type=Path,
                        help="use previously verified fixed-socket images and calibration")
    parser.add_argument("--key-already-placed", action="store_true",
                        help="skip the initial key-placement prompt")
    args = parser.parse_args(argv)
    if not _EXP_NAME.fullmatch(args.exp_name):
        parser.error("exp name must be a single path-safe directory name")
    ports = (args.init_port_mask, args.init_port_pose, args.init_port_cmd)
    if any(port is not None for port in ports) and any(
            port is None for port in ports):
        parser.error("supply all three precision init daemon ports together")
    init_ports = None if all(port is None for port in ports) else ports
    if init_ports is not None and (any(not 0 < port < 65536 for port in ports)
                                   or len(set(ports)) != 3):
        parser.error("init daemon ports must be distinct TCP ports")
    if args.preflight:
        if args.integration not in (None,
                                    "precision_insertion.franka_integration:factory"):
            parser.error("static preflight supports the built-in FR3 integration only")
        report = inspect_live_inputs(
            launch_config_path=args.launch_config,
            endpoint_catalog_path=args.endpoint_catalog,
            robot_config_path=args.robot_config)
        report["exp_name"] = args.exp_name
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if init_ports is None:
        parser.error("live run requires explicit frame-bound init daemon ports")
    if args.integration is None:
        parser.error("--integration is required for a live run")
    output_dir = args.output_dir or default_run_output(
        args.launch_config, args.exp_name)
    if args.output_dir is None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    options = RunLiveOptions(
        launch_config_path=args.launch_config,
        endpoint_catalog_path=args.endpoint_catalog,
        run_output_dir=output_dir,
        max_xy_retries=args.max_xy_retries,
        xy_update_weight=args.xy_update_weight,
        max_steps=args.max_steps,
        enable_robot_motion=args.enable_robot_motion,
        robot_config_path=args.robot_config,
        label_mode=args.label_mode, annotator=args.annotator,
        init_ports=init_ports, stop_after_planning=args.stop_after_planning,
        reuse_session_evidence_dir=args.reuse_session_evidence,
        key_already_placed=args.key_already_placed)
    with profile_run(output_dir) as profile:
        with timed("live.session", phase="session"):
            result = run_integrated_live_session(
                options, integration_spec=args.integration)
        if result.status != "success":
            profile.outcome = result.status
    payload = result.to_record()
    output = options.run_output_dir.expanduser().resolve()
    payload["exp_name"] = args.exp_name
    payload["run_output_dir"] = str(output)
    payload["timing_dir"] = str(timing_directory(output))
    report_path = output / "action_loop_result.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"result": payload, "report_path": str(report_path)},
                     indent=2, sort_keys=True))
    return 0 if result.status in ("success", "planning_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
