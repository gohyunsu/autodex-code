#!/usr/bin/env python3
"""Run one precision-insertion session with an AutoDex-PC integration module.

This CLI is intentionally opt-in for robot motion. It calls the demo-local
session loop, not stock run_pipeline.py or run_auto.py. The integration module
supplies commissioned robot/camera adapters and all ten action components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for module_root in (str(REPO_ROOT), str(SRC_DIR), str(DEMO_DIR)):
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

from precision_insertion.live_runtime import (  # noqa: E402
    RunLiveOptions, run_integrated_live_session,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integration", required=True,
                        help="importable AutoDex-PC module:factory")
    parser.add_argument("--robot-config", type=Path,
                        help="FR3 config for precision_insertion.franka_integration:factory")
    parser.add_argument(
        "--max-xy-retries", type=int,
        help="optional cap on failed-trial XY advisories; default has no arbitrary count cap")
    parser.add_argument(
        "--xy-update-weight", type=float, default=1.0,
        help="gain applied to each failed-trial XY residual before adding to the next trial (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--label-mode", choices=(
        "vlm_only", "manual", "evidence_fused"), default="vlm_only",
        help="task labels: VLM only, interactive human, or legacy sensor fusion")
    parser.add_argument("--annotator", help="required with --label-mode manual")
    parser.add_argument("--enable-robot-motion", action="store_true",
                        help="required: initial clear-view and trial motions")
    args = parser.parse_args(argv)
    options = RunLiveOptions(
        launch_config_path=args.launch_config,
        endpoint_catalog_path=args.endpoint_catalog,
        run_output_dir=args.output_dir,
        max_xy_retries=args.max_xy_retries,
        xy_update_weight=args.xy_update_weight,
        max_steps=args.max_steps,
        enable_robot_motion=args.enable_robot_motion,
        robot_config_path=args.robot_config,
        label_mode=args.label_mode, annotator=args.annotator)
    result = run_integrated_live_session(
        options, integration_spec=args.integration)
    payload = result.to_record()
    output = options.run_output_dir.expanduser().resolve()
    report_path = output / "action_loop_result.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"result": payload, "report_path": str(report_path)},
                     indent=2, sort_keys=True))
    return 0 if result.status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
