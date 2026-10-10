#!/usr/bin/env python3
"""Compare saved VLM XY diagnostics against independent held-key metrology.

Example:
  python evaluate_grounded_alignment.py --manifest /data/eval/manifest.json \
      --output /data/eval/report.json

The output is evidence only and is created exclusively; it cannot command a
robot or certify that the external metrology was correctly calibrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from precision_insertion.grounding_eval import evaluate_grounding_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON path; existing report is not overwritten")
    args = parser.parse_args(argv)
    report = evaluate_grounding_manifest(args.manifest)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "report": str(output),
        "samples": report["sample_count"],
        "advice": report["advice_count"],
        "false_advice": report["false_advice_count"],
        "robot_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
