#!/usr/bin/env python3
"""Build the successful-trial reset preview by reversing a validated preview.

The input must be a pick-to-insertion geometric preview produced by
``build_tabletop_pose_animation_set.py``.  Reversal preserves every sampled
robot configuration and collision result: the closed hand extracts the key,
returns it to its original tabletop pose, opens, and only then retreats.
This is still a sampled geometric preview, not a controller or a physical
success claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASES = {
    "final hold": "hold seated key",
    "insert to CAD seated pose": "extract from socket",
    "descend to preinsert": "rise above socket",
    "reorient above socket": "undo insertion orientation",
    "transfer above socket": "return above tabletop pose",
    "lift key from table": "lower key to tabletop",
    "close on handle sides and rear": "open after tabletop support",
    "approach tabletop key": "retreat from released key",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.trajectory.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(source.stem + "_reset.npz")
    )

    with np.load(source, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    status = str(arrays["preview_status"].item())
    if status != "sampled_geometric_preview_passed_not_physical_validation":
        raise ValueError(f"reset requires a passing sampled preview, got {status}")

    frame_fields = (
        "qpos",
        "object_pose",
        "collision_counts",
        "hand_socket_collision_counts",
        "hand_table_collision_counts",
        "key_socket_collision_counts",
        "key_table_collision_counts",
    )
    for name in frame_fields:
        arrays[name] = arrays[name][::-1].copy()
    source_phases = [str(value) for value in arrays["phase"][::-1].tolist()]
    unknown = sorted(set(source_phases) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown source phases: {unknown}")
    arrays["phase"] = np.asarray([PHASES[value] for value in source_phases])
    arrays["preview_kind"] = np.asarray("sampled_geometric_reset")
    arrays["source_trajectory"] = np.asarray(str(source))

    # The reverse trajectory must be collision-equivalent to its source.
    if np.any(arrays["collision_counts"]):
        raise RuntimeError("source contains sampled collisions")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {
        "schema_version": 1,
        "status": status,
        "kind": "successful_trial_reset",
        "source_trajectory": str(source),
        "output_trajectory": str(output),
        "frame_count": int(len(arrays["qpos"])),
        "sequence_phases": list(dict.fromkeys(arrays["phase"].tolist())),
        "invariants": {
            "closed_until_tabletop_support": True,
            "opens_before_retreat": True,
            "exact_reverse_robot_samples": True,
            "sampled_collision_count_max": int(arrays["collision_counts"].max()),
        },
        "not_validated": [
            "cuRobo continuous collision planning",
            "force-controlled extraction",
            "physical execution",
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    print(output.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
