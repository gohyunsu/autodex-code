#!/usr/bin/env python3
"""Build reset-only or complete trial-and-reset presentation trajectories.

The input must be a pick-to-insertion geometric preview produced by
``build_tabletop_pose_animation_set.py``.  The default reset extracts the key,
transfers it to the safe high pose above its original tabletop region, opens,
and lets the key drop.  It deliberately does not place the key back along the
exact inverse pickup trajectory.  ``--reset-mode reverse-place`` retains the
old diagnostic for comparison only.
With ``--full-trial`` the output shows the complete requested presentation
sequence: pick, rigid transfer, insertion, release, retreat and observation
hold, re-approach, re-grasp, extraction, return, release, and final retreat.
The added open-hand socket stages are deliberately marked unvalidated; this is
still a presentation preview, not a controller or a physical success claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASES = {
    "final hold": "hold inserted key",
    "insert to CAD seated pose": "extract from socket",
    "insert 20 mm to verification depth": "extract from verification depth",
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
    parser.add_argument(
        "--full-trial", action="store_true",
        help="prepend the successful trial and add release/confirmation before reset",
    )
    parser.add_argument(
        "--reset-mode",
        choices=("drop", "reverse-place"),
        default="drop",
        help=(
            "drop releases above the original tabletop region; reverse-place "
            "is the legacy exact inverse diagnostic"
        ),
    )
    args = parser.parse_args()
    if not args.full_trial and args.reset_mode == "drop":
        parser.error(
            "drop reset composition requires --full-trial; use "
            "--reset-mode reverse-place only for the legacy reset-only diagnostic"
        )
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
    original = {name: value.copy() for name, value in arrays.items()}
    source_phases = [str(value) for value in original["phase"].tolist()]
    unknown = sorted(set(source_phases) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown source phases: {unknown}")

    if args.full_trial:
        qpos = original["qpos"]
        phases = np.asarray(source_phases)
        inserted = original["object_pose"][-1]
        open_hand = qpos[0, 7:]
        closed_hand = qpos[-1, 7:]
        # Reverse only insertion+descent to obtain a known arm retreat from the
        # socket.  The key remains seated and the hand is open.  These frames
        # are visually faithful but require a future continuous collision plan.
        near_mask = np.isin(
            phases, [
                "descend to preinsert",
                "insert to CAD seated pose",
                "insert 20 mm to verification depth",
            ]
        )
        near_indices = np.flatnonzero(near_mask)
        retreat_indices = near_indices[::-1]
        retreat_q = qpos[retreat_indices].copy()
        retreat_q[:, 7:] = open_hand
        approach_q = retreat_q[::-1].copy()
        release_alpha = np.linspace(0.0, 1.0, 12)[:, None]
        release_q = np.repeat(qpos[-1:, :], len(release_alpha), axis=0)
        release_q[:, 7:] = (
            (1.0 - release_alpha) * closed_hand + release_alpha * open_hand
        )
        reclose_q = release_q[::-1].copy()
        observation_q = np.repeat(retreat_q[-1:], 24, axis=0)

        # Do not replay the ten-frame final hold when beginning reset; reclose
        # already establishes the inserted closed-hand state.
        final_hold_start = int(np.flatnonzero(phases == "final hold")[0])
        if args.reset_mode == "drop":
            # Stop at the top of the original pickup lift.  This is a safe
            # high reset region, not an exact-placement goal.  The transfer
            # samples are reused only as a presentation placeholder until a
            # fresh cuRobo reset plan is generated online.
            lift_indices = np.flatnonzero(phases == "lift key from table")
            reset_stop = int(lift_indices[-1])
        else:
            reset_stop = 0
        reverse_indices = np.arange(
            final_hold_start - 1, reset_stop - 1, -1
        )
        reverse_q = qpos[reverse_indices].copy()
        reverse_objects = original["object_pose"][reverse_indices].copy()
        reverse_phases = np.asarray([
            PHASES[source_phases[index]] for index in reverse_indices
        ])
        if args.reset_mode == "drop":
            reverse_phases[-1] = "carry above reset drop zone"

        reset_extra_q: list[np.ndarray] = []
        reset_extra_objects: list[np.ndarray] = []
        reset_extra_phases: list[np.ndarray] = []
        if args.reset_mode == "drop":
            drop_open_q = np.repeat(reverse_q[-1:, :], 12, axis=0)
            drop_alpha = np.linspace(0.0, 1.0, len(drop_open_q))[:, None]
            drop_open_q[:, 7:] = (
                (1.0 - drop_alpha) * closed_hand + drop_alpha * open_hand
            )
            drop_start = reverse_objects[-1]
            drop_target = original["object_pose"][0]
            falling_objects = np.repeat(drop_start[None], 18, axis=0)
            fall_alpha = np.linspace(0.0, 1.0, 18)
            falling_objects[:, :3, 3] = (
                (1.0 - fall_alpha[:, None]) * drop_start[:3, 3]
                + fall_alpha[:, None] * drop_target[:3, 3]
            )
            falling_q = np.repeat(drop_open_q[-1:, :], 18, axis=0)
            reset_extra_q = [drop_open_q, falling_q]
            reset_extra_objects = [
                np.repeat(drop_start[None], len(drop_open_q), axis=0),
                falling_objects,
            ]
            reset_extra_phases = [
                np.repeat("release above reset drop zone", len(drop_open_q)),
                np.repeat("drop key into reset zone", len(falling_q)),
            ]

        q_segments = [
            qpos, release_q, retreat_q, observation_q,
            approach_q, reclose_q, reverse_q, *reset_extra_q,
        ]
        object_segments = [
            original["object_pose"],
            np.repeat(inserted[None], len(release_q), axis=0),
            np.repeat(inserted[None], len(retreat_q), axis=0),
            np.repeat(inserted[None], len(observation_q), axis=0),
            np.repeat(inserted[None], len(approach_q), axis=0),
            np.repeat(inserted[None], len(reclose_q), axis=0),
            reverse_objects,
            *reset_extra_objects,
        ]
        phase_segments = [
            phases,
            np.repeat("release after insertion", len(release_q)),
            np.repeat("retreat after release", len(retreat_q)),
            np.repeat("confirm insertion success", len(observation_q)),
            np.repeat("approach inserted key for reset", len(approach_q)),
            np.repeat("regrasp inserted key", len(reclose_q)),
            reverse_phases,
            *reset_extra_phases,
        ]
        arrays["qpos"] = np.concatenate(q_segments)
        arrays["object_pose"] = np.concatenate(object_segments)
        arrays["phase"] = np.concatenate(phase_segments)
        # Collision evidence exists only for source/reverse frames.  -1 makes
        # the missing validation machine-readable instead of silently claiming
        # zero collisions for composed release/regrasp stages.
        for name in frame_fields[2:]:
            known_forward = original[name]
            unknown_values = [
                np.full(len(segment), -1, dtype=known_forward.dtype)
                for segment in q_segments[1:6]
            ]
            reset_unknown = [
                np.full(len(segment), -1, dtype=known_forward.dtype)
                for segment in reset_extra_q
            ]
            arrays[name] = np.concatenate(
                [known_forward, *unknown_values,
                 known_forward[reverse_indices], *reset_unknown]
            )
        arrays["preview_status"] = np.asarray(
            "composed_full_trial_contains_unvalidated_release_regrasp_stages"
        )
        arrays["preview_kind"] = np.asarray("full_success_trial_then_reset_preview")
    else:
        for name in frame_fields:
            arrays[name] = original[name][::-1].copy()
        reversed_phases = source_phases[::-1]
        arrays["phase"] = np.asarray([PHASES[value] for value in reversed_phases])
        arrays["preview_kind"] = np.asarray("sampled_geometric_reset")
    arrays["source_trajectory"] = np.asarray(str(source))

    # A reset-only reverse trajectory must be collision-equivalent to source.
    if not args.full_trial and np.any(arrays["collision_counts"]):
        raise RuntimeError("source contains sampled collisions")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {
        "schema_version": 2,
        "status": str(arrays["preview_status"].item()),
        "kind": (
            "full_success_trial_then_reset_preview"
            if args.full_trial else "successful_trial_reset"
        ),
        "source_trajectory": str(source),
        "output_trajectory": str(output),
        "frame_count": int(len(arrays["qpos"])),
        "reset_mode": args.reset_mode,
        "sequence_phases": list(dict.fromkeys(arrays["phase"].tolist())),
        "invariants": {
            "reset_grasp_closed_until_release_or_tabletop_support": True,
            "opens_before_each_unloaded_retreat": True,
            "exact_reverse_robot_samples_for_reset": (
                args.reset_mode == "reverse-place"
            ),
            "reset_objective_is_exact_original_pose": (
                args.reset_mode == "reverse-place"
            ),
            "drop_release_above_reset_region": args.reset_mode == "drop",
            "key_fixed_in_socket_during_release_confirmation": bool(args.full_trial),
            "sampled_collision_count_max_for_known_frames": int(
                arrays["collision_counts"][arrays["collision_counts"] >= 0].max()
            ),
        },
        "not_validated": [
            "cuRobo continuous collision planning",
            *(
                [
                    "release, open-hand retreat, reset re-approach, and "
                    "regrasp collision checking"
                ]
                if args.full_trial else []
            ),
            "force-controlled extraction",
            *(
                ["drop dynamics and resulting tabletop pose"]
                if args.reset_mode == "drop" else []
            ),
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
