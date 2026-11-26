#!/usr/bin/env python3
"""Validate generic precision evidence against the cylinder pilot backend.

This is an integration check, not an RSS benchmark result.  It imports the
independent blind-assembly reproduction, extracts evidence through the generic
MuJoCo adapter, and compares the adapter's socket-axis distance with the
environment's separately implemented metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from precision_layer.evaluator import evaluate_predicate
from precision_layer.mujoco_adapter import (
    FrameBinding,
    ManifoldBinding,
    MuJoCoEvidenceSpec,
    PairDistanceBinding,
    RelativeTransformBinding,
    extract_mujoco_evidence,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind-assembly-root", type=Path,
        default=Path("/home/hyunsu/blind-assembly-reproduction"))
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/cylinder_socket_key_gap5_paper_success.yaml"))
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("generated/cylinder_socket_gap5"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _predicate(
    predicate_id: str,
    predicate_type: str,
    entities: tuple[str, ...],
    tolerance_m: float,
    parameters: dict | None = None,
) -> dict:
    return {
        "predicate_id": predicate_id,
        "type": predicate_type,
        "entities": list(entities),
        "tolerance": {"value": tolerance_m, "unit": "m"},
        "parameters": parameters or {},
    }


def main() -> None:
    args = _arguments()
    root = args.blind_assembly_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from blind_assembly.env import CylinderSocketEnv

    config = args.config if args.config.is_absolute() else root / args.config
    model_dir = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
    environment = CylinderSocketEnv(config, model_dir, seed=args.seed)
    _, info = environment.reset(seed=args.seed, centered=False)

    socket_rotation = environment.data.xmat[
        environment.socket_body_id].reshape(3, 3).copy()
    entry = environment.data.site_xpos[
        environment.socket_entry_site_id].copy()
    socket_geom_names = tuple(
        environment.model.geom(index).name
        for index in sorted(environment.socket_collision_geom_ids)
    )
    specification = MuJoCoEvidenceSpec(
        relative_transforms={
            ("payload", "goal"): RelativeTransformBinding(
                FrameBinding("site", "key_tip"),
                FrameBinding("site", "socket_entry")),
        },
        pair_distances={
            ("payload", "boundary"): PairDistanceBinding(
                ("key_collision",), socket_geom_names),
        },
        manifolds={
            ("payload", "axis"): ManifoldBinding(
                FrameBinding("site", "key_tip"),
                "line",
                origin=tuple(entry),
                direction_or_normal=tuple(-socket_rotation[:, 2])),
        },
    )
    evidence = extract_mujoco_evidence(
        environment.model, environment.data, specification)
    transform = evidence.relative_transforms[("payload", "goal")]
    radial_adapter = evidence.manifold_distance_m[("payload", "axis")]
    radial_environment = float(info["radial_offset_m"])
    if not np.isclose(radial_adapter, radial_environment, atol=1e-10, rtol=0.0):
        raise RuntimeError(
            "adapter/environment radial metrics disagree: "
            f"{radial_adapter} versus {radial_environment}")

    axis_margin = evaluate_predicate(
        _predicate(
            "axis_alignment",
            "pose_equality",
            ("payload", "goal"),
            tolerance_m=0.001,
            parameters={
                "orientation_mode": "axis_alignment",
                "local_axis": [0.0, 0.0, 1.0],
                "target_axis": [0.0, 0.0, -1.0],
                "allow_axis_sign_flip": False,
                "rotation_length_scale_m_per_rad": 0.05,
            }),
        evidence)

    print(json.dumps({
        "claim_scope": "integration_semantics_only",
        "seed": args.seed,
        "reset_attempts": int(info["reset_sampling_attempts"]),
        "tip_in_socket_frame_m": transform[:3, 3].tolist(),
        "radial_from_adapter_m": radial_adapter,
        "radial_from_independent_environment_m": radial_environment,
        "tilt_from_independent_environment_deg": float(info["tilt_deg"]),
        "axis_alignment_margin_m": axis_margin.signed_margin,
        "minimum_signed_key_socket_distance_m": (
            evidence.pair_separation_m[("payload", "boundary")]),
        "socket_collision_geom_count": len(socket_geom_names),
        "passes_R1": False,
        "R1_missing": [
            "procedural_physics_geometry_families",
            "corrupted_nominal_trajectories",
            "joint_Z1_Z4_physics_split",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
