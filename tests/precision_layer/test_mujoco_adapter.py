import numpy as np
import pytest

from precision_layer.mujoco_adapter import (
    DisplacementBinding,
    FrameBinding,
    ManifoldBinding,
    MuJoCoEvidenceSpec,
    PairDistanceBinding,
    RelativeTransformBinding,
    extract_mujoco_evidence,
)


mujoco = pytest.importorskip("mujoco")


def test_mujoco_adapter_extracts_metric_evidence_without_task_dispatch():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="moving" pos="0 0 0">
          <geom name="moving_geom" type="sphere" size="0.1"/>
          <site name="moving_frame"/>
        </body>
        <body name="target" pos="0.25 0 0">
          <geom name="target_geom" type="sphere" size="0.1"/>
          <site name="target_frame"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spec = MuJoCoEvidenceSpec(
        relative_transforms={
            ("payload", "goal"): RelativeTransformBinding(
                FrameBinding("site", "moving_frame"),
                FrameBinding("site", "target_frame")),
        },
        pair_distances={
            ("payload", "boundary"): PairDistanceBinding(
                ("moving_geom",), ("target_geom",)),
        },
        displacements={
            ("payload",): DisplacementBinding(
                FrameBinding("body", "moving"), (0.0, 0.0, 0.0)),
        },
        manifolds={
            ("payload", "guide"): ManifoldBinding(
                FrameBinding("site", "moving_frame"), "line",
                origin=(0.0, 0.0, 0.0), direction_or_normal=(1.0, 0.0, 0.0)),
        },
    )
    evidence = extract_mujoco_evidence(model, data, spec)
    assert evidence.relative_transforms[("payload", "goal")][:3, 3] == pytest.approx(
        [-0.25, 0.0, 0.0])
    assert evidence.pair_separation_m[("payload", "boundary")] == pytest.approx(0.05)
    assert evidence.entity_displacement_m[("payload",)] == pytest.approx(0.0)
    assert evidence.manifold_distance_m[("payload", "guide")] == pytest.approx(0.0)


def test_mujoco_adapter_reports_penetration_with_negative_distance():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body pos="0 0 0"><geom name="a" type="sphere" size="0.1"/></body>
      <body pos="0.15 0 0"><geom name="b" type="sphere" size="0.1"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evidence = extract_mujoco_evidence(model, data, MuJoCoEvidenceSpec(
        pair_distances={("a", "b"): PairDistanceBinding(("a",), ("b",))}))
    assert evidence.pair_separation_m[("a", "b")] < 0.0
