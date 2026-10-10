"""Contracts for read-only, socket-frame, multi-view XY choice proposals."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.xy_voting import (  # noqa: E402
    ViewVote, XYChoice, axis_1mm_proposals, project_choice_anchors,
    resolve_multiview_choice, validate_choices,
)


CHOICES = (
    XYChoice("C0", (0.0, 0.0)),
    XYChoice("C1", (0.001, 0.0)),
    XYChoice("C2", (-0.001, 0.0)),
)


def _vote(camera, selected="C2", *, time=1.0, visible=True,
          failure="misaligned"):
    return ViewVote(camera, time, visible, selected, failure)


def _resolve(votes, *, current=(0.0, 0.0), grasp_held=True,
             hard_abort=False, decision_time=1.01):
    return resolve_multiview_choice(
        CHOICES, votes, current_offset_socket_m=current,
        grasp_held=grasp_held, hard_abort=hard_abort,
        max_step_m=0.001, max_total_m=0.003,
        max_timestamp_skew_s=0.02, decision_timestamp_s=decision_time,
        max_frame_age_s=0.2,
    )


def test_two_agreeing_views_propose_exact_catalog_offset_not_average():
    result = _resolve([
        _vote("front", "C2", time=10.0),
        _vote("side", "C2", time=10.01, failure="rim_jam"),
        _vote("top", "abstain", time=10.01, visible=False, failure="occluded"),
    ], decision_time=10.02)
    assert result.status == "propose"
    assert result.choice_id == "C2"
    assert result.offset_socket_m == (-0.001, 0.0)
    assert result.supporting_cameras == ("front", "side")
    assert result.to_record()["scope"].startswith("read_only")


def test_tie_single_view_and_unsynchronized_images_abstain():
    assert _resolve([_vote("front", "C1"), _vote("side", "C2")]).status == "abstain"
    assert _resolve([_vote("front")]).reason == "insufficient_camera_views"
    assert _resolve([_vote("front", time=1.0),
                     _vote("side", time=1.1)]).reason == "unsynchronized_images"
    assert _resolve([_vote("front", time=1.0),
                     _vote("side", time=1.01)],
                    decision_time=2.0).reason == "stale_or_future_images"


def test_slip_or_hard_abort_stops_without_xy_motion():
    assert _resolve([_vote("front", failure="slip"),
                     _vote("side")]).status == "stop"
    assert _resolve([_vote("front"), _vote("side")], grasp_held=False).status == "stop"
    assert _resolve([_vote("front"), _vote("side")], hard_abort=True).status == "stop"
    assert _resolve([_vote("front"), _vote("side")], hard_abort=None).status == "stop"


def test_budget_and_no_op_are_not_motion_authorizations():
    assert _resolve([_vote("front"), _vote("side")],
                    current=(0.001, 0.0)).reason == "step_budget_exceeded"
    no_op = _resolve([_vote("front", "C0"), _vote("side", "C0")])
    assert no_op.status == "no_correction"
    assert no_op.choice_id == "C0"
    assert no_op.to_record()["scope"].endswith("not_motion_authorization")


def test_invalid_choices_or_occluded_vote_fail_closed():
    with pytest.raises(ValueError, match="duplicate XY choice ID"):
        validate_choices([CHOICES[0], CHOICES[0]])
    with pytest.raises(ValueError, match="same XY offset"):
        validate_choices([CHOICES[0], XYChoice("other", (0.0, 0.0))])
    with pytest.raises(ValueError, match="unknown XY choice"):
        _resolve([_vote("front", "C77"), _vote("side")])
    with pytest.raises(ValueError, match="occluded view"):
        _resolve([_vote("front", visible=False), _vote("side")])


def test_projection_uses_metric_socket_frame_and_rejects_behind_camera():
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    T_camera_socket = np.eye(4)
    T_camera_socket[2, 3] = 1.0
    projection = project_choice_anchors(
        CHOICES, T_camera_socket=T_camera_socket, intrinsics=K,
        socket_plane_z_m=0.0, image_size_wh=(1000, 800),
    )
    assert projection["C0"] == (500.0, 400.0)
    assert projection["C1"] == pytest.approx((501.0, 400.0))
    assert projection["C2"] == pytest.approx((499.0, 400.0))
    T_camera_socket[2, 3] = -1.0
    assert project_choice_anchors(
        CHOICES, T_camera_socket=T_camera_socket, intrinsics=K,
        socket_plane_z_m=0.0, image_size_wh=(1000, 800),
    ) == {}


def test_one_mm_axis_choices_and_noncompliant_votes_fail_closed():
    generated = axis_1mm_proposals((0.002, -0.001))
    assert [choice.choice_id for choice in generated] == [
        "hold", "x_plus_1mm", "x_minus_1mm", "y_plus_1mm", "y_minus_1mm",
    ]
    for choice in generated[1:]:
        delta = np.subtract(choice.offset_socket_m, generated[0].offset_socket_m)
        assert np.linalg.norm(delta) == pytest.approx(0.001)
        assert np.count_nonzero(np.abs(delta) > 1e-12) == 1
    for wrong in ((0.0005, 0.0), (0.001, 0.001)):
        result = resolve_multiview_choice(
            [XYChoice("hold", (0.0, 0.0)), XYChoice("wrong", wrong)],
            [_vote("front", "wrong"), _vote("side", "wrong")],
            current_offset_socket_m=(0.0, 0.0),
            grasp_held=True, hard_abort=False,
            max_step_m=0.002, max_total_m=0.003,
            max_timestamp_skew_s=0.02, decision_timestamp_s=1.01,
            max_frame_age_s=0.2,
        )
        assert result.status == "abstain"
        assert result.reason == "not_one_mm_axis_step"
