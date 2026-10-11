"""A failed trial's metric advice shifts only the next wrist command."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.next_trial_xy import (  # noqa: E402
    compensated_key_hand_relation,
)


def test_socket_xy_error_becomes_wrist_only_compensation(tmp_path):
    mode = select_mode("cylinder", 20)
    angle = np.pi / 2
    T_socket_key = np.eye(4)
    T_socket_key[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    path = AssetPaths(tmp_path, mode).task_geometry
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "T_socket_key_verification": T_socket_key.tolist(),
    }), encoding="utf-8")
    held = np.eye(4)
    held[:3, 3] = [0.01, -0.02, 0.03]
    delta = (0.0006, -0.0008)
    corrected = compensated_key_hand_relation(
        mode=mode, shared_root=tmp_path, T_key_hand=held,
        wrist_correction_socket_m=delta)
    displacement = (T_socket_key @ corrected)[:3, 3] - (
        T_socket_key @ held)[:3, 3]
    assert displacement == pytest.approx([*delta, 0.0])
    assert corrected[:3, :3] == pytest.approx(held[:3, :3])
    assert held[:3, 3] == pytest.approx([0.01, -0.02, 0.03])
    accumulated = compensated_key_hand_relation(
        mode=mode, shared_root=tmp_path, T_key_hand=held,
        wrist_correction_socket_m=(0.0011, 0.0))
    displacement = (T_socket_key @ accumulated)[:3, 3] - (
        T_socket_key @ held)[:3, 3]
    assert displacement == pytest.approx([0.0011, 0.0, 0.0])


def test_square_relation_cannot_receive_xy_error_correction(tmp_path):
    mode = select_mode("square", 1.5)
    relation = np.eye(4)
    assert compensated_key_hand_relation(
        mode=mode, shared_root=tmp_path, T_key_hand=relation,
        wrist_correction_socket_m=(0.0, 0.0)) == pytest.approx(relation)
    with pytest.raises(ValueError, match="square mode has no XY"):
        compensated_key_hand_relation(
            mode=mode, shared_root=tmp_path, T_key_hand=relation,
            wrist_correction_socket_m=(0.001, 0.0))
