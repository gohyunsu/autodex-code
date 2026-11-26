import json

import numpy as np

from precision_insertion.mvgotrack import MVGoTrackPose, MVGoTrackPoseJournal


def _pose(seq: int, *, valid: bool = True) -> MVGoTrackPose:
    return MVGoTrackPose(
        seq=seq,
        object_name="precision_key_cylinder_r15_h80",
        pose_world=np.eye(4),
        valid=valid,
        status="tracking" if valid else "lost",
        score=None,
        frame_ids={"camera_a": seq, "camera_b": seq + 1},
        stamp_capture=float(seq),
        stamp_publish=float(seq) + 0.1,
        calibration_session="calibration:test",
    )


def test_pose_journal_preserves_valid_and_invalid_native_records(tmp_path):
    path = tmp_path / "mvgotrack_pose_stream.jsonl"
    journal = MVGoTrackPoseJournal(path)
    journal.append(_pose(1))
    journal.append(_pose(2, valid=False))
    journal.append(_pose(3))
    journal.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["valid"] for row in rows] == [True, False, True]
    assert all(row["received_unix_ns"] > 0 for row in rows)
    assert all(row["received_monotonic_ns"] > 0 for row in rows)
    assert journal.error is None


def test_journal_close_is_idempotent(tmp_path):
    path = tmp_path / "mvgotrack_pose_stream.jsonl"
    journal = MVGoTrackPoseJournal(path)
    journal.append(_pose(4))

    journal.close()
    journal.close()

    assert json.loads(path.read_text())["seq"] == 4
