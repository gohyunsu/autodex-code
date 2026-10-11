"""The live physical route can locate a source-bound v8 grasp calibration."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.insertion_grasp_calibration import (  # noqa: E402
    build_insertion_grasp_calibration, calibration_path_for_candidate,
)
from precision_insertion.physical_grasp_calibration import (  # noqa: E402
    verify_physical_held_relation,
)
from test_physical_grasp_calibration import _sample  # noqa: E402


MODE = select_mode("square", 1.5)
KEY = ("table", "000", "084")


def test_insertion_grasp_calibration_uses_eligible_exact_v8_grasp(
    tmp_path, monkeypatch,
):
    import precision_insertion.insertion_grasp_calibration as module

    candidate = AssetPaths(tmp_path, MODE).candidate_dir / Path(*KEY)
    candidate.mkdir(parents=True)
    np.save(candidate / "wrist_se3.npy", np.eye(4))
    catalog_path = tmp_path / "endpoint_catalog.json"
    catalog = {"candidates": [{"key": list(KEY),
                               "tabletop_pose_stem": "000"}]}
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    def select(record, *, expected_mode, tabletop_pose_stem):
        assert record == catalog
        assert expected_mode == MODE
        assert tabletop_pose_stem == "000"
        return {"status": "candidates_available", "candidates": [
            {"key": list(KEY), "candidate_dir": str(candidate)}]}

    monkeypatch.setattr(module, "select_pose_candidates", select)
    samples = [_sample(tmp_path, i, KEY) for i in range(5)]
    kwargs = dict(
        shared_root=tmp_path, mode=MODE, catalog_path=catalog_path,
        candidate_id="/".join(KEY),
        sample_paths=[Path(row["evidence_path"]) for row in samples],
        output_root=tmp_path / "calibrations",
        max_nominal_translation_drift_m=.003,
        max_nominal_rotation_drift_deg=5.)
    output = build_insertion_grasp_calibration(**kwargs)
    assert output == calibration_path_for_candidate(
        kwargs["output_root"], MODE, "/".join(KEY))
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["sample_count"] == 5
    assert record["robot_ready"] is False
    assert verify_physical_held_relation(
        record=record, mode=MODE, shared_root=tmp_path,
        candidate_key=KEY, candidate_dir=candidate) == record
    with pytest.raises(FileExistsError):
        build_insertion_grasp_calibration(**kwargs)


def test_grasp_calibration_rejects_unsafe_ids_and_unselected_grasp(
    tmp_path, monkeypatch,
):
    import precision_insertion.insertion_grasp_calibration as module

    with pytest.raises(ValueError, match="candidate ID"):
        calibration_path_for_candidate(tmp_path, MODE, "../000/084")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"candidates": [
        {"key": list(KEY), "tabletop_pose_stem": "000"}]}), encoding="utf-8")
    monkeypatch.setattr(module, "select_pose_candidates", lambda *a, **kw: {
        "status": "no_eligible_in_screened_pool", "candidates": []})
    with pytest.raises(ValueError, match="endpoint-eligible"):
        build_insertion_grasp_calibration(
            shared_root=tmp_path, mode=MODE, catalog_path=catalog_path,
            candidate_id="/".join(KEY), sample_paths=[],
            output_root=tmp_path / "calibrations",
            max_nominal_translation_drift_m=.003,
            max_nominal_rotation_drift_deg=5.)
