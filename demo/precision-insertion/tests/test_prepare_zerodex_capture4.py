"""A rebuilt capture4 stage uses the selected machine profile values."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_zerodex_capture4 as prepare  # noqa: E402

build = prepare.build


def test_builder_uses_configured_capture_ip_and_camera_ids(tmp_path):
    stage = tmp_path / "source"
    calibration = tmp_path / "calibration"
    output = tmp_path / "output"
    (stage / "static_preflight_board9_gap20_20261013").mkdir(parents=True)
    (stage / "live_board9_gap20_20261013").mkdir()
    calibration.mkdir()
    for name in ("intrinsics.json", "extrinsics.json"):
        (calibration / name).write_text("{}\n", encoding="utf-8")
    (stage / "static_preflight_board9_gap20_20261013" /
     "base_scene.json").write_text("{}\n", encoding="utf-8")
    (stage / "live_board9_gap20_20261013" /
     "launch.json").write_text("{}\n", encoding="utf-8")
    (stage / "franka_runtime_board9_gap20_distributed.json").write_text(
        json.dumps({"vlm": {"mode": "distributed", "workers": {}},
                    "held_motion_monitor": {}}), encoding="utf-8")
    c2r = tmp_path / "C2R.npy"
    c2r.write_bytes(b"placeholder")
    result = build(
        source_stage=stage, shared_root=tmp_path / "shared_data",
        calibration_dir=calibration, c2r_npy=c2r,
        output_dir=output, qwen_model_dir=tmp_path / "qwen",
        capture_ip="192.0.2.4", camera_ids=("100", "200", "300", "400"),
        board_fit_camera_ids=("100", "200", "400"))
    launch = json.loads(Path(result["launch"]).read_text(encoding="utf-8"))
    assert launch["mode"] == {"family": "cylinder", "gap_mm": 5}
    assert launch["capture_ips"] == ["192.0.2.4"]
    assert launch["camera_ids"] == ["100", "200", "300", "400"]
    assert launch["board_fit_camera_ids"] == ["100", "200", "400"]
    runtime = json.loads((output / "franka_runtime.json").read_text(
        encoding="utf-8"))
    assert runtime["held_motion_monitor"]["daemon_binary_path"] == str(
        output / "robot_deps/restored/home/robot/paradex/cpp/"
                 "franka_daemon/build/franka_daemon")


def test_cli_defaults_use_independent_project_roots(
    tmp_path, monkeypatch, capsys,
):
    nas = tmp_path / "nas_project"
    local = tmp_path / "local_project"
    monkeypatch.setenv("PRECISION_NAS_PROJECT_ROOT", str(nas))
    monkeypatch.setenv("PRECISION_LOCAL_PROJECT_ROOT", str(local))
    monkeypatch.setattr(prepare, "build", lambda **kwargs: {
        key: str(value) if isinstance(value, Path) else value
        for key, value in kwargs.items()
    })
    result = prepare.main([
        "--source-stage", str(tmp_path / "source"),
        "--calibration-dir", str(tmp_path / "calibration"),
        "--c2r-npy", str(tmp_path / "C2R.npy"),
    ])
    assert result == 0
    record = json.loads(capsys.readouterr().out)
    shared = nas / "runtime/active_shared_data"
    assert record["shared_root"] == str(shared)
    assert record["output_dir"] == str(
        shared / "AutoDex/precision_insertion/zerodex_capture4")
    assert record["qwen_model_dir"] == str(
        local / "cache/models/Qwen3-VL-2B-Instruct")
