"""Distributed Qwen staging supports one host serving four cameras."""

import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_capture_qwen  # noqa: E402


def test_setup_uses_explicit_shared_host_mapping(tmp_path, capsys):
    source = tmp_path / "franka_runtime.json"
    output = tmp_path / "franka_runtime_capture4_qwen.json"
    worker_root = tmp_path / "qwen_worker"
    workers_file = tmp_path / "workers.json"
    source.write_text(json.dumps({"vlm": {
        "mode": "local", "model_id": "/local/model",
    }}), encoding="utf-8")
    shared = {
        "host": "capture4@192.0.2.4", "ssh_port": 77,
        "python": "/home/capture4/python",
        "model_dir": "/home/capture4/model",
        "worker_root": "/home/capture4/qwen_worker",
    }
    workers_file.write_text(json.dumps({
        camera: shared for camera in ("100", "200", "300", "400")
    }), encoding="utf-8")
    setup_capture_qwen.main([
        "--source-config", str(source),
        "--output-config", str(output),
        "--worker-root", str(worker_root),
        "--workers-json", str(workers_file),
    ])
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["vlm"]["mode"] == "distributed"
    assert record["vlm"]["workers"]["400"] == shared
    assert record["vlm"]["worker_root"] == str(worker_root)
    worker = worker_root / "precision_insertion/distributed_qwen_worker.py"
    assert worker.is_file()
    manifest = json.loads(
        (worker_root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "precision_insertion_qwen_worker_stage_v1"
    assert manifest["workers"] == {
        camera: shared for camera in ("100", "200", "300", "400")
    }
    declared = {row["path"]: row for row in manifest["worker_files"]}
    relative = "precision_insertion/distributed_qwen_worker.py"
    assert declared[relative]["sha256"] == hashlib.sha256(
        worker.read_bytes()).hexdigest()
    assert manifest["output_config"]["sha256"] == hashlib.sha256(
        output.read_bytes()).hexdigest()
    output_text = capsys.readouterr().out
    assert str(output) in output_text
    assert str(worker_root / "MANIFEST.json") in output_text
