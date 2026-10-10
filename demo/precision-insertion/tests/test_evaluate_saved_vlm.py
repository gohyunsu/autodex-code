"""A batch VLM comparison is an image audit, not a robot success gate."""

import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_saved_vlm  # noqa: E402


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, task, truth in (
            ("grasp_miss", "lift", "miss"),
            ("rim_jam", "insertion_visual", "rim_jam")):
        annotation = root / f"{case_id}_annotation.json"
        annotation.write_text(json.dumps({
            "schema": "precision_insertion_visual_annotation_v1",
            "case_id": case_id, "task": task, "label": truth,
            "source": "independent_human_review", "reviewer_id": "r1",
            "evidence": "review of both raw views",
        }), encoding="utf-8")
        before, after = (("before_grasp", "after_lift") if task == "lift"
                         else ("preinsert", "final_or_abort"))
        views = []
        for camera in ("front", "side"):
            view = {"camera_id": camera}
            for index, phase in enumerate((before, after)):
                image = root / f"{case_id}_{camera}_{phase}.png"
                Image.new("RGB", (16, 12), (50 * index, 10, 50)).save(image)
                view[phase] = {
                    "path": image.name, "sha256": _digest(image),
                    "timestamp_s": float(10 + index) + (
                        .001 if camera == "side" else 0),
                }
            views.append(view)
        cases.append({
            "case_id": case_id, "task": task,
            "annotation": {"path": annotation.name,
                           "sha256": _digest(annotation)},
            "views": views,
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "precision_insertion_vlm_visual_benchmark_v1",
        "max_phase_skew_s": .01, "cases": cases,
    }), encoding="utf-8")
    return manifest


def test_benchmark_loads_one_model_and_reports_false_positives(tmp_path,
                                                               monkeypatch):
    manifest = _dataset(tmp_path / "dataset")
    answers = [
        '{"class":"held","evidence_views":["front","side"],"evidence":"visible"}',
        '{"visual_class":"normal_appearance","evidence_views":'
        '["front","side"],"evidence":"looks aligned"}',
    ]
    loads, calls = [], []

    class Backend:
        def infer(self, images, prompt):
            calls.append((images, prompt))
            return answers.pop(0)

    def load(**kwargs):
        loads.append(kwargs)
        return Backend()

    monkeypatch.setattr(evaluate_saved_vlm, "load_vlm_backend", load)
    output = tmp_path / "benchmark.json"
    assert evaluate_saved_vlm.main([
        "--manifest", str(manifest), "--backend", "local",
        "--model-id", "Qwen/test", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(loads) == 1
    assert loads[0]["require_native_pixels"] is False
    assert len(calls) == 2 and all(len(images) == 4 for images, _ in calls)
    assert report["robot_ready"] is False
    assert report["summary"]["lift"]["target_class"] == "held"
    assert report["summary"]["lift"]["target_class_false_positives"] == 1
    assert report["summary"]["insertion_visual"]["target_class"] == (
        "normal_appearance")
    assert report["summary"]["insertion_visual"][
        "target_class_false_positives"] == 1
    assert report["cases"][0]["correct"] is False
    with pytest.raises(FileExistsError):
        evaluate_saved_vlm.main([
            "--manifest", str(manifest), "--backend", "local",
            "--model-id", "Qwen/test", "--output", str(output)])


def test_changed_image_or_annotation_rejected_before_model_load(tmp_path,
                                                                 monkeypatch):
    manifest = _dataset(tmp_path / "dataset")
    monkeypatch.setattr(evaluate_saved_vlm, "load_vlm_backend",
                        lambda **kwargs: pytest.fail("must not load model"))
    changed = tmp_path / "dataset" / "grasp_miss_front_after_lift.png"
    Image.new("RGB", (16, 12), "green").save(changed)
    with pytest.raises(ValueError, match="bytes changed"):
        evaluate_saved_vlm.main([
            "--manifest", str(manifest), "--backend", "local",
            "--model-id", "Qwen/test", "--output", str(tmp_path / "out.json")])
    _dataset(tmp_path / "dataset")
    annotation = tmp_path / "dataset" / "rim_jam_annotation.json"
    annotation.write_text(annotation.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        evaluate_saved_vlm.main([
            "--manifest", str(manifest), "--backend", "local",
            "--model-id", "Qwen/test", "--output", str(tmp_path / "out.json")])


def test_gemini_batch_requires_explicit_external_image_consent(tmp_path):
    manifest = _dataset(tmp_path / "dataset")
    with pytest.raises(SystemExit, match="2"):
        evaluate_saved_vlm.main([
            "--manifest", str(manifest), "--backend", "gemini",
            "--model-id", "gemini-test", "--output", str(tmp_path / "out.json")])
