"""Capture4 MV-GoTrack lifecycle tests without camera or GPU access."""

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np


def _load(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reinit_reuses_active_path_for_identical_mesh(tmp_path, monkeypatch):
    manager = _load(
        "capture4_mvgotrack_manager_lifecycle", "capture4/manage_mvgotrack.py")
    state = tmp_path / "state"
    monkeypatch.setattr(manager, "STATE", state)
    first_mesh = tmp_path / "first/key.obj"
    first_mesh.parent.mkdir()
    first_mesh.write_bytes(b"same geometry")
    first = SimpleNamespace(
        object="key", mesh=first_mesh, session=tmp_path / "session_1")
    manager._write_active_tracker(first)

    next_mesh = tmp_path / "second/key.obj"
    next_mesh.parent.mkdir()
    next_mesh.write_bytes(first_mesh.read_bytes())
    second = SimpleNamespace(
        object="key", mesh=next_mesh, session=tmp_path / "session_2",
        init_pose=tmp_path / "pose.txt", endpoint="tcp://capture4:5591")

    assert manager._reinit_mesh(second) == first_mesh.resolve()
    calls = []
    monkeypatch.setattr(manager, "_ensure_stream", lambda: calls.append("stream"))
    monkeypatch.setattr(manager, "_pid", lambda name: 123 if name == "tracker" else None)
    monkeypatch.setattr(
        manager.subprocess, "run",
        lambda command, check: calls.append((command, check)))
    monkeypatch.setattr(
        manager, "_stop",
        lambda _name: (_ for _ in ()).throw(AssertionError("no restart")))

    manager.start_or_reinit(second)
    command = calls[1][0]
    assert command[command.index("--mesh") + 1] == str(first_mesh.resolve())
    assert str(next_mesh) not in command


def test_changed_mesh_requires_tracker_restart(tmp_path, monkeypatch):
    manager = _load(
        "capture4_mvgotrack_manager_restart", "capture4/manage_mvgotrack.py")
    state = tmp_path / "state"
    monkeypatch.setattr(manager, "STATE", state)
    old = tmp_path / "old.obj"
    old.write_bytes(b"old")
    manager._write_active_tracker(SimpleNamespace(
        object="key", mesh=old, session=tmp_path / "old_session"))
    new = tmp_path / "new.obj"
    new.write_bytes(b"new")
    args = SimpleNamespace(
        object="key", mesh=new, session=tmp_path / "new_session",
        init_pose=tmp_path / "pose.txt", endpoint="tcp://capture4:5591")
    calls = []
    monkeypatch.setattr(manager, "_ensure_stream", lambda: None)
    monkeypatch.setattr(manager, "_pid", lambda name: 123 if name == "tracker" else None)
    monkeypatch.setattr(manager, "_stop", lambda name: calls.append(("stop", name)))
    monkeypatch.setattr(
        manager, "_spawn", lambda name, command: calls.append((name, command)))
    monkeypatch.setattr(manager, "_write_active_tracker",
                        lambda value: calls.append(("active", value)))
    monkeypatch.setattr(manager, "_tracker_command", lambda _args: ["tracker"])

    manager.start_or_reinit(args)
    assert calls == [
        ("stop", "tracker"), ("tracker", ["tracker"]), ("active", args)]


def test_empty_shared_memory_slot_is_not_encoded(monkeypatch):
    stream = _load(
        "capture4_mvgotrack_stream_empty", "capture4/mvgotrack_stream.py")
    monkeypatch.setattr(
        stream.cv2, "imencode",
        lambda *_args: (_ for _ in ()).throw(AssertionError("empty encoded")))
    assert stream.encode_frame(None, []) is None
    assert stream.encode_frame(np.empty((0, 0, 3), dtype=np.uint8), []) is None


def test_valid_shared_memory_slot_is_encoded(monkeypatch):
    stream = _load(
        "capture4_mvgotrack_stream_valid", "capture4/mvgotrack_stream.py")
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(
        stream.cv2, "imencode",
        lambda extension, value, parameters: (
            extension == ".jpg" and value is frame and parameters == [7],
            np.asarray([1, 2, 3], dtype=np.uint8)))
    assert stream.encode_frame(frame, [7]) == b"\x01\x02\x03"
