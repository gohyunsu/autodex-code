"""One persistent Qwen process per selected capture-PC GPU.

SSH carries only the selected, lossless camera pixels and the unchanged
observer prompt. The robot PC owns frame identity, evidence and voting; the
capture PCs perform inference without opening a second camera connection.
Several cameras on one PC share one loaded model and one request stream.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import re
import select
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping

from PIL import Image

from .observer import resize_classification_images


_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_SSH_DESTINATION = re.compile(r"[A-Za-z0-9_.@-]+\Z")
_WORKER_LOCK = "/tmp/precision-insertion-qwen-worker.lock"
_LOCK_BUSY_EXIT = 73
_CAMERA_IN_PROMPT = (
    re.compile(r"Valid camera IDs: ([A-Za-z0-9_-]+)\."),
    re.compile(r"\bCamera ([A-Za-z0-9_-]+)[;.]"),
    re.compile(r"\bCamera: ([A-Za-z0-9_-]+)\."),
)


def _remote_home_path(path: Path, host: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    if len(parts) < 4 or parts[:2] != ("/", "home"):
        raise ValueError("distributed Qwen assets must be under a local home")
    suffix = Path(*parts[3:])
    remote_user = host.split("@", 1)[0]
    return Path("/home") / remote_user / suffix


class DistributedQwenBackend:
    """Route per-camera Qwen calls to persistent host/GPU workers."""

    preferred_image_batch_size = 2
    supports_visible_grip_lift = True
    independent_remote_inference = True
    native_pixel_coordinates = True

    def __init__(self, *, model_id: str, worker_root: Path,
                 workers: Mapping[str, Mapping[str, str]],
                 max_input_size: tuple[int, int], max_new_tokens: int,
                 classification_image_size: tuple[int, int] | None,
                 log_path: Path | None = None):
        if (not isinstance(workers, Mapping) or not 1 <= len(workers) <= 8 or
                type(max_new_tokens) is not int or max_new_tokens <= 0 or
                len(max_input_size) != 2 or
                any(type(x) is not int or x <= 0 for x in max_input_size)):
            raise ValueError("distributed Qwen needs cameras and valid limits")
        self.model = model_id
        self.max_new_tokens = max_new_tokens
        self.classification_image_size = classification_image_size
        self.max_parallel_views = 1
        self.early_quorum_views = None
        self.log_path = None if log_path is None else Path(log_path)
        self._log_lock = threading.Lock()
        self._channels: dict[str, tuple[subprocess.Popen, threading.Lock, Path]] = {}
        self._background_pools: list[ThreadPoolExecutor] = []
        groups: dict[tuple[str, int | None, str, str | None, str], list[str]] = {}
        try:
            for camera, spec in workers.items():
                host, python = spec.get("host"), spec.get("python")
                ssh_port = spec.get("ssh_port")
                model_dir = spec.get("model_dir")
                remote_worker_root = spec.get("worker_root")
                if (not isinstance(camera, str) or not _ID.fullmatch(camera) or
                        not isinstance(host, str) or
                        not _SSH_DESTINATION.fullmatch(host) or
                        host.startswith("-") or
                        not host.split("@", 1)[0].startswith("capture") or
                        ssh_port is not None and (
                            type(ssh_port) is not int or
                            not 1 <= ssh_port <= 65535) or
                        not isinstance(python, str) or
                        not Path(python).is_absolute() or
                        model_dir is not None and (
                            not isinstance(model_dir, str) or
                            not Path(model_dir).is_absolute()) or
                        remote_worker_root is not None and (
                            not isinstance(remote_worker_root, str) or
                            not Path(remote_worker_root).is_absolute())):
                    raise ValueError(
                        "each Qwen camera needs a capture PC and absolute paths")
                remote_root = (Path(remote_worker_root)
                               if remote_worker_root is not None else
                               _remote_home_path(worker_root, host))
                groups.setdefault((host, ssh_port, python, model_dir,
                                   str(remote_root)), []).append(camera)
            if len({group[0] for group in groups}) != len(groups):
                raise ValueError(
                    "one capture host must use one Qwen model and worker root")
            self.max_parallel_views = len(groups)
            # Three or more independent GPUs can return a deterministic
            # quorum early. One or two hosts instead batch every camera per
            # GPU so a four-camera capture4 rig loads and decodes one model.
            self.early_quorum_views = (min(3, len(workers))
                                       if len(groups) >= 3 else None)
            for (host, ssh_port, python, model_dir,
                 remote_root_value), cameras in groups.items():
                cameras = sorted(cameras)
                remote_root = Path(remote_root_value)
                remote_model = (Path(model_dir) if model_dir is not None else
                                _remote_home_path(Path(model_id), host))
                command = shlex.join([
                    "flock", "-n", "-E", str(_LOCK_BUSY_EXIT), _WORKER_LOCK,
                    "env", f"PYTHONPATH={remote_root}", python, "-u", "-m",
                    "precision_insertion.distributed_qwen_worker",
                    "--camera-ids", ",".join(cameras),
                    "--model-id", str(remote_model),
                    "--max-width", str(max_input_size[0]),
                    "--max-height", str(max_input_size[1]),
                ])
                stderr = tempfile.NamedTemporaryFile(
                    prefix=f"precision-qwen-{host}-", suffix=".log",
                    delete=False)
                ssh_command = ["ssh", "-T", "-o", "BatchMode=yes", "-o",
                               "ConnectTimeout=5"]
                if ssh_port is not None:
                    ssh_command.extend(["-p", str(ssh_port)])
                ssh_command.extend([host, command])
                try:
                    process = subprocess.Popen(
                        ssh_command,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=stderr, text=True, bufsize=1)
                finally:
                    stderr.close()
                channel = (process, threading.Lock(), Path(stderr.name))
                for camera in cameras:
                    self._channels[camera] = channel
            self._wait_ready(timeout_s=300.)
        except BaseException:
            self.close()
            raise

    def _wait_ready(self, *, timeout_s: float) -> None:
        pending: dict[int, tuple[subprocess.Popen, set[str], Path]] = {}
        for camera, (process, _, path) in self._channels.items():
            row = pending.setdefault(id(process), (process, set(), path))
            row[1].add(camera)
        deadline = time.monotonic() + timeout_s
        while pending:
            streams = {process.stdout: process_id
                       for process_id, (process, _, _) in pending.items()}
            ready, _, _ = select.select(
                list(streams), [], [], max(0., deadline - time.monotonic()))
            if not ready:
                raise TimeoutError(f"capture-PC Qwen model load timed out: {sorted(pending)}")
            for stream in ready:
                process_id = streams[stream]
                process, cameras, path = pending[process_id]
                line = stream.readline()
                if not line:
                    try:
                        exit_code = process.wait(timeout=1.)
                    except subprocess.TimeoutExpired:
                        exit_code = None
                    if exit_code == _LOCK_BUSY_EXIT:
                        raise RuntimeError(
                            f"capture-PC Qwen {sorted(cameras)} already has an active "
                            "worker on this GPU")
                    raise RuntimeError(
                        f"capture-PC Qwen {sorted(cameras)} exited; see {path}")
                message = json.loads(line)
                ready_cameras = message.get("camera_ids")
                if ready_cameras is None and message.get("camera_id") is not None:
                    ready_cameras = [message["camera_id"]]
                if (message.get("type") != "ready" or
                        set(ready_cameras or []) != cameras):
                    raise RuntimeError(
                        f"capture-PC Qwen {sorted(cameras)} did not become ready")
                del pending[process_id]

    def prepare_classification_images(
        self, images: list[Image.Image],
    ) -> tuple[list[Image.Image], dict | None]:
        if self.classification_image_size is None:
            return images, None
        return resize_classification_images(images, self.classification_image_size)

    def _camera_from_prompt(self, prompt: str) -> str:
        for pattern in _CAMERA_IN_PROMPT:
            match = pattern.search(prompt)
            if match is not None and match.group(1) in self._channels:
                return match.group(1)
        raise ValueError("distributed Qwen request needs one selected camera ID")

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        return self.infer_for_camera(self._camera_from_prompt(prompt), images, prompt)

    @staticmethod
    def _encode_images(images: list[Image.Image]) -> list[str]:
        encoded = []
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
        return encoded

    @staticmethod
    def _retire_channel(process: subprocess.Popen) -> None:
        """Drop an unbound request/reply stream before another request."""
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            pass

    def infer_for_camera(
        self, camera: str, images: list[Image.Image], prompt: str,
    ) -> str:
        if (not isinstance(prompt, str) or not prompt.strip() or
                not 1 <= len(images) <= 2 or
                any(not isinstance(image, Image.Image) or image.mode != "RGB"
                    for image in images)):
            raise ValueError("distributed Qwen needs one or two ordered RGB images")
        if camera not in self._channels:
            raise ValueError("distributed Qwen camera is not selected")
        process, lock, log_file = self._channels[camera]
        encoded = self._encode_images(images)
        request_id = uuid.uuid4().hex
        request = {"type": "infer", "request_id": request_id,
                   "camera_id": camera, "images_png_base64": encoded,
                   "prompt": prompt, "max_new_tokens": self.max_new_tokens,
                   "native_pixels": self.native_pixel_coordinates}
        with lock:
            if process.poll() is not None:
                raise RuntimeError(f"capture-PC Qwen {camera} exited; see {log_file}")
            started = time.perf_counter()
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except OSError as exc:
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen {camera} request failed; see {log_file}") from exc
            ready, _, _ = select.select([process.stdout], [], [], 300.)
            if not ready:
                self._retire_channel(process)
                raise TimeoutError(f"capture-PC Qwen {camera} inference timed out")
            line = process.stdout.readline()
            if not line:
                self._retire_channel(process)
                raise RuntimeError(f"capture-PC Qwen {camera} exited; see {log_file}")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError as exc:
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen {camera} sent invalid JSON") from exc
            if (not isinstance(reply, dict) or
                    reply.get("request_id") != request_id or
                    reply.get("camera_id") != camera):
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen {camera} response identity changed")
            if reply.get("type") not in {"result", "error"}:
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen {camera} response type changed")
            elapsed = time.perf_counter() - started
        if reply.get("type") != "result" or not isinstance(reply.get("answer"), str):
            raise RuntimeError(f"capture-PC Qwen {camera}: {reply.get('error')}")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"camera_id": camera, "request_id": request_id,
                   "roundtrip_s": elapsed,
                   "inference_s": reply["inference_s"],
                   "cuda_peak_bytes": reply["cuda_peak_bytes"],
                   "image_sizes": reply["image_sizes"]}
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
        return reply["answer"]

    def _infer_many_channel(
        self,
        channel: tuple[subprocess.Popen, threading.Lock, Path],
        requests: list[tuple[str, list[Image.Image], str]],
    ) -> dict[str, str]:
        process, lock, log_file = channel
        request_id = uuid.uuid4().hex
        request = {
            "type": "infer_many", "request_id": request_id,
            "requests": [
                {"camera_id": camera,
                 "images_png_base64": self._encode_images(images),
                 "prompt": prompt}
                for camera, images, prompt in requests
            ],
            "max_new_tokens": self.max_new_tokens,
            "native_pixels": self.native_pixel_coordinates,
        }
        with lock:
            if process.poll() is not None:
                raise RuntimeError(
                    f"capture-PC Qwen batch exited; see {log_file}")
            started = time.perf_counter()
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except OSError as exc:
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen batch request failed; see {log_file}") from exc
            ready, _, _ = select.select([process.stdout], [], [], 300.)
            if not ready:
                self._retire_channel(process)
                raise TimeoutError("capture-PC Qwen batch inference timed out")
            line = process.stdout.readline()
            if not line:
                self._retire_channel(process)
                raise RuntimeError(
                    f"capture-PC Qwen batch exited; see {log_file}")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError as exc:
                self._retire_channel(process)
                raise RuntimeError(
                    "capture-PC Qwen batch sent invalid JSON") from exc
            if (not isinstance(reply, dict) or
                    reply.get("request_id") != request_id or
                    reply.get("type") not in {"batch_result", "error"}):
                self._retire_channel(process)
                raise RuntimeError("capture-PC Qwen batch response changed")
            elapsed = time.perf_counter() - started
        if reply.get("type") != "batch_result":
            raise RuntimeError(f"capture-PC Qwen batch: {reply.get('error')}")
        results = reply.get("results")
        expected = {camera for camera, _, _ in requests}
        if (not isinstance(results, list) or len(results) != len(requests) or
                {row.get("camera_id") for row in results
                 if isinstance(row, dict)} != expected or
                any(not isinstance(row, dict) or
                    not isinstance(row.get("answer"), str)
                    for row in results)):
            raise RuntimeError("capture-PC Qwen batch result identities changed")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "camera_ids": sorted(expected), "request_id": request_id,
                "batch_size": len(requests), "roundtrip_s": elapsed,
                "inference_s": reply["inference_s"],
                "cuda_peak_bytes": reply["cuda_peak_bytes"],
                "image_sizes": {result["camera_id"]: result["image_sizes"]
                                for result in results},
            }
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
        return {row["camera_id"]: row["answer"] for row in results}

    def infer_many(
        self, requests: list[tuple[list[Image.Image], str]],
    ) -> list[str]:
        if (not requests or any(
                not 1 <= len(images) <= 2 or
                any(not isinstance(image, Image.Image) or image.mode != "RGB"
                    for image in images) or
                not isinstance(prompt, str) or not prompt.strip()
                for images, prompt in requests)):
            raise ValueError(
                "distributed Qwen batch needs ordered RGB image requests")
        indexed = []
        groups: dict[int, tuple[tuple, list[tuple[str, list[Image.Image], str]]]] = {}
        for images, prompt in requests:
            camera = self._camera_from_prompt(prompt)
            channel = self._channels[camera]
            group = groups.setdefault(id(channel[0]), (channel, []))[1]
            if any(existing[0] == camera for existing in group):
                raise ValueError("distributed Qwen batch repeats a camera")
            group.append((camera, images, prompt))
            indexed.append(camera)

        def run(item):
            channel, camera_requests = item
            return self._infer_many_channel(channel, camera_requests)

        grouped = list(groups.values())
        if len(grouped) == 1:
            results = run(grouped[0])
        else:
            with ThreadPoolExecutor(max_workers=len(grouped),
                                    thread_name_prefix="precision-qwen-host") as pool:
                partial = list(pool.map(run, grouped))
            results = {camera: answer for row in partial
                       for camera, answer in row.items()}
        return [results[camera] for camera in indexed]

    def close(self) -> None:
        for pool in self._background_pools:
            pool.shutdown(wait=True)
        unique = {}
        for camera, channel in self._channels.items():
            unique.setdefault(id(channel[0]), (camera, channel))
        for camera, (process, lock, _) in unique.values():
            if process.stdin is not None and not process.stdin.closed:
                with lock:
                    if process.poll() is None:
                        try:
                            process.stdin.write(json.dumps({
                                "type": "shutdown", "camera_id": camera,
                            }) + "\n")
                            process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
        for _, (process, _, _) in unique.values():
            try:
                process.wait(timeout=15.)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.)

    def register_view_pool(self, pool: ThreadPoolExecutor) -> None:
        self._background_pools.append(pool)
