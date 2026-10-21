"""One persistent Qwen process per selected AutoDex capture PC camera.

SSH carries only the selected, lossless camera pixels and the unchanged
observer prompt. The robot PC owns frame identity, evidence and voting; the
capture PCs perform inference without opening a second camera connection.
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
_WORKER_LOCK = "/tmp/precision-insertion-qwen-worker.lock"
_LOCK_BUSY_EXIT = 73
_CAMERA_IN_PROMPT = (
    re.compile(r"Valid camera IDs: ([A-Za-z0-9_-]+)\."),
    re.compile(r"\bCamera ([A-Za-z0-9_-]+)[;.]"),
    re.compile(r"\bCamera: ([A-Za-z0-9_-]+)\."),
)


def _remote_home_path(path: Path, host: str) -> Path:
    try:
        suffix = Path(path).expanduser().resolve().relative_to("/home/robot")
    except ValueError as exc:
        raise ValueError("distributed Qwen assets must be under /home/robot") from exc
    return Path("/home") / host / suffix


class DistributedQwenBackend:
    """Route independent per-camera Qwen calls to five persistent SSH workers."""

    preferred_image_batch_size = 2
    supports_visible_grip_lift = True
    independent_remote_inference = True
    native_pixel_coordinates = True

    def __init__(self, *, model_id: str, worker_root: Path,
                 workers: Mapping[str, Mapping[str, str]],
                 max_input_size: tuple[int, int], max_new_tokens: int,
                 classification_image_size: tuple[int, int] | None,
                 log_path: Path | None = None):
        if (not isinstance(workers, Mapping) or len(workers) != 5 or
                type(max_new_tokens) is not int or max_new_tokens <= 0 or
                len(max_input_size) != 2 or
                any(type(x) is not int or x <= 0 for x in max_input_size)):
            raise ValueError("distributed Qwen needs five cameras and valid limits")
        self.model = model_id
        self.max_new_tokens = max_new_tokens
        self.classification_image_size = classification_image_size
        self.max_parallel_views = len(workers)
        self.early_quorum_views = 3
        self.log_path = None if log_path is None else Path(log_path)
        self._log_lock = threading.Lock()
        self._channels: dict[str, tuple[subprocess.Popen, threading.Lock, Path]] = {}
        self._background_pools: list[ThreadPoolExecutor] = []
        hosts = set()
        try:
            for camera, spec in workers.items():
                host, python = spec.get("host"), spec.get("python")
                model_dir = spec.get("model_dir")
                if (not isinstance(camera, str) or not _ID.fullmatch(camera) or
                        not isinstance(host, str) or not _ID.fullmatch(host) or
                        not host.startswith("capture") or host in hosts or
                        not isinstance(python, str) or
                        not Path(python).is_absolute() or
                        model_dir is not None and (
                            not isinstance(model_dir, str) or
                            not Path(model_dir).is_absolute())):
                    raise ValueError("each Qwen camera needs a unique capture PC and Python")
                hosts.add(host)
                remote_root = _remote_home_path(worker_root, host)
                remote_model = (Path(model_dir) if model_dir is not None else
                                _remote_home_path(Path(model_id), host))
                command = shlex.join([
                    "flock", "-n", "-E", str(_LOCK_BUSY_EXIT), _WORKER_LOCK,
                    "env", f"PYTHONPATH={remote_root}", python, "-u", "-m",
                    "precision_insertion.distributed_qwen_worker",
                    "--camera-id", camera, "--model-id", str(remote_model),
                    "--max-width", str(max_input_size[0]),
                    "--max-height", str(max_input_size[1]),
                ])
                stderr = tempfile.NamedTemporaryFile(
                    prefix=f"precision-qwen-{host}-{camera}-", suffix=".log",
                    delete=False)
                try:
                    process = subprocess.Popen(
                        ["ssh", "-T", "-o", "BatchMode=yes", "-o",
                         "ConnectTimeout=5", host, command],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=stderr, text=True, bufsize=1)
                finally:
                    stderr.close()
                self._channels[camera] = (
                    process, threading.Lock(), Path(stderr.name))
            self._wait_ready(timeout_s=300.)
        except BaseException:
            self.close()
            raise

    def _wait_ready(self, *, timeout_s: float) -> None:
        pending = set(self._channels)
        deadline = time.monotonic() + timeout_s
        while pending:
            streams = {self._channels[camera][0].stdout: camera for camera in pending}
            ready, _, _ = select.select(
                list(streams), [], [], max(0., deadline - time.monotonic()))
            if not ready:
                raise TimeoutError(f"capture-PC Qwen model load timed out: {sorted(pending)}")
            for stream in ready:
                camera = streams[stream]
                line = stream.readline()
                if not line:
                    path = self._channels[camera][2]
                    process = self._channels[camera][0]
                    try:
                        exit_code = process.wait(timeout=1.)
                    except subprocess.TimeoutExpired:
                        exit_code = None
                    if exit_code == _LOCK_BUSY_EXIT:
                        raise RuntimeError(
                            f"capture-PC Qwen {camera} already has an active "
                            "worker on this GPU")
                    raise RuntimeError(
                        f"capture-PC Qwen {camera} exited; see {path}")
                message = json.loads(line)
                if (message.get("type") != "ready" or
                        message.get("camera_id") != camera):
                    raise RuntimeError(f"capture-PC Qwen {camera} did not become ready")
                pending.remove(camera)

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
        encoded = []
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
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
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], 300.)
            if not ready:
                raise TimeoutError(f"capture-PC Qwen {camera} inference timed out")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"capture-PC Qwen {camera} exited; see {log_file}")
            reply = json.loads(line)
            elapsed = time.perf_counter() - started
        if (reply.get("request_id") != request_id or
                reply.get("camera_id") != camera):
            raise ValueError("capture-PC Qwen response identity changed")
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

    def close(self) -> None:
        for pool in self._background_pools:
            pool.shutdown(wait=True)
        for camera, (process, lock, _) in self._channels.items():
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
        for process, _, _ in self._channels.values():
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
