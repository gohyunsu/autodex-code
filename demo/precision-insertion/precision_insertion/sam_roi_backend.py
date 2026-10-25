"""Keep a local SAM3 ROI process warm for same-frame saved XY diagnosis."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import select
import subprocess
import threading
from uuid import uuid4

import numpy as np
from PIL import Image


class SamRoiBackend:
    def __init__(
        self, *, python_path: Path, checkpoint: Path, stderr_path: Path,
        worker_path: Path | None = None, startup_timeout_s: float = 180.,
        response_timeout_s: float = 120.,
    ) -> None:
        python = Path(python_path).expanduser().resolve(strict=True)
        model = Path(checkpoint).expanduser().resolve(strict=True)
        worker = (Path(worker_path).expanduser().resolve(strict=True)
                  if worker_path is not None else
                  Path(__file__).resolve().parents[1] / "sam_roi_worker.py")
        if not worker.is_file():
            raise FileNotFoundError("SAM3 ROI worker script is missing")
        log_path = Path(stderr_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._response_timeout_s = response_timeout_s
        self._failed = False
        self.model = str(model)
        self._process = subprocess.Popen(
            [str(python), str(worker), "--checkpoint", str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1)
        try:
            ready = self._read_line(startup_timeout_s)
            if ready.get("type") != "ready" or ready.get("model") != str(model):
                raise RuntimeError("SAM3 ROI worker did not load the selected model")
        except BaseException:
            self.close()
            raise

    def _read_line(self, timeout_s: float) -> dict:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("SAM3 ROI worker has no output pipe")
        ready, _, _ = select.select([stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError("SAM3 ROI worker response timed out")
        line = stdout.readline()
        if not line:
            raise RuntimeError("SAM3 ROI worker exited unexpectedly")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SAM3 ROI worker sent invalid JSON") from exc

    def _retire_worker(self) -> None:
        """Discard a worker whose pending response can no longer be bound."""
        self._failed = True
        process = self._process
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            pass

    def segment(self, image: Image.Image, *, camera_id: str,
                prompt: str) -> tuple[np.ndarray | None, dict]:
        if (not isinstance(image, Image.Image) or image.mode != "RGB" or
                not isinstance(camera_id, str) or not camera_id or
                not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("SAM3 ROI needs an RGB image, camera and prompt")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        request_id = uuid4().hex
        request = {
            "type": "segment", "request_id": request_id,
            "camera_id": camera_id, "prompt": prompt,
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
        }
        with self._lock:
            process = self._process
            if self._failed or process.poll() is not None or process.stdin is None:
                raise RuntimeError("SAM3 ROI worker is not running")
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
            except BrokenPipeError as exc:
                self._failed = True
                raise RuntimeError("SAM3 ROI worker exited unexpectedly") from exc
            try:
                reply = self._read_line(self._response_timeout_s)
            except (OSError, RuntimeError):
                # Timeout, EOF or malformed JSON leaves the request/reply
                # stream unbound. A later camera must start a new worker.
                self._retire_worker()
                raise
            if (not isinstance(reply, dict) or
                    reply.get("request_id") != request_id or
                    reply.get("camera_id") != camera_id or
                    reply.get("type") not in {"result", "error"}):
                self._retire_worker()
                raise RuntimeError("SAM3 ROI response identity changed")
        if reply["type"] == "error":
            raise RuntimeError(f"SAM3 ROI {camera_id}: {reply.get('error')}")
        if reply.get("image_size_wh") != [image.width, image.height]:
            raise ValueError("SAM3 ROI response image size changed")
        encoded = reply.get("mask_png_base64")
        mask = None
        if encoded is not None:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as output:
                mask = np.asarray(output.convert("L"), dtype=np.uint8).copy()
            if (mask.shape != (image.height, image.width) or
                    not np.all((mask == 0) | (mask == 255))):
                raise ValueError("SAM3 ROI response mask is invalid")
        metadata = {key: reply[key] for key in (
            "score", "inference_s", "cuda_peak_bytes")}
        metadata["model"] = self.model
        return mask, metadata

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write('{"type":"close"}\n')
                    process.stdin.flush()
                process.wait(timeout=5.)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()
