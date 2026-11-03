"""Reusable four-view calibration monitor independent of OpenCV HighGUI."""

from __future__ import annotations

import cv2
import numpy as np

from .charuco import detect, draw_detection


_HISTORY_COLOURS = (
    (255, 128, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
    (180, 80, 255), (80, 220, 120), (255, 80, 80), (80, 180, 255),
)


def load_capture_points(session_dir, camera_ids) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Load prior full-resolution ChArUco points for monitor history."""
    history = {str(camera): [] for camera in camera_ids}
    if not session_dir.exists():
        return history
    for capture_dir in sorted(
            path for path in session_dir.iterdir()
            if path.is_dir() and path.name.isdigit()):
        index = int(capture_dir.name)
        for camera in history:
            path = capture_dir / "markers_2d" / f"{camera}_corner.npy"
            if not path.is_file():
                continue
            corners = np.load(path).reshape(-1, 2)
            if len(corners):
                history[camera].append((index, corners.astype(np.float32)))
    return history


def append_capture_points(history, index: int, snapshot, detector) -> None:
    """Append a just-saved observation without re-reading the session."""
    for camera, image in snapshot.images_bgr.items():
        corners, ids = detect(detector, image)
        if len(ids):
            history.setdefault(camera, []).append(
                (index, corners.astype(np.float32)))


def _draw_points(image, current_corners, prior) -> np.ndarray:
    overlay = image.copy()
    for index, corners in prior:
        colour = _HISTORY_COLOURS[index % len(_HISTORY_COLOURS)]
        for x, y in np.asarray(corners).reshape(-1, 2):
            cv2.circle(overlay, (int(round(x)), int(round(y))),
                       4, colour, -1, cv2.LINE_AA)
        x, y = np.mean(corners, axis=0)
        point = (int(round(x)), int(round(y)))
        cv2.circle(overlay, point, 10, colour, 3, cv2.LINE_AA)
        cv2.putText(overlay, str(index), (point[0] + 12, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    if len(current_corners):
        centre = np.mean(current_corners, axis=0)
        point = (int(round(centre[0])), int(round(centre[1])))
        cv2.drawMarker(overlay, point, (255, 255, 255),
                       cv2.MARKER_CROSS, 38, 8, cv2.LINE_AA)
        cv2.circle(overlay, point, 18, (255, 0, 255), 5, cv2.LINE_AA)
    return overlay


def preview_grid(snapshot, detector, *, tile_width: int,
                 history=None) -> tuple[np.ndarray, dict]:
    """Build a display-only 2x2 grid; saved images remain full resolution."""
    order = ("26053248", "25322639", "25305462", "25322642")
    tiles = []
    counts = {}
    for serial in order:
        image = snapshot.images_bgr[serial]
        corners, ids = detect(detector, image)
        counts[serial] = int(len(ids))
        overlay = draw_detection(image, corners, ids)
        overlay = _draw_points(
            overlay, corners, [] if history is None else history.get(serial, []))
        tile_height = int(round(tile_width * image.shape[0] / image.shape[1]))
        tile = cv2.resize(
            overlay, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        label = f"{serial} | {len(ids)} corners"
        cv2.putText(tile, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))), counts


class TkMonitor:
    """Small GUI independent of the headless OpenCV wheel used for planning."""

    def __init__(self, title: str) -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._command: str | None = None
        self._root.bind_all("<KeyPress-c>", lambda _event: self._set("capture"))
        self._root.bind_all("<KeyPress-q>", lambda _event: self._set("quit"))
        self._root.protocol("WM_DELETE_WINDOW", lambda: self._set("quit"))
        self._photo = None

    def _set(self, command: str) -> None:
        self._command = command

    def show(self, image_bgr: np.ndarray) -> str | None:
        from PIL import Image, ImageTk

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()
        command, self._command = self._command, None
        return command

    def close(self) -> None:
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass
