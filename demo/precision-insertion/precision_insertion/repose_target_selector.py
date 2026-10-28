"""Choose directed v8 reset targets after the current key pose is exhausted.

For the cylinder, the first release site preserves the observed CAD center
when a lying key is stood upright. The socket-aware reset preflight decides
whether a site, grasp and complete release/exit path are feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np

from autodex.utils.conversion import cart2se3

from .assets import AssetPaths
from .endpoint import _load_mesh
from .original_pose_restoration import original_center_release_xy


class ReposeTargetsExhausted(RuntimeError):
    """No untried target pose/height/release site remains this capture."""


@dataclass
class DirectedReposeTargetSelector:
    """Rank known insertion poses and try one release site at a time."""

    heights_cm: tuple[int, ...] = (12,)
    xy_offsets_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0), (0.03, 0.0), (-0.03, 0.0),
        (0.0, 0.03), (0.0, -0.03))
    _capture_id: str | None = field(default=None, init=False)
    _choices: list[dict] = field(default_factory=list, init=False)
    _index: int = field(default=-1, init=False)
    _attempted: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if (not self.heights_cm or
                any(type(value) is not int or value not in (4, 8, 12)
                    for value in self.heights_cm) or
                len(set(self.heights_cm)) != len(self.heights_cm) or
                not self.xy_offsets_m or
                any(len(offset) != 2 or
                    any(type(value) not in (int, float) or
                        not math.isfinite(value) for value in offset)
                    for offset in self.xy_offsets_m)):
            raise ValueError("repose selector needs distinct v8 heights and finite XY cells")

    def _start_capture(
        self, runner, latest,
        seed_available: Callable[[str, int], bool] | None = None,
        site_available: Callable[[str, tuple[float, float]], bool] | None = None,
    ) -> None:
        trial = latest.preflight
        source = trial.pose_class["stem"]
        targets = tuple(trial.repose_target_stems)
        if not targets or source in targets:
            raise ValueError("exhausted pose has no distinct insertion target")
        key_pose = np.asarray(cart2se3(np.asarray(
            trial.trial_scene["mesh"]["target"]["pose"], dtype=float)),
            dtype=float)
        xy = key_pose[:2, 3]
        if key_pose.shape != (4, 4) or not np.all(np.isfinite(xy)):
            raise ValueError("repose needs the measured robot-frame key XY")
        counts = {stem: 0 for stem in targets}
        for row in runner.catalog["candidates"]:
            stem = row.get("tabletop_pose_stem")
            if stem in counts and row.get("eligible") is True:
                counts[stem] += 1
        ordered = sorted(targets, key=lambda stem: (-counts[stem], stem))
        target_xy = {stem: xy for stem in ordered}
        board_center_xy = None
        if getattr(getattr(runner, "mode", None), "family", None) == "cylinder":
            mesh = _load_mesh(AssetPaths(
                runner.shared_root, runner.mode).raw_mesh(runner.mode.key_object))
            center_local = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
            center_xy = (key_pose[:3, :3] @ center_local + key_pose[:3, 3])[:2]
            target_xy = {
                stem: original_center_release_xy(
                    shared_root=runner.shared_root, mode=runner.mode,
                    target_pose_stem=stem,
                    original_center_xy_robot_m=tuple(map(float, center_xy)),
                    key_center_local_m=tuple(map(float, center_local)))
                for stem in ordered}
            measured_board = getattr(getattr(runner, "calibration", None),
                                     "board", {})
            board_center = measured_board.get("center_robot_m")
            if board_center is not None:
                board_center = np.asarray(board_center, dtype=float)
                if board_center.shape == (3,) and np.all(np.isfinite(board_center)):
                    board_center_xy = {
                        stem: original_center_release_xy(
                            shared_root=runner.shared_root, mode=runner.mode,
                            target_pose_stem=stem,
                            original_center_xy_robot_m=tuple(board_center[:2]),
                            key_center_local_m=tuple(map(float, center_local)))
                        for stem in ordered}
        self._choices = [
            {"to_pose_stem": stem, "height_cm": height,
             "release_xy_robot_m": (
                 float(site[0] + dx),
                 float(site[1] + dy))}
            for stem in ordered for height in self.heights_cm
            if seed_available is None or seed_available(stem, height)
            for site in ([target_xy[stem]] +
                         ([] if board_center_xy is None else
                          [board_center_xy[stem]]))
            for dx, dy in self.xy_offsets_m
            if site_available is None or site_available(
                stem, (float(site[0] + dx),
                       float(site[1] + dy)))]
        self._capture_id = latest.observation.capture_id
        self._index = -1
        self._attempted = []

    def __call__(
        self, runner, decision, latest, *,
        seed_available: Callable[[str, int], bool] | None = None,
        site_available: Callable[[str, tuple[float, float]], bool] | None = None,
    ) -> dict:
        capture_id = latest.observation.capture_id
        if self._capture_id != capture_id:
            self._start_capture(
                runner, latest, seed_available=seed_available,
                site_available=site_available)
        if decision.action in {"continue_repose_seed_preflight",
                               "preflight_repose_release"}:
            if self._index < 0:
                raise ValueError("repose continuation lacks its previous target")
        elif decision.action in {"preflight_repose",
                                 "preflight_alternative_repose_cell"}:
            previous = (None if self._index < 0 else
                        self._choices[self._index])
            self._index += 1
            # Missing directed seeds are independent of the release XY; do
            # not repeat the same absent asset for every nearby site.
            if (decision.action == "preflight_alternative_repose_cell" and
                    decision.reason == "no_verified_reset_seed" and
                    previous is not None):
                while (self._index < len(self._choices) and
                       self._choices[self._index]["to_pose_stem"] ==
                       previous["to_pose_stem"] and
                       self._choices[self._index]["height_cm"] ==
                       previous["height_cm"]):
                    self._index += 1
        else:
            raise ValueError("selector received a non-repose decision")
        if self._index >= len(self._choices):
            raise ReposeTargetsExhausted(
                f"searched all {len(self._choices)} directed release choices")
        selected = dict(self._choices[self._index])
        if (decision.action in {"preflight_repose",
                                "preflight_alternative_repose_cell"}):
            self._attempted.append(selected)
        return selected

    def attempted_choices(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._attempted)
