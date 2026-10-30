"""Associate a held key face across successive calibrated preinsert images.

The robot's measured XY motion predicts where the same face centre should
appear in each camera.  A new SAM ellipse is used as feedback only when its
displacement agrees with that prediction.  This prevents a newly detected
socket rim or a fallen key from becoming the next wrist target.
"""

from __future__ import annotations

import math

import numpy as np


def _visible_views(observation: dict) -> dict[str, dict]:
    return {
        row["camera_id"]: row for row in observation["alignment"].get(
            "sam_half_ellipse_views", [])
        if row.get("admitted") and row.get("sam_half_ellipse")
    }


def _camera_geometry(camera: str, calibration: dict,
                     socket_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cameras = calibration["camera_calibration"]
    K = np.asarray(cameras["intrinsics_full"][camera]["K_undist"], float)
    E = (np.asarray(cameras["extrinsics_full"][camera], float)
         @ np.asarray(calibration["c2r"], float) @ socket_robot)
    return K, E


def _on_tip_plane(pixel: np.ndarray, z: float,
                  K: np.ndarray, E: np.ndarray) -> np.ndarray:
    origin = -E[:3, :3].T @ E[:3, 3]
    ray = E[:3, :3].T @ np.linalg.solve(K, np.r_[pixel, 1.])
    return origin + ((z - origin[2]) / ray[2]) * ray


def _project(point: np.ndarray, K: np.ndarray,
             E: np.ndarray) -> np.ndarray:
    camera_point = E[:3, :3] @ point + E[:3, 3]
    pixel = K @ camera_point
    return pixel[:2] / pixel[2]


def update_held_key_xy(*, previous: dict, current: dict,
                       measured_xy_socket_m, measured_axis_socket_m: float,
                       calibration: dict, socket_robot: np.ndarray,
                       previous_camera_id: str,
                       previous_xy_socket_m,
                       key_radius_m: float,
                       anchor: dict | None = None,
                       cumulative_xy_socket_m=None,
                       cumulative_axis_socket_m: float = 0.) -> dict:
    """Fuse a same-camera key-face observation with measured robot motion.

    The image innovation is normalized by the observed ellipse radii.  A
    quarter-radius is a deliberately broad pixel uncertainty for a partially
    visible face; both frames contribute to the three-sigma association test.
    This is a data-association test, not an insertion-success threshold.
    """
    old = _visible_views(previous)
    new = _visible_views(current)
    anchored = _visible_views(anchor) if anchor is not None else {}
    delta = np.asarray(measured_xy_socket_m, float)
    prior = np.asarray(previous_xy_socket_m, float)
    if delta.shape != (2,) or prior.shape != (2,):
        raise ValueError("held-key feedback needs two socket XY coordinates")
    if not math.isfinite(key_radius_m) or key_radius_m <= 0:
        raise ValueError("held-key feedback needs the key CAD radius")
    predicted_xy = prior + delta
    z_old = float(previous["alignment"]["nominal_tip_z_socket_m"])
    z_new = float(current["alignment"]["nominal_tip_z_socket_m"])
    motion = np.array([delta[0], delta[1], measured_axis_socket_m], float)
    rows = []
    for camera in sorted(set(old) & set(new)):
        before = old[camera]["sam_half_ellipse"]
        after = new[camera]["sam_half_ellipse"]
        uv_old = np.asarray(before["center_full_px"], float)
        uv_new = np.asarray(after["center_full_px"], float)
        K, E = _camera_geometry(camera, calibration, socket_robot)
        old_xyz = _on_tip_plane(uv_old, z_old, K, E)
        expected_uv = _project(old_xyz + motion, K, E)
        observed_xy = _on_tip_plane(uv_new, z_new, K, E)[:2]
        sigma_old = max(1., float(before["major_radius_px"]) / 4.)
        sigma_new = max(1., float(after["major_radius_px"]) / 4.)
        sigma_motion_px = 1.0
        sigma = math.sqrt(sigma_old**2 + sigma_new**2 + sigma_motion_px**2)
        error = float(np.linalg.norm(uv_new - expected_uv))
        anchor_response = None
        if camera in anchored and cumulative_xy_socket_m is not None:
            first = anchored[camera]["sam_half_ellipse"]
            first_uv = np.asarray(first["center_full_px"], float)
            first_z = float(anchor["alignment"]["nominal_tip_z_socket_m"])
            first_xyz = _on_tip_plane(first_uv, first_z, K, E)
            cumulative = np.r_[np.asarray(cumulative_xy_socket_m, float),
                               cumulative_axis_socket_m]
            expected_from_first = _project(first_xyz + cumulative, K, E)
            projected_motion = expected_from_first - first_uv
            actual_motion = uv_new - first_uv
            projected_length = float(np.linalg.norm(projected_motion))
            sigma_first = max(1., float(first["major_radius_px"]) / 4.)
            response_sigma = math.hypot(sigma_first, sigma_new)
            response_ratio = (float(np.dot(actual_motion, projected_motion)) /
                              projected_length**2 if projected_length > 0 else 1.)
            anchor_response = {
                "predicted_displacement_px": projected_motion.tolist(),
                "observed_displacement_px": actual_motion.tolist(),
                "response_ratio": response_ratio,
                "resolvable": projected_length >= 2. * response_sigma,
            }
        follows_anchor = (anchor_response is None or
                          not anchor_response["resolvable"] or
                          anchor_response["response_ratio"] >= .25)
        rows.append({
            "camera_id": camera,
            "previous_pixel": uv_old.tolist(),
            "current_pixel": uv_new.tolist(),
            "expected_pixel": expected_uv.tolist(),
            "innovation_px": error,
            "innovation_sigma": error / sigma,
            "observed_xy_socket_m": observed_xy.tolist(),
            "anchor_response": anchor_response,
            "associated": error <= 3. * sigma and follows_anchor,
        })
    associated = [row for row in rows if row["associated"]]
    # Keep the same face/view when possible. A different view is eligible only
    # when its previous point agreed with the previous fused 3D estimate.
    ranked = []
    for row in associated:
        camera = row["camera_id"]
        if camera != previous_camera_id:
            K, E = _camera_geometry(camera, calibration, socket_robot)
            old_xy = _on_tip_plane(np.asarray(
                old[camera]["sam_half_ellipse"]["center_full_px"], float),
                z_old, K, E)[:2]
            if np.linalg.norm(old_xy - prior) > key_radius_m:
                continue
        ranked.append(row)
    ranked.sort(key=lambda row: (
        row["camera_id"] != previous_camera_id,
        row["innovation_sigma"]))
    if not ranked:
        return {"status": "lost", "reason": "no_face_follows_measured_motion",
                "predicted_xy_socket_m": predicted_xy.tolist(),
                "camera_rows": rows}
    chosen = ranked[0]
    # Complementary update: a coherent new image can revise the predicted
    # location, while a marginally associated ellipse contributes less.
    weight = .5 * max(0., 1. - (chosen["innovation_sigma"] / 3.)**2)
    observed_xy = np.asarray(chosen["observed_xy_socket_m"], float)
    fused = predicted_xy + weight * (observed_xy - predicted_xy)
    return {
        "status": "tracked", "camera_id": chosen["camera_id"],
        "predicted_xy_socket_m": predicted_xy.tolist(),
        "observed_xy_socket_m": observed_xy.tolist(),
        "fused_xy_socket_m": fused.tolist(),
        "image_weight": weight,
        "camera_rows": rows,
    }
