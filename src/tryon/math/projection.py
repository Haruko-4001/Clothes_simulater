"""Pinhole projection — the ONLY projection implementation site.

IMPL: src/tryon/math/projection.py

Convention (SPEC §1.2 / §2.2):
  * world: right-handed, Y-up, metres
  * camera: right-handed, Z-forward, metres
  * image: origin top-left, x→right, y→down, pixels
Camera basis is X-right / Y-down / Z-forward (a 180° rotation about world X
from the Y-up world frame), so the standard positive-focal K applies directly:
``uv_h = K [R|t] p / depth``.
"""
from __future__ import annotations

import numpy as np

from ..types import TryonError


def project(
    K: np.ndarray,
    T_CW: np.ndarray,
    p_W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into pixels.

    Returns ``(uv (N,2), depth (N,))``.
    raises E_MATH_DEPTH when any depth <= 0 (point behind / on the camera).
    """
    K = np.asarray(K, dtype=np.float64)
    T_CW = np.asarray(T_CW, dtype=np.float64)
    p = np.asarray(p_W, dtype=np.float64)
    if K.shape != (3, 3) or T_CW.shape != (4, 4):
        raise ValueError("K must be 3x3 and T_CW must be 4x4")
    single = p.ndim == 1
    P = p.reshape(-1, 3)
    p_cam = P @ T_CW[:3, :3].T + T_CW[:3, 3]
    depth = p_cam[:, 2]
    if np.any(depth <= 0.0) or not np.all(np.isfinite(depth)):
        raise TryonError(
            "E_MATH_DEPTH",
            "projection requires strictly positive finite depth",
            context={"min_depth": float(np.min(depth)) if depth.size else 0.0},
        )
    uvw = p_cam @ K.T
    uv = uvw[:, :2] / depth[:, None]
    return (uv[0], depth[0]) if single else (uv, depth)


def backproject(
    K: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Back-project pixels with known depth into CAMERA coordinates.

    ``Π_K^{-1}(u)`` is a ray; fixing depth selects one point on it (SPEC §2.2).
    """
    K = np.asarray(K, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")
    single = uv.ndim == 1
    UV = uv.reshape(-1, 2)
    d = depth.reshape(-1)
    if d.size != UV.shape[0]:
        raise ValueError("depth must match uv")
    K_inv = np.linalg.inv(K)
    homogeneous = np.concatenate([UV, np.ones((UV.shape[0], 1))], axis=1)
    rays = homogeneous @ K_inv.T          # (N,3), normalized image plane
    pts = rays * d[:, None]
    return pts[0] if single else pts


def camera_to_world(T_CW: np.ndarray, p_C: np.ndarray) -> np.ndarray:
    T_CW = np.asarray(T_CW, dtype=np.float64)
    p = np.asarray(p_C, dtype=np.float64)
    single = p.ndim == 1
    P = p.reshape(-1, 3)
    out = (P - T_CW[:3, 3]) @ T_CW[:3, :3]
    return out[0] if single else out


def world_to_camera(T_CW: np.ndarray, p_W: np.ndarray) -> np.ndarray:
    T_CW = np.asarray(T_CW, dtype=np.float64)
    p = np.asarray(p_W, dtype=np.float64)
    single = p.ndim == 1
    P = p.reshape(-1, 3)
    out = P @ T_CW[:3, :3].T + T_CW[:3, 3]
    return out[0] if single else out


def make_intrinsics(width: int, height: int, fov_deg: float = 45.0) -> np.ndarray:
    """Positive-focal pinhole intrinsics for the image-aligned CV camera."""
    f = 0.5 * float(height) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def visible_mask(
    uv: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.0,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return (
        (uv[:, 0] >= margin)
        & (uv[:, 0] <= width - 1 - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] <= height - 1 - margin)
    )


__all__ = [
    "project",
    "backproject",
    "camera_to_world",
    "world_to_camera",
    "make_intrinsics",
    "visible_mask",
]
