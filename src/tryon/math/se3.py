"""SE(3) and rotation helpers — the ONLY rotation implementation site.

IMPL: src/tryon/math/se3.py
"""
from __future__ import annotations

import math

import numpy as np

from ..types import TryonError
from .quaternion import is_unit_quat, normalize_quat


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a unit quaternion (w, x, y, z).

    INV: R.T @ R == I and det(R) == 1.
    raises E_MATH_QUAT when q is not a finite unit quaternion.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if not is_unit_quat(q):
        raise TryonError(
            "E_MATH_QUAT",
            f"quaternion must be finite and unit-norm, got norm={np.linalg.norm(q)!r}",
        )
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    q = normalize_quat(q)
    return q if q[0] >= 0.0 else -q


def compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 SE(3) from rotation and translation."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N,3) points."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")
    single = pts.ndim == 1
    P = pts.reshape(-1, 3)
    out = P @ T[:3, :3].T + T[:3, 3]
    return out.reshape(3) if single else out


def inverse_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose_Ts(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for T in Ts:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def is_SE3(T: np.ndarray, atol: float = 1.0e-6) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False
    if abs(float(np.linalg.det(R)) - 1.0) > atol:
        return False
    return bool(np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol))


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World→camera transform for a right-handed, Z-forward CV camera.

    Camera axes (in world): X right, Y down, Z forward. The returned
    ``T_CW = (R_cam→world)^T [I | -C]`` maps world points to camera coords.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    z = target - eye
    nz = float(np.linalg.norm(z))
    if nz <= 1.0e-12:
        raise ValueError("eye and target coincide")
    z = z / nz
    # Right = forward × up; down = forward × right.
    x = np.cross(z, up)
    nx = float(np.linalg.norm(x))
    if nx <= 1.0e-12:
        # up parallel to view direction: pick a stable fallback
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(z, fallback)
        nx = float(np.linalg.norm(x))
    x = x / nx
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], axis=1)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ eye
    return compose_T(R_w2c, t_w2c), R_c2w


def rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1.0e-12)
    b = b / max(np.linalg.norm(b), 1.0e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1.0e-12:
        if c > 0:
            return np.eye(3)
        # 180° rotation around any perpendicular axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    Kx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + Kx + (Kx @ Kx) * ((1.0 - c) / (s * s))


__all__ = [
    "quat_to_R",
    "R_to_quat",
    "compose_T",
    "transform",
    "inverse_T",
    "compose_Ts",
    "is_SE3",
    "look_at",
    "rotation_between_vectors",
]
