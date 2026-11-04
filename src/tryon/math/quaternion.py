"""Unit-quaternion utilities (q = [w, x, y, z]).

IMPL: src/tryon/math/quaternion.py
Rotations are stored as unit quaternions; Euler angles are never stored
(SPEC §1.2).
"""
from __future__ import annotations

import math

import numpy as np

from ..types import TryonError


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n <= 1.0e-12:
        raise TryonError("E_MATH_QUAT", "zero or non-finite quaternion")
    return q / n


def is_unit_quat(q: np.ndarray, atol: float = 1.0e-6) -> bool:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return bool(q.size == 4 and np.all(np.isfinite(q)) and abs(np.linalg.norm(q) - 1.0) <= atol)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, w-first convention."""
    aw, ax, ay, az = normalize_quat(a)
    bw, bx, by, bz = normalize_quat(b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * float(angle)
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    q = normalize_quat(q)
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return q[1:] / s, angle


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat(q0 + float(t) * (q1 - q0))
    theta0 = math.acos(float(np.clip(dot, -1.0, 1.0)))
    theta = theta0 * float(t)
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return normalize_quat(s0 * q0 + s1 * q1)


__all__ = [
    "normalize_quat",
    "is_unit_quat",
    "quat_mul",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_to_axis_angle",
    "slerp",
]
