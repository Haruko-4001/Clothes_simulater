"""Numerical helpers shared by identification / gates.

IMPL: src/tryon/math/numerics.py
No projection / quaternion / SE(3) code lives here (SPEC §4).
"""
from __future__ import annotations

import math

import numpy as np


def is_finite(a: np.ndarray | float) -> bool:
    return bool(np.all(np.isfinite(np.asarray(a, dtype=np.float64))))


def condition_number(A: np.ndarray) -> float:
    """2-norm condition number; infinity for singular / non-finite matrices."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("condition_number expects a square matrix")
    if A.size == 0 or not np.all(np.isfinite(A)):
        return math.inf
    sv = np.linalg.svd(A, compute_uv=False)
    if sv.size == 0:
        return math.inf
    smax = float(sv[0])
    smin = float(sv[-1])
    if smax <= 0.0:
        return math.inf
    if smin <= max(smax * 1.0e-15, 1.0e-300):
        return math.inf
    return smax / smin


def solve_psd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cholesky solve with SVD/lstsq fallback (SPEC §10.3 step 3)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x


def clip_mahalanobis(
    delta: np.ndarray,
    P: np.ndarray,
    delta_max: float,
) -> np.ndarray:
    """Scale ΔΘ so that ‖ΔΘ‖_{P⁻¹} ≤ δ_max (SPEC §10.3, D14 guard)."""
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    P = np.asarray(P, dtype=np.float64)
    if not math.isfinite(delta_max) or delta_max <= 0.0:
        return np.zeros_like(delta)
    try:
        z = np.linalg.solve(P, delta)
        norm2 = float(delta @ z)
    except np.linalg.LinAlgError:
        norm2 = float(delta @ delta)
    if not math.isfinite(norm2):
        return np.zeros_like(delta)
    norm = math.sqrt(max(norm2, 0.0))
    if norm > delta_max and norm > 0.0:
        delta = delta * (delta_max / norm)
    return delta


def project_box(
    theta: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    return np.clip(np.asarray(theta, dtype=np.float64), lo, hi)


def weighted_geometric_mean(values: np.ndarray, weights: np.ndarray, eps: float = 1e-6) -> float:
    v = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    if v.size != w.size or w.sum() <= 0:
        raise ValueError("bad geometric mean inputs")
    return float(np.exp(np.sum(w * np.log(v)) / w.sum()))


def empirical_p95(samples: list[float] | np.ndarray) -> float:
    """Empirical 95th percentile (never a mean — SPEC §1.3 / §7)."""
    a = np.asarray(samples, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, 95.0))


def empirical_p50(samples: list[float] | np.ndarray) -> float:
    a = np.asarray(samples, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, 50.0))


def safe_scalar(x: float, default: float = 0.0) -> float:
    x = float(x)
    return x if math.isfinite(x) else default


def relative_step(value: float, eps: float = 1.0e-3) -> float:
    """Finite-difference step respecting parameter scale."""
    return float(eps * max(abs(value), 1.0))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d))) if d.size else 0.0
