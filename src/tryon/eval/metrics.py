"""Evaluation metrics defined BEFORE implementation (SPEC §7).

IMPL: src/tryon/eval/metrics.py
"""
from __future__ import annotations

import math

import numpy as np

from ..math import empirical_p50, empirical_p95


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    A = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    # Chunked to bound memory.
    total = 0.0
    for i in range(0, len(A), 512):
        d = np.linalg.norm(A[i : i + 512, None, :] - B[None, :, :], axis=2)
        total += float(d.min(axis=1).sum())
    for i in range(0, len(B), 512):
        d = np.linalg.norm(B[i : i + 512, None, :] - A[None, :, :], axis=2)
        total += float(d.min(axis=1).sum())
    return total / (len(A) + len(B))


def normal_error(n_pred: np.ndarray, n_true: np.ndarray) -> float:
    a = np.asarray(n_pred, dtype=np.float64)
    b = np.asarray(n_true, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("normal arrays must match and be non-empty")
    na = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    nb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    dots = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
    return float(np.mean(np.arccos(dots)))


def ade(pred: np.ndarray, true: np.ndarray, horizon: int | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("pred/true shapes must match")
    if horizon is not None:
        p = p[: int(horizon)]
        t = t[: int(horizon)]
    if p.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(p - t, axis=-1)))


def fde(pred: np.ndarray, true: np.ndarray, horizon: int | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("pred/true shapes must match")
    idx = -1 if horizon is None else min(int(horizon), len(p)) - 1
    return float(np.mean(np.linalg.norm(p[idx] - t[idx], axis=-1)))


def penetration_rate(counts_or_flags, n_vertices: int) -> float:
    if n_vertices <= 0:
        return 0.0
    arr = np.asarray(counts_or_flags, dtype=np.float64)
    return float(np.mean(arr) / n_vertices) if arr.ndim else 0.0


def parameter_error(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    h = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    s = np.asarray(theta_star, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(s))
    if denom <= 0:
        return float("inf")
    return float(np.linalg.norm(h - s) / denom)


def energy_decay_ok(energies: list[float], gamma: float, dt: float, tol: float = 1.0e-12) -> bool:
    e = np.asarray(energies, dtype=np.float64)
    if len(e) < 2:
        return True
    factors = 1.0 - float(gamma) * float(dt)
    return bool(np.all(e[1:] <= e[:-1] * factors + tol))


def convergence(variances: list[float], windows: int = 5, rel_tol: float = 0.02) -> bool:
    v = [float(x) for x in variances]
    if len(v) < windows:
        return False
    tail = v[-windows:]
    for i in range(1, len(tail)):
        if tail[i] >= tail[i - 1]:
            return False
        rel = abs(tail[i - 1] - tail[i]) / max(abs(tail[i - 1]), 1e-12)
        if rel >= rel_tol:
            return False
    return True


def latency_percentiles(samples_ms: list[float]) -> tuple[float, float]:
    return empirical_p50(samples_ms), empirical_p95(samples_ms)


def framing_compliance_rate(verdicts: list[str]) -> float:
    if not verdicts:
        return 0.0
    ok = sum(1 for v in verdicts if str(v) == "OK")
    return float(ok / len(verdicts))


__all__ = [
    "iou",
    "chamfer_distance",
    "normal_error",
    "ade",
    "fde",
    "penetration_rate",
    "parameter_error",
    "energy_decay_ok",
    "convergence",
    "latency_percentiles",
    "framing_compliance_rate",
]
