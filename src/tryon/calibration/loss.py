"""Differentiable-by-forward-simulation calibration loss (SPEC §10.3(1)(2)).

IMPL: src/tryon/calibration/loss.py::compute

r is produced by FORWARD-INTEGRATING the physics solver with Θ; observed
residuals alone are not substituted for the differentiable residual.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image, ImageDraw

from ..math import condition_number, project, relative_step
from ..types import (
    GarmentAsset,
    LossWeights,
    MaterialParams,
    ObservationBatch,
    TryonError,
    material_bounds_arrays,
)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _base_asset(sim) -> GarmentAsset:
    asset = getattr(sim, "asset", None)
    if not isinstance(asset, GarmentAsset):
        raise TryonError(
            "E_CALIB_UNIDENTIFIABLE",
            "calibration simulator must expose a GarmentAsset on .asset",
        )
    return asset


def _asset_with_theta(sim, theta: np.ndarray) -> GarmentAsset:
    base = _base_asset(sim)
    mat = MaterialParams.from_vector(theta, template=base.material)
    return dataclasses.replace(base, material=mat)


def simulate_forward(
    batch: ObservationBatch,
    theta: np.ndarray,
    sim,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-integrate the window from the first observed state."""
    asset = _asset_with_theta(sim, theta)
    state = batch.states[0].copy()
    T = len(batch.states)
    xs = np.zeros((T, *state.positions.shape), dtype=np.float64)
    vs = np.zeros((T, *state.velocities.shape), dtype=np.float64)
    xs[0] = state.positions
    vs[0] = state.velocities
    for tau in range(1, T):
        state = sim.step(state, asset, [batch.contacts[tau]], float(batch.dt))
        xs[tau] = state.positions
        vs[tau] = state.velocities
    return xs, vs


def _phi_pen(x: np.ndarray, asset: GarmentAsset) -> float:
    col = getattr(asset, "collision_body", None)
    centers = getattr(col, "centers", None)
    radii = getattr(col, "radii", None)
    if centers is None or radii is None or len(centers) == 0:
        return 0.0
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1, 3)
    q = (x[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    qn = np.linalg.norm(q, axis=2)
    scale = float(np.min(radii))
    g = (qn - 1.0) * scale
    neg = np.maximum(0.0, -g)
    return float(np.sum(neg * neg))


def _silhouette_iou(uv: np.ndarray, mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    pts = np.asarray(uv, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    for x, y in pts:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=255)
    pred = np.asarray(img) > 127
    obs = np.asarray(mask) > 0
    union = np.logical_or(pred, obs).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, obs).sum() / union)


def residual_vector(
    batch: ObservationBatch,
    theta: np.ndarray,
    w: LossWeights,
    sim,
) -> np.ndarray:
    T = len(batch.states)
    xs, vs = simulate_forward(batch, theta, sim)
    asset = _asset_with_theta(sim, theta)
    obs_x = np.stack([s.positions for s in batch.states], axis=0)
    obs_v = np.stack([s.velocities for s in batch.states], axis=0)
    N = xs.shape[1]

    reproj = np.zeros((T, 2 * N), dtype=np.float64)
    if batch.K is not None and batch.T_CW is not None:
        try:
            uv_pred, _ = project(batch.K, batch.T_CW, xs.reshape(-1, 3))
            uv_obs, _ = project(batch.K, batch.T_CW, obs_x.reshape(-1, 3))
            reproj = (uv_pred - uv_obs).reshape(T, 2 * N)
        except TryonError:
            reproj = np.zeros((T, 2 * N), dtype=np.float64)

    velocity = (obs_v - vs).reshape(T, 3 * N)

    silhouette = np.zeros(T, dtype=np.float64)
    if batch.seg_masks and batch.K is not None and batch.T_CW is not None:
        for tau in range(T):
            try:
                uv, _ = project(batch.K, batch.T_CW, xs[tau])
            except TryonError:
                continue
            silhouette[tau] = 1.0 - _silhouette_iou(uv, batch.seg_masks[tau])

    penetration = np.array(
        [np.sqrt(max(_phi_pen(xs[tau], asset), 0.0)) for tau in range(T)],
        dtype=np.float64,
    )

    return np.concatenate(
        [
            (np.sqrt(w.w_r) * reproj).reshape(-1),
            (np.sqrt(w.w_v) * velocity).reshape(-1),
            np.sqrt(w.w_s) * silhouette,
            np.sqrt(w.w_p) * penetration,
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    batch: ObservationBatch,
    w: LossWeights,
    theta: np.ndarray,
    theta0: np.ndarray,
    P_inv: np.ndarray,
    sim,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (L, g = JᵀΣ⁻¹r, J = ∂r/∂Θ). J has 7 columns."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size != 7 or theta0.size != 7:
        raise ValueError("theta / theta0 must have 7 components")
    lo, hi = material_bounds_arrays()
    if np.all((theta < lo) | (theta > hi)):
        raise TryonError(
            "E_CALIB_GATE_REJECT",
            "all material components are outside the feasible domain",
        )
    theta = np.clip(theta, lo, hi)

    r0 = residual_vector(batch, theta, w, sim)
    dim = r0.size
    # Σ⁻¹ defaults to the identity. Never materialise a dense D×D identity:
    # D = T·(5N+2) is tens of thousands for a modest window.
    sigma_inv = None
    if w.sigma_inv is not None:
        sigma_inv = np.asarray(w.sigma_inv, dtype=np.float64)
        if sigma_inv.shape != (dim, dim):
            raise ValueError(f"sigma_inv must be ({dim},{dim}), got {sigma_inv.shape}")

    J = np.zeros((dim, 7), dtype=np.float64)
    for i in range(7):
        step = relative_step(theta[i], 1.0e-3)
        tp = theta.copy()
        tp[i] = min(max(theta[i] + step, lo[i]), hi[i])
        if abs(tp[i] - theta[i]) < 1e-15:
            tp[i] = max(theta[i] - step, lo[i])
        actual = tp[i] - theta[i]
        if abs(actual) < 1e-15:
            continue
        rp = residual_vector(batch, tp, w, sim)
        J[:, i] = (rp - r0) / actual

    if sigma_inv is None:
        g = J.T @ r0
        data_loss = float(r0 @ r0)
    else:
        g = J.T @ (sigma_inv @ r0)
        data_loss = float(r0 @ (sigma_inv @ r0))
    prior = theta - theta0
    P_inv = np.asarray(P_inv, dtype=np.float64).reshape(7, 7)
    L = float(data_loss + w.lam * (prior @ (P_inv @ prior)))
    if not np.isfinite(L):
        raise TryonError("E_CALIB_GATE_REJECT", "non-finite calibration loss")
    return max(L, 0.0), g, J


def fisher_information(J: np.ndarray, sigma_inv: np.ndarray | None = None) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    if sigma_inv is None:
        return J.T @ J
    return J.T @ np.asarray(sigma_inv, dtype=np.float64) @ J


def fisher_condition(J: np.ndarray, sigma_inv: np.ndarray | None = None) -> float:
    return condition_number(fisher_information(J, sigma_inv))


__all__ = [
    "compute",
    "simulate_forward",
    "residual_vector",
    "fisher_information",
    "fisher_condition",
    "_phi_pen",
    "_asset_with_theta",
]
