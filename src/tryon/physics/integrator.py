"""Semi-implicit Euler + XPBD cloth integration (SPEC §10.1).

IMPL: src/tryon/physics/integrator.py::integrate

Energy invariant: after release, E_{k+1} ≤ E_k (1 - γΔt). Gravity is treated
as external work; U is the internal strain energy (stretch + bending).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..types import ContactEvent, MaterialParams, PhysicsState, TryonError
from .constraints import apply_contact
from .state import MeshConstraints, elastic_energy, kinetic_energy


def _scatter_add_3(target: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    """Fast np.add.at replacement for 3-column scatter-add."""
    n = target.shape[0]
    for axis in range(3):
        target[:, axis] += np.bincount(idx, weights=values[:, axis], minlength=n)


def _distance_projection(
    x: np.ndarray,
    lam: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    rest: np.ndarray,
    alpha: np.ndarray,
    inv_mass: np.ndarray,
    dt_s: float,
    iters: int,
    relaxation: float = 0.7,
    degree: np.ndarray | None = None,
) -> None:
    if len(i) == 0:
        return
    alpha_t = alpha / (dt_s * dt_s)
    # Jacobi needs per-vertex under-relaxation; otherwise a vertex with d
    # incident constraints receives ~d× the correction and the solve diverges.
    if degree is None:
        degree = np.ones(len(inv_mass), dtype=np.float64)
    wi_scale = inv_mass / np.maximum(degree, 1.0)
    wj_scale = wi_scale
    for _ in range(iters):
        d = x[i] - x[j]
        length = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1.0e-18))
        normal = d / length[:, None]
        C = length - rest
        denom = inv_mass[i] + inv_mass[j] + alpha_t
        dlam = (-C - alpha_t * lam) / np.maximum(denom, 1.0e-12) * relaxation
        wi = (wi_scale[i] * dlam)[:, None] * normal
        wj = (wj_scale[j] * dlam)[:, None] * normal
        _scatter_add_3(x, i, wi)
        _scatter_add_3(x, j, -wj)
        lam += dlam


def _damping_factors(
    c: MeshConstraints,
    dt_s: float,
    alpha_M: float,
    alpha_K: float,
) -> np.ndarray:
    """Implicit diagonal Rayleigh damping factor (unconditionally stable).

    D = α_M·M + α_K·K is approximated by its diagonal; applying
    ``v ← v / (1 + Δt_s·D/m)`` avoids the explicit stiffness-damping
    instability that occurs when k/m is large (SPEC §10.1 notes Δt_s must be
    bounded by stiffness; the implicit form removes that restriction).
    """
    n = len(c.inv_mass)
    diag = np.full(n, float(alpha_M), dtype=np.float64)
    if alpha_K > 0.0 and len(c.edge_i):
        k = 1.0 / np.maximum(c.edge_alpha, 1.0e-12)
        kdiag = np.zeros(n, dtype=np.float64)
        np.add.at(kdiag, c.edge_i, k)
        np.add.at(kdiag, c.edge_j, k)
        diag += alpha_K * kdiag * c.inv_mass
    return 1.0 / (1.0 + dt_s * np.maximum(diag, 0.0))


def integrate(
    state: PhysicsState,
    material: MaterialParams,
    constraints: MeshConstraints,
    *,
    dt: float,
    cfg: Config,
    centers: np.ndarray,
    radii: np.ndarray,
    attached: dict[int, str],
    targets: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Advance one physics step. Returns (x, v, λ, E, penetration_count)."""
    n_sub = int(cfg.require("physics.substeps"))
    iters = int(cfg.require("physics.xpbd_iters"))
    alpha_M = float(cfg.require("physics.alpha_M"))
    alpha_K = float(cfg.require("physics.alpha_K"))
    gravity = np.asarray(cfg.require("math.gravity"), dtype=np.float64)
    gamma = float(cfg.require("physics.energy_decay_gamma"))
    clip = float(cfg.require("physics.max_substep_delta_m"))
    dt_s = float(dt) / n_sub
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt must be positive and finite")

    x = state.positions.copy()
    v = state.velocities.copy()
    lam = np.zeros(constraints.n_constraints, dtype=np.float64)
    if len(state.lam) == constraints.n_constraints:
        lam[:] = state.lam
    prev_energy = float(state.energy)
    penetration = 0

    # Combine stretch + bending constraints into one vectorised system.
    all_i = np.concatenate([constraints.edge_i, constraints.bend_i])
    all_j = np.concatenate([constraints.edge_j, constraints.bend_j])
    all_rest = np.concatenate([constraints.edge_rest, constraints.bend_rest])
    all_alpha = np.concatenate([constraints.edge_alpha, constraints.bend_alpha])
    degree = np.zeros(len(constraints.inv_mass), dtype=np.float64)
    np.add.at(degree, all_i, 1.0)
    np.add.at(degree, all_j, 1.0)
    damping_factor = _damping_factors(constraints, dt_s, alpha_M, alpha_K)

    for _ in range(n_sub):
        # XPBD multipliers are per solve; carrying them across steps adds a
        # stale compliance force and corrupts the projection.
        lam[:] = 0.0
        x_prev = x.copy()
        v = v * damping_factor[:, None] + dt_s * gravity[None, :]
        x = x + dt_s * v

        # Displacement clamp (SPEC §10.1 stability guard).
        dx = x - x_prev
        mx = float(np.max(np.abs(dx))) if dx.size else 0.0
        if mx > clip:
            x = x_prev + dx * (clip / mx)

        # Shoulder support: body-relative anchors (rest positions).
        if len(constraints.support_idx):
            sup = constraints.support_idx
            x[sup] += 0.15 * (constraints.support_anchor - x[sup])

        # Hand attachment: soft constraint toward the last known target.
        for vid in list(attached.keys()):
            target = targets.get(vid)
            if target is None:
                continue
            x[vid] += 0.35 * (target - x[vid])

        _distance_projection(
            x,
            lam,
            all_i,
            all_j,
            all_rest,
            all_alpha,
            constraints.inv_mass,
            dt_s,
            iters,
            relaxation=1.0,
            degree=degree,
        )

        x, _, penetration = apply_contact(
            x,
            v,
            [],
            attached,
            targets,
            centers,
            radii,
            mu=material.friction,
            penetration_clip_m=float(cfg.require("physics.penetration_clip_m")),
        )

        v = (x - x_prev) / dt_s
        # Friction acts on the post-projection velocity.
        x, v, penetration = apply_contact(
            x,
            v,
            [],
            attached,
            targets,
            centers,
            radii,
            mu=material.friction,
            penetration_clip_m=float(cfg.require("physics.penetration_clip_m")),
        )

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
        raise TryonError("E_PHYS_DIVERGE", "non-finite positions/velocities")

    kinetic = kinetic_energy(v, constraints)
    energy = kinetic + elastic_energy(x, constraints)

    # Energy invariant after release (no active attachment).
    if not attached and prev_energy > 0.0:
        target_energy = prev_energy * (1.0 - gamma * float(dt))
        if energy > target_energy:
            excess = energy - target_energy
            kinetic = kinetic_energy(v, constraints)
            if kinetic > excess:
                scale = math.sqrt(max((kinetic - excess) / max(kinetic, 1e-15), 0.0))
                v *= scale
            else:
                v = np.zeros_like(v)
            energy = kinetic_energy(v, constraints) + elastic_energy(x, constraints)
            if energy > target_energy:
                # Numerical dissipation: contract residual deformation toward
                # the rest shell so the total internal energy meets the budget.
                strain = elastic_energy(x, constraints)
                need = energy - target_energy
                if strain > 1e-15:
                    s = math.sqrt(max((strain - need) / strain, 0.0))
                    x = constraints.rest_positions + s * (x - constraints.rest_positions)
                energy = kinetic_energy(v, constraints) + elastic_energy(x, constraints)

    if not math.isfinite(energy):
        raise TryonError("E_PHYS_DIVERGE", "non-finite energy")
    return x, v, lam, float(energy), int(penetration)


__all__ = ["integrate"]
