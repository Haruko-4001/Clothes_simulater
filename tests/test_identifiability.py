import dataclasses

import numpy as np

from tryon.calibration.identifier import SystemIdentifier
from tryon.calibration.loss import compute, fisher_information
from tryon.math import condition_number, look_at, make_intrinsics
from tryon.types import (
    ContactEvent,
    ContactKind,
    LossWeights,
    MaterialParams,
    ObservationBatch,
    PhysicsState,
)


class SyntheticClothSolver:
    """Known-Θ* synthetic dynamics; fast, differentiable-by-forward-sim."""

    def __init__(self, asset) -> None:
        self.asset = asset
        self.rest = asset.rest_mesh.vertices.copy()

    def step(self, state, asset, contacts, dt):
        th = asset.material.to_vector()
        rho, E, kb, nu, mu, cd, h = th
        x = np.array(state.positions, copy=True)
        v = np.array(state.velocities, copy=True)
        idx = np.arange(len(x), dtype=np.float64)
        pattern = np.stack(
            [np.sin(0.7 * idx), np.cos(1.1 * idx), np.sin(0.3 * idx + 1.0)], axis=1
        )
        a = np.zeros_like(x)
        # In static equilibrium (v=0) E and kb are exactly collinear; the
        # velocity-dependent terms break that degeneracy under excitation.
        a += -0.02 * (E / 2000.0) * (x - self.rest)
        a += -0.03 * (kb / 1.0e-3) * (x - self.rest)
        a += 0.02 * (kb / 1.0e-3) * v * pattern
        a[:, 1] += 0.30 * (rho / 0.25)
        a[:, 2] -= 0.20 * (rho / 0.25)
        a += -0.40 * (cd / 2.0) * v
        a += -0.20 * (nu / 0.30) * v * pattern
        a += -0.15 * (mu / 0.40) * v * np.abs(v)
        a += 0.10 * (h / 1.0e-3) * v * np.sign(pattern)
        v = v + float(dt) * a
        x = x + float(dt) * v
        return PhysicsState(
            positions=x,
            velocities=v,
            attached={},
            lam=np.zeros(1),
            step=state.step + 1,
            energy=float(0.5 * np.sum(v * v)),
        )


def _contact(t: float) -> ContactEvent:
    return ContactEvent(t, ContactKind.SLIDE, np.zeros(3), 0, 0.0, "right")


def _initial(asset, dynamic: bool, seed: int = 0) -> PhysicsState:
    rng = np.random.default_rng(seed)
    rest = asset.rest_mesh.vertices
    x = rest + 0.01 * rng.normal(size=rest.shape)
    v = (0.5 * rng.normal(size=rest.shape)) if dynamic else np.zeros_like(rest)
    return PhysicsState(
        positions=x,
        velocities=v,
        attached={},
        lam=np.zeros(1),
        step=0,
        energy=0.0,
    )


def _camera():
    K = make_intrinsics(320, 240, 50.0)
    T_CW, _ = look_at(
        np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    return K, T_CW


def _batch(solver, asset, dynamic: bool, T: int = 18):
    state = _initial(asset, dynamic)
    states = [state.copy()]
    contacts = [_contact(0.0)]
    for k in range(1, T):
        state = solver.step(state, asset, [_contact(k * 33.0)], 1.0 / 30.0)
        states.append(state.copy())
        contacts.append(_contact(k * 33.0))
    K, T_CW = _camera()
    return ObservationBatch(
        contacts=contacts,
        states=states,
        camera_views=[np.zeros((8, 8, 3), dtype=np.uint8)] * T,
        K=K,
        T_CW=T_CW,
        dt=1.0 / 30.0,
    )


def _static_batch(asset) -> ObservationBatch:
    state = _initial(asset, dynamic=False)
    states = [state.copy(), state.copy()]
    contacts = [_contact(0.0), _contact(33.0)]
    K, T_CW = _camera()
    return ObservationBatch(
        contacts=contacts,
        states=states,
        camera_views=[np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
        K=K,
        T_CW=T_CW,
        dt=1.0 / 30.0,
    )


def _normalised_cond(J: np.ndarray) -> float:
    norms = np.linalg.norm(J, axis=0)
    active = norms > 1e-10 * max(float(norms.max()), 1.0)
    if not np.any(active):
        return float("inf")
    Jn = J[:, active] / norms[active][None, :]
    return condition_number(Jn.T @ Jn)


def _fisher(batch, solver, asset):
    theta = asset.material.to_vector()
    P_inv = np.eye(7) * 1.0e-6
    L, g, J = compute(batch, LossWeights(), theta, theta, P_inv, solver)
    return fisher_information(J), J


def test_static_fisher_is_rank_deficient_and_dynamic_is_richer(stub_asset, mock_cfg):
    solver = SyntheticClothSolver(stub_asset)
    static_batch = _static_batch(stub_asset)
    dynamic_batch = _batch(solver, stub_asset, dynamic=True)
    I_static, J_static = _fisher(static_batch, solver, stub_asset)
    I_dynamic, J_dynamic = _fisher(dynamic_batch, solver, stub_asset)

    rank_static = int(np.linalg.matrix_rank(I_static, tol=1e-8 * max(np.linalg.norm(I_static), 1.0)))
    rank_dynamic = int(np.linalg.matrix_rank(I_dynamic, tol=1e-8 * max(np.linalg.norm(I_dynamic), 1.0)))
    assert rank_static < 7
    assert rank_dynamic > rank_static
    cond_static = _normalised_cond(J_static)
    cond_dynamic = _normalised_cond(J_dynamic)
    kappa_max = float(mock_cfg.require("calibration.kappa_max"))
    assert cond_static > kappa_max          # static gate would reject
    assert np.isfinite(cond_dynamic)
    assert cond_dynamic < cond_static / 100.0  # excitation restores conditioning


def test_parameter_error_decreases_against_known_theta_star(stub_asset, mock_cfg):
    true_theta = np.array([0.55, 2.4e4, 6.0e-3, 0.28, 0.62, 4.5, 2.0e-3])
    true_material = MaterialParams.from_vector(true_theta, template=stub_asset.material)
    true_asset = dataclasses.replace(stub_asset, material=true_material)
    solver = SyntheticClothSolver(stub_asset)  # base asset = prior Θ₀
    batch = _batch(solver, true_asset, dynamic=True, T=20)
    # Simulator must use the prior as its template, not the hidden truth.
    solver.asset = stub_asset

    ident = SystemIdentifier(mock_cfg, solver)
    initial_error = float(
        np.linalg.norm(stub_asset.material.to_vector() - true_theta)
        / np.linalg.norm(true_theta)
    )
    res = ident.update(batch)
    intermediate = float(
        np.linalg.norm(res.params.to_vector() - true_theta) / np.linalg.norm(true_theta)
    )
    if intermediate >= initial_error:
        res = ident.update(batch)
        intermediate = float(
            np.linalg.norm(res.params.to_vector() - true_theta) / np.linalg.norm(true_theta)
        )
    assert intermediate < initial_error
    assert res.loss_after <= res.loss_before
