import numpy as np
import pytest

from tryon.types import ContactEvent, ContactKind, PhysicsState, TryonError


def test_release_clears_attachments_but_keeps_velocities(synth_trajectory, mock_cfg):
    pulled = synth_trajectory["pulled"]
    released = synth_trajectory["released"]
    vid = synth_trajectory["vertex_id"]
    assert vid in pulled.attached
    assert released.attached == {}
    assert np.linalg.norm(released.velocities[vid]) > 1.0e-6
    assert np.all(np.isfinite(released.positions))
    assert synth_trajectory["solver"].last_penetration_count == 0


def test_energy_monotonically_decays_after_release(synth_trajectory, mock_cfg):
    gamma = float(mock_cfg.require("physics.energy_decay_gamma"))
    dt = synth_trajectory["dt"]
    energies = [synth_trajectory["released"].energy] + [
        s.energy for s in synth_trajectory["rebound"]
    ]
    for prev, nxt in zip(energies, energies[1:]):
        assert nxt <= prev * (1.0 - gamma * dt) + 1.0e-10
    assert energies[-1] < energies[0]


def test_attachment_persists_under_long_pull(stub_asset, physics_solver, mock_cfg):
    asset = stub_asset
    dt = float(mock_cfg.require("physics.dt"))
    state = physics_solver.initial_state(asset)
    for _ in range(6):
        state = physics_solver.step(state, asset, [], dt)
    vid = int(np.argmin(np.linalg.norm(state.positions - np.array([-0.22, 1.2, 0.0]), axis=1)))
    hand = state.positions[vid].copy()
    state = physics_solver.step(
        state, asset, [ContactEvent(0, ContactKind.GRASP, hand, vid, 0, "right")], dt
    )
    for k in range(120):  # 2 s at 60 Hz
        hand = hand + np.array([-0.0008, -0.0004, 0.0005])
        state = physics_solver.step(
            state, asset, [ContactEvent(k, ContactKind.PULL, hand, vid, 0, "right")], dt
        )
        assert vid in state.attached
    assert physics_solver.last_penetration_count == 0


def test_occlusion_hold_releases_after_timeout(stub_asset, physics_solver, mock_cfg):
    asset = stub_asset
    dt = float(mock_cfg.require("physics.dt"))
    hold_s = float(mock_cfg.require("framing.contact_hold_s"))
    state = physics_solver.initial_state(asset)
    for _ in range(6):
        state = physics_solver.step(state, asset, [], dt)
    vid = int(np.argmin(np.linalg.norm(state.positions - np.array([-0.22, 1.2, 0.0]), axis=1)))
    hand = state.positions[vid].copy()
    state = physics_solver.step(
        state, asset, [ContactEvent(0, ContactKind.GRASP, hand, vid, 0, "right")], dt
    )
    assert vid in state.attached
    hold_steps = int(np.ceil(hold_s / dt))
    for _ in range(hold_steps):
        state = physics_solver.step(state, asset, [], dt)
        assert vid in state.attached
    for _ in range(3):
        state = physics_solver.step(state, asset, [], dt)
    assert state.attached == {}
    assert np.all(np.isfinite(state.velocities))


def test_soft_constraint_relaxation_is_read_from_config(stub_asset, mock_cfg):
    """SPEC §0.2: no second hardcoded copy of a physics threshold exists.

    Changing ``physics.support_relax`` / ``physics.attach_relax`` must change
    the trajectory; otherwise the config keys would be decorative.
    """
    from tryon.physics.solver import PhysicsSolver

    dt = float(mock_cfg.require("physics.dt"))
    anchor = np.array([-0.22, 1.20, 0.0])

    def settle(cfg):
        solver = PhysicsSolver(cfg)
        state = solver.initial_state(stub_asset)
        for _ in range(6):
            state = solver.step(state, stub_asset, [], dt)
        return solver, state

    def support_positions(relax):
        cfg = mock_cfg.clone()
        cfg.set("physics.support_relax", relax)
        solver, state = settle(cfg)
        for _ in range(8):
            state = solver.step(state, stub_asset, [], dt)
        return state.positions.copy()

    assert not np.allclose(support_positions(0.05), support_positions(0.90))

    def attach_positions(relax):
        cfg = mock_cfg.clone()
        cfg.set("physics.attach_relax", relax)
        solver, state = settle(cfg)
        vid = int(np.argmin(np.linalg.norm(state.positions - anchor, axis=1)))
        hand = state.positions[vid].copy()
        state = solver.step(
            state, stub_asset,
            [ContactEvent(0.0, ContactKind.GRASP, hand, vid, 0.0, "right")], dt,
        )
        for k in range(10):
            hand = hand + np.array([-0.004, -0.002, 0.003])
            state = solver.step(
                state, stub_asset,
                [ContactEvent(float(k), ContactKind.PULL, hand, vid, 0.0, "right")], dt,
            )
        return state.positions[vid].copy()

    assert not np.allclose(attach_positions(0.05), attach_positions(0.90))


def test_nan_triggers_divergence_and_fallback_is_finite(stub_asset, physics_solver, mock_cfg):
    asset = stub_asset
    dt = float(mock_cfg.require("physics.dt"))
    state = physics_solver.initial_state(asset)
    for _ in range(3):
        state = physics_solver.step(state, asset, [], dt)
    bad = state.copy()
    bad.positions[0, 0] = np.nan
    with pytest.raises(TryonError) as exc:
        physics_solver.step(bad, asset, [], dt)
    assert exc.value.code == "E_PHYS_DIVERGE"
    assert exc.value.fallback is not None
    assert np.all(np.isfinite(exc.value.fallback.positions))
    assert np.all(np.isfinite(exc.value.fallback.velocities))
