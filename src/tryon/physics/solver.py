"""PhysicsSolver — O(N) XPBD step, attachment persistence, divergence fallback.

IMPL: src/tryon/physics/solver.py::PhysicsSolver
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from ..config import Config
from ..types import (
    ContactEvent,
    GarmentAsset,
    MaterialParams,
    PhysicsState,
    SimReadyMesh,
    TryonError,
)
from .constraints import process_contact_events
from .integrator import integrate
from .state import MeshConstraints, build_constraints, initial_state


class PhysicsSolver:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._constraints: MeshConstraints | None = None
        self._constraint_key: tuple | None = None
        self._targets: dict[int, np.ndarray] = {}
        self._target_step: dict[int, int] = {}
        self._last_stable: PhysicsState | None = None
        self._asset_id: str | None = None
        self.asset: GarmentAsset | None = None
        self.last_penetration_count: int = 0
        self.last_step_ms: float = 0.0
        self.n_physics_steps: int = 0

    # ------------------------------------------------------------------
    def initial_state(self, asset: GarmentAsset) -> PhysicsState:
        c = self._ensure_constraints(asset)
        state = initial_state(asset.rest_mesh, c.n_constraints)
        state.lam = np.zeros(c.n_constraints, dtype=np.float64)
        self._last_stable = state.copy()
        self._targets.clear()
        self._target_step.clear()
        self._asset_id = asset.asset_id
        self.asset = asset
        return state

    def reset(self, asset: GarmentAsset) -> PhysicsState:
        return self.initial_state(asset)

    # ------------------------------------------------------------------
    def _ensure_constraints(self, asset: GarmentAsset) -> MeshConstraints:
        material = asset.material
        key = (
            asset.asset_id,
            asset.rest_mesh.n_vertices,
            asset.rest_mesh.n_faces,
            material.to_vector().tobytes(),
            float(asset.rest_mesh.thickness),
        )
        if self._constraints is not None and self._constraint_key == key:
            return self._constraints
        c = build_constraints(asset.rest_mesh, material)
        self._constraints = c
        self._constraint_key = key
        return c

    # ------------------------------------------------------------------
    def step(
        self,
        state: PhysicsState,
        asset: GarmentAsset,
        contacts: list[ContactEvent],
        dt: float,
    ) -> PhysicsState:
        t0 = time.perf_counter()
        if not isinstance(asset.rest_mesh, SimReadyMesh):
            raise TypeError(
                "D13 violated: only a SimReadyMesh physics shell may enter the solver"
            )
        if not (np.all(np.isfinite(state.positions)) and np.all(np.isfinite(state.velocities))):
            raise TryonError(
                "E_PHYS_DIVERGE",
                "input state is non-finite",
                fallback=self._last_stable.copy() if self._last_stable is not None else None,
            )

        c = self._ensure_constraints(asset)
        self.asset = asset
        work = state.copy()
        if len(work.lam) != c.n_constraints:
            lam = np.zeros(c.n_constraints, dtype=np.float64)
            n = min(len(work.lam), c.n_constraints)
            if n:
                lam[:n] = work.lam[:n]
            work.lam = lam

        # Contact events → attached / targets (RELEASE keeps velocities).
        process_contact_events(work, contacts, self._targets, self._target_step)

        # Occlusion hold: release attachments that have not been refreshed.
        hold_steps = int(
            math.ceil(float(self.cfg.require("framing.contact_hold_s")) / max(dt, 1e-9))
        )
        for vid in list(work.attached.keys()):
            last = self._target_step.get(vid, work.step)
            if work.step - last > hold_steps:
                work.attached.pop(vid, None)
                self._targets.pop(vid, None)
                self._target_step.pop(vid, None)

        collider = asset.collision_body
        centers = np.asarray(getattr(collider, "centers", np.zeros((0, 3))), dtype=np.float64)
        radii = np.asarray(getattr(collider, "radii", np.zeros((0, 3))), dtype=np.float64)

        try:
            x, v, lam, energy, pen = integrate(
                work,
                asset.material,
                c,
                dt=float(dt),
                cfg=self.cfg,
                centers=centers,
                radii=radii,
                attached=work.attached,
                targets=self._targets,
            )
        except TryonError as exc:
            if exc.code == "E_PHYS_DIVERGE":
                exc.fallback = self._last_stable.copy() if self._last_stable is not None else None
            raise

        new_state = PhysicsState(
            positions=x,
            velocities=v,
            attached=dict(work.attached),
            lam=lam,
            step=work.step + 1,
            energy=float(energy),
        )
        if not (np.all(np.isfinite(new_state.positions)) and np.all(np.isfinite(new_state.velocities))):
            raise TryonError(
                "E_PHYS_DIVERGE",
                "integration produced non-finite state",
                fallback=self._last_stable.copy() if self._last_stable is not None else None,
            )
        self._last_stable = new_state.copy()
        self.last_penetration_count = int(pen)
        self.n_physics_steps += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return new_state

    # ------------------------------------------------------------------
    @property
    def constraints(self) -> MeshConstraints | None:
        return self._constraints


class DiffClothSolver:
    """Real differentiable-cloth adapter (optional; guarded integration point).

    Missing dependency / unfinished integration raises ImportError so the
    factory can fall back to the internal XPBD solver (E_ADAPTER_FALLBACK).
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import diffcloth  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("diffcloth is not installed") from exc
        raise ImportError(
            "DiffCloth integration is a Phase-5 guarded integration point; "
            "falling back to internal XPBD"
        )

    def step(self, state, asset, contacts, dt):  # pragma: no cover
        raise NotImplementedError("DiffCloth solver is not implemented")


__all__ = ["PhysicsSolver", "DiffClothSolver"]
