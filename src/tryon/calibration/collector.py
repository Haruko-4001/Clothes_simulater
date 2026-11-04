"""Trajectory collection and excitation classification (SPEC §10.3(4)).

IMPL: src/tryon/calibration/collector.py::TrajectoryCollector
Light shaking updates confidence only — it never unlocks parameter updates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..types import ContactEvent, ContactKind, PhysicsState


@dataclass
class CollectedSample:
    contact: ContactEvent
    state: PhysicsState


class TrajectoryCollector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.window_frames = int(cfg.require("calibration.window_frames"))
        self.epsilon_s = float(cfg.require("calibration.epsilon_s"))
        self.epsilon_a = float(cfg.require("calibration.epsilon_a"))
        self.rho_excitation = float(cfg.require("calibration.rho_excitation"))
        self.samples: deque[CollectedSample] = deque(maxlen=self.window_frames)
        self.n_interactions = 0
        self.last_mask: list[str] = []
        self.last_reason: str = "none"
        self._last_was_grasp = False

    # ------------------------------------------------------------------
    def observe(self, ev: ContactEvent, state: PhysicsState) -> None:
        self.samples.append(CollectedSample(contact=ev, state=state.copy()))
        if ev.kind == ContactKind.GRASP and not self._last_was_grasp:
            self.n_interactions += 1
        if ev.kind == ContactKind.RELEASE:
            self._last_was_grasp = False
        else:
            self._last_was_grasp = ev.kind in (ContactKind.GRASP, ContactKind.PULL, ContactKind.LIFT)

    # ------------------------------------------------------------------
    def is_excited(self) -> bool:
        self.last_mask = []
        self.last_reason = "none"
        if len(self.samples) < 3:
            return False
        positions = np.stack([s.contact.world_pos for s in self.samples], axis=0)
        times = np.array([s.contact.t_ms for s in self.samples], dtype=np.float64) / 1000.0
        energies = np.array([s.state.energy for s in self.samples], dtype=np.float64)
        kinds = [s.contact.kind for s in self.samples]

        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_disp = float(np.linalg.norm(positions[-1] - positions[0]))
        max_step = float(diffs.max()) if diffs.size else 0.0
        dt_s = np.maximum(np.diff(times), 1.0e-3)
        speeds = diffs / dt_s
        accel = np.abs(np.diff(speeds)) / np.maximum(dt_s[1:], 1.0e-3) if speeds.size > 1 else np.zeros(1)
        max_accel = float(accel.max()) if accel.size else 0.0

        centered = positions - positions.mean(axis=0, keepdims=True)
        if len(positions) >= 3:
            cov = centered.T @ centered / max(len(positions) - 1, 1)
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            ratio = float(eig[0] / max(eig[1], 1.0e-12)) if eig[0] > 1.0e-12 else 0.0
        else:
            ratio = 0.0

        # Light shake: tiny displacement AND tiny acceleration.
        if max_step < self.epsilon_s and max_accel < self.epsilon_a:
            self.last_reason = "light_shake"
            return False

        has_release = ContactKind.RELEASE in kinds
        active_contact = kinds[-1] in (ContactKind.GRASP, ContactKind.PULL, ContactKind.LIFT)

        if ratio > self.rho_excitation and max_step >= self.epsilon_s:
            self.last_reason = "single_axis_pull"
            self.last_mask = ["youngs_modulus", "damping"]
        if total_disp > self.epsilon_s and self._is_sliding(positions):
            self.last_reason = "relative_slide"
            self.last_mask = sorted(set(self.last_mask) | {"friction"})
        if (not active_contact) and max_accel > self.epsilon_a:
            self.last_reason = "free_hang"
            self.last_mask = sorted(set(self.last_mask) | {"areal_density", "bending_stiffness"})
        if has_release and len(energies) > 2 and energies[-1] <= energies[0]:
            self.last_reason = "release_rebound" if self.last_reason == "none" else self.last_reason
            self.last_mask = sorted(set(self.last_mask) | {"damping", "youngs_modulus"})
        return bool(self.last_mask)

    def _is_sliding(self, positions: np.ndarray) -> bool:
        if len(positions) < 3:
            return False
        v = positions[-1] - positions[-2]
        tangential = np.linalg.norm(v - np.array([0.0, v[1], 0.0]))
        return bool(tangential > self.epsilon_s)

    # ------------------------------------------------------------------
    @property
    def states(self) -> list[PhysicsState]:
        return [s.state for s in self.samples]

    @property
    def contacts(self) -> list[ContactEvent]:
        return [s.contact for s in self.samples]

    def ready(self) -> bool:
        return len(self.samples) >= self.window_frames

    def clear(self) -> None:
        self.samples.clear()
        self.last_mask = []
        self.last_reason = "none"


__all__ = ["TrajectoryCollector", "CollectedSample"]
