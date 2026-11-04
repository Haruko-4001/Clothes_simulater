"""Formal state machines: asset / interaction / keyframe (SPEC §6).

IMPL: src/tryon/interaction/state_machine.py
"""
from __future__ import annotations

import math
from enum import Enum

from ..config import Config
from ..types import ContactKind, GateVerdict


class AssetState(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    GEOMETRY_READY = "GEOMETRY_READY"
    MATERIAL_PRIOR = "MATERIAL_PRIOR"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CALIBRATING = "CALIBRATING"
    CALIBRATED = "CALIBRATED"
    DEGRADED = "DEGRADED"


class InteractionState(str, Enum):
    IDLE = "IDLE"
    GRASPED = "GRASPED"
    PULLING = "PULLING"
    LIFTING = "LIFTING"
    REBOUNDING = "REBOUNDING"


class KeyframeState(str, Enum):
    TRACKING = "TRACKING"
    STANDBY = "STANDBY"
    GENERATING = "GENERATING"
    FUSING = "FUSING"


class AssetStateMachine:
    """§6.1 transition table."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = AssetState.BOOTSTRAP
        self.needs_human_review = False
        self.version = 0
        self.scene_tag = ""
        self.ttl_s = 1800.0
        self.history: list[tuple[AssetState, AssetState, str]] = []

    def _to(self, nxt: AssetState, event: str) -> AssetState:
        self.history.append((self.state, nxt, event))
        self.state = nxt
        return self.state

    def g1(self, passed: bool) -> AssetState:
        if self.state != AssetState.BOOTSTRAP:
            return self.state
        if passed:
            return self._to(AssetState.GEOMETRY_READY, "G1.pass")
        self.needs_human_review = True
        return self._to(AssetState.DEGRADED, "G1.fail")

    def confidence_review(self, overall: float) -> AssetState:
        if self.state != AssetState.GEOMETRY_READY:
            return self.state
        tau = float(self.cfg.require("asset.review_threshold"))
        if float(overall) >= tau:
            return self._to(AssetState.MATERIAL_PRIOR, "conf.overall>=tau_review")
        return self._to(AssetState.NEEDS_REVIEW, "conf.overall<tau_review")

    def excitation(self, is_excited: bool, fisher_cond: float) -> AssetState:
        if self.state != AssetState.MATERIAL_PRIOR:
            return self.state
        kappa_max = float(self.cfg.require("calibration.kappa_max"))
        if is_excited and float(fisher_cond) < kappa_max:
            return self._to(AssetState.CALIBRATING, "is_excited ∧ cond<kappa")
        return self.state

    def calibration(self, verdict: GateVerdict, *, scene_tag: str = "", ttl_s: float = 1800.0) -> AssetState:
        if self.state != AssetState.CALIBRATING:
            return self.state
        if verdict == GateVerdict.COMMIT:
            self.version += 1
            self.scene_tag = scene_tag
            self.ttl_s = float(ttl_s)
            return self._to(AssetState.CALIBRATED, "G3.COMMIT")
        if verdict == GateVerdict.DEGRADE:
            return self._to(AssetState.DEGRADED, "G3.DEGRADE")
        return self.state

    def interaction_update(self, n_interactions: int, converged: bool) -> AssetState:
        if self.state != AssetState.CALIBRATING:
            return self.state
        limit = int(self.cfg.require("calibration.max_interactions_before_degrade"))
        if int(n_interactions) > limit and not converged:
            return self._to(AssetState.DEGRADED, "n_interaction>limit ∧ ¬converged")
        return self.state

    def manual_review(self, confirmed: bool) -> AssetState:
        if self.state != AssetState.NEEDS_REVIEW:
            return self.state
        if confirmed:
            self.needs_human_review = False
            return self._to(AssetState.GEOMETRY_READY, "manual.confirm")
        return self.state

    @classmethod
    def reachable_states(cls) -> set[AssetState]:
        return set(AssetState)


class InteractionStateMachine:
    """§6.2. RELEASE clears attachments but velocities are preserved downstream."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = InteractionState.IDLE
        self.last_energy = 0.0
        self.attached_cleared = False
        self.release_count = 0
        self.energy_settle = float(cfg.require("physics.energy_settle_j"))

    def on_event(self, kind: ContactKind) -> InteractionState:
        if kind == ContactKind.GRASP and self.state == InteractionState.IDLE:
            self.state = InteractionState.GRASPED
        elif self.state in (InteractionState.GRASPED, InteractionState.PULLING):
            if kind == ContactKind.PULL:
                self.state = InteractionState.PULLING
            elif kind == ContactKind.LIFT:
                self.state = InteractionState.LIFTING
            elif kind == ContactKind.SLIDE:
                self.state = InteractionState.PULLING
            elif kind == ContactKind.RELEASE:
                self.state = InteractionState.REBOUNDING
                self.attached_cleared = True
                self.release_count += 1
        elif self.state == InteractionState.LIFTING:
            if kind in (ContactKind.PULL, ContactKind.SLIDE):
                self.state = InteractionState.PULLING
            elif kind == ContactKind.RELEASE:
                self.state = InteractionState.REBOUNDING
                self.attached_cleared = True
                self.release_count += 1
        elif self.state == InteractionState.REBOUNDING and kind == ContactKind.GRASP:
            self.state = InteractionState.GRASPED
        return self.state

    def update_energy(self, energy: float) -> InteractionState:
        self.last_energy = float(energy)
        if self.state == InteractionState.REBOUNDING and self.last_energy < self.energy_settle:
            self.state = InteractionState.IDLE
        return self.state

    @staticmethod
    def energy_invariant(energy_prev: float, energy_next: float, dt: float, gamma: float) -> bool:
        return bool(energy_next <= energy_prev * (1.0 - gamma * dt) + 1.0e-12)

    def reset(self) -> None:
        self.state = InteractionState.IDLE
        self.attached_cleared = False


class KeyframeStateMachine:
    """§6.3 four-state machine with hysteresis and expiry."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = KeyframeState.TRACKING
        self.epsilon_in = float(cfg.require("keyframe.epsilon_in"))
        self.epsilon_out = float(cfg.require("keyframe.epsilon_out"))
        self.min_interval_ms = float(cfg.require("keyframe.min_interval_ms"))
        self.stable_count = 0
        self.m = int(math.ceil(float(cfg.require("keyframe.stable_ms")) / (1000.0 / 30.0)))
        self.history: list[tuple[KeyframeState, KeyframeState, str]] = []

    def _to(self, nxt: KeyframeState, event: str) -> KeyframeState:
        self.history.append((self.state, nxt, event))
        self.state = nxt
        return self.state

    def observe(
        self,
        dbar: float,
        *,
        garment_changed: bool = False,
        elapsed_ms: float = 0.0,
        bbox_crossed: bool = False,
    ) -> bool:
        fire = False
        if self.state == KeyframeState.TRACKING:
            if dbar < self.epsilon_in:
                self.stable_count += 1
            else:
                self.stable_count = 0
            if self.stable_count >= self.m:
                self._to(KeyframeState.STANDBY, "stable")
                self.stable_count = 0
        elif self.state == KeyframeState.STANDBY:
            if dbar > self.epsilon_out:
                self._to(KeyframeState.TRACKING, "hysteresis")
            elif garment_changed or elapsed_ms >= self.min_interval_ms or bbox_crossed:
                fire = True
                self._to(KeyframeState.GENERATING, "trigger")
        return fire

    def result_ready(self, expired: bool = False) -> KeyframeState:
        if self.state != KeyframeState.GENERATING:
            return self.state
        if expired:
            return self._to(KeyframeState.STANDBY, "expired")
        return self._to(KeyframeState.FUSING, "result")

    def fuse_done(self) -> KeyframeState:
        if self.state != KeyframeState.FUSING:
            return self.state
        return self._to(KeyframeState.TRACKING, "fused")

    @classmethod
    def reachable_states(cls) -> set[KeyframeState]:
        return set(KeyframeState)


__all__ = [
    "AssetState",
    "InteractionState",
    "KeyframeState",
    "AssetStateMachine",
    "InteractionStateMachine",
    "KeyframeStateMachine",
]
