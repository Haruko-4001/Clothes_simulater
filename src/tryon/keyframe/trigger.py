"""Keyframe trigger with hysteresis + framing hard gate (SPEC §10.6).

IMPL: src/tryon/keyframe/trigger.py::KeyframeTrigger
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import Config
from ..types import FramingVerdict, TriggerContext


class KeyframeTrigger:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.epsilon_in = float(cfg.require("keyframe.epsilon_in"))
        self.epsilon_out = float(cfg.require("keyframe.epsilon_out"))
        if not (0.0 < self.epsilon_in < self.epsilon_out):
            raise ValueError("epsilon_in must be < epsilon_out")
        self.m = int(
            math.ceil(float(cfg.require("keyframe.stable_ms")) / (1000.0 / 30.0))
        )
        self.min_interval_ms = float(cfg.require("keyframe.min_interval_ms"))
        self.state = "TRACKING"
        self._stable_count = 0
        self._last_garment_id: str | None = None
        self.last_dbar: float = float("inf")

    # ------------------------------------------------------------------
    def dbar(self, history: deque) -> float:
        thetas = []
        for b in history:
            theta = b.smpl_params.get("theta")
            if theta is None:
                continue
            thetas.append(np.asarray(theta, dtype=np.float64).reshape(-1))
        if len(thetas) < 2:
            return float("inf")
        recent = thetas[-(self.m + 1) :]
        ds = [float(np.linalg.norm(recent[i] - recent[i - 1])) for i in range(1, len(recent))]
        return float(np.mean(ds)) if ds else float("inf")

    def should_fire(self, ctx: TriggerContext) -> bool:
        # Hard gate: no full-image try-on unless framing is OK.
        if ctx.framing.verdict != FramingVerdict.OK:
            self.state = "TRACKING"
            self._stable_count = 0
            return False

        dbar = self.dbar(ctx.pose_history)
        self.last_dbar = dbar

        if self.state == "GENERATING":
            return False
        if self.state == "FUSING":
            return False

        if self.state == "TRACKING":
            if dbar < self.epsilon_in:
                self._stable_count += 1
            else:
                self._stable_count = 0
            if self._stable_count >= self.m:
                self.state = "STANDBY"
                self._stable_count = 0
            return False

        # STANDBY
        if dbar > self.epsilon_out:
            self.state = "TRACKING"
            self._stable_count = 0
            return False
        garment_changed = self._last_garment_id is not None and ctx.garment_id != self._last_garment_id
        if self._last_garment_id is None:
            garment_changed = True
        if garment_changed or ctx.elapsed_since_kf_ms >= self.min_interval_ms or ctx.bbox_crossed:
            self.state = "GENERATING"
            self._last_garment_id = ctx.garment_id
            return True
        return False

    # External state-machine notifications.
    def notify_fusing(self) -> None:
        self.state = "FUSING"

    def notify_tracking(self) -> None:
        self.state = "TRACKING"
        self._stable_count = 0

    def force_standby(self) -> None:
        self.state = "STANDBY"


__all__ = ["KeyframeTrigger"]
