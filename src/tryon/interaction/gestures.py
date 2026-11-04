"""Gesture classification helpers (visual proxies only)."""
from __future__ import annotations

import numpy as np

from ..types import ContactEvent, ContactKind


def classify_hand_motion(
    prev_pos: np.ndarray | None,
    pos: np.ndarray,
    *,
    speed: float,
    release_speed: float = 2.0,
    lift_dy: float = 0.03,
    slide_speed: float = 0.55,
) -> ContactKind:
    if prev_pos is None:
        return ContactKind.GRASP
    if speed > release_speed:
        return ContactKind.RELEASE
    dy = float(pos[1] - prev_pos[1])
    if dy > lift_dy:
        return ContactKind.LIFT
    if speed > slide_speed:
        return ContactKind.SLIDE
    return ContactKind.PULL


def event_velocity(prev: ContactEvent | None, cur: ContactEvent) -> np.ndarray:
    if prev is None:
        return np.zeros(3, dtype=np.float64)
    dt_s = max((cur.t_ms - prev.t_ms) / 1000.0, 1.0e-4)
    return (cur.world_pos - prev.world_pos) / dt_s


__all__ = ["classify_hand_motion", "event_velocity"]
