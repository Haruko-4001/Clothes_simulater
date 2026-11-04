"""Contact extraction: GRASP / PULL / LIFT / SLIDE / RELEASE (SPEC §9.3 / §10.2).

IMPL: src/tryon/perception/contact.py::ContactExtractor

force_proxy is a visual proxy only; it must never be passed to the physics
solver as a real force input (SPEC §17.18).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..types import BodyState, ContactEvent, ContactKind, HandState, PhysicsState


class ContactExtractor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._attached: dict[int, str] = {}
        self._targets: dict[int, np.ndarray] = {}
        self._prev_palm: np.ndarray | None = None
        self._prev_t_ms: float | None = None
        self._last_seen_ms: float | None = None
        self._grasp_time: dict[int, float] = {}
        self._grasp_radius = 0.065       # metres (visual proximity)
        self._release_speed = 2.0        # m/s
        self._slide_speed = 0.55         # m/s
        self._hold_s = float(cfg.require("framing.contact_hold_s"))
        self.last_events: list[ContactEvent] = []

    # ------------------------------------------------------------------
    def extract(
        self,
        hands: HandState,
        body: BodyState,
        phys: PhysicsState,
    ) -> list[ContactEvent]:
        palm = hands.palm
        now_ms = float(body.smpl_params.get("t_ms", 0.0))
        dt = 1.0 / 30.0
        speed = 0.0
        if self._prev_palm is not None and self._prev_t_ms is not None:
            dt_s = max((now_ms - self._prev_t_ms) / 1000.0, 1e-4)
            speed = float(np.linalg.norm(palm - self._prev_palm) / dt_s)
        self._prev_palm = palm.copy()
        self._prev_t_ms = now_ms

        events: list[ContactEvent] = []
        positions = phys.positions
        vertex_id, dist = _nearest_vertex(positions, palm)

        if hands.occluded:
            self._last_seen_ms = now_ms
            if self._attached:
                held_ms = now_ms - self._grasp_time.get(vertex_id, now_ms)
                if held_ms > self._hold_s * 1000.0:
                    for vid, hand_id in list(self._attached.items()):
                        events.append(self._event(now_ms, ContactKind.RELEASE, palm, vid, speed, hand_id))
                        self._attached.pop(vid, None)
                        self._targets.pop(vid, None)
            self.last_events = events
            return events

        if not self._attached and vertex_id is not None and dist < self._grasp_radius:
            self._attached[vertex_id] = hands.handedness
            self._targets[vertex_id] = palm.copy()
            self._grasp_time[vertex_id] = now_ms
            events.append(
                self._event(now_ms, ContactKind.GRASP, palm, vertex_id, speed, hands.handedness)
            )
        elif self._attached:
            if speed > self._release_speed:
                for vid, hand_id in list(self._attached.items()):
                    events.append(self._event(now_ms, ContactKind.RELEASE, palm, vid, speed, hand_id))
                    self._attached.pop(vid, None)
                    self._targets.pop(vid, None)
                    self._grasp_time.pop(vid, None)
            else:
                for vid, hand_id in list(self._attached.items()):
                    self._targets[vid] = palm.copy()
                    kind = self._classify_motion(phys, vid, palm, speed)
                    events.append(self._event(now_ms, kind, palm, vid, speed, hand_id))
        self.last_events = events
        return events

    # ------------------------------------------------------------------
    def _classify_motion(
        self,
        phys: PhysicsState,
        vid: int,
        palm: np.ndarray,
        speed: float,
    ) -> ContactKind:
        if speed > self._slide_speed:
            return ContactKind.SLIDE
        if vid < len(phys.positions):
            dy = palm[1] - float(phys.positions[vid, 1])
            if dy > 0.03:
                return ContactKind.LIFT
        return ContactKind.PULL

    def _event(
        self,
        t_ms: float,
        kind: ContactKind,
        palm: np.ndarray,
        vertex_id: int | None,
        speed: float,
        hand_id: str,
    ) -> ContactEvent:
        return ContactEvent(
            t_ms=t_ms,
            kind=kind,
            world_pos=np.asarray(palm, dtype=np.float64),
            vertex_id=vertex_id,
            force_proxy=float(0.5 * speed * speed),
            hand_id=hand_id,
        )

    @property
    def attached(self) -> dict[int, str]:
        return dict(self._attached)


def _nearest_vertex(positions: np.ndarray, palm: np.ndarray) -> tuple[int | None, float]:
    if positions.size == 0:
        return None, math.inf
    d = np.linalg.norm(positions - palm[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


__all__ = ["ContactExtractor", "_nearest_vertex"]
