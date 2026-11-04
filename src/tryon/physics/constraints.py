"""Contact attachment, non-penetration and Coulomb friction (SPEC §10.2).

IMPL: src/tryon/physics/constraints.py::apply_contact
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..types import ContactEvent, ContactKind, PhysicsState


def process_contact_events(
    state: PhysicsState,
    contacts: list[ContactEvent],
    targets: dict[int, np.ndarray],
    target_step: dict[int, int],
) -> None:
    """Apply GRASP / PULL / LIFT / SLIDE / RELEASE semantics in place.

    INV: RELEASE clears ``attached`` but KEEPS ``velocities``.
    """
    for ev in contacts:
        vid = ev.vertex_id
        if vid is None:
            continue
        if ev.kind == ContactKind.GRASP:
            state.attached[vid] = ev.hand_id
            state.positions[vid] = ev.world_pos
            state.velocities[vid] = 0.0
            targets[vid] = np.asarray(ev.world_pos, dtype=np.float64).copy()
            target_step[vid] = state.step
        elif ev.kind in (ContactKind.PULL, ContactKind.LIFT, ContactKind.SLIDE):
            if vid in state.attached:
                targets[vid] = np.asarray(ev.world_pos, dtype=np.float64).copy()
                target_step[vid] = state.step
        elif ev.kind == ContactKind.RELEASE:
            state.attached.pop(vid, None)
            targets.pop(vid, None)
            target_step.pop(vid, None)
            # velocities intentionally preserved


def apply_contact(
    x: np.ndarray,
    v: np.ndarray,
    contacts: list[ContactEvent],
    attached: dict[int, str],
    targets: dict[int, np.ndarray],
    centers: np.ndarray,
    radii: np.ndarray,
    mu: float,
    penetration_clip_m: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resolve ellipsoid non-penetration and friction; return penetration count.

    ``force_proxy`` from ContactEvent is never read here (SPEC §17.18).
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if centers is None or len(centers) == 0:
        return x, v, 0
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1, 3)
    q = (x[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    d2 = np.sum(q * q, axis=2)
    inside = d2 < 1.0
    near = d2 < 1.0 + 2.0e-3
    # Resolve the deepest violating ellipsoid per vertex.
    if np.any(inside):
        kbest = np.argmin(d2, axis=1)
        rows = np.arange(len(x))
        hit = inside[rows, kbest]
        kk = kbest[hit]
        xh = x[hit]
        c = centers[kk]
        r = radii[kk]
        qh = (xh - c) / r
        n = np.linalg.norm(qh, axis=1)
        n = np.maximum(n, 1.0e-9)
        # Clamp to the ellipsoid surface.
        new_q = qh / n[:, None]
        x[hit] = c + new_q * r

    # Coulomb-style tangential damping for vertices in contact.
    if np.any(near):
        kbest = np.argmin(d2, axis=1)
        rows = np.arange(len(x))
        hit = near[rows, kbest]
        kk = kbest[hit]
        c = centers[kk]
        r = radii[kk]
        qh = (x[hit] - c) / r
        normal = qh / np.maximum(np.linalg.norm(qh, axis=1), 1e-9)[:, None]
        vn = np.sum(v[hit] * normal, axis=1, keepdims=True)
        vt = v[hit] - vn * normal
        vt *= max(0.0, 1.0 - 0.5 * float(np.clip(mu, 0.0, 1.5)))
        v[hit] = np.minimum(vn, 0.0) * normal + vt

    q2 = (x[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    post = np.sum(q2 * q2, axis=2)
    pen = int(np.sum(np.any(post < 1.0 - 1.0e-6, axis=1)))
    return x, v, pen


def friction_limit(normal_impulse: float, mu: float) -> float:
    return float(max(normal_impulse, 0.0) * max(mu, 0.0))


__all__ = ["process_contact_events", "apply_contact", "friction_limit"]
