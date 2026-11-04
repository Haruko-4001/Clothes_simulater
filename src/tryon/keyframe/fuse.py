"""Asynchronous keyframe fusion (SPEC §10.5).

IMPL: src/tryon/keyframe/fuse.py
α_k = α₀ · exp(-(t_k - t_kf)/τ_fuse), exactly ``fuse_frames`` frames.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config


def alpha_sequence(cfg: Config, n: int | None = None, step_ms: float | None = None) -> np.ndarray:
    n = int(cfg.require("keyframe.fuse_frames")) if n is None else int(n)
    step_ms = (1000.0 / 30.0) if step_ms is None else float(step_ms)
    alpha0 = float(cfg.require("keyframe.alpha0"))
    tau = float(cfg.require("keyframe.tau_fuse_ms"))
    return np.array([alpha0 * math.exp(-(i * step_ms) / tau) for i in range(n)], dtype=np.float64)


def is_expired(
    pose_now: np.ndarray,
    pose_req: np.ndarray,
    delta_pose: float,
) -> bool:
    a = np.asarray(pose_now, dtype=np.float64).reshape(-1)
    b = np.asarray(pose_req, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        return True
    return bool(np.linalg.norm(a - b) > float(delta_pose))


class FusionBlender:
    """Tracks a single 6-frame fusion window; never generates extra frames."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.n_frames = int(cfg.require("keyframe.fuse_frames"))
        self.alpha0 = float(cfg.require("keyframe.alpha0"))
        self.tau = float(cfg.require("keyframe.tau_fuse_ms"))
        self.active = False
        self.count = 0
        self.kf_time_ms = 0.0

    def begin(self, kf_time_ms: float) -> None:
        self.active = True
        self.count = 0
        self.kf_time_ms = float(kf_time_ms)

    def step(self, now_ms: float) -> tuple[float, bool]:
        if not self.active:
            return 0.0, False
        if self.count >= self.n_frames:
            self.active = False
            return 0.0, False
        alpha = self.alpha0 * math.exp(-max(0.0, now_ms - self.kf_time_ms) / self.tau)
        self.count += 1
        finished = self.count >= self.n_frames
        if finished:
            self.active = False
        return float(alpha), True

    def reset(self) -> None:
        self.active = False
        self.count = 0


def blend(
    render_image: np.ndarray,
    keyframe_image: np.ndarray,
    alpha: float,
    keyframe_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend keyframe into the physics render outside contact-protected pixels."""
    render = np.asarray(render_image)
    kf = np.asarray(keyframe_image)
    if kf.shape != render.shape:
        return render.copy()
    m = np.asarray(keyframe_mask) > 0
    if protected_mask is not None:
        m &= np.asarray(protected_mask) == 0
    if not m.any() or alpha <= 0.0:
        return render.copy()
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    out = np.array(render, copy=True)
    sub_mask = m[y0:y1, x0:x1]
    a = float(np.clip(alpha, 0.0, 1.0))
    reg = out[y0:y1, x0:x1].astype(np.float32)
    kreg = kf[y0:y1, x0:x1].astype(np.float32)
    reg[sub_mask] = (1.0 - a) * reg[sub_mask] + a * kreg[sub_mask]
    out[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    return out


__all__ = ["alpha_sequence", "is_expired", "FusionBlender", "blend"]
