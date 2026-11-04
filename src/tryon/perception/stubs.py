"""Synthetic perception stack — deterministic, no weights, no GUI.

IMPL: src/tryon/perception/stubs.py

The camera, body tracker, and hand tracker share one scripted scene so that
2D pixels, 3D keypoints, and contact positions are mutually consistent.
"""
from __future__ import annotations

from collections import deque
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from ..math import project
from ..types import BodyState, Frame, HandState, RGBImage
from .camera import CameraModel, default_camera


DT = 1.0 / 30.0

# 17-joint standing skeleton (world metres, Y-up, feet at y≈0).
BASE_KEYPOINTS = np.array(
    [
        [0.00, 1.70, 0.00],   # 0 head
        [0.00, 1.50, 0.00],   # 1 neck
        [-0.18, 1.47, 0.00],  # 2 L shoulder
        [-0.28, 1.23, 0.01],  # 3 L elbow
        [-0.31, 1.00, 0.02],  # 4 L wrist
        [0.18, 1.47, 0.00],   # 5 R shoulder
        [0.28, 1.23, 0.01],   # 6 R elbow
        [0.31, 1.00, 0.02],   # 7 R wrist
        [-0.10, 0.95, 0.00],  # 8 L hip
        [-0.10, 0.52, 0.00],  # 9 L knee
        [-0.10, 0.08, 0.00],  # 10 L ankle
        [0.10, 0.95, 0.00],   # 11 R hip
        [0.10, 0.52, 0.00],   # 12 R knee
        [0.10, 0.08, 0.00],   # 13 R ankle
        [0.00, 0.95, 0.00],   # 14 pelvis
        [0.00, 1.20, 0.00],   # 15 spine
        [0.00, 1.40, 0.00],   # 16 chest
    ],
    dtype=np.float64,
)

BONES = [
    (0, 1), (1, 16), (16, 15), (15, 14),
    (16, 2), (2, 3), (3, 4), (16, 5), (5, 6), (6, 7),
    (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]


def camera_model(cfg: Config) -> CameraModel:
    return default_camera(
        int(cfg.require("camera.width")),
        int(cfg.require("camera.height")),
        fov_deg=45.0,
    )


def scripted_body_keypoints(frame_id: int) -> np.ndarray:
    t = frame_id * DT
    sway = np.array(
        [0.004 * math.sin(0.9 * t), 0.002 * math.sin(1.7 * t), 0.0],
        dtype=np.float64,
    )
    kp = BASE_KEYPOINTS.copy() + sway[None, :]
    # Tiny arm sway; stays far below the keyframe stability threshold.
    kp[3, 0] += 0.004 * math.sin(0.7 * t)
    kp[4, 0] += 0.006 * math.sin(0.7 * t + 0.2)
    kp[6, 0] -= 0.004 * math.sin(0.7 * t + 0.4)
    kp[7, 0] -= 0.006 * math.sin(0.7 * t + 0.6)
    return kp


def scripted_pose_theta(frame_id: int) -> np.ndarray:
    t = frame_id * DT
    return np.array(
        [0.0020 * math.sin(0.9 * t), 0.0015 * math.sin(1.3 * t), 0.0008 * math.sin(0.5 * t)],
        dtype=np.float64,
    )


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    u = float(np.clip(u, 0.0, 1.0))
    return a + u * (b - a)


IDLE_HAND = np.array([-0.46, 0.95, 0.12], dtype=np.float64)
GRASP_HAND = np.array([-0.238, 1.22, 0.02], dtype=np.float64)
PULL_HAND = np.array([-0.44, 1.06, 0.20], dtype=np.float64)
AWAY_HAND = np.array([-0.62, 0.98, 0.32], dtype=np.float64)


def scripted_hand_palm(frame_id: int) -> np.ndarray:
    """Two grasp→pull→release interactions per 240-frame cycle."""
    f = frame_id % 240
    if f < 55:
        return IDLE_HAND.copy()
    if f < 72:
        return _lerp(IDLE_HAND, GRASP_HAND, (f - 55) / 17.0)
    if f < 100:
        return _lerp(GRASP_HAND, PULL_HAND, (f - 72) / 28.0)
    if f < 103:
        return _lerp(PULL_HAND, AWAY_HAND, (f - 100) / 3.0)
    if f < 130:
        return _lerp(AWAY_HAND, IDLE_HAND, (f - 103) / 27.0)
    if f < 147:
        return _lerp(IDLE_HAND, GRASP_HAND, (f - 130) / 17.0)
    if f < 175:
        return _lerp(GRASP_HAND, PULL_HAND, (f - 147) / 28.0)
    if f < 178:
        return _lerp(PULL_HAND, AWAY_HAND, (f - 175) / 3.0)
    return _lerp(AWAY_HAND, IDLE_HAND, (f - 178) / 62.0)


def hand_landmarks(palm: np.ndarray) -> np.ndarray:
    pts = [palm]
    for k in range(20):
        ang = 2.0 * math.pi * k / 20.0
        r = 0.035 + 0.010 * ((k % 3) / 2.0)
        z = 0.012 * math.sin(ang * 2.0)
        pts.append(palm + np.array([r * math.cos(ang), r * math.sin(ang) * 0.8, z]))
    return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stub body tracker (D12 temporal accumulation)
# ---------------------------------------------------------------------------


class StubBodyTracker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.accumulate = int(cfg.get("framing.temporal_accumulate_frames", 30))
        self._model = camera_model(cfg)

    def track(self, frame: Frame, history: deque | None = None) -> BodyState:
        n_hist = len(history) if history is not None else 0
        n = min(n_hist + 1, max(1, self.accumulate))
        confidence = float(0.55 + 0.40 * (1.0 - math.exp(-n / 8.0)))
        kp = scripted_body_keypoints(frame.frame_id)
        if history:
            recent = list(history)[-4:]
            if recent:
                stacked = np.stack([b.pose_keypoints for b in recent], axis=0)
                kp = 0.5 * kp + 0.5 * stacked.mean(axis=0)
        uv, depth = project(self._model.K, self._model.T_CW, kp)
        x0 = int(np.floor(uv[:, 0].min())) - 8
        y0 = int(np.floor(uv[:, 1].min())) - 8
        x1 = int(np.ceil(uv[:, 0].max())) + 8
        y1 = int(np.ceil(uv[:, 1].max())) + 8
        theta = scripted_pose_theta(frame.frame_id)
        return BodyState(
            smpl_params={
                "beta": np.zeros(10, dtype=np.float64),
                "theta": theta,
                "transl": np.array(kp[14] - BASE_KEYPOINTS[14], dtype=np.float64),
                "facing": "frontal",
                "occlusion": 0.02,
                "keypoints_2d": uv,
                # Simulated timeline: contact speeds must not depend on the
                # host's achieved wall-clock fps.
                "t_ms": frame.frame_id * (1000.0 / 30.0),
            },
            pose_keypoints=kp,
            bbox=(x0, y0, x1, y1),
            confidence=confidence,
            fitted_from_frames=n,
        )


# ---------------------------------------------------------------------------
# Stub hand tracker
# ---------------------------------------------------------------------------


class StubHandTracker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def track(self, frame: Frame) -> HandState:
        palm = scripted_hand_palm(frame.frame_id)
        return HandState(
            landmarks=hand_landmarks(palm),
            mesh=None,
            handedness="right",
            occluded=False,
        )


# ---------------------------------------------------------------------------
# Synthetic camera
# ---------------------------------------------------------------------------


def _background_image(model: CameraModel) -> Image.Image:
    w, h = model.width, model.height
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    bg = (np.array([214.0, 220.0, 228.0])[None, None, :] * (1.0 - 0.18 * yy[:, :, None])).astype(
        np.uint8
    )
    image = np.repeat(bg, w, axis=1)
    return Image.fromarray(image)


def render_scene(
    frame_id: int,
    cfg: Config,
    model: CameraModel | None = None,
    background: Image.Image | None = None,
) -> RGBImage:
    model = model or camera_model(cfg)
    w, h = model.width, model.height
    pil = (background or _background_image(model)).copy()
    draw = ImageDraw.Draw(pil)

    kp = scripted_body_keypoints(frame_id)
    uv, _ = project(model.K, model.T_CW, kp)
    lw = max(2, int(round(h * 0.020)))

    # Torso + head.
    torso = [tuple(uv[i]) for i in (2, 5, 11, 8)]
    draw.polygon(torso, fill=(58, 92, 124))
    head_r = max(3, int(round(h * 0.028)))
    cx, cy = uv[0]
    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=(196, 154, 126))
    for a, b in BONES:
        draw.line([tuple(uv[a]), tuple(uv[b])], fill=(70, 104, 138), width=lw)
    for i in (9, 12, 10, 13):
        draw.ellipse(
            [uv[i][0] - lw * 0.7, uv[i][1] - lw * 0.7, uv[i][0] + lw * 0.7, uv[i][1] + lw * 0.7],
            fill=(70, 104, 138),
        )

    # Hand blob (visual consistency with the hand tracker).
    palm = scripted_hand_palm(frame_id)
    puv, pdepth = project(model.K, model.T_CW, palm.reshape(1, 3))
    pr = max(3, int(round(model.focal() * 0.05 / max(float(pdepth[0]), 0.1))))
    draw.ellipse(
        [puv[0, 0] - pr, puv[0, 1] - pr, puv[0, 0] + pr, puv[0, 1] + pr],
        fill=(205, 165, 135),
        outline=(150, 110, 85),
    )

    return np.asarray(pil, dtype=np.uint8)


class StubCamera:
    def __init__(self, cfg: Config, *, seed: int = 0) -> None:
        self.cfg = cfg
        self.model = camera_model(cfg)
        self.K = self.model.K
        self.T_CW = self.model.T_CW
        self.width = self.model.width
        self.height = self.model.height
        self._id = 0
        self._seed = int(seed)
        self._background = _background_image(self.model)

    def capture(self) -> Frame:
        self._id += 1
        image = render_scene(self._id, self.cfg, self.model, self._background)
        return Frame(
            frame_id=self._id,
            pts_ms=time.time() * 1000.0,
            image=image,
        )


__all__ = [
    "StubBodyTracker",
    "StubHandTracker",
    "StubCamera",
    "camera_model",
    "render_scene",
    "scripted_body_keypoints",
    "scripted_hand_palm",
    "scripted_pose_theta",
    "hand_landmarks",
    "BASE_KEYPOINTS",
]
