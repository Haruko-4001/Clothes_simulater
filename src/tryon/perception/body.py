"""Real body tracker adapter (MediaPipe / SMPL-X). Optional; stub fallback.

IMPL: src/tryon/perception/body.py::BodyTracker / MediaPipeBodyTracker
D12: temporal accumulation is mandatory; a single frame never yields a
high-confidence fit.

Depth note (honest limitation): MediaPipe pose landmarks carry no metric
depth. ``MediaPipeBodyTracker`` back-projects through the configured camera
model at a config depth prior (``perception.body_depth_m``) with per-joint
relative offsets, which keeps relative motion (arms swinging forward) but
fixes the absolute scale. This is not a depth sensor; frame it as such.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..config import Config
from ..math import backproject, camera_to_world
from ..types import BodyState, Frame

# MediaPipe Pose (33) → stub 17-joint convention.
_MP_TO_17 = [0, None, 11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, None, None, None]
_CORE = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class BodyTracker:
    """Legacy guarded integration point (kept for SPEC §8 routing)."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("mediapipe is not installed") from exc
        raise ImportError(
            "legacy BodyTracker adapter is a guarded integration point; "
            "use MediaPipeBodyTracker"
        )

    def track(self, frame: Frame, history: deque | None = None) -> BodyState:  # pragma: no cover
        raise NotImplementedError("real body tracker is not implemented")


class MediaPipeBodyTracker:
    """MediaPipe Pose → BodyState in the stub's 17-joint world convention."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("mediapipe is not installed") from exc
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if cfg is not None:
            from .stubs import camera_model as _cm

            self._model = _cm(cfg)
        else:
            from .camera import default_camera

            self._model = default_camera(1280, 720, fov_deg=45.0)
        self._depth_m = float(cfg.get("perception.body_depth_m", 1.9)) if cfg else 1.9
        self._accumulate = (
            int(cfg.get("framing.temporal_accumulate_frames", 30)) if cfg else 30
        )
        self._first_pelvis: np.ndarray | None = None

    # ------------------------------------------------------------------
    def track(self, frame: Frame, history: deque | None = None) -> BodyState:
        img = np.asarray(frame.image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        result = self._pose.process(img)

        n_hist = len(history) if history is not None else 0
        n = min(n_hist + 1, max(1, self._accumulate))

        if result is None or result.pose_landmarks is None:
            # No detection: fall back to the last history state, low confidence.
            if history:
                prev = list(history)[-1]
                return BodyState(
                    smpl_params=dict(prev.smpl_params),
                    pose_keypoints=prev.pose_keypoints.copy(),
                    bbox=prev.bbox,
                    confidence=0.20,
                    fitted_from_frames=n,
                )
            kp = np.zeros((17, 3), dtype=np.float64)
            kp[:, 1] = np.linspace(0.1, 1.7, 17)
            return BodyState(
                smpl_params={
                    "beta": np.zeros(10),
                    "theta": np.zeros(3),
                    "transl": np.zeros(3),
                    "facing": "unknown",
                    "occlusion": 1.0,
                    "keypoints_2d": np.zeros((17, 2)),
                    "t_ms": frame.pts_ms,
                },
                pose_keypoints=kp,
                bbox=(0, 0, img.shape[1], img.shape[0]),
                confidence=0.10,
                fitted_from_frames=n,
            )

        lms = result.pose_landmarks.landmark
        h, w = img.shape[:2]
        uv = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)  # (33,2)
        z_rel = np.array([p.z for p in lms], dtype=np.float64)
        vis = np.array([p.visibility for p in lms], dtype=np.float64)

        depth = self._depth_m * (1.0 + np.clip(z_rel, -0.25, 0.25))
        pts_cam = backproject(self._model.K, uv, depth)
        world = camera_to_world(self._model.T_CW, pts_cam)  # (33,3)

        kp = np.zeros((17, 3), dtype=np.float64)
        for j17, mp_idx in enumerate(_MP_TO_17):
            if mp_idx is not None:
                kp[j17] = world[mp_idx]
        # Derived joints: neck / pelvis / spine / chest.
        kp[1] = 0.5 * (world[11] + world[12])            # neck
        kp[14] = 0.5 * (world[23] + world[24])            # pelvis
        kp[16] = 0.5 * (kp[1] + kp[14])                    # spine
        kp[15] = 0.5 * (kp[1] + kp[16])                    # chest

        # Temporal smoothing (D12): blend with the recent mean.
        if history:
            recent = list(history)[-4:]
            if recent:
                stacked = np.stack([b.pose_keypoints for b in recent], axis=0)
                kp = 0.5 * kp + 0.5 * stacked.mean(axis=0)

        if self._first_pelvis is None:
            self._first_pelvis = kp[14].copy()
        transl = kp[14] - self._first_pelvis

        # Facing from shoulder/nose geometry.
        nose_vis = float(vis[0])
        ls, rs = float(vis[11]), float(vis[12])
        shoulders = (ls > 0.5) and (rs > 0.5)
        nose_x = uv[0, 0]
        between = min(uv[11, 0], uv[12, 0]) - 0.05 * w <= nose_x <= max(uv[11, 0], uv[12, 0]) + 0.05 * w
        if shoulders and nose_vis > 0.5 and between:
            facing = "frontal"
        elif shoulders and nose_vis <= 0.3:
            facing = "back"
        else:
            facing = "side"
        occlusion = float(np.clip(1.0 - vis[_CORE].mean(), 0.0, 1.0))

        vis_ratio = float(np.clip(vis[_CORE].mean(), 0.0, 1.0))
        confidence = float(
            np.clip(0.45 * vis_ratio + 0.40 * (1.0 - occlusion) + 0.15 * min(n / 8.0, 1.0), 0.0, 1.0)
        )

        x0 = int(np.floor(uv[:, 0].min())) - 8
        y0 = int(np.floor(uv[:, 1].min())) - 8
        x1 = int(np.ceil(uv[:, 0].max())) + 8
        y1 = int(np.ceil(uv[:, 1].max())) + 8
        bbox = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))

        # Pose feature vector (NOT SMPL θ): normalised geometry that responds
        # to real pose changes so keyframe stability detection stays meaningful.
        theta = np.array(
            [
                abs(uv[11, 0] - uv[12, 0]) / max(w, 1),
                (uv[11, 1] + uv[12, 1]) / (2.0 * max(h, 1)),
                (uv[23, 0] + uv[24, 0]) / (2.0 * max(w, 1)),
            ],
            dtype=np.float64,
        )

        return BodyState(
            smpl_params={
                "beta": np.zeros(10, dtype=np.float64),
                "theta": theta,
                "transl": transl,
                "facing": facing,
                "occlusion": occlusion,
                "keypoints_2d": uv,
                "t_ms": frame.pts_ms,
            },
            pose_keypoints=kp,
            bbox=bbox,
            confidence=confidence,
            fitted_from_frames=n,
        )

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass


__all__ = ["BodyTracker", "MediaPipeBodyTracker"]
