"""Real hand tracker adapter (MediaPipe Hands default, SPEC §13).

IMPL: src/tryon/perception/hands.py::HandTracker / MediaPipeHandTracker
Missing dependency ⇒ ImportError ⇒ factory falls back to StubHandTracker.

Depth note (honest limitation): monocular 2D landmarks carry no metric depth,
so ``MediaPipeHandTracker`` back-projects through the configured camera model
at a config depth prior (``perception.hand_depth_m``). Grasp interactions at
a roughly constant webcam distance behave correctly; this is not a depth
sensor and must not be presented as one.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..math import backproject, camera_to_world
from ..types import Frame, HandState


class HandTracker:
    """Legacy guarded integration point (kept for SPEC §8 routing)."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("mediapipe is not installed") from exc
        raise ImportError(
            "legacy HandTracker adapter is a guarded integration point; "
            "use MediaPipeHandTracker"
        )

    def track(self, frame: Frame) -> HandState:  # pragma: no cover
        raise NotImplementedError("real hand tracker is not implemented")


class MediaPipeHandTracker:
    """MediaPipe Hands → HandState (world metres via a config depth prior)."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("mediapipe is not installed") from exc
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if cfg is not None:
            from .stubs import camera_model as _cm

            self._model = _cm(cfg)
        else:
            from .camera import default_camera

            self._model = default_camera(1280, 720, fov_deg=45.0)
        self._depth_m = float(cfg.get("perception.hand_depth_m", 1.7)) if cfg else 1.7
        self._last_landmarks: np.ndarray | None = None
        self._last_handedness: str = "right"

    def track(self, frame: Frame) -> HandState:
        img = np.asarray(frame.image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        result = self._hands.process(img)
        landmarks = None
        handedness = self._last_handedness
        if result is not None and result.multi_hand_landmarks:
            hands = result.multi_hand_landmarks
            lm = hands[0].landmark
            h, w = img.shape[:2]
            uv = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
            z_rel = np.array([p.z for p in lm], dtype=np.float64)
            depth = self._depth_m * (1.0 + np.clip(z_rel, -0.3, 0.3))
            pts_cam = backproject(self._model.K, uv, depth)
            world = camera_to_world(self._model.T_CW, pts_cam)
            landmarks = world
            if result.multi_handedness:
                label = result.multi_handedness[0].classification[0].label
                handedness = "left" if label == "Left" else "right"
            self._last_landmarks = world
            self._last_handedness = handedness
            occluded = False
        else:
            landmarks = self._last_landmarks
            occluded = True
        if landmarks is None:
            landmarks = np.zeros((21, 3), dtype=np.float64)
        return HandState(
            landmarks=landmarks,
            mesh=None,
            handedness=handedness,
            occluded=occluded,
        )

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass


__all__ = ["HandTracker", "MediaPipeHandTracker"]
