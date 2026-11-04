"""Camera model and real webcam adapter (SPEC §1.2).

IMPL: src/tryon/perception/camera.py
All projection math delegates to ``tryon.math`` — no local reimplementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..math import look_at, make_intrinsics, project, visible_mask
from ..types import Frame, RGBImage


@dataclass
class CameraModel:
    K: np.ndarray
    T_CW: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.T_CW = np.asarray(self.T_CW, dtype=np.float64).reshape(4, 4)
        self.width = int(self.width)
        self.height = int(self.height)

    def project(self, points_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return project(self.K, self.T_CW, points_W)

    def inside(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return visible_mask(uv, self.width, self.height, margin)

    def focal(self) -> float:
        return float(self.K[0, 0])


def default_camera(width: int, height: int, *, fov_deg: float = 45.0) -> CameraModel:
    K = make_intrinsics(width, height, fov_deg=fov_deg)
    T_CW, _ = look_at(
        eye=np.array([0.0, 1.00, 3.60]),
        target=np.array([0.0, 0.90, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    return CameraModel(K=K, T_CW=T_CW, width=width, height=height)


class RealCamera:
    """OpenCV webcam adapter. Missing cv2 / device ⇒ ImportError (stub fallback)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("opencv-python is not installed") from exc
        width = int(cfg.require("camera.width"))
        height = int(cfg.require("camera.height"))
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ImportError("no camera device available")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cv2 = cv2
        self._cap = cap
        self._model = default_camera(width, height)
        self.K = self._model.K
        self.T_CW = self._model.T_CW
        self._id = 0
        self._t0 = cv2.getTickCount() / cv2.getTickFrequency()

    def capture(self) -> Frame:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)  # BGR is forbidden downstream
        now_ms = 1000.0 * (self._cv2.getTickCount() / self._cv2.getTickFrequency() - self._t0)
        self._id += 1
        return Frame(frame_id=self._id, pts_ms=now_ms, image=rgb)

    def release(self) -> None:
        self._cap.release()


__all__ = ["CameraModel", "default_camera", "RealCamera"]
