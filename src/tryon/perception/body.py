"""Real body tracker adapter (MediaPipe / SMPL-X). Optional; stub fallback.

IMPL: src/tryon/perception/body.py::BodyTracker
D12: temporal accumulation is mandatory; a single frame never yields a
high-confidence fit.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..config import Config
from ..types import BodyState, Frame


class BodyTracker:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("mediapipe is not installed") from exc
        raise ImportError(
            "MediaPipe/SMPL-X body adapter is a guarded integration point "
            "(Phase 5); using stub body tracker"
        )

    def track(self, frame: Frame, history: deque | None = None) -> BodyState:  # pragma: no cover
        raise NotImplementedError("real body tracker is not implemented")


__all__ = ["BodyTracker"]
