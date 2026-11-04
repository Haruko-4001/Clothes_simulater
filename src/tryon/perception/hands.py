"""Real hand tracker adapter (MediaPipe Hands default, SPEC §13).

IMPL: src/tryon/perception/hands.py::HandTracker
Missing dependency ⇒ ImportError ⇒ factory falls back to StubHandTracker.
"""
from __future__ import annotations

from ..config import Config
from ..types import Frame, HandState


class HandTracker:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import mediapipe  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("mediapipe is not installed") from exc
        raise ImportError(
            "MediaPipe Hands adapter is a guarded integration point (Phase 5); "
            "using stub hand tracker"
        )

    def track(self, frame: Frame) -> HandState:  # pragma: no cover
        raise NotImplementedError("real hand tracker is not implemented")


__all__ = ["HandTracker"]
