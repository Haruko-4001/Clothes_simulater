"""Stub keyframe backend aliases."""
from __future__ import annotations

from .tryon import KeyframeService, StubKeyframeTryOn

StubKeyframeBackend = StubKeyframeTryOn

__all__ = ["StubKeyframeTryOn", "StubKeyframeBackend", "KeyframeService"]
