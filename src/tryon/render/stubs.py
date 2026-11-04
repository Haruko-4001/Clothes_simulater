"""Stub renderer: PIL mesh rasterisation of the physics state."""
from __future__ import annotations

from .mesh_render import PILMeshRenderer

StubRenderer = PILMeshRenderer
StubPILRenderer = PILMeshRenderer

__all__ = ["StubRenderer", "StubPILRenderer", "PILMeshRenderer"]
