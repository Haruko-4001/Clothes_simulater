"""Visible-layer rendering (physics state, never generative)."""
from .mesh_render import PILMeshRenderer
from .gaussian import GaussianSplatRenderer
from .stubs import StubRenderer

__all__ = ["PILMeshRenderer", "GaussianSplatRenderer", "StubRenderer"]
