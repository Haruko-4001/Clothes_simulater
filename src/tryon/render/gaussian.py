"""3D Gaussian splatting renderer adapter (optional; not required in mock).

IMPL: src/tryon/render/gaussian.py
D10: any-view splatting is offline/asynchronous only — never claim realtime
free-viewpoint rendering from a single camera frame.
"""
from __future__ import annotations

from ..config import Config
from ..types import Frame, GarmentAsset, PhysicsState, RenderOutput


class GaussianSplatRenderer:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import gsplat  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("gsplat is not installed") from exc
        raise ImportError(
            "3DGS renderer is an async/offline integration point (D10); "
            "falling back to stub renderer"
        )

    def render(
        self,
        frame: Frame,
        asset: GarmentAsset,
        state: PhysicsState,
        camera,
    ) -> RenderOutput:  # pragma: no cover
        raise NotImplementedError("3DGS renderer is not implemented")


__all__ = ["GaussianSplatRenderer"]
