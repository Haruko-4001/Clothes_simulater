"""Real geometry cold-start adapter (SewFormer / GarmentRec / InstantMesh).

IMPL: src/tryon/asset/geometry.py::GeometryEstimator

Only the protocol is required for mock mode. Real adapters are probed lazily;
any missing dependency or unimplemented integration raises ImportError so the
factory can fall back to StubGeometryEstimator (E_ADAPTER_FALLBACK).
"""
from __future__ import annotations

from ..config import Config
from ..types import GeometryDraft, GarmentClass, RGBImage, Segmentation


class GeometryEstimator:
    """Panel/stitch estimator. Currently a guarded integration point."""

    REQUIRED_MODULE = "sewformer"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            __import__(self.REQUIRED_MODULE)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(f"{self.REQUIRED_MODULE} is not installed") from exc
        # Integration is intentionally not claimed: the real adapter is a
        # Phase-5 task (SPEC §15). Never fabricate a mesh here.
        raise ImportError(
            "SewFormer adapter is a guarded integration point (not implemented); "
            "falling back to stub geometry"
        )

    def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft:  # pragma: no cover
        raise NotImplementedError("real geometry adapter is not implemented")


class InstantMeshBackPrior:
    """Visual-only back prior (D13: never enters the physics layer)."""

    REQUIRED_MODULE = "instantmesh"

    def __init__(self) -> None:
        try:
            __import__(self.REQUIRED_MODULE)
        except Exception as exc:  # pragma: no cover
            raise ImportError("instantmesh is not installed") from exc
        raise ImportError("InstantMesh integration TODO (visual mesh only, D13)")


__all__ = ["GeometryEstimator", "InstantMeshBackPrior"]
