"""Pipeline A: garment image → GarmentAsset (offline, non-realtime)."""
from .asset_builder import AppearanceMesh, AssetBuilder, CollisionBody
from .registry import AssetRegistry
from .segment import Segmenter
from .geometry import GeometryEstimator
from .mesh_repair import MeshRepairer
from .drape import DrapeAligner
from .material_prior import MaterialPrior
from .stubs import StubSegmenter, StubGeometryEstimator

__all__ = [
    "AppearanceMesh",
    "AssetBuilder",
    "CollisionBody",
    "AssetRegistry",
    "Segmenter",
    "GeometryEstimator",
    "MeshRepairer",
    "DrapeAligner",
    "MaterialPrior",
    "StubSegmenter",
    "StubGeometryEstimator",
]
