"""Pipeline A orchestration: garment image → GarmentAsset v0 (SPEC §5).

IMPL: src/tryon/asset/asset_builder.py::AssetBuilder
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np

from ..config import Config
from ..types import (
    BodyState,
    Confidence,
    DrapeReport,
    GarmentAsset,
    GarmentClass,
    GeometryProvenance,
    MaterialParams,
    RGBImage,
    Segmentation,
    SimReadyMesh,
    TryonError,
    overall_confidence,
)
from .drape import DrapeAligner
from .geometry import GeometryEstimator
from .material_prior import MaterialPrior, material_prior_level
from .mesh_repair import MeshRepairer
from .registry import AssetRegistry, asset_id_for
from .segment import Segmenter
from .uv import cylindrical_uvs


@dataclass
class AppearanceMesh:
    """Visual mesh. D13: kept strictly separate from the physics shell."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    source_image: np.ndarray
    shell_mapping: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)
        self.shell_mapping = np.asarray(self.shell_mapping, dtype=np.int64)
        if len(self.shell_mapping) != len(self.vertices):
            raise ValueError("appearance mesh must map 1:1 to shell vertices")


@dataclass
class CollisionBody:
    """Ellipsoid collision proxy in the world frame (metres)."""

    centers: np.ndarray   # (K,3)
    radii: np.ndarray     # (K,3)
    rest_centers: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(-1, 3)
        self.radii = np.asarray(self.radii, dtype=np.float64).reshape(-1, 3)
        self.rest_centers = np.asarray(self.rest_centers, dtype=np.float64).reshape(-1, 3)
        if self.centers.shape != self.rest_centers.shape:
            raise ValueError("rest_centers must match centers")

    @classmethod
    def from_mesh(cls, mesh: SimReadyMesh) -> "CollisionBody":
        v = mesh.vertices
        cx = float(v[:, 0].mean())
        cz = float(v[:, 2].mean())
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        rx = max(0.05, float(np.percentile(np.abs(v[:, 0] - cx), 95)) - 0.02)
        rz = max(0.05, float(np.percentile(np.abs(v[:, 2] - cz), 95)) - 0.02)
        centers = np.array([[cx, 0.5 * (y0 + y1), cz]], dtype=np.float64)
        radii = np.array([[rx, 0.5 * (y1 - y0), rz]], dtype=np.float64)
        return cls(centers=centers, radii=radii, rest_centers=centers.copy())

    def update_from_body(self, body: BodyState | None) -> None:
        if body is None:
            self.centers = self.rest_centers.copy()
            return
        transl = np.asarray(body.smpl_params.get("transl", np.zeros(3)), dtype=np.float64).reshape(3)
        # Translation is small in mock; keep feet fixed and shift the torso.
        self.centers = self.rest_centers + transl[None, :]


class AssetBuilder:
    """Cold-start pipeline A. No step here may enter the realtime loop."""

    def __init__(
        self,
        cfg: Config,
        segmenter: Segmenter,
        geometry: GeometryEstimator,
        repairer: MeshRepairer,
        drape: DrapeAligner,
        material_prior: MaterialPrior,
        registry: AssetRegistry | None = None,
    ) -> None:
        self.cfg = cfg
        self.segmenter = segmenter
        self.geometry = geometry
        self.repairer = repairer
        self.drape = drape
        self.material_prior = material_prior
        self.registry = registry
        self.last_segmentation: Segmentation | None = None
        self.last_drape: DrapeReport | None = None
        self.last_g1_pass: bool = False

    def _check_timeout(self, t0: float, timeout_s: float, stage: str) -> None:
        elapsed = time.monotonic() - t0
        if elapsed > float(timeout_s):
            raise TryonError(
                "E_ASSET_TIMEOUT",
                f"pipeline A exceeded {timeout_s}s at stage {stage}",
                context={"elapsed_s": elapsed, "stage": stage},
            )

    def build(self, image: RGBImage, *, timeout_s: float = 300.0) -> GarmentAsset:
        t0 = time.monotonic()
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("AssetBuilder expects an (H,W,3) RGB image")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        seg = self.segmenter.run(img)
        self._check_timeout(t0, timeout_s, "segmentation")
        self.last_segmentation = seg

        draft = self.geometry.estimate(img, seg)
        self._check_timeout(t0, timeout_s, "geometry")
        if len(draft.stitches) == 0:
            raise TryonError("E_ASSET_GEOM_FAIL", "geometry draft has no stitches")

        mesh = self.repairer.repair(draft)
        self._check_timeout(t0, timeout_s, "mesh_repair")

        report = self.drape.align(mesh, seg)
        self._check_timeout(t0, timeout_s, "drape")
        self.last_drape = report

        material = self.material_prior.prior(img, seg)
        if not isinstance(material, MaterialParams):
            raise TypeError("MaterialPrior.prior must return MaterialParams")
        self._check_timeout(t0, timeout_s, "material_prior")
        from ..types import MaterialState

        if material.state != MaterialState.AI_PRIOR:
            raise ValueError("material prior must be AI_PRIOR, never a claimed truth")

        provenance = dict(draft.provenance)
        if bool(self.cfg.get("calibration.ablation.disable_back_prior", False)):
            # Ablation: do not use the back-geometry prior (D14 guard).
            provenance["back"] = GeometryProvenance.PRIOR_PATTERN
        back_prov = provenance.get("back", GeometryProvenance.PRIOR_BACK)
        if back_prov == GeometryProvenance.OBSERVED:
            back_conf = 0.90
        elif back_prov == GeometryProvenance.PRIOR_PATTERN:
            back_conf = 0.20
        else:
            back_conf = 0.35
        confidence = Confidence(
            segmentation=float(seg.confidence),
            visible_geometry=float(np.clip(seg.confidence, 0.0, 1.0)),
            back_geometry=float(back_conf),
            topology=1.0 if mesh.is_closed and not mesh.has_isolated_edge else 0.4,
            material=float(material_prior_level(material)),
            overall=0.0,
        )
        weights = {k: float(v) for k, v in dict(self.cfg.require("asset.confidence_weights")).items()}
        confidence.overall = overall_confidence(confidence, weights)
        conf = confidence

        g1_iou = float(self.cfg.require("asset.g1_iou_threshold"))
        g1_pass = (
            report.silhouette_iou >= g1_iou
            and not mesh.has_isolated_edge
            and mesh.self_intersection_count == 0
            and report.flyaway_faces == 0
        )
        self.last_g1_pass = bool(g1_pass)
        review_threshold = float(self.cfg.require("asset.review_threshold"))
        needs_review = (conf.overall < review_threshold) or (not g1_pass)

        appearance = AppearanceMesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            uvs=cylindrical_uvs(mesh.vertices),
            source_image=img.copy(),
            shell_mapping=np.arange(mesh.n_vertices, dtype=np.int64),
        )
        collider = CollisionBody.from_mesh(mesh)

        asset = GarmentAsset(
            asset_id=asset_id_for(np.ascontiguousarray(img).tobytes()),
            version=0,
            garment_class=seg.garment_class,
            rest_mesh=mesh,
            appearance_mesh=appearance,
            sewing_pattern={
                "panels": draft.panels,
                "stitches": draft.stitches,
                "topology": {
                    "is_closed": mesh.is_closed,
                    "has_isolated_edge": mesh.has_isolated_edge,
                    "self_intersection_count": mesh.self_intersection_count,
                },
                "g1": {
                    "iou": report.silhouette_iou,
                    "flyaway_faces": report.flyaway_faces,
                    "pass": bool(g1_pass),
                },
                "g2": {
                    "silhouette_iou": report.silhouette_iou,
                    "chamfer_distance": report.chamfer_distance,
                    "normal_error": report.normal_error,
                },
            },
            material=material,
            collision_body=collider,
            confidence=conf,
            provenance=provenance,
            created_at=time.time(),
            needs_human_review=bool(needs_review),
        )
        if self.registry is not None:
            self.registry.register(asset)
        self._check_timeout(t0, timeout_s, "finalise")
        return asset


__all__ = ["AssetBuilder", "AppearanceMesh", "CollisionBody"]
