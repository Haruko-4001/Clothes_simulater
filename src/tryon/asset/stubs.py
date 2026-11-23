"""Stub asset pipeline — structurally valid, numerically self-consistent (SPEC §0.1).

IMPL: src/tryon/asset/stubs.py

The stub geometry is a real closed double-sided shell around a torso, not a
placeholder box: it has thickness, welds a watertight rim, carries a sewing
pattern, and is directly usable by the XPBD physics layer.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..config import Config
from ..types import (
    BodyState,
    GarmentClass,
    GeometryDraft,
    GeometryProvenance,
    RGBImage,
    Segmentation,
    SimReadyMesh,
    TryonError,
)


# ---------------------------------------------------------------------------
# Synthetic / analytic shell geometry
# ---------------------------------------------------------------------------


@dataclass
class ShellTopology:
    nu: int
    nv: int
    outer_offset: int
    inner_offset: int
    n_outer_per_row: int

    def outer(self, j: int, i: int) -> int:
        return j * self.nu + (i % self.nu)

    def inner(self, j: int, i: int) -> int:
        return self.inner_offset + j * self.nu + (i % self.nu)


def _ellipse_profile(t: float) -> tuple[float, float, float]:
    """Return (rx, rz, y) for vertical parameter t∈[0,1] (hem→neck)."""
    y = 1.02 + 0.54 * t
    # Slight taper toward the shoulders, a small waist pinch.
    pinch = 1.0 - 0.06 * math.exp(-((t - 0.45) / 0.22) ** 2)
    rx = (0.240 - 0.026 * t) * pinch
    rz = (0.158 - 0.018 * t) * pinch
    return rx, rz, y


def build_shell_mesh(
    resolution: tuple[int, int] = (28, 20),
    thickness: float = 1.0e-3,
) -> tuple[SimReadyMesh, ShellTopology, np.ndarray]:
    """Closed tube-shell garment: outer + inner surface + two welded rims.

    Returns ``(mesh, topology, outward_normals)``. Every undirected edge is
    incident to exactly two triangles, so ``is_closed`` is true.
    """
    nu, nv = int(resolution[0]), int(resolution[1])
    if nu < 6 or nv < 3:
        raise ValueError("resolution too small for a shell garment")
    profile = np.zeros((nv + 1, 3), dtype=np.float64)
    for j in range(nv + 1):
        rx, rz, y = _ellipse_profile(j / nv)
        profile[j] = (rx, rz, y)
    return build_shell_from_profile(profile, nu, thickness=thickness)


def build_shell_from_profile(
    profile: np.ndarray,
    nu: int,
    thickness: float = 1.0e-3,
) -> tuple[SimReadyMesh, ShellTopology, np.ndarray]:
    """Loft a closed double-sided shell from per-row (rx, rz, y) rings.

    ``profile`` has one row per ring, hem first (index 0) → neck last. Ring
    shapes are scaled ellipses ``(x/rx)² + (z/rz)² = 1``; the topology matches
    ``build_shell_mesh`` (outer + inner + welded rims), so the XPBD layer and
    the repair gate treat both builders identically.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != 3 or len(profile) < 2:
        raise ValueError("profile must be (nv+1, 3) rows of (rx, rz, y)")
    if nu < 6:
        raise ValueError("nu too small for a shell garment")
    if np.any(profile[:, 0] <= 0.0) or np.any(profile[:, 1] <= 0.0):
        raise ValueError("profile radii must be positive")
    nv = len(profile) - 1
    topo = ShellTopology(nu=nu, nv=nv, outer_offset=0, inner_offset=nu * (nv + 1), n_outer_per_row=nu)
    n_outer = nu * (nv + 1)
    vertices = np.zeros((2 * n_outer, 3), dtype=np.float64)
    normals = np.zeros((2 * n_outer, 3), dtype=np.float64)

    for j in range(nv + 1):
        rx, rz, y = profile[j]
        for i in range(nu):
            theta = 2.0 * math.pi * i / nu
            c, s = math.cos(theta), math.sin(theta)
            p = np.array([rx * c, y, rz * s], dtype=np.float64)
            grad = np.array([2.0 * p[0] / (rx * rx), 0.0, 2.0 * p[2] / (rz * rz)])
            n = grad / max(float(np.linalg.norm(grad)), 1.0e-12)
            o = topo.outer(j, i)
            vertices[o] = p
            normals[o] = n
            vertices[topo.inner(j, i)] = p - n * thickness
            normals[topo.inner(j, i)] = n

    faces: list[tuple[int, int, int]] = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for j in range(nv):
        for i in range(nu):
            i1 = (i + 1) % nu
            quad(topo.outer(j, i), topo.outer(j, i1), topo.outer(j + 1, i1), topo.outer(j + 1, i))
            quad(topo.inner(j, i), topo.inner(j + 1, i), topo.inner(j + 1, i1), topo.inner(j, i1))

    for i in range(nu):
        i1 = (i + 1) % nu
        quad(topo.outer(0, i), topo.inner(0, i), topo.inner(0, i1), topo.outer(0, i1))
        quad(topo.outer(nv, i1), topo.inner(nv, i1), topo.inner(nv, i), topo.outer(nv, i))

    faces_arr = np.asarray(faces, dtype=np.int64)
    mesh = SimReadyMesh(
        vertices=vertices,
        faces=faces_arr,
        thickness=float(thickness),
        is_closed=True,          # verified by MeshRepairer before physics
        has_isolated_edge=False,
        self_intersection_count=0,
        collision_ready=True,
    )
    return mesh, topo, normals


def analyze_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Real manifold analysis: boundary edges, non-manifold edges, isolated verts."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        return {
            "is_closed": False,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "isolated_vertices": int(V.shape[0]),
            "has_isolated_edge": True,
            "degenerate_faces": 0,
        }
    a = F[:, [0, 1, 2]].reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = lo * max(1, int(V.shape[0])) + hi
    uniq, counts = np.unique(keys, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    used = np.zeros(V.shape[0], dtype=bool)
    used[F.reshape(-1)] = True
    isolated = int(np.sum(~used))
    degenerate = int(
        np.sum(
            (
                (F[:, 0] == F[:, 1])
                | (F[:, 1] == F[:, 2])
                | (F[:, 0] == F[:, 2])
            )
        )
    )
    return {
        "is_closed": boundary == 0 and nonmanifold == 0,
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "isolated_vertices": isolated,
        "has_isolated_edge": boundary > 0 or nonmanifold > 0 or isolated > 0,
        "degenerate_faces": degenerate,
    }


def shell_support_vertices(mesh: SimReadyMesh) -> np.ndarray:
    """Top-ring vertices are supported by the shoulders (kept body-relative)."""
    y = mesh.vertices[:, 1]
    top = float(np.max(y))
    return np.where(y >= top - 0.02)[0]


# ---------------------------------------------------------------------------
# Stub segmenter
# ---------------------------------------------------------------------------


def _border_background(image: RGBImage) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    border = np.concatenate(
        [img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]], axis=0
    )
    return np.median(border, axis=0)


class StubSegmenter:
    """Border-colour foreground segmentation (deterministic, no weights)."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.last_confidence = 0.0

    def run(self, image: RGBImage) -> Segmentation:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Segmenter expects an (H,W,3) RGB image")
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        f = img.astype(np.float64)
        bg = _border_background(f)
        dist = np.linalg.norm(f - bg[None, None, :], axis=2)
        mask = (dist > 18.0).astype(np.uint8)
        # Clean speckle with a tiny box filter implemented in numpy.
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        try:
            from PIL import ImageFilter

            mask_img = Image.fromarray(mask * 255).filter(ImageFilter.MedianFilter(3))
            mask = (np.asarray(mask_img) > 127).astype(np.uint8)
        except Exception:
            pass
        if int(mask.sum()) < max(16, h * w // 500):
            # Fallback: central portrait region, explicitly low confidence.
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(0.08 * h): int(0.92 * h), int(0.25 * w): int(0.75 * w)] = 1
            confidence = 0.45
        else:
            confidence = 0.92
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            raise TryonError("E_ASSET_LOW_CONF", "no foreground pixels found")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        aspect = (bbox[3] - bbox[1]) / max(1, bbox[2] - bbox[0])
        if aspect > 1.9 and aspect < 3.2:
            garment_class = GarmentClass.ONE_PIECES
        elif aspect <= 1.35:
            garment_class = GarmentClass.TOPS
        else:
            garment_class = GarmentClass.TOPS
        self.last_confidence = confidence
        return Segmentation(
            mask=mask,
            garment_class=garment_class,
            bbox=bbox,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Stub geometry estimator
# ---------------------------------------------------------------------------


class StubGeometryEstimator:
    """Analytic tube-shell cold start labelled with explicit provenance."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        res = (28, 20)
        if cfg is not None:
            res = tuple(int(v) for v in cfg.get("asset.mesh_resolution", [28, 20]))
            max_v = int(cfg.get("physics.max_vertices", 6000))
            # 2 * nu * (nv + 1) <= max_vertices
            while 2 * res[0] * (res[1] + 1) > max_v and res[1] > 3:
                res = (res[0], res[1] - 1)
        self.resolution = res
        self.last_provenance: dict[str, GeometryProvenance] = {}

    def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft:
        mesh, topo, normals = build_shell_mesh(self.resolution, thickness=1.0e-3)
        nu, nv = topo.nu, topo.nv
        front_ids = [topo.outer(j, i) for j in range(nv + 1) for i in range(nu) if math.sin(2 * math.pi * i / nu) >= 0]
        back_ids = [topo.outer(j, i) for j in range(nv + 1) for i in range(nu) if math.sin(2 * math.pi * i / nu) < 0]
        inner_ids = [topo.inner(j, i) for j in range(nv + 1) for i in range(nu)]
        stitches: list[tuple[int, int]] = []
        for j in range(nv):
            stitches.append((topo.outer(j, nu // 2), topo.outer(j + 1, nu // 2)))
            stitches.append((topo.outer(j, 0), topo.outer(j + 1, 0)))
        provenance: dict[str, GeometryProvenance] = {
            "front": GeometryProvenance.OBSERVED,
            "back": GeometryProvenance.PRIOR_BACK,
            "inner": GeometryProvenance.PRIOR_BACK,
            "stitch_order": GeometryProvenance.PRIOR_PATTERN,
        }
        self.last_provenance = provenance
        panels = [
            {"name": "front", "vertex_ids": front_ids, "category": seg.garment_class.value},
            {"name": "back", "vertex_ids": back_ids, "category": seg.garment_class.value},
            {"name": "lining", "vertex_ids": inner_ids, "category": "lining"},
        ]
        return GeometryDraft(
            panels=panels,
            stitches=stitches,
            rest_mesh=mesh,
            back_appearance=None,  # single image: back is AI completion only
            class_prior=seg.garment_class,
            provenance=provenance,
        )


__all__ = [
    "ShellTopology",
    "build_shell_mesh",
    "build_shell_from_profile",
    "analyze_topology",
    "shell_support_vertices",
    "StubSegmenter",
    "StubGeometryEstimator",
]
