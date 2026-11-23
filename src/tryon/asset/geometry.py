"""Real geometry cold-start adapter (SewFormer / GarmentRec / InstantMesh).

IMPL: src/tryon/asset/geometry.py::GeometryEstimator

Only the protocol is required for mock mode. Real adapters are probed lazily;
any missing dependency or unimplemented integration raises ImportError so the
factory can fall back to StubGeometryEstimator (E_ADAPTER_FALLBACK).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..types import GeometryDraft, GarmentClass, RGBImage, Segmentation
from .stubs import ShellTopology, build_shell_from_profile


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


# ---------------------------------------------------------------------------
# Image-driven lofting (P3): segmentation rows drive the shell profile.
# ---------------------------------------------------------------------------

# Class priors (Appendix-B style constants, metres). These are pattern-level
# priors C_cls, not SKU registrations (SPEC §2.5).
_CLASS_PRIORS: dict[GarmentClass, dict[str, float]] = {
    GarmentClass.TOPS: {"max_half_width_m": 0.26, "length_lo_m": 0.35, "length_hi_m": 0.85},
    GarmentClass.BOTTOMS: {"max_half_width_m": 0.23, "length_lo_m": 0.50, "length_hi_m": 1.10},
    GarmentClass.ONE_PIECES: {"max_half_width_m": 0.26, "length_lo_m": 0.60, "length_hi_m": 1.20},
    GarmentClass.ACCESSORY: {"max_half_width_m": 0.15, "length_lo_m": 0.20, "length_hi_m": 0.40},
    GarmentClass.UNKNOWN: {"max_half_width_m": 0.26, "length_lo_m": 0.35, "length_hi_m": 1.20},
}
_NECK_RING_Y_M = 1.47   # shoulder line (world Y-up, matches stub scene)
_DEPTH_RATIO = 0.65      # frontal width → depth prior for a flat garment tube


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def silhouette_profile(
    seg: Segmentation,
    nv: int,
) -> tuple[np.ndarray, float]:
    """Per-row (rx, rz, y) rings from the segmentation mask.

    Returns ``(profile, length_m)``. The mask's row span maps to the garment
    length; each row's foreground half-width maps to ``rx``; ``rz`` follows a
    constant depth prior so the single view is never turned into a depth
    claim (SPEC §2.2 proposition 1).
    """
    mask = np.asarray(seg.mask) > 0
    if mask.ndim != 2 or not mask.any():
        raise ValueError("silhouette_profile needs a non-empty mask")
    rows = np.where(mask.any(axis=1))[0]
    y_top_px, y_bot_px = int(rows.min()), int(rows.max())
    span = max(1, y_bot_px - y_top_px)

    prior = _CLASS_PRIORS.get(seg.garment_class, _CLASS_PRIORS[GarmentClass.UNKNOWN])
    x0, _, x1, _ = seg.bbox
    bw = max(1, x1 - x0)
    aspect = (y_bot_px - y_top_px + 1) / bw
    width_m = prior["max_half_width_m"] * 2.0
    length_m = float(np.clip(width_m * aspect, prior["length_lo_m"], prior["length_hi_m"]))

    # Sample nv+1 rows uniformly over the mask span and measure half-widths.
    sample_rows = np.linspace(y_top_px, y_bot_px, nv + 1)
    half_px = np.zeros(nv + 1, dtype=np.float64)
    for k, ry in enumerate(sample_rows):
        r0, r1 = int(math.floor(ry)), int(math.ceil(ry)) + 1
        seg_rows = mask[r0:r1]
        xs = np.where(seg_rows.any(axis=0))[0]
        if xs.size:
            half_px[k] = 0.5 * (xs.max() - xs.min() + 1)
        # empty rows keep 0 and are repaired by interpolation below
    good = half_px > 1e-6
    if not good.any():
        half_px[:] = bw * 0.5
    else:
        idx = np.arange(nv + 1)
        half_px[~good] = np.interp(idx[~good], idx[good], half_px[good])

    half_px = _smooth(half_px)
    max_half = float(half_px.max())
    scale_m_per_px = prior["max_half_width_m"] / max(max_half, 1e-9)
    rx = np.maximum(half_px * scale_m_per_px, 0.02)
    rx = _smooth(rx)
    rz = np.maximum(rx * _DEPTH_RATIO, 0.012)
    y = _NECK_RING_Y_M - np.linspace(0.0, length_m, nv + 1)

    profile = np.stack([rx, rz, y], axis=1)
    return profile, length_m


class SilhouetteGeometryEstimator:
    """Loft the physics shell from the garment's own silhouette rows (P3).

    Unlike ``StubGeometryEstimator`` the ring profile varies with the input
    image: a fitted tank top and a loose dress produce different meshes. The
    visible side is still only *frontal*; back/inner remain explicit priors.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        res = (28, 20)
        if cfg is not None:
            res = tuple(int(v) for v in cfg.get("asset.mesh_resolution", [28, 20]))
            max_v = int(cfg.get("physics.max_vertices", 6000))
            while 2 * res[0] * (res[1] + 1) > max_v and res[1] > 3:
                res = (res[0], res[1] - 1)
        self.resolution = res
        self.last_profile: np.ndarray | None = None
        self.last_length_m: float | None = None

    def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft:
        nu, nv = self.resolution
        profile, length_m = silhouette_profile(seg, nv)
        mesh, topo, _normals = build_shell_from_profile(profile, nu, thickness=1.0e-3)
        self.last_profile = profile
        self.last_length_m = length_m

        front_ids = [
            topo.outer(j, i)
            for j in range(nv + 1)
            for i in range(nu)
            if math.sin(2 * math.pi * i / nu) >= 0
        ]
        back_ids = [
            topo.outer(j, i)
            for j in range(nv + 1)
            for i in range(nu)
            if math.sin(2 * math.pi * i / nu) < 0
        ]
        inner_ids = [topo.inner(j, i) for j in range(nv + 1) for i in range(nu)]
        stitches: list[tuple[int, int]] = []
        for j in range(nv):
            stitches.append((topo.outer(j, nu // 2), topo.outer(j + 1, nu // 2)))
            stitches.append((topo.outer(j, 0), topo.outer(j + 1, 0)))
        from ..types import GeometryProvenance

        provenance = {
            "front": GeometryProvenance.OBSERVED,      # driven by the mask rows
            "back": GeometryProvenance.PRIOR_BACK,     # single image: prior only
            "inner": GeometryProvenance.PRIOR_BACK,
            "stitch_order": GeometryProvenance.PRIOR_PATTERN,
        }
        panels = [
            {"name": "front", "vertex_ids": front_ids, "category": seg.garment_class.value},
            {"name": "back", "vertex_ids": back_ids, "category": seg.garment_class.value},
            {"name": "lining", "vertex_ids": inner_ids, "category": "lining"},
        ]
        return GeometryDraft(
            panels=panels,
            stitches=stitches,
            rest_mesh=mesh,
            back_appearance=None,
            class_prior=seg.garment_class,
            provenance=provenance,
        )


__all__ = [
    "GeometryEstimator",
    "InstantMeshBackPrior",
    "SilhouetteGeometryEstimator",
    "silhouette_profile",
]
