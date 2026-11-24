"""Cylindrical UV unwrap for the shell garment (single-image texturing).

IMPL: src/tryon/asset/uv.py::cylindrical_uvs

The physics shell is a tube lofted around the Y axis, so a cylindrical
projection is the natural parameterisation: ``u`` follows the angle around the
body and ``v`` follows height. The back half is folded onto the front half
(``φ = min(θ, 2π−θ)``), which wraps the single frontal photo around the whole
garment with the seam at the sides — the standard, honest treatment of a
single-view texture (SPEC §2.2: the back is a prior, not an observation).

This module only produces UVs; sampling and shading live in
``render/texture.py`` and the renderers.
"""
from __future__ import annotations

import numpy as np


def cylindrical_uvs(vertices: np.ndarray) -> np.ndarray:
    """Return per-vertex (u, v) in [0,1] for a Y-axis tube garment.

    ``u`` = 0 at the left silhouette and 1 at the right silhouette, mirrored
    across the back. ``v`` = 0 at the top (neck) and 1 at the bottom (hem), so
    it matches image row order directly.
    """
    V = np.asarray(vertices, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must be (N,3)")
    cx = float(V[:, 0].mean())
    cz = float(V[:, 2].mean())
    theta = np.arctan2(V[:, 2] - cz, V[:, 0] - cx)  # (-π, π]
    # Fold the back half onto the front: φ ∈ [0, π], 0 = +X side, π = −X side.
    phi = np.abs(theta)
    u = 1.0 - phi / np.pi

    y = V[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    span = max(y1 - y0, 1e-9)
    v = (y1 - y) / span
    return np.stack([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)], axis=1)


__all__ = ["cylindrical_uvs"]
