"""Per-face garment texture sampling (UV → source image colour).

IMPL: src/tryon/render/texture.py::garment_face_colors

The renderers used a single mean foreground colour for the whole mesh, which
threw away every pattern, stripe and print in the garment photo. This module
samples the source image through the appearance mesh's UVs instead, so a
striped shirt renders as stripes and a printed tee keeps its print.

Sampling is per FACE (4 taps: three vertices + centroid) rather than per pixel:
a per-triangle affine warp would cost a Python call per face (thousands per
frame) and break the 30 fps budget, while per-face sampling is one vectorised
lookup and already carries most of the visual signal at this mesh density.
"""
from __future__ import annotations

import numpy as np

from ..types import GarmentAsset

_CACHE: dict[tuple, np.ndarray] = {}


def _foreground_mean(arr: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    fg = arr[dist > 18.0]
    if fg.size == 0:
        return (78, 115, 158)
    mean = fg.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def garment_face_colors(asset: GarmentAsset) -> np.ndarray:
    """Return (M,3) float colours for the physics shell faces.

    Falls back to the mean foreground colour when the asset has no UVs or no
    source image, so stubs and older assets keep rendering.
    """
    faces = np.asarray(asset.rest_mesh.faces, dtype=np.int64)
    appearance = getattr(asset, "appearance_mesh", None)
    src = getattr(appearance, "source_image", None)
    uvs = getattr(appearance, "uvs", None)
    if src is None:
        return np.full((len(faces), 3), 78.0, dtype=np.float64)

    arr = np.asarray(src)
    key = (asset.asset_id, id(arr), len(faces), 0 if uvs is None else len(uvs))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if uvs is None or len(uvs) != asset.rest_mesh.n_vertices:
        mean = np.asarray(_foreground_mean(arr.astype(np.float64)), dtype=np.float64)
        out = np.repeat(mean[None, :], len(faces), axis=0)
        _CACHE[key] = out
        return out

    h, w = arr.shape[:2]
    uv = np.asarray(uvs, dtype=np.float64)
    tri = uv[faces]                       # (M,3,2)
    centroid = tri.mean(axis=1)           # (M,2)
    # Four taps per face, averaged: corners + centroid.
    taps = np.concatenate([tri, centroid[:, None, :]], axis=1)  # (M,4,2)
    xs = np.clip((taps[:, :, 0] * (w - 1)).round().astype(np.int64), 0, w - 1)
    ys = np.clip((taps[:, :, 1] * (h - 1)).round().astype(np.int64), 0, h - 1)
    samples = arr[ys, xs].astype(np.float64)                    # (M,4,3)
    out = samples.mean(axis=1)

    # Pixels that fell on the image background are not garment colour: replace
    # those taps with the foreground mean before averaging.
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg = np.median(border.astype(np.float64), axis=0)
    mean = np.asarray(_foreground_mean(arr.astype(np.float64)), dtype=np.float64)
    dist = np.linalg.norm(samples - bg[None, None, :], axis=2)
    is_bg = dist < 18.0
    if is_bg.any():
        samples = np.where(is_bg[:, :, None], mean[None, None, :], samples)
        out = samples.mean(axis=1)
    _CACHE[key] = out
    return out


def clear_cache() -> None:
    _CACHE.clear()


__all__ = ["garment_face_colors", "clear_cache"]
