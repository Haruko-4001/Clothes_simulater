"""G2 drape check: silhouette IoU / Chamfer / normal error on the VISIBLE side.

IMPL: src/tryon/asset/drape.py::DrapeAligner
G2 never substitutes for dynamic verification (SPEC §7 / §11).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from ..types import DrapeReport, Segmentation, SimReadyMesh


def _face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(a, b)
    norm = np.linalg.norm(n, axis=1)
    norm[norm < 1e-14] = 1.0
    return n / norm[:, None]


def _boundary_points(mask: np.ndarray, max_points: int = 400) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]
    interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]
    interior[:, :-1] &= m[:, 1:]
    edge = m & ~interior
    ys, xs = np.where(edge)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    total = 0.0
    # Chunked pairwise distances keep memory bounded.
    chunk = 256
    for i in range(0, len(a), chunk):
        d = np.linalg.norm(a[i : i + chunk, None, :] - b[None, :, :], axis=2)
        total += float(d.min(axis=1).sum())
    for i in range(0, len(b), chunk):
        d = np.linalg.norm(b[i : i + chunk, None, :] - a[None, :, :], axis=2)
        total += float(d.min(axis=1).sum())
    return total / (len(a) + len(b))


class DrapeAligner:
    """Orthographic front-view silhouette comparison against the segmentation."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg

    def align(self, mesh: SimReadyMesh, seg: Segmentation) -> DrapeReport:
        V = mesh.vertices
        F = mesh.faces
        mask = (np.asarray(seg.mask) > 0).astype(np.uint8)
        h, w = mask.shape
        x0, y0, x1, y1 = seg.bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)

        # Fit the mesh's front projection into the segmented bounding box.
        xmin, xmax = float(V[:, 0].min()), float(V[:, 0].max())
        ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
        sx = bw / max(xmax - xmin, 1e-9)
        sy = bh / max(ymax - ymin, 1e-9)
        px = x0 + (V[:, 0] - xmin) * sx
        py = y1 - (V[:, 1] - ymin) * sy
        proj = np.stack([px, py], axis=1)

        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for tri in F:
            pts = [tuple(proj[i]) for i in tri]
            draw.polygon(pts, fill=255)
        sil = np.asarray(canvas) > 127

        inter = np.logical_and(sil, mask > 0).sum()
        union = np.logical_or(sil, mask > 0).sum()
        iou = float(inter / union) if union else 0.0

        bp_sil = _boundary_points(sil.astype(np.uint8))
        bp_seg = _boundary_points(mask)
        cd = _chamfer(bp_sil, bp_seg)
        if not math.isfinite(cd):
            cd = float(max(w, h))

        # Normal smoothness of adjacent faces (visible side proxy).
        n = _face_normals(V, F)
        edge_faces: dict[tuple[int, int], list[int]] = {}
        for fi, tri in enumerate(F):
            for u, v in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
                key = (u, v) if u < v else (v, u)
                edge_faces.setdefault(key, []).append(fi)
        angles = []
        for fs in edge_faces.values():
            if len(fs) == 2:
                c = float(np.clip(n[fs[0]] @ n[fs[1]], -1.0, 1.0))
                angles.append(math.acos(c))
        normal_error = float(np.mean(angles)) if angles else 0.0

        # Faces projected outside the segmentation bbox (expanded 10%).
        pad_x, pad_y = 0.10 * bw, 0.10 * bh
        cx = proj[F][:, :, 0].mean(axis=1)
        cy = proj[F][:, :, 1].mean(axis=1)
        outside = (
            (cx < x0 - pad_x)
            | (cx > x1 + pad_x)
            | (cy < y0 - pad_y)
            | (cy > y1 + pad_y)
        )
        flyaway = int(outside.sum())

        return DrapeReport(
            silhouette_iou=iou,
            chamfer_distance=float(cd),
            normal_error=normal_error,
            flyaway_faces=flyaway,
        )


__all__ = ["DrapeAligner", "_boundary_points", "_chamfer"]
