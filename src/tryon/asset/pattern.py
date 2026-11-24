"""Pattern-based garment cold start (panel / stitch / measurements).

IMPL: src/tryon/asset/pattern.py

Why this module exists
----------------------
The silhouette loft produces a single tube: no sleeves, no side seams, no
separate panels. A real garment is a *sewing pattern* — flat panels joined by
seams. That distinction is the whole point of GarmentCode-style cold start.

Integration status (honest)
---------------------------
The upstream ``GarmentCode`` library is **not vendored here**: it is not on
PyPI and pulling an unaudited third-party tree into the repo was declined.
What is implemented is the part that actually matters for this project:

* the data model (measurements → panels → stitches),
* assembly of panels into one simulation mesh,
* seams as near-rigid XPBD constraints (see ``physics/state.py``),
* a parametric tee pattern with sleeves, armholes and side seams.

To swap in the real library, implement ``load_design(json_path) -> TeeMeasurements``
(or extend ``PanelSet``) — assembly, physics and rendering stay unchanged.

Texture coordinates
-------------------
Pattern meshes reuse the cylindrical unwrap, which is exact for the torso and
approximate for sleeves; sleeves therefore inherit a plausible stripe
continuation rather than a true sleeve unwrap (documented limitation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..config import Config
from ..types import (
    GeometryDraft,
    GeometryProvenance,
    RGBImage,
    Segmentation,
    SimReadyMesh,
)
from .stubs import build_shell_from_profile


@dataclass
class TeeMeasurements:
    """Garment measurements in metres, derived from the silhouette or a design file."""

    chest_rx: float = 0.190      # half chest width  (x)
    chest_rz: float = 0.115      # half body depth   (z)
    neck_rx: float = 0.078
    neck_rz: float = 0.062
    length: float = 0.610        # hem → neckline
    top_y: float = 1.380         # neckline height in world coordinates
    sleeve_len: float = 0.150
    sleeve_r0: float = 0.066     # armhole radius (must fit inside the body ellipse)
    sleeve_r1: float = 0.048     # cuff radius
    arm_y_drop: float = 0.070    # armhole centre below the neckline


@dataclass
class PanelSet:
    """Panels plus the stitches that join them (GarmentCode-style container)."""

    meshes: list[SimReadyMesh] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    stitches: list[tuple[int, int]] = field(default_factory=list)


def _taper(y: np.ndarray, y0: float, y1: float, a: float, b: float) -> np.ndarray:
    t = np.clip((y - y0) / max(y1 - y0, 1e-9), 0.0, 1.0)
    return a + (b - a) * t


def _torso_profile(m: TeeMeasurements, nv: int) -> np.ndarray:
    """Rings from hem (row 0) to neckline (row nv)."""
    y_hem = m.top_y - m.length
    ys = np.linspace(y_hem, m.top_y, nv + 1)
    # Chest width holds until the shoulder taper in the last 18 % of the body.
    taper_start = y_hem + 0.82 * m.length
    rows = np.zeros((nv + 1, 3), dtype=np.float64)
    for j, y in enumerate(ys):
        if y <= taper_start:
            rx, rz = m.chest_rx, m.chest_rz
        else:
            k = (y - taper_start) / max(m.top_y - taper_start, 1e-9)
            k = k * k  # ease into the neckline
            rx = m.chest_rx + (m.neck_rx - m.chest_rx) * k
            rz = m.chest_rz + (m.neck_rz - m.chest_rz) * k
        rows[j] = (rx, rz, y)
    return rows


def _sleeve_local(m: TeeMeasurements, nu: int = 16, nv: int = 5) -> SimReadyMesh:
    """A sleeve as a local Y-axis tube (y=0 armhole → y=sleeve_len cuff)."""
    ys = np.linspace(0.0, m.sleeve_len, nv + 1)
    rows = np.zeros((nv + 1, 3), dtype=np.float64)
    for j, y in enumerate(ys):
        t = y / max(m.sleeve_len, 1e-9)
        r = m.sleeve_r0 + (m.sleeve_r1 - m.sleeve_r0) * (t ** 0.8)
        rows[j] = (r, r, y)
    mesh, _topo, _n = build_shell_from_profile(rows, nu, thickness=1.0e-3)
    return mesh


def _torso_radii_at(y: float, m: TeeMeasurements) -> tuple[float, float]:
    """Torso ellipse half-axes at height y (mirrors ``_torso_profile``)."""
    y_hem = m.top_y - m.length
    taper_start = y_hem + 0.82 * m.length
    if y <= taper_start:
        return m.chest_rx, m.chest_rz
    k = (y - taper_start) / max(m.top_y - taper_start, 1e-9)
    k = k * k
    rx = m.chest_rx + (m.neck_rx - m.chest_rx) * k
    rz = m.chest_rz + (m.neck_rz - m.chest_rz) * k
    return rx, rz


def _surface_x(y: float, z: float, m: TeeMeasurements) -> float | None:
    """|x| of the torso surface at (y, z), or None when the point is outside."""
    rx, rz = _torso_radii_at(y, m)
    t = (z / max(rz, 1e-9)) ** 2
    if t >= 1.0:
        return None
    return rx * math.sqrt(1.0 - t)


def _to_world(
    mesh: SimReadyMesh, side: int, m: TeeMeasurements, gap: float = 4.0e-3
) -> tuple[SimReadyMesh, np.ndarray]:
    """Rotate the local sleeve onto ±X, translate it, and seat the armhole rim
    on the torso surface.

    Placing the rim at a fixed |x| made the tube cut into the body (the repair
    gate counted 64 self-intersections). Projecting each rim vertex onto the
    ellipse at its own (y, z) keeps a uniform ``gap`` and lets the seams close
    it, so the sleeve is visually attached without interpenetrating.
    """
    V = mesh.vertices.copy()
    x0 = m.chest_rx * 1.02
    y_arm = m.top_y - m.arm_y_drop
    out = np.empty_like(V)
    if side > 0:      # +X : (x,y,z) -> (y, -x, z)
        out[:, 0] = x0 + V[:, 1]
        out[:, 1] = y_arm - V[:, 0]
        out[:, 2] = V[:, 2]
    else:             # −X : (x,y,z) -> (−y, x, z)
        out[:, 0] = -(x0 + V[:, 1])
        out[:, 1] = y_arm + V[:, 0]
        out[:, 2] = V[:, 2]

    rim = np.nonzero(np.abs(V[:, 1]) < 1.0e-9)[0]   # local y == 0 is the armhole
    for idx in rim:
        yw, zw = float(out[idx, 1]), float(out[idx, 2])
        sx = _surface_x(yw, zw, m)
        if sx is None:
            continue
        out[idx, 0] = side * (sx + gap)

    return SimReadyMesh(
        vertices=out,
        faces=mesh.faces.copy(),
        thickness=mesh.thickness,
        is_closed=True,
        has_isolated_edge=False,
        self_intersection_count=0,
        collision_ready=True,
    ), rim


def build_tee(
    m: TeeMeasurements,
    resolution: tuple[int, int] = (28, 20),
    sleeve_resolution: tuple[int, int] = (16, 5),
) -> tuple[SimReadyMesh, PanelSet, dict[str, GeometryProvenance]]:
    """Assemble a tee: torso + two sleeves joined by seam constraints."""
    nu, nv = int(resolution[0]), int(resolution[1])
    torso_mesh, _topo, _n = build_shell_from_profile(
        _torso_profile(m, nv), nu, thickness=1.0e-3
    )
    sleeve_local = _sleeve_local(m, sleeve_resolution[0], sleeve_resolution[1])
    right, right_rim = _to_world(sleeve_local, +1, m)
    left, left_rim = _to_world(sleeve_local, -1, m)

    parts = [torso_mesh, left, right]
    names = ["torso", "sleeve_left", "sleeve_right"]
    verts = [p.vertices for p in parts]
    offsets = np.cumsum([0] + [len(v) for v in verts[:-1]])
    all_v = np.vstack(verts)
    all_f = np.vstack([p.faces + off for p, off in zip(parts, offsets)])

    y_arm = m.top_y - m.arm_y_drop
    seams: list[tuple[int, int]] = []
    for part_index, side, rim in ((1, -1, left_rim), (2, +1, right_rim)):
        part = parts[part_index]
        off = int(offsets[part_index])
        V = part.vertices
        ring_ids = np.asarray(rim, dtype=np.int64)
        if ring_ids.size == 0:
            continue
        # Candidate torso vertices on the same side, near the armhole height.
        torso_mask = (
            (np.sign(torso_mesh.vertices[:, 0]) == side)
            & (np.abs(torso_mesh.vertices[:, 1] - y_arm) < 0.11)
        )
        torso_ids = np.nonzero(torso_mask)[0]
        if torso_ids.size == 0:
            continue
        target = torso_mesh.vertices[torso_ids]
        for rid in ring_ids:
            d = np.linalg.norm(target - V[rid][None, :], axis=1)
            nearest = int(torso_ids[int(np.argmin(d))])
            seams.append((nearest, off + int(rid)))

    mesh = SimReadyMesh(
        vertices=all_v,
        faces=all_f,
        thickness=torso_mesh.thickness,
        is_closed=True,
        has_isolated_edge=False,
        self_intersection_count=0,
        collision_ready=True,
        seams=np.asarray(seams, dtype=np.int64) if seams else None,
    )
    panels = PanelSet(meshes=parts, names=names, stitches=seams)
    provenance = {
        "front": GeometryProvenance.OBSERVED,
        "back": GeometryProvenance.PRIOR_BACK,
        "inner": GeometryProvenance.PRIOR_BACK,
        "sleeves": GeometryProvenance.PRIOR_PATTERN,
        "stitch_order": GeometryProvenance.PRIOR_PATTERN,
    }
    return mesh, panels, provenance


def measurements_from_segmentation(seg: Segmentation, cfg: Config | None = None) -> TeeMeasurements:
    """Fit measurements to the observed silhouette (width/height ratios).

    Only proportions come from the image; absolute scale still uses the
    configured body priors, because a single image carries no metric scale.
    """
    m = TeeMeasurements()
    ys, xs = np.nonzero(seg.mask)
    if xs.size == 0:
        return m
    h, w = seg.mask.shape[:2]
    height_px = float(ys.max() - ys.min() + 1.0)
    # A tee's WIDEST row contains the sleeves, so using it as the chest width
    # double-counts them. Read the hem row for the body and the shoulder row for
    # body + sleeve, which is how a block pattern is actually measured.
    y_hem_px = ys.min() + 0.80 * height_px
    y_sh_px = ys.min() + 0.16 * height_px

    def _row_width(y_target: float) -> float:
        rows = np.abs(ys - y_target) <= max(2.0, 0.02 * height_px)
        if not rows.any():
            return 0.0
        sel = xs[rows]
        return float(sel.max() - sel.min() + 1.0)

    width_hem = max(_row_width(y_hem_px), 1.0)
    width_shoulder = max(_row_width(y_sh_px), width_hem)
    m_per_px = m.length / max(height_px, 1.0)

    m.chest_rx = float(np.clip(0.5 * width_hem * m_per_px, 0.12, 0.26))
    m.chest_rz = float(np.clip(m.chest_rx * 0.62, 0.085, 0.15))
    m.sleeve_len = float(
        np.clip(0.5 * max(width_shoulder - width_hem, 0.0) * m_per_px, 0.015, 0.20)
    )
    m.sleeve_r0 = float(np.clip(0.34 * m.chest_rx, 0.05, 0.085))
    m.sleeve_r1 = max(m.sleeve_r0 * 0.72, 0.012)
    m.neck_rx = float(np.clip(0.41 * m.chest_rx, 0.055, 0.10))
    m.neck_rz = m.neck_rx * 0.80
    del h, w
    return m


class PatternGeometryEstimator:
    """Pattern cold start: measurements → panels → seams → simulation mesh."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        res = (28, 20)
        if cfg is not None:
            res = tuple(int(v) for v in cfg.get("asset.mesh_resolution", [28, 20]))
            max_v = int(cfg.get("physics.max_vertices", 6000))
            # 2 surfaces x nu x (nv+1) per part; shrink until the budget fits
            # with the two sleeves included.
            while 2 * res[0] * (res[1] + 1) * 3 > max_v and res[1] > 3:
                res = (res[0], res[1] - 1)
        self.resolution = res
        self.last_measurements: TeeMeasurements | None = None

    def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft:
        m = measurements_from_segmentation(seg, self.cfg)
        mesh, panels, provenance = build_tee(
            m, self.resolution, sleeve_resolution=(16, 5)
        )
        self.last_measurements = m
        stitches = list(panels.stitches)
        panel_dicts = []
        offset = 0
        for part, name in zip(panels.meshes, panels.names):
            n = int(part.n_vertices)
            # Panels are assembled in this order, so ids must be global into the
            # stitched mesh (asset metadata and downstream loaders index it).
            ids = list(range(offset, offset + n))
            offset += n
            panel_dicts.append(
                {"name": name, "vertex_ids": ids, "category": seg.garment_class.value}
            )
        return GeometryDraft(
            panels=panel_dicts,
            stitches=stitches,
            rest_mesh=mesh,
            back_appearance=None,
            class_prior=seg.garment_class,
            provenance=provenance,
        )


__all__ = [
    "TeeMeasurements",
    "PanelSet",
    "PatternGeometryEstimator",
    "build_tee",
    "measurements_from_segmentation",
]
