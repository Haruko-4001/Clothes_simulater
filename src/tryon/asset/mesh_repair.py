"""Mesh repair / topology validation — G1 hard gate (SPEC §8 / §10.3).

IMPL: src/tryon/asset/mesh_repair.py::MeshRepairer

A draft that cannot be made closed, edge-manifold, non-self-intersecting and
thick is rejected with E_ASSET_GEOM_FAIL and must never enter physics.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from ..config import Config
from ..types import GeometryDraft, SimReadyMesh, TryonError


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges[key].append(fi)
    return edges


def _compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.reshape(-1)] = True
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return vertices[used], remap[faces]


def _fill_boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Fan-triangulate boundary loops around their centroid.

    Returns (vertices, faces, filled_loops). Non-manifold edges are not
    repairable here and are left for the caller to reject.
    """
    edges = _edge_map(faces)
    boundary = {k: v[0] for k, v in edges.items() if len(v) == 1}
    if not boundary:
        return vertices, faces, 0
    # Build directed boundary half-edges in face winding order.
    directed: list[tuple[int, int]] = []
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in boundary and boundary[key] == fi:
                directed.append((u, v))
    out_map: dict[int, list[int]] = defaultdict(list)
    for u, v in directed:
        out_map[u].append(v)

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for u0, v0 in directed:
        if (u0, v0) in visited_edges:
            continue
        loop = [u0, v0]
        visited_edges.add((u0, v0))
        prev, cur = u0, v0
        for _ in range(len(directed) + 2):
            if cur == u0:
                loop.pop()
                break
            nxts = [w for w in out_map.get(cur, []) if (cur, w) not in visited_edges]
            if not nxts:
                break
            nxt = nxts[0]
            visited_edges.add((cur, nxt))
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3 and loop[0] == u0:
            loops.append(loop)

    if not loops:
        return vertices, faces, 0
    verts = [v for v in vertices]
    new_faces = [tuple(int(x) for x in f) for f in faces]
    for loop in loops:
        centroid = np.mean(vertices[loop], axis=0)
        cid = len(verts)
        verts.append(centroid)
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            # Reverse boundary orientation to produce an outward cap.
            new_faces.append((b, a, cid))
    return np.asarray(verts, dtype=np.float64), np.asarray(new_faces, dtype=np.int64), len(loops)


# ---------------------------------------------------------------------------
# Self-intersection (AABB broad phase + exact tri-tri narrow phase)
# ---------------------------------------------------------------------------


def _tri_tri_intersect(v: np.ndarray, u: np.ndarray) -> bool:
    """Conservative triangle-triangle intersection test (non-coplanar path)."""
    n1 = np.cross(v[1] - v[0], v[2] - v[0])
    n2 = np.cross(u[1] - u[0], u[2] - u[0])
    if np.linalg.norm(n1) < 1e-14 or np.linalg.norm(n2) < 1e-14:
        return False
    d1 = (u - v[0]) @ n1
    d2 = (v - u[0]) @ n2
    eps = 1e-12
    if (np.all(d1 > eps) or np.all(d1 < -eps)) and (np.all(d2 > eps) or np.all(d2 < -eps)):
        return False
    if np.all(np.abs(d1) <= eps) and np.all(np.abs(d2) <= eps):
        # Coplanar: 2D triangle overlap via separating axis of edge normals.
        axis = n1 / np.linalg.norm(n1)
        basis_x = np.array([1.0, 0.0, 0.0])
        if abs(axis @ basis_x) > 0.9:
            basis_x = np.array([0.0, 1.0, 0.0])
        ex = basis_x - axis * (axis @ basis_x)
        ex /= np.linalg.norm(ex)
        ey = np.cross(axis, ex)
        p2 = np.stack([v @ ex, v @ ey], axis=1)
        q2 = np.stack([u @ ex, u @ ey], axis=1)
        return _tri2d_overlap(p2, q2)
    # Segment where each triangle meets the other's plane; test containment.
    for tri, other, d in ((v, u, d1), (u, v, d2)):
        pts: list[np.ndarray] = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            da, db = (a - other[0]) @ (np.cross(other[1] - other[0], other[2] - other[0])), (
                b - other[0]
            ) @ (np.cross(other[1] - other[0], other[2] - other[0]))
            if abs(da - db) < 1e-15:
                continue
            if da * db <= 0.0:
                t = da / (da - db)
                pts.append(a + t * (b - a))
        if len(pts) >= 2:
            # Any point inside the other triangle ⇒ intersection.
            for p in pts:
                if _point_in_tri(p, other):
                    return True
    return False


def _point_in_tri(p: np.ndarray, tri: np.ndarray) -> bool:
    v0, v1, v2 = tri
    a = np.cross(v1 - v0, p - v0)
    b = np.cross(v2 - v1, p - v1)
    c = np.cross(v0 - v2, p - v2)
    return bool((a @ b >= -1e-12) and (b @ c >= -1e-12) and (c @ a >= -1e-12))


def _tri2d_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    for tri in (p, q):
        for i in range(3):
            e = tri[(i + 1) % 3] - tri[i]
            axis = np.array([-e[1], e[0]])
            if np.linalg.norm(axis) < 1e-14:
                continue
            pp = p @ axis
            qq = q @ axis
            if pp.max() < qq.min() - 1e-12 or qq.max() < pp.min() - 1e-12:
                return False
    return True


def self_intersection_count(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_candidates: int = 400_000,
) -> int:
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.shape[0] < 2:
        return 0
    tris = V[F]
    lo = tris.min(axis=1)
    hi = tris.max(axis=1)
    # Spatial hash accelerated broad phase.
    extent = np.maximum(hi.max(axis=0) - lo.min(axis=0), 1e-6)
    cell = float(np.max(extent) / 48.0)
    keys_per_face: list[list[tuple[int, int, int]]] = []
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for fi in range(F.shape[0]):
        k0 = np.floor((lo[fi] - lo.min(axis=0)) / cell).astype(int)
        k1 = np.floor((hi[fi] - lo.min(axis=0)) / cell).astype(int)
        cells = [
            (int(x), int(y), int(z))
            for x in range(k0[0], k1[0] + 1)
            for y in range(k0[1], k1[1] + 1)
            for z in range(k0[2], k1[2] + 1)
        ]
        keys_per_face.append(cells)
        for c in cells:
            buckets[c].append(fi)
    pairs: set[tuple[int, int]] = set()
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        ids = sorted(set(ids))
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                pairs.add((ids[ai], ids[bi]))
                if len(pairs) > max_candidates:
                    break
            if len(pairs) > max_candidates:
                break
        if len(pairs) > max_candidates:
            break
    count = 0
    for a, b in pairs:
        fa, fb = set(F[a].tolist()), set(F[b].tolist())
        if fa & fb:                       # adjacent faces share an edge/vertex
            continue
        # AABB reject.
        if np.any(hi[a] < lo[b] - 1e-12) or np.any(hi[b] < lo[a] - 1e-12):
            continue
        if _tri_tri_intersect(tris[a], tris[b]):
            count += 1
            if count > 64:
                return count
    return count


# ---------------------------------------------------------------------------
# MeshRepairer
# ---------------------------------------------------------------------------


class MeshRepairer:
    """Topology repair + G1 assertions. Pure geometry, safe in mock/real mode."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self._cache: dict[str, tuple[int, dict]] = {}

    def repair(self, draft: GeometryDraft) -> SimReadyMesh:
        raw = draft.rest_mesh
        if isinstance(raw, SimReadyMesh):
            V = raw.vertices.copy()
            F = raw.faces.copy()
            thickness = float(raw.thickness)
        elif raw is not None and hasattr(raw, "vertices") and hasattr(raw, "faces"):
            V = np.asarray(raw.vertices, dtype=np.float64)
            F = np.asarray(raw.faces, dtype=np.int64)
            thickness = float(getattr(raw, "thickness", 1.0e-3))
        else:
            raise TryonError("E_ASSET_GEOM_FAIL", "geometry draft has no rest mesh")

        if not np.all(np.isfinite(V)):
            raise TryonError("E_ASSET_GEOM_FAIL", "non-finite rest vertices")
        if F.ndim != 2 or F.shape[1] != 3 or F.size == 0:
            raise TryonError("E_ASSET_GEOM_FAIL", "rest mesh has no valid triangles")
        if thickness <= 0.0:
            # A zero-thickness sheet cannot be a closed physics shell.
            raise TryonError("E_ASSET_GEOM_FAIL", "thickness must be > 0")

        # Drop degenerate triangles.
        nondeg = (
            (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        )
        F = F[nondeg]
        if F.size == 0:
            raise TryonError("E_ASSET_GEOM_FAIL", "all faces degenerate")

        edges = _edge_map(F)
        nonmanifold = [k for k, v in edges.items() if len(v) > 2]
        if nonmanifold:
            raise TryonError(
                "E_ASSET_GEOM_FAIL",
                f"non-manifold edges: {len(nonmanifold)}",
                context={"nonmanifold_edges": len(nonmanifold)},
            )
        boundary = [k for k, v in edges.items() if len(v) == 1]
        filled = 0
        if boundary:
            V, F, filled = _fill_boundary_loops(V, F)
            if filled == 0:
                raise TryonError("E_ASSET_GEOM_FAIL", "open mesh cannot be closed")

        V, F = _compact(V, F)

        key = hashlib.sha1(
            np.ascontiguousarray(V).tobytes() + np.ascontiguousarray(F).tobytes()
        ).hexdigest()
        if key in self._cache:
            si_count, topo = self._cache[key]
        else:
            si_count = self_intersection_count(V, F)
            topo = _topology_summary(V, F)
            self._cache[key] = (si_count, topo)
        if si_count > 0:
            raise TryonError(
                "E_ASSET_GEOM_FAIL",
                f"self intersections: {si_count}",
                context={"self_intersection_count": si_count},
            )
        if not topo["is_closed"]:
            raise TryonError("E_ASSET_GEOM_FAIL", "repaired mesh is not closed")

        return SimReadyMesh(
            vertices=V,
            faces=F,
            thickness=thickness,
            is_closed=True,
            has_isolated_edge=False,
            self_intersection_count=0,
            collision_ready=True,
            # Seams are pattern data, not topology: repair only drops faces and
            # must not silently disconnect sleeves from the body.
            seams=getattr(draft.rest_mesh, "seams", None),
        )


def _topology_summary(V: np.ndarray, F: np.ndarray) -> dict:
    edges = _edge_map(F)
    boundary = sum(1 for v in edges.values() if len(v) == 1)
    nonmanifold = sum(1 for v in edges.values() if len(v) > 2)
    used = np.zeros(len(V), dtype=bool)
    used[F.reshape(-1)] = True
    isolated = int(np.sum(~used))
    return {
        "is_closed": boundary == 0 and nonmanifold == 0,
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "isolated_vertices": isolated,
        "has_isolated_edge": boundary > 0 or nonmanifold > 0 or isolated > 0,
    }


__all__ = ["MeshRepairer", "self_intersection_count", "_topology_summary"]
