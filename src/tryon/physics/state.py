"""Physics state construction and constraint assembly (SPEC §10.1).

IMPL: src/tryon/physics/state.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MaterialParams, PhysicsState, SimReadyMesh


@dataclass
class MeshConstraints:
    masses: np.ndarray          # (N,) kg
    inv_mass: np.ndarray        # (N,)
    vertex_area: np.ndarray     # (N,) m²
    edge_i: np.ndarray          # (E,)
    edge_j: np.ndarray
    edge_rest: np.ndarray
    edge_alpha: np.ndarray
    bend_i: np.ndarray          # (B,)
    bend_j: np.ndarray
    bend_rest: np.ndarray
    bend_alpha: np.ndarray
    support_idx: np.ndarray
    support_anchor: np.ndarray  # (S,3) m, world
    rest_positions: np.ndarray  # (N,3) m
    n_constraints: int

    def all_compliances(self) -> np.ndarray:
        return np.concatenate([self.edge_alpha, self.bend_alpha])


def unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = faces[:, [0, 1, 2]].reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    out = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(out, faces[:, 0], area / 3.0)
    np.add.at(out, faces[:, 1], area / 3.0)
    np.add.at(out, faces[:, 2], area / 3.0)
    return np.maximum(out, 1.0e-9)


def support_vertices(vertices: np.ndarray, band: float = 0.02) -> np.ndarray:
    y = vertices[:, 1]
    return np.where(y >= float(y.max()) - band)[0]


def bending_pairs(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One bending spring per interior edge (opposite-vertex pair)."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(fi)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for (u, v), fids in edge_faces.items():
        if len(fids) != 2:
            continue
        f0, f1 = faces[fids[0]], faces[fids[1]]
        o0 = next((int(x) for x in f0 if int(x) not in (u, v)), None)
        o1 = next((int(x) for x in f1 if int(x) not in (u, v)), None)
        if o0 is None or o1 is None or o0 == o1:
            continue
        key = (o0, o1) if o0 < o1 else (o1, o0)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def build_constraints(
    mesh: SimReadyMesh,
    material: MaterialParams,
) -> MeshConstraints:
    V = mesh.vertices
    F = mesh.faces
    areas = vertex_areas(V, F)
    masses = np.maximum(material.areal_density * areas, 1.0e-10)
    inv_mass = 1.0 / masses

    ei, ej = unique_edges(F)
    rest = np.linalg.norm(V[ei] - V[ej], axis=1)
    rest = np.maximum(rest, 1.0e-6)
    # Stretch stiffness: E is the SPEC "stretch-stiffness proxy" in N/m.
    k_edge = np.maximum(material.youngs_modulus, 1.0e-3)
    alpha_edge = np.full(len(ei), 1.0 / k_edge, dtype=np.float64)

    bi, bj = bending_pairs(V, F)
    if len(bi):
        brest = np.linalg.norm(V[bi] - V[bj], axis=1)
        brest = np.maximum(brest, 1.0e-6)
        kb = np.maximum(material.bending_stiffness / (brest * brest), 1.0e-4)
        alpha_bend = 1.0 / kb
    else:
        brest = np.zeros(0)
        alpha_bend = np.zeros(0)

    sup = support_vertices(V)
    anchors = V[sup].copy()
    return MeshConstraints(
        masses=masses,
        inv_mass=inv_mass,
        vertex_area=areas,
        edge_i=ei,
        edge_j=ej,
        edge_rest=rest,
        edge_alpha=alpha_edge,
        bend_i=bi,
        bend_j=bj,
        bend_rest=brest,
        bend_alpha=alpha_bend,
        support_idx=sup,
        support_anchor=anchors,
        rest_positions=V.copy(),
        n_constraints=int(len(ei) + len(bi)),
    )


def initial_state(mesh: SimReadyMesh, n_constraints: int = 0) -> PhysicsState:
    x = mesh.vertices.copy()
    v = np.zeros_like(x)
    return PhysicsState(
        positions=x,
        velocities=v,
        attached={},
        lam=np.zeros(max(n_constraints, 1), dtype=np.float64),
        step=0,
        energy=0.0,
    )


def elastic_energy(
    x: np.ndarray,
    c: MeshConstraints,
) -> float:
    """Internal strain energy U (gravity is external work, not material energy)."""
    e = 0.0
    if len(c.edge_i):
        d = np.linalg.norm(x[c.edge_i] - x[c.edge_j], axis=1) - c.edge_rest
        k = 1.0 / np.maximum(c.edge_alpha, 1e-12)
        e += float(np.sum(0.5 * k * d * d))
    if len(c.bend_i):
        d = np.linalg.norm(x[c.bend_i] - x[c.bend_j], axis=1) - c.bend_rest
        k = 1.0 / np.maximum(c.bend_alpha, 1e-12)
        e += float(np.sum(0.5 * k * d * d))
    return float(max(e, 0.0))


def kinetic_energy(v: np.ndarray, c: MeshConstraints) -> float:
    return float(0.5 * np.sum(c.masses[:, None] * v * v))


def total_energy(x: np.ndarray, v: np.ndarray, c: MeshConstraints) -> float:
    return elastic_energy(x, c) + kinetic_energy(v, c)


def penetration_count(
    x: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    tol: float = 1.0e-6,
) -> int:
    if centers.size == 0:
        return 0
    q = (x[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    d2 = np.sum(q * q, axis=2)
    return int(np.sum(np.any(d2 < 1.0 - tol, axis=1)))


__all__ = [
    "MeshConstraints",
    "build_constraints",
    "initial_state",
    "unique_edges",
    "vertex_areas",
    "support_vertices",
    "bending_pairs",
    "elastic_energy",
    "kinetic_energy",
    "total_energy",
    "penetration_count",
]
