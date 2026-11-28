"""Newton VBD physics backend (optional, experimental).

IMPL: src/tryon/physics/newton_solver.py::NewtonVBDSolver

Uses NVIDIA Newton's Vertex Block Descent solver on a cloth triangle mesh.
This is an **optional** backend: importing Newton is guarded, and the factory
falls back to the internal XPBD solver when it is missing or fails to build
(SPEC §8 ``E_ADAPTER_FALLBACK``). The default path remains the internal XPBD.

Mapping from the project model:
  * ``SimReadyMesh`` vertices/faces -> Newton cloth mesh
  * the project's closed double-sided shell is collapsed onto one layer first;
    Newton treats the two layers as independent sheets and lets the inner one
    fall, which produces false stretching (verified on the pattern asset)
  * pattern seams -> extra Newton springs (near-rigid distance constraints)
  * body collider ellipsoids -> static Newton bodies with ellipsoid shapes
  * shoulder supports -> zero-mass particles fixed before ``finalize()``
  * hand contacts -> position corrections before/after a step

Known differences from the XPBD backend (honest):
  * Newton does not expose our XPBD multipliers; ``PhysicsState.lam`` is zeros.
  * The energy invariant is reported, not enforced; energy gates still run.
  * Building the model and the first step trigger Warp kernel compilation.

Enable with:
    pip install "newton>=1.8"
    physics.solver: newton_vbd   (or --physics-solver newton_vbd)
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..config import Config
from ..types import (
    ContactEvent,
    GarmentAsset,
    PhysicsState,
    SimReadyMesh,
    TryonError,
)
from .constraints import process_contact_events
from .state import MeshConstraints, build_constraints, elastic_energy, kinetic_energy

try:  # optional dependency
    import warp as wp
    import newton

    _NEWTON_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only when installed
    _NEWTON_AVAILABLE = False
    _NEWTON_IMPORT_ERROR = exc


def _collapse_shell(
    vertices: np.ndarray, faces: np.ndarray, thickness: float, seams: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a closed double-sided shell onto one cloth layer.

    Returns ``(vertices, faces, reduce_map, seams)`` where ``reduce_map[i]`` is
    the reduced index of original vertex ``i``. Vertices closer than
    ``2 * thickness`` are merged; duplicate faces (the inner layer mapped onto
    the outer one) and degenerate faces are removed.
    """
    n = len(vertices)
    tol = max(float(thickness) * 3.0, 3.0e-4)
    cells: dict[tuple[int, int, int], list[int]] = {}
    reduce_map = np.full(n, -1, dtype=np.int64)
    representatives: list[int] = []
    for i, p in enumerate(vertices):
        key = tuple(int(v) for v in np.floor(p / tol))
        found = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for cand in cells.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if np.linalg.norm(vertices[cand] - p) <= tol:
                            found = int(reduce_map[cand])
                            break
                    if found >= 0:
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found < 0:
            found = len(representatives)
            representatives.append(i)
            cells.setdefault(key, []).append(i)
        reduce_map[i] = found

    kept: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for f in np.asarray(faces, dtype=np.int64):
        rf = tuple(int(reduce_map[v]) for v in f)
        if len(set(rf)) < 3:
            continue
        key = tuple(sorted(rf))
        if key in seen:
            continue
        seen.add(key)
        kept.append(rf)  # outer faces are emitted first in build_shell_from_profile
    if len(kept) < 10:
        return (
            np.asarray(vertices, dtype=np.float64).copy(),
            np.asarray(faces, dtype=np.int64).copy(),
            np.arange(n, dtype=np.int64),
            np.asarray(seams, dtype=np.int64).reshape(-1, 2) if seams is not None else np.zeros((0, 2), dtype=np.int64),
        )
    reduced = np.asarray(vertices, dtype=np.float64)[representatives]
    reduced_faces = np.asarray(kept, dtype=np.int64)
    reduced_seams = np.zeros((0, 2), dtype=np.int64)
    if seams is not None and len(seams):
        pairs = []
        seen_pairs = set()
        for i, j in np.asarray(seams, dtype=np.int64):
            a, b = int(reduce_map[i]), int(reduce_map[j])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(key)
        reduced_seams = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return reduced, reduced_faces, reduce_map, reduced_seams


def _collapse_by_panels(
    vertices: np.ndarray,
    faces: np.ndarray,
    panels: list[dict],
    thickness: float,
    seams: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse each sewing panel independently, then combine.

    A global collapse can merge armhole-rim vertices across the torso/sleeve
    boundary (they are only ~4 mm apart), which creates a bad bridge. The
    pattern metadata already gives exact panel membership, so collapse within
    each panel and keep the seams as springs.
    """
    labels = np.full(len(faces), -1, dtype=np.int64)
    for index, panel in enumerate(panels):
        ids = np.asarray(panel.get("vertex_ids", []), dtype=np.int64)
        if ids.size == 0:
            continue
        mask = np.isin(faces, ids).all(axis=1) & (labels < 0)
        labels[mask] = index
    if np.any(labels < 0):
        return _collapse_shell(vertices, faces, thickness, seams)

    reduce_map = np.full(len(vertices), -1, dtype=np.int64)
    out_vertices: list[np.ndarray] = []
    out_faces: list[np.ndarray] = []
    offset = 0
    for index, panel in enumerate(panels):
        ids = np.asarray(panel.get("vertex_ids", []), dtype=np.int64)
        if ids.size == 0:
            continue
        face_mask = labels == index
        local = np.full(len(vertices), -1, dtype=np.int64)
        local[ids] = np.arange(len(ids))
        sub_faces = local[faces[face_mask]] if np.any(face_mask) else np.zeros((0, 3), dtype=np.int64)
        reduced_v, reduced_f, panel_map, _ = _collapse_shell(
            vertices[ids], sub_faces, thickness, None
        )
        reduce_map[ids] = offset + panel_map
        out_vertices.append(reduced_v)
        if len(reduced_f):
            out_faces.append(reduced_f + offset)
        offset += len(reduced_v)

    missing = np.nonzero(reduce_map < 0)[0]
    for vid in missing:
        reduce_map[vid] = offset
        out_vertices.append(vertices[vid : vid + 1])
        offset += 1

    reduced_vertices = np.vstack(out_vertices) if out_vertices else vertices.copy()
    reduced_faces = (
        np.vstack(out_faces) if out_faces else np.zeros((0, 3), dtype=np.int64)
    )
    reduced_seams = np.zeros((0, 2), dtype=np.int64)
    if seams is not None and len(seams):
        pairs = []
        seen_pairs = set()
        for i, j in np.asarray(seams, dtype=np.int64):
            a, b = int(reduce_map[i]), int(reduce_map[j])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(key)
        reduced_seams = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return reduced_vertices, reduced_faces, reduce_map, reduced_seams


class NewtonVBDSolver:
    """Newton SolverVBD behind the project's PhysicsSolver protocol."""

    def __init__(self, cfg: Config) -> None:
        if not _NEWTON_AVAILABLE:
            raise ImportError(
                "newton is not installed; run `pip install newton>=1.8` "
                "or use physics.solver: stub_xpbd"
            )
        self.cfg = cfg
        self._constraints: MeshConstraints | None = None
        self._constraint_key: tuple | None = None
        self._targets: dict[int, np.ndarray] = {}
        self._target_step: dict[int, int] = {}
        self._last_stable: PhysicsState | None = None
        self._asset_id: str | None = None
        self._model = None
        self._solver = None
        self._state0 = None
        self._state1 = None
        self._control = None
        self._reduce_map = np.zeros(0, dtype=np.int64)
        self._n_reduced = 0
        self._support_reduced = np.zeros(0, dtype=np.int64)
        self._support_anchor_reduced = np.zeros((0, 3), dtype=np.float64)
        self.asset: GarmentAsset | None = None
        self.last_penetration_count = 0
        self.last_step_ms = 0.0
        self.n_physics_steps = 0

    # ------------------------------------------------------------------
    def _ensure_constraints(self, asset: GarmentAsset) -> MeshConstraints:
        material = asset.material
        key = (
            asset.asset_id,
            asset.rest_mesh.n_vertices,
            asset.rest_mesh.n_faces,
            material.to_vector().tobytes(),
            float(asset.rest_mesh.thickness),
        )
        if self._constraints is not None and self._constraint_key == key:
            return self._constraints
        c = build_constraints(asset.rest_mesh, material)
        self._constraints = c
        self._constraint_key = key
        return c

    def _value(self, key: str, default: float) -> float:
        value = self.cfg.get(key)
        return float(value) if value is not None else float(default)

    # ------------------------------------------------------------------
    def _build_model(self, asset: GarmentAsset) -> None:
        mesh = asset.rest_mesh
        if not isinstance(mesh, SimReadyMesh):
            raise TypeError(
                "D13 violated: only a SimReadyMesh physics shell may enter the solver"
            )
        material = asset.material
        c = self._ensure_constraints(asset)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        seams = getattr(mesh, "seams", None)
        if bool(self.cfg.get("physics.newton_collapse_shell", True)):
            panels = asset.sewing_pattern.get("panels") if asset.sewing_pattern else None
            if panels:
                vertices, faces, reduce_map, seams = _collapse_by_panels(
                    vertices, faces, panels, float(mesh.thickness), seams
                )
            else:
                vertices, faces, reduce_map, seams = _collapse_shell(
                    vertices, faces, float(mesh.thickness), seams
                )
        else:
            reduce_map = np.arange(len(vertices), dtype=np.int64)
        self._reduce_map = np.asarray(reduce_map, dtype=np.int64)
        self._n_reduced = len(vertices)

        builder = newton.ModelBuilder()
        gravity = self.cfg.require("math.gravity")
        builder.gravity = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(x), float(y), float(z)) for x, y, z in vertices],
            indices=faces.reshape(-1).tolist(),
            density=float(material.areal_density),
            tri_ke=self._value("physics.newton_tri_ke", material.youngs_modulus),
            tri_ka=self._value("physics.newton_tri_ka", material.youngs_modulus),
            tri_kd=self._value("physics.newton_tri_kd", material.damping),
            edge_ke=self._value(
                "physics.newton_edge_ke", max(material.bending_stiffness * 1.0e4, 1.0)
            ),
            edge_kd=self._value("physics.newton_edge_kd", 0.0),
            particle_radius=self._value(
                "physics.newton_particle_radius", max(mesh.thickness * 0.5, 1.0e-3)
            ),
        )
        if seams is not None and len(seams):
            seam_ke = self._value("physics.newton_seam_ke", 1.0e3)
            seam_kd = self._value("physics.newton_seam_kd", 0.0)
            if seam_ke > 0.0:
                for i, j in np.asarray(seams, dtype=np.int64):
                    builder.add_spring(int(i), int(j), ke=seam_ke, kd=seam_kd, control=0.0)

        # Shoulder support: zero mass must be applied to the builder BEFORE
        # finalize; editing model.particle_inv_mass afterwards does not affect
        # the solver's cached inverse masses.
        support_ids = np.asarray(c.support_idx, dtype=np.int64)
        support_reduced = self._reduce_map[support_ids]
        support = np.unique(support_reduced)
        self._support_reduced = support.astype(np.int64)
        self._support_anchor_reduced = np.zeros((len(support), 3), dtype=np.float64)
        for r, rid in enumerate(support):
            self._support_anchor_reduced[r] = c.support_anchor[support_reduced == rid].mean(axis=0)
        for rid in support:
            builder.particle_mass[int(rid)] = 0.0

        centers = np.asarray(
            getattr(asset.collision_body, "centers", np.zeros((0, 3))), dtype=np.float64
        ).reshape(-1, 3)
        radii = np.asarray(
            getattr(asset.collision_body, "radii", np.zeros((0, 3))), dtype=np.float64
        ).reshape(-1, 3)
        for center, radius in zip(centers, radii):
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                    q=wp.quat_identity(),
                ),
                mass=0.0,
            )
            builder.add_shape_ellipsoid(
                body,
                rx=float(radius[0]),
                ry=float(radius[1]),
                rz=float(radius[2]),
            )
        # VBD needs color groups for both particles and rigid bodies, so this
        # must run after every body/shape has been added.
        builder.color(include_bending=True)
        device = self.cfg.get("physics.newton_device") or None
        model = builder.finalize(device=device)
        model.soft_contact_ke = self._value("physics.newton_contact_ke", 1.0e2)
        model.soft_contact_kd = self._value("physics.newton_contact_kd", 1.0e1)
        model.soft_contact_mu = float(material.friction)

        iterations = int(self.cfg.get("physics.newton_iterations") or 2)
        self_contact = bool(self.cfg.get("physics.self_collision", False))
        contact_kwargs = {}
        if self_contact:
            margin = float(self.cfg.get("physics.self_collision_radius_m", 0.004))
            contact_kwargs = {
                "particle_self_contact_margin": margin,
                "particle_self_contact_gap": margin * 0.5,
            }
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=self._value("physics.newton_contact_gap", 0.01),
        )
        solver = newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            particle_enable_self_contact=self_contact,
            collision_pipeline=collision_pipeline,
            # DAT (Divide and Truncate) is the penetration-free rigid-soft
            # mechanism discussed in the engine evaluation; keep it opt-out.
            rigid_soft_enable_dat=bool(self.cfg.get("physics.newton_dat", True)),
            rigid_soft_dat_use_interval_arithmetic=bool(
                self.cfg.get("physics.newton_dat_interval", True)
            ),
            **contact_kwargs,
        )
        self._model = model
        self._solver = solver
        self._state0 = model.state()
        self._state1 = model.state()
        self._control = model.control()
        self._asset_id = asset.asset_id

    def _ensure_model(self, asset: GarmentAsset) -> None:
        if self._model is None or self._asset_id != asset.asset_id:
            self._build_model(asset)

    # ------------------------------------------------------------------
    def initial_state(self, asset: GarmentAsset) -> PhysicsState:
        c = self._ensure_constraints(asset)
        self._build_model(asset)
        positions = np.asarray(asset.rest_mesh.vertices, dtype=np.float64).copy()
        self._last_stable = PhysicsState(
            positions=positions.copy(),
            velocities=np.zeros_like(positions),
            attached={},
            lam=np.zeros(c.n_constraints, dtype=np.float64),
            step=0,
            energy=0.0,
        )
        self._targets.clear()
        self._target_step.clear()
        self.asset = asset
        return self._last_stable.copy()

    def reset(self, asset: GarmentAsset) -> PhysicsState:
        return self.initial_state(asset)

    # ------------------------------------------------------------------
    def _penetration_count(self, q: np.ndarray, asset: GarmentAsset) -> int:
        centers = np.asarray(
            getattr(asset.collision_body, "centers", np.zeros((0, 3))), dtype=np.float64
        ).reshape(-1, 3)
        radii = np.asarray(
            getattr(asset.collision_body, "radii", np.zeros((0, 3))), dtype=np.float64
        ).reshape(-1, 3)
        if centers.size == 0:
            return 0
        qq = (q[:, None, :] - centers[None, :, :]) / radii[None, :, :]
        d2 = np.sum(qq * qq, axis=2)
        return int(np.sum(np.any(d2 < 1.0 - 1.0e-6, axis=1)))

    def _group_average(self, values: np.ndarray) -> np.ndarray:
        counts = np.bincount(self._reduce_map, minlength=self._n_reduced).astype(np.float64)
        out = np.zeros((self._n_reduced, values.shape[1]), dtype=np.float64)
        np.add.at(out, self._reduce_map, values)
        return out / np.maximum(counts, 1.0)[:, None]

    def step(
        self,
        state: PhysicsState,
        asset: GarmentAsset,
        contacts: list[ContactEvent],
        dt: float,
    ) -> PhysicsState:
        t0 = time.perf_counter()
        if not isinstance(asset.rest_mesh, SimReadyMesh):
            raise TypeError(
                "D13 violated: only a SimReadyMesh physics shell may enter the solver"
            )
        if not (np.all(np.isfinite(state.positions)) and np.all(np.isfinite(state.velocities))):
            raise TryonError(
                "E_PHYS_DIVERGE",
                "input state is non-finite",
                fallback=self._last_stable.copy() if self._last_stable is not None else None,
            )
        self._ensure_model(asset)
        c = self._ensure_constraints(asset)
        self.asset = asset

        work = state.copy()
        process_contact_events(work, contacts, self._targets, self._target_step)
        hold_steps = int(
            math.ceil(float(self.cfg.require("framing.contact_hold_s")) / max(dt, 1e-9))
        )
        for vid in list(work.attached.keys()):
            last = self._target_step.get(vid, work.step)
            if work.step - last > hold_steps:
                work.attached.pop(vid, None)
                self._targets.pop(vid, None)
                self._target_step.pop(vid, None)

        q_red = self._group_average(np.asarray(work.positions, dtype=np.float64))
        v_red = self._group_average(np.asarray(work.velocities, dtype=np.float64))
        q_prev_red = q_red.copy()
        attach_relax = float(self.cfg.require("physics.attach_relax"))

        if len(self._support_reduced):
            q_red[self._support_reduced] = self._support_anchor_reduced
        attached_red: dict[int, np.ndarray] = {}
        for vid in work.attached:
            target = self._targets.get(vid)
            if target is None:
                continue
            rid = int(self._reduce_map[vid])
            q_red[rid] += attach_relax * (target - q_red[rid])
            attached_red[rid] = np.asarray(target, dtype=np.float64)

        self._state0.particle_q.assign(q_red.astype(np.float32))
        self._state0.particle_qd.assign(v_red.astype(np.float32))
        self._state0.clear_forces()
        self._solver.step(self._state0, self._state1, self._control, None, float(dt))
        self._state0, self._state1 = self._state1, self._state0
        q_red = self._state0.particle_q.numpy().astype(np.float64)
        v_red = self._state0.particle_qd.numpy().astype(np.float64)

        if len(self._support_reduced):
            q_red[self._support_reduced] = self._support_anchor_reduced
            v_red[self._support_reduced] = 0.0
        for rid, target in attached_red.items():
            q_red[rid] += attach_relax * (target - q_red[rid])
            v_red[rid] = (q_red[rid] - q_prev_red[rid]) / max(dt, 1.0e-9)

        q = q_red[self._reduce_map]
        v = v_red[self._reduce_map]
        energy = kinetic_energy(v, c) + elastic_energy(q, c)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v)) and math.isfinite(energy)):
            raise TryonError(
                "E_PHYS_DIVERGE",
                "Newton VBD produced a non-finite state",
                fallback=self._last_stable.copy() if self._last_stable is not None else None,
            )
        new_state = PhysicsState(
            positions=q,
            velocities=v,
            attached=dict(work.attached),
            lam=np.zeros(c.n_constraints, dtype=np.float64),
            step=work.step + 1,
            energy=float(energy),
        )
        self._last_stable = new_state.copy()
        self.last_penetration_count = self._penetration_count(q, asset)
        self.n_physics_steps += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return new_state

    # ------------------------------------------------------------------
    @property
    def constraints(self) -> MeshConstraints | None:
        return self._constraints

    @property
    def model(self):
        return self._model


__all__ = ["NewtonVBDSolver"]
