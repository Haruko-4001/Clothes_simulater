"""Differentiable rollout gradients for pipeline C (P4).

IMPL: src/tryon/calibration/warp_grad.py::rollout_loss_and_grad

Replaces the 7-column finite-difference Jacobian *inside the LM loop* with
exact reverse-mode gradients from a Warp-taped XPBD rollout. One forward +
one backward pass yields (L, ∂L/∂Θ) instead of 8 forward simulations per LM
iteration. The surrogate matches ``physics.warp_solver``'s model family
(Jacobi XPBD distance + bending springs, mass damping, ellipsoid contact
friction, soft attachments).

Honest scope notes:
  * Poisson ratio and thickness have no effect inside the surrogate, so their
    gradients are exactly zero; the FD path stays responsible for those
    columns and for the Fisher matrix used by the gates.
  * fp32 arithmetic: gradients match the FD direction (cosine similarity
    checked in tests) but not to double precision.
"""
from __future__ import annotations

import numpy as np

from ..types import GarmentAsset, ObservationBatch
from ..physics.state import build_constraints

try:  # Warp is optional: its absence must not break module import.
    import warp as _wp

    _WARP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _wp = None
    _WARP_AVAILABLE = False


if _WARP_AVAILABLE:

    @_wp.func
    def _ellipsoid_q(p: _wp.vec3, c: _wp.vec3, r: _wp.vec3) -> _wp.vec3:
        d = p - c
        return _wp.vec3(d[0] / r[0], d[1] / r[1], d[2] / r[2])

    @_wp.func
    def _scale_vec(v: _wp.vec3, s: _wp.vec3) -> _wp.vec3:
        return _wp.vec3(v[0] * s[0], v[1] * s[1], v[2] * s[2])

    @_wp.kernel
    def _build_inv_mass_kernel(
        vertex_area: _wp.array(dtype=_wp.float32),
        theta: _wp.array(dtype=_wp.float32),
        inv_mass: _wp.array(dtype=_wp.float32),
    ):
        i = _wp.tid()
        rho = theta[0]  # areal_density
        m = _wp.max(rho * vertex_area[i], 1.0e-10)
        inv_mass[i] = 1.0 / m

    @_wp.kernel
    def _build_edge_alpha_kernel(
        theta: _wp.array(dtype=_wp.float32),
        edge_alpha: _wp.array(dtype=_wp.float32),
    ):
        c = _wp.tid()
        youngs = theta[1]  # youngs_modulus
        edge_alpha[c] = 1.0 / _wp.max(youngs, 1.0e-3)

    @_wp.kernel
    def _build_bend_alpha_kernel(
        bend_rest: _wp.array(dtype=_wp.float32),
        theta: _wp.array(dtype=_wp.float32),
        bend_alpha: _wp.array(dtype=_wp.float32),
    ):
        c = _wp.tid()
        kbend = theta[2]  # bending_stiffness
        brest = bend_rest[c]
        kb = _wp.max(kbend / (brest * brest), 1.0e-4)
        bend_alpha[c] = 1.0 / kb

    @_wp.kernel
    def _build_damping_kernel(
        theta: _wp.array(dtype=_wp.float32),
        damping: _wp.array(dtype=_wp.float32),
    ):
        i = _wp.tid()
        cd = theta[5]  # damping
        damping[i] = 1.0 / (1.0 + cd)

    @_wp.kernel
    def _zero_vec_kernel(delta: _wp.array(dtype=_wp.vec3)):
        i = _wp.tid()
        delta[i] = _wp.vec3(0.0, 0.0, 0.0)

    @_wp.kernel
    def _pred_kernel(
        x: _wp.array(dtype=_wp.vec3),
        x_prev: _wp.array(dtype=_wp.vec3),
        v: _wp.array(dtype=_wp.vec3),
        dt: float,
        gravity: _wp.vec3,
        damping: _wp.array(dtype=_wp.float32),
    ):
        i = _wp.tid()
        x_prev[i] = x[i]
        vi = v[i] * damping[i] + dt * gravity
        x[i] = x[i] + dt * vi
        v[i] = vi

    @_wp.kernel
    def _adjacent_kernel(
        x: _wp.array(dtype=_wp.vec3),
        lam: _wp.array(dtype=_wp.float32),
        ci: _wp.array(dtype=_wp.int32),
        cj: _wp.array(dtype=_wp.int32),
        rest: _wp.array(dtype=_wp.float32),
        edge_alpha: _wp.array(dtype=_wp.float32),
        bend_alpha: _wp.array(dtype=_wp.float32),
        inv_mass: _wp.array(dtype=_wp.float32),
        degree: _wp.array(dtype=_wp.float32),
        delta: _wp.array(dtype=_wp.vec3),
        n_edge: int,
        dt: float,
    ):
        c = _wp.tid()
        i = ci[c]
        j = cj[c]
        d = x[i] - x[j]
        length = _wp.max(_wp.length(d), 1.0e-9)
        n = d / length
        C = length - rest[c]
        if c < n_edge:
            alpha = edge_alpha[c]
        else:
            alpha = bend_alpha[c - n_edge]
        alpha_t = alpha / (dt * dt)
        denom = inv_mass[i] + inv_mass[j] + alpha_t
        dlam = (-C - alpha_t * lam[c]) / _wp.max(denom, 1.0e-12)
        wi = inv_mass[i] / _wp.max(degree[i], 1.0) * dlam
        wj = inv_mass[j] / _wp.max(degree[j], 1.0) * dlam
        _wp.atomic_add(delta, i, n * wi)
        _wp.atomic_add(delta, j, -n * wj)
        lam[c] = lam[c] + dlam

    @_wp.kernel
    def _apply_delta_kernel(
        x: _wp.array(dtype=_wp.vec3),
        delta: _wp.array(dtype=_wp.vec3),
    ):
        i = _wp.tid()
        x[i] = x[i] + delta[i]

    @_wp.kernel
    def _attach_grad_kernel(
        x: _wp.array(dtype=_wp.vec3),
        idx: _wp.array(dtype=_wp.int32),
        target: _wp.array(dtype=_wp.vec3),
        relax: float,
    ):
        k = _wp.tid()
        i = idx[k]
        x[i] = x[i] + relax * (target[k] - x[i])

    @_wp.kernel
    def _support_grad_kernel(
        x: _wp.array(dtype=_wp.vec3),
        idx: _wp.array(dtype=_wp.int32),
        anchor: _wp.array(dtype=_wp.vec3),
        relax: float,
    ):
        k = _wp.tid()
        i = idx[k]
        x[i] = x[i] + relax * (anchor[k] - x[i])

    @_wp.kernel
    def _contact_grad_kernel(
        x: _wp.array(dtype=_wp.vec3),
        v: _wp.array(dtype=_wp.vec3),
        centers: _wp.array(dtype=_wp.vec3),
        radii: _wp.array(dtype=_wp.vec3),
        n_centers: int,
        theta: _wp.array(dtype=_wp.float32),
    ):
        i = _wp.tid()
        p = x[i]
        best_d2 = float(1.0e30)
        best_k = int(-1)
        for k in range(n_centers):
            q = _ellipsoid_q(p, centers[k], radii[k])
            d2 = _wp.dot(q, q)
            if d2 < best_d2:
                best_d2 = d2
                best_k = k
        if best_k >= 0 and best_d2 < 1.0 + 2.0e-3:
            c = centers[best_k]
            r = radii[best_k]
            q = _ellipsoid_q(p, c, r)
            n = _wp.max(_wp.length(q), 1.0e-9)
            normal = q / n
            vn = _wp.dot(v[i], normal)
            vt = v[i] - normal * vn
            mu = theta[4]  # friction
            vt = vt * (1.0 - 0.5 * _wp.clamp(mu, 0.0, 1.5))
            v[i] = normal * _wp.min(vn, 0.0) + vt
            if best_d2 < 1.0:
                x[i] = c + _scale_vec(q / n, r)

    @_wp.kernel
    def _fin_kernel(
        x: _wp.array(dtype=_wp.vec3),
        x_prev: _wp.array(dtype=_wp.vec3),
        v: _wp.array(dtype=_wp.vec3),
        dt: float,
    ):
        i = _wp.tid()
        v[i] = (x[i] - x_prev[i]) / dt

    @_wp.kernel
    def _copy_frame_kernel(
        x: _wp.array(dtype=_wp.vec3),
        v: _wp.array(dtype=_wp.vec3),
        traj: _wp.array(dtype=_wp.vec3),
        vtraj: _wp.array(dtype=_wp.vec3),
        frame: int,
        n: int,
    ):
        i = _wp.tid()
        traj[frame * n + i] = x[i]
        vtraj[frame * n + i] = v[i]

    @_wp.kernel
    def _loss_kernel(
        traj: _wp.array(dtype=_wp.vec3),
        vtraj: _wp.array(dtype=_wp.vec3),
        obs_x: _wp.array(dtype=_wp.vec3),
        obs_v: _wp.array(dtype=_wp.vec3),
        w_x: float,
        w_v: float,
        loss: _wp.array(dtype=_wp.float32),
    ):
        k = _wp.tid()
        dx = traj[k] - obs_x[k]
        dv = vtraj[k] - obs_v[k]
        # ½ factor keeps ∂L/∂Θ on the same scale as the FD path's g = Jᵀr
        # (gradient of ½·rᵀr), so the LM step sizes stay comparable.
        _wp.atomic_add(loss, 0, 0.5 * (w_x * _wp.dot(dx, dx) + w_v * _wp.dot(dv, dv)))


def warp_available() -> bool:
    return _WARP_AVAILABLE


def _frame_attachments(batch: ObservationBatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-frame (vertex_ids, targets) from the flat per-frame contact list.

    ``ObservationBatch.contacts`` carries exactly one event per frame (the
    collector's convention); a SLIDE/PULL on a non-attached vertex is a no-op,
    mirroring ``physics.constraints.process_contact_events``.
    """
    from ..types import ContactKind

    out: list[tuple[np.ndarray, np.ndarray]] = []
    held: dict[int, np.ndarray] = {}
    for ev in batch.contacts:
        vid = ev.vertex_id
        if vid is not None:
            if ev.kind == ContactKind.GRASP:
                held[vid] = np.asarray(ev.world_pos, dtype=np.float64)
            elif ev.kind == ContactKind.RELEASE:
                held.pop(vid, None)
            elif vid in held:
                held[vid] = np.asarray(ev.world_pos, dtype=np.float64)
        if held:
            ids_sorted = sorted(held.keys())
            ids = np.asarray(ids_sorted, dtype=np.int32)
            tgt = np.stack([held[i] for i in ids_sorted])
        else:
            ids = np.zeros(0, dtype=np.int32)
            tgt = np.zeros((0, 3))
        out.append((ids, tgt))
    return out


def rollout_loss_and_grad(
    asset: GarmentAsset,
    batch: ObservationBatch,
    theta: np.ndarray,
    *,
    w_x: float = 1.0,
    w_v: float = 0.5,
    substeps: int = 2,
    iters: int = 4,
    attach_relax: float = 0.35,
    support_relax: float = 0.15,
) -> tuple[float, np.ndarray] | None:
    """One tape pass → (L_data, ∂L_data/∂Θ) with Θ ∈ R^7.

    Returns None when warp is unavailable (caller falls back to FD). The
    prior term is added by the caller (analytic, cheap).
    """
    if not _WARP_AVAILABLE:  # pragma: no cover - optional dependency
        return None
    wp = _wp
    wp.init()

    c = build_constraints(asset.rest_mesh, asset.material)
    T = len(batch.states)
    n = len(c.inv_mass)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size != 7:
        raise ValueError("theta must be 7-vector")

    all_i = np.concatenate([c.edge_i, c.bend_i]).astype(np.int32)
    all_j = np.concatenate([c.edge_j, c.bend_j]).astype(np.int32)
    all_rest = np.concatenate([c.edge_rest, c.bend_rest]).astype(np.float32)
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, all_i, 1.0)
    np.add.at(degree, all_j, 1.0)
    n_con = int(all_i.shape[0])
    n_edge = int(len(c.edge_i))
    n_bend = int(len(c.bend_i))
    dt_s = float(batch.dt) / max(1, substeps)
    gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)

    collider = asset.collision_body
    centers = np.asarray(getattr(collider, "centers", np.zeros((0, 3))), dtype=np.float64)
    radii = np.asarray(getattr(collider, "radii", np.zeros((0, 3))), dtype=np.float64)
    if centers.size == 0:
        centers = np.zeros((1, 3))
        radii = np.ones((1, 3)) * 1.0e9
    n_centers = int(len(centers))

    x0 = batch.states[0].positions.astype(np.float32)
    v0 = batch.states[0].velocities.astype(np.float32)
    obs_x = np.concatenate([s.positions for s in batch.states], axis=0).astype(np.float32)
    obs_v = np.concatenate([s.velocities for s in batch.states], axis=0).astype(np.float32)
    attach_frames = _frame_attachments(batch)
    support_idx = c.support_idx.astype(np.int32)
    support_anchor = c.support_anchor.astype(np.float32)

    theta_dev = wp.array(theta.astype(np.float32), dtype=wp.float32, requires_grad=True)
    # The whole mutable rollout state must participate in the tape, otherwise
    # the chain Θ → inv_mass/α → x/v is cut at the first kernel write.
    x_dev = wp.array(x0, dtype=wp.vec3, requires_grad=True)
    v_dev = wp.array(v0, dtype=wp.vec3, requires_grad=True)
    x_prev = wp.zeros_like(x_dev)
    delta = wp.zeros_like(x_dev)
    lam_dev = wp.zeros(n_con, dtype=wp.float32, requires_grad=True)
    inv_mass = wp.zeros(n, dtype=wp.float32, requires_grad=True)
    edge_alpha = wp.zeros(n_edge, dtype=wp.float32, requires_grad=True)
    bend_alpha = wp.zeros(n_bend, dtype=wp.float32, requires_grad=True)
    damping = wp.zeros(n, dtype=wp.float32, requires_grad=True)
    traj = wp.zeros(T * n, dtype=wp.vec3, requires_grad=True)
    vtraj = wp.zeros(T * n, dtype=wp.vec3, requires_grad=True)
    obs_x_dev = wp.array(obs_x, dtype=wp.vec3)
    obs_v_dev = wp.array(obs_v, dtype=wp.vec3)
    centers_dev = wp.array(centers.astype(np.float32), dtype=wp.vec3)
    radii_dev = wp.array(radii.astype(np.float32), dtype=wp.vec3)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    area_dev = wp.array(c.vertex_area.astype(np.float32), dtype=wp.float32)
    ci_dev = wp.array(all_i, dtype=wp.int32)
    cj_dev = wp.array(all_j, dtype=wp.int32)
    rest_dev = wp.array(all_rest, dtype=wp.float32)
    degree_dev = wp.array(degree.astype(np.float32), dtype=wp.float32)
    empty_i = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32)
    empty_v = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)

    with wp.Tape() as tape:
        wp.launch(_build_inv_mass_kernel, dim=n, inputs=[area_dev, theta_dev, inv_mass])
        if n_edge:
            wp.launch(_build_edge_alpha_kernel, dim=n_edge, inputs=[theta_dev, edge_alpha])
        if n_bend:
            bend_rest_dev = wp.array(c.bend_rest.astype(np.float32), dtype=wp.float32)
            wp.launch(_build_bend_alpha_kernel, dim=n_bend, inputs=[bend_rest_dev, theta_dev, bend_alpha])
        wp.launch(_build_damping_kernel, dim=n, inputs=[theta_dev, damping])
        for t in range(T):
            for _ in range(max(1, substeps)):
                wp.launch(
                    _pred_kernel,
                    dim=n,
                    inputs=[x_dev, x_prev, v_dev, wp.float32(dt_s), wp.vec3(*gravity), damping],
                )
                if len(support_idx):
                    wp.launch(
                        _support_grad_kernel,
                        dim=len(support_idx),
                        inputs=[
                            x_dev,
                            wp.array(support_idx, dtype=wp.int32),
                            wp.array(support_anchor, dtype=wp.vec3),
                            wp.float32(support_relax),
                        ],
                    )
                ids, tgt = attach_frames[t]
                if len(ids):
                    wp.launch(
                        _attach_grad_kernel,
                        dim=len(ids),
                        inputs=[
                            x_dev,
                            wp.array(ids, dtype=wp.int32),
                            wp.array(tgt.astype(np.float32), dtype=wp.vec3),
                            wp.float32(attach_relax),
                        ],
                    )
                for _ in range(max(1, iters)):
                    wp.launch(_zero_vec_kernel, dim=n, inputs=[delta])
                    wp.launch(
                        _adjacent_kernel,
                        dim=n_con,
                        inputs=[
                            x_dev,
                            lam_dev,
                            ci_dev,
                            cj_dev,
                            rest_dev,
                            edge_alpha,
                            bend_alpha,
                            inv_mass,
                            degree_dev,
                            delta,
                            wp.int32(n_edge),
                            wp.float32(dt_s),
                        ],
                    )
                    wp.launch(_apply_delta_kernel, dim=n, inputs=[x_dev, delta])
                wp.launch(
                    _contact_grad_kernel,
                    dim=n,
                    inputs=[x_dev, v_dev, centers_dev, radii_dev, n_centers, theta_dev],
                )
                wp.launch(_fin_kernel, dim=n, inputs=[x_dev, x_prev, v_dev, wp.float32(dt_s)])
                wp.launch(
                    _contact_grad_kernel,
                    dim=n,
                    inputs=[x_dev, v_dev, centers_dev, radii_dev, n_centers, theta_dev],
                )
            wp.launch(
                _copy_frame_kernel,
                dim=n,
                inputs=[x_dev, v_dev, traj, vtraj, wp.int32(t), wp.int32(n)],
            )
        wp.launch(
            _loss_kernel,
            dim=T * n,
            inputs=[traj, vtraj, obs_x_dev, obs_v_dev, wp.float32(w_x), wp.float32(w_v), loss],
        )

    tape.backward(loss)
    grad = theta_dev.grad.numpy().astype(np.float64) if theta_dev.grad is not None else np.zeros(7)
    L = float(loss.numpy()[0])
    return L, grad


__all__ = ["rollout_loss_and_grad", "warp_available"]
