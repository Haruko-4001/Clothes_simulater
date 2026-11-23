"""Online system identification: LM + EKF covariance (SPEC §10.3(3)).

IMPL: src/tryon/calibration/identifier.py::SystemIdentifier
Must run off the realtime thread; this class is synchronous by design.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from ..config import Config
from ..math import clip_mahalanobis, condition_number, solve_psd
from ..types import (
    IdentificationResult,
    MaterialParams,
    MaterialState,
    ObservationBatch,
    PARAM_NAMES,
    material_bounds_arrays,
)
from .loss import _asset_with_theta, compute, fisher_information, simulate_forward


class SystemIdentifier:
    def __init__(self, cfg: Config, sim=None) -> None:
        self.cfg = cfg
        self.sim = sim
        self.theta: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.last_jacobian: np.ndarray | None = None
        self.last_fisher: np.ndarray | None = None
        self.update_count = 0

    # ------------------------------------------------------------------
    def bind(self, sim) -> "SystemIdentifier":
        self.sim = sim
        return self

    def _ensure_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sim is None:
            raise RuntimeError("SystemIdentifier.bind(sim) must be called first")
        base = getattr(self.sim, "asset", None)
        if base is None:
            raise RuntimeError("calibration simulator has no asset")
        if self.theta is None:
            self.theta = base.material.to_vector().copy()
        if self.P is None:
            var = base.material.variance
            diag = np.array(
                [
                    max(float(var.get(n, (0.3 * max(abs(v), 1.0)) ** 2)), 1.0e-10)
                    for n, v in zip(PARAM_NAMES, self.theta)
                ],
                dtype=np.float64,
            )
            self.P = np.diag(diag)
        return self.theta, self.P

    # ------------------------------------------------------------------
    def _warp_eval(
        self,
        batch: ObservationBatch,
        theta: np.ndarray,
        theta0: np.ndarray,
        P_inv: np.ndarray,
    ) -> tuple[float, np.ndarray] | None:
        """(L_total, ∂L_data/∂Θ) from the differentiable Warp rollout (P4)."""
        from .warp_grad import rollout_loss_and_grad

        out = rollout_loss_and_grad(
            self.sim.asset,
            batch,
            theta,
            substeps=int(self.cfg.require("physics.substeps")),
            iters=int(self.cfg.require("physics.xpbd_iters")),
            attach_relax=float(self.cfg.require("physics.attach_relax")),
            support_relax=float(self.cfg.require("physics.support_relax")),
        )
        if out is None:
            return None
        L_data, g_data = out
        prior = np.asarray(theta, dtype=np.float64) - theta0
        lam_prior = float(self._weights().lam)
        L = float(L_data + lam_prior * float(prior @ (P_inv @ prior)))
        return L, g_data

    # ------------------------------------------------------------------
    def update(self, batch: ObservationBatch) -> IdentificationResult:
        if self.sim is None:
            raise RuntimeError("SystemIdentifier.bind(sim) must be called first")
        theta, P = self._ensure_state()
        orig_asset = getattr(self.sim, "asset", None)
        base = getattr(self.sim, "asset").material
        theta0 = base.to_vector().copy()
        P = P + np.eye(7) * 1.0e-12
        P_inv = np.linalg.inv(P)

        ablation = dict(self.cfg.get("calibration.ablation", {}) or {})
        if bool(ablation.get("disable_material_opt", False)):
            return self._no_update(theta, base, "material_opt_disabled")

        lam_LM = float(self.cfg.require("calibration.lambda_LM"))
        max_iter = int(self.cfg.require("calibration.max_iter"))
        delta_max = float(self.cfg.require("calibration.delta_max"))
        eps_J = float(self.cfg.require("calibration.epsilon_J"))
        lo, hi = material_bounds_arrays()

        # P4: reverse-mode Warp gradients replace FD inside the LM loop when
        # configured; the FD Jacobian still feeds the Fisher/active-mask gates.
        use_warp_grad = False
        if str(self.cfg.get("calibration.grad_backend", "fd")) == "warp":
            try:
                from .warp_grad import warp_available

                use_warp_grad = warp_available() and getattr(self.sim, "asset", None) is not None
            except Exception:
                use_warp_grad = False

        L, g, J = compute(batch, self._weights(), theta, theta0, P_inv, self.sim)
        L_before = L
        if use_warp_grad:
            wout = self._warp_eval(batch, theta, theta0, P_inv)
            if wout is None:
                use_warp_grad = False
            else:
                L, g = wout
        active = self._active_mask(J, ablation, eps_J)
        for _ in range(max_iter):
            fisher = fisher_information(J, self._sigma_inv())
            A = fisher + lam_LM * np.eye(7) + P_inv
            delta = solve_psd(A, -g)
            delta = clip_mahalanobis(delta, P, delta_max)
            trial = np.clip(theta + delta, lo, hi)
            trial[~active] = theta[~active]
            if np.max(np.abs(trial - theta)) < 1.0e-9:
                break
            if use_warp_grad:
                wout = self._warp_eval(batch, trial, theta0, P_inv)
                if wout is None:
                    use_warp_grad = False
                    continue
                L_trial, g_trial = wout
                J_trial = J
            else:
                L_trial, g_trial, J_trial = compute(
                    batch, self._weights(), trial, theta0, P_inv, self.sim
                )
            if L_trial < L:
                rel_improvement = (L - L_trial) / max(L, 1.0e-12)
                theta, L, g = trial, L_trial, g_trial
                if not use_warp_grad:
                    J = J_trial
                lam_LM = max(lam_LM / 3.0, 1.0e-9)
                if rel_improvement < 1.0e-4:
                    break
            else:
                lam_LM = min(lam_LM * 3.0, 1.0e9)
            if float(np.linalg.norm(delta)) < 1.0e-8:
                break

        if use_warp_grad:
            # Final FD pass at the accepted θ so gates see a full Jacobian.
            L, g, J = compute(batch, self._weights(), theta, theta0, P_inv, self.sim)

        active = self._active_mask(J, ablation, eps_J)
        fisher = fisher_information(J, self._sigma_inv())
        self.last_jacobian = J
        self.last_fisher = fisher
        # Condition number is meaningful only on the identifiable columns;
        # a deliberately frozen column (friction/ν/thickness) would make the
        # full 7×7 Fisher matrix singular and block every COMMIT (SPEC §3.2).
        if np.any(active):
            cond = condition_number(fisher[np.ix_(active, active)])
        else:
            cond = float("inf")
        try:
            P_new = np.linalg.inv(P_inv + fisher)
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(P_inv + fisher)
        P_new = 0.5 * (P_new + P_new.T)
        self.theta = theta
        self.P = P_new
        self.update_count += 1

        predicts = self._predicts_next_frame(batch, theta, theta0)
        candidate = MaterialParams.from_vector(theta, template=base)
        candidate = dataclasses.replace(
            candidate,
            state=MaterialState.CALIBRATING,
            calibrated_at=None,
            scene_tag="",
        )
        variance = {n: float(max(P_new[i, i], 0.0)) for i, n in enumerate(PARAM_NAMES)}
        result = IdentificationResult(
            params=candidate,
            loss_before=float(L_before),
            loss_after=float(L),
            param_variance=variance,
            fisher_cond=float(cond),
            predicts_next_frame=bool(predicts),
        )
        if orig_asset is not None:
            self.sim.asset = orig_asset
        return result

    # ------------------------------------------------------------------
    def _weights(self):
        from ..types import LossWeights

        return LossWeights()

    def _sigma_inv(self):
        w = self._weights()
        return w.sigma_inv

    def _active_mask(self, J: np.ndarray, ablation: dict, eps_J: float) -> np.ndarray:
        names = set(self.cfg.get("calibration.params", PARAM_NAMES))
        mask = np.array([n in names for n in PARAM_NAMES], dtype=bool)
        if bool(ablation.get("disable_friction_opt", False)):
            mask[PARAM_NAMES.index("friction")] = False
        norms = np.linalg.norm(np.asarray(J, dtype=np.float64), axis=0)
        mask &= norms > float(eps_J)
        return mask

    def _no_update(self, theta: np.ndarray, base: MaterialParams, reason: str) -> IdentificationResult:
        candidate = dataclasses.replace(
            MaterialParams.from_vector(theta, template=base),
            state=MaterialState.CALIBRATING,
            calibrated_at=None,
            scene_tag="",
        )
        return IdentificationResult(
            params=candidate,
            loss_before=0.0,
            loss_after=0.0,
            param_variance={n: float(self.P[i, i]) if self.P is not None else 0.0 for i, n in enumerate(PARAM_NAMES)},
            fisher_cond=float("inf"),
            predicts_next_frame=False,
        )

    def _predicts_next_frame(
        self,
        batch: ObservationBatch,
        theta: np.ndarray,
        theta0: np.ndarray,
    ) -> bool:
        horizon_ms = float(self.cfg.require("calibration.predict_horizon_ms"))
        eps_ade = float(self.cfg.require("calibration.epsilon_ade"))
        H = max(1, int(round((horizon_ms / 1000.0) / float(batch.dt))))
        if len(batch.states) < H + 2:
            return False

        original_asset = getattr(self.sim, "asset", None)

        def ade_for(th: np.ndarray) -> float:
            asset = _asset_with_theta(self.sim, th)
            state = batch.states[-H - 1].copy()
            total = 0.0
            for h in range(H):
                state = self.sim.step(
                    state, asset, [batch.contacts[-H + h]], float(batch.dt)
                )
                obs = batch.states[-H + h].positions
                total += float(np.mean(np.linalg.norm(state.positions - obs, axis=1)))
            return total / H

        try:
            ade_new = ade_for(theta)
            ade_prior = ade_for(theta0)
        except Exception:
            return False
        finally:
            if original_asset is not None:
                self.sim.asset = original_asset
        return bool(ade_new < ade_prior - eps_ade)


def run_identification_job(payload: dict) -> dict:
    """Top-level process-worker entry: pipeline C runs off the realtime process.

    Payload keys: cfg, asset, batch, theta (optional), P (optional).
    Returns a picklable result dict; never raises across the process boundary.
    """
    try:
        from ..physics.solver import PhysicsSolver

        cfg = payload["cfg"]
        asset = payload["asset"]
        batch = payload["batch"]
        sim = PhysicsSolver(cfg)
        sim.asset = asset
        ident = SystemIdentifier(cfg, sim)
        theta = payload.get("theta")
        P = payload.get("P")
        if theta is not None:
            ident.theta = np.asarray(theta, dtype=np.float64).copy()
        if P is not None:
            ident.P = np.asarray(P, dtype=np.float64).copy()
        result = ident.update(batch)
        return {
            "ok": True,
            "result": result,
            "theta": None if ident.theta is None else ident.theta.copy(),
            "P": None if ident.P is None else ident.P.copy(),
        }
    except Exception as exc:  # pragma: no cover - process boundary
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "code": getattr(exc, "code", "E_CALIB_UNIDENTIFIABLE"),
        }


def calibration_process_worker(payload: dict, out_queue) -> None:
    """Daemon-process entry point for pipeline C."""
    out_queue.put(run_identification_job(payload))


__all__ = ["SystemIdentifier", "run_identification_job", "calibration_process_worker"]
