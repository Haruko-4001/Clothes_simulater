"""Three acceptance gates G1/G2/G3 (SPEC §11 / §10.3(5)).

IMPL: src/tryon/calibration/gates.py::CalibrationGate
"""
from __future__ import annotations

from ..config import Config
from ..types import (
    CalibHistory,
    Confidence,
    DrapeReport,
    GateVerdict,
    IdentificationResult,
    SimReadyMesh,
)


def evaluate_g1(
    iou: float,
    mesh: SimReadyMesh,
    flyaway_faces: int,
    cfg: Config,
) -> bool:
    threshold = float(cfg.require("asset.g1_iou_threshold"))
    return bool(
        float(iou) >= threshold
        and not mesh.has_isolated_edge
        and int(mesh.self_intersection_count) == 0
        and int(flyaway_faces) == 0
    )


def evaluate_g2(report: DrapeReport, cfg: Config) -> bool:
    g2 = cfg.require("asset.g2")
    return bool(
        report.silhouette_iou >= float(g2["silhouette_iou"])
        and report.chamfer_distance <= float(g2["chamfer_px"])
        and report.normal_error <= float(g2["normal_error_rad"])
    )


def needs_ai_completion(conf: Confidence, cfg: Config) -> bool:
    return bool(float(conf.back_geometry) < float(cfg.require("asset.back_confidence_threshold")))


class CalibrationGate:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.kappa_max = float(cfg.require("calibration.kappa_max"))
        self.eps_L = float(cfg.require("calibration.commit_min_loss_drop"))
        self.max_interactions = int(cfg.require("calibration.max_interactions_before_degrade"))

    # ------------------------------------------------------------------
    @staticmethod
    def _variance_decreasing(hist: CalibHistory) -> bool:
        v = hist.variances
        if len(v) < 2:
            return False
        tail = v[-5:] if len(v) >= 5 else v
        return all(tail[i] < tail[i - 1] for i in range(1, len(tail)))

    @staticmethod
    def converged(hist: CalibHistory, rel_tol: float = 0.02) -> bool:
        """tr(P) strictly decreasing for 5 windows AND relative change <2%."""
        v = hist.variances
        if len(v) < 5:
            return False
        tail = v[-5:]
        for i in range(1, len(tail)):
            if tail[i] >= tail[i - 1]:
                return False
            rel = abs(tail[i - 1] - tail[i]) / max(abs(tail[i - 1]), 1e-12)
            if rel >= rel_tol:
                return False
        return True

    # ------------------------------------------------------------------
    def evaluate(self, res: IdentificationResult, hist: CalibHistory) -> GateVerdict:
        loss0 = max(float(res.loss_before), 1.0e-12)
        loss_drop = (res.loss_before - res.loss_after) / loss0
        cond_ok = float(res.fisher_cond) < self.kappa_max
        dL_ok = loss_drop > self.eps_L
        var_ok = self._variance_decreasing(hist)
        predict_ok = bool(res.predicts_next_frame)

        if cond_ok and dL_ok and var_ok and predict_ok:
            return GateVerdict.COMMIT

        converged = self.converged(hist)
        if int(hist.n_interactions) > self.max_interactions and not converged:
            return GateVerdict.DEGRADE
        return GateVerdict.KEEP_PRIOR


__all__ = ["CalibrationGate", "evaluate_g1", "evaluate_g2", "needs_ai_completion"]
