"""Evaluation metrics and reports."""
from .metrics import (
    ade,
    chamfer_distance,
    convergence,
    energy_decay_ok,
    fde,
    framing_compliance_rate,
    iou,
    latency_percentiles,
    normal_error,
    parameter_error,
    penetration_rate,
)
from .report import build_report

__all__ = [
    "ade",
    "chamfer_distance",
    "convergence",
    "energy_decay_ok",
    "fde",
    "framing_compliance_rate",
    "iou",
    "latency_percentiles",
    "normal_error",
    "parameter_error",
    "penetration_rate",
    "build_report",
]
