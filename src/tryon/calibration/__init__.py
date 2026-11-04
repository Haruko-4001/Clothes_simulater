"""Online material identification (pipeline C)."""
from .collector import TrajectoryCollector
from .loss import compute, fisher_condition, fisher_information, simulate_forward
from .identifier import SystemIdentifier
from .gates import CalibrationGate, evaluate_g1, evaluate_g2, needs_ai_completion

__all__ = [
    "TrajectoryCollector",
    "compute",
    "fisher_condition",
    "fisher_information",
    "simulate_forward",
    "SystemIdentifier",
    "CalibrationGate",
    "evaluate_g1",
    "evaluate_g2",
    "needs_ai_completion",
]
