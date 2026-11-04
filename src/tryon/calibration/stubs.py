"""Calibration is algorithm-only; stub aliases point at the real implementation."""
from __future__ import annotations

from .collector import TrajectoryCollector
from .gates import CalibrationGate
from .identifier import SystemIdentifier

StubTrajectoryCollector = TrajectoryCollector
StubSystemIdentifier = SystemIdentifier
StubCalibrationGate = CalibrationGate

__all__ = [
    "TrajectoryCollector",
    "CalibrationGate",
    "SystemIdentifier",
    "StubTrajectoryCollector",
    "StubSystemIdentifier",
    "StubCalibrationGate",
]
