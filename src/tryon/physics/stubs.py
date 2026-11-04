"""Stub physics — the §10.1 integrator is real, not a placeholder.

IMPL: src/tryon/physics/stubs.py

There is no neural or vendor dependency in mock mode; the reference XPBD
solver is used directly so drag / rebound / energy invariants are physical.
"""
from __future__ import annotations

from .solver import PhysicsSolver

StubXPBDPhysicsSolver = PhysicsSolver

__all__ = ["StubXPBDPhysicsSolver", "PhysicsSolver"]
