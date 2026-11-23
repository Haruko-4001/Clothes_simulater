"""Stub physics — the §10.1 integrator is real, not a placeholder.

IMPL: src/tryon/physics/stubs.py

There is no neural or vendor dependency in mock mode; the reference XPBD
solver is used directly so drag / rebound / energy invariants are physical.
"""
from __future__ import annotations

import numpy as np

from .solver import PhysicsSolver

StubXPBDPhysicsSolver = PhysicsSolver


def shell_index_band(mesh) -> int:
    """Max vertex-index gap across any mesh edge (structured-grid prior).

    Vertex-vertex self-collision exempts pairs inside this band because they
    are (or border on) topological neighbours; for shell meshes lofted by
    ``asset.stubs.build_shell_mesh`` this equals the ring size ``nu``.
    """
    F = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), dtype=np.int64)), dtype=np.int64)
    if F.size == 0:
        return 0
    a = F[:, [0, 1, 2]].reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    return int(np.max(np.abs(a - b)))


__all__ = ["StubXPBDPhysicsSolver", "PhysicsSolver", "shell_index_band"]
