"""Physics layer: XPBD state / constraints / integrator / solver."""
from .state import (
    MeshConstraints,
    bending_pairs,
    build_constraints,
    elastic_energy,
    initial_state,
    kinetic_energy,
    penetration_count,
    support_vertices,
    total_energy,
    unique_edges,
    vertex_areas,
)
from .constraints import apply_contact, friction_limit, process_contact_events
from .integrator import integrate
from .solver import PhysicsSolver
from .stubs import StubXPBDPhysicsSolver

__all__ = [
    "MeshConstraints",
    "bending_pairs",
    "build_constraints",
    "elastic_energy",
    "initial_state",
    "kinetic_energy",
    "penetration_count",
    "support_vertices",
    "total_energy",
    "unique_edges",
    "vertex_areas",
    "apply_contact",
    "friction_limit",
    "process_contact_events",
    "integrate",
    "PhysicsSolver",
    "StubXPBDPhysicsSolver",
]
