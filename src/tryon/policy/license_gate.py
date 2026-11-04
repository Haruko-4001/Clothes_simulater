"""License gate: NC / ND / research-only weights are blocked in real mode.

IMPL: src/tryon/policy/license_gate.py

Repo MIT/Apache ≠ weights commercially usable (SPEC §13 red line).
"""
from __future__ import annotations

from ..config import Config
from ..types import TryonError


NC_BACKENDS = {
    "catvton",
    "idm_vton",
    "idm-vton",
    "ootd",
    "ootdiffusion",
    "stableviton",
    "causvid",
}
ND_BACKENDS = {"wilor"}
RESEARCH_ONLY = {"smplx", "smpl-x", "mano", "bcnet", "rtv"}
NC_LICENSES = {"nc", "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "non-commercial"}
RESEARCH_LICENSES = {"research-only", "research_only", "non-commercial-research"}


def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


class LicenseGate:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.allow_nc = bool(cfg.get("policy.allow_nc_weights", False))

    def check(self, backend: str, *, mode: str, weights_license: str = "") -> None:
        if mode != "real":
            return
        if self.allow_nc:
            return
        name = _norm(backend)
        lic = _norm(weights_license)
        if name in NC_BACKENDS or name in ND_BACKENDS or lic in NC_LICENSES:
            raise TryonError(
                "E_LICENSE_NC_BLOCKED",
                f"backend {backend!r} has NC/ND weights; commercial/real use is blocked",
                context={"backend": backend, "weights_license": weights_license},
            )
        if name in RESEARCH_ONLY or lic in RESEARCH_LICENSES:
            raise TryonError(
                "E_LICENSE_NC_BLOCKED",
                f"backend {backend!r} is research-only (SMPL-X/MANO/BCNET); blocked in real mode",
                context={"backend": backend, "weights_license": weights_license},
            )

    def assert_startup_allowed(self, cfg: Config) -> None:
        mode = str(cfg.get("mode", "mock"))
        if mode != "real":
            return
        backends = {
            "geometry": cfg.get("asset.geometry"),
            "keyframe": cfg.get("keyframe.backend"),
            "physics": cfg.get("physics.solver"),
            "render": cfg.get("render.backend"),
        }
        for layer, backend in backends.items():
            self.check(str(backend), mode=mode)
        # Default FASHN VTON is Apache-2.0; the separate human parser terms
        # must be reviewed before any redistribution (see SPEC §13).


__all__ = ["LicenseGate", "NC_BACKENDS", "ND_BACKENDS", "RESEARCH_ONLY"]
