"""Dictionary material prior Θ₀ (SPEC §2.3 / §8).

IMPL: src/tryon/asset/material_prior.py::MaterialPrior

A single static image cannot identify Θ (rank-deficient Fisher information);
this class therefore always returns ``state=AI_PRIOR`` and never claims truth.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import (
    GarmentClass,
    MaterialParams,
    MaterialState,
    RGBImage,
    Segmentation,
    weighted_geometric_mean,
)


# Fabric presets are priors, not measured values. All components lie inside
# the Appendix-B feasible domain.
FABRIC_PRESETS: dict[str, dict[str, float]] = {
    "cotton": {
        "areal_density": 0.24,
        "youngs_modulus": 2.2e3,
        "bending_stiffness": 1.2e-3,
        "poisson_ratio": 0.30,
        "friction": 0.42,
        "damping": 2.4,
        "thickness": 1.0e-3,
    },
    "denim": {
        "areal_density": 0.55,
        "youngs_modulus": 2.4e4,
        "bending_stiffness": 6.0e-3,
        "poisson_ratio": 0.28,
        "friction": 0.62,
        "damping": 4.5,
        "thickness": 2.0e-3,
    },
    "silk": {
        "areal_density": 0.09,
        "youngs_modulus": 5.5e2,
        "bending_stiffness": 8.0e-5,
        "poisson_ratio": 0.35,
        "friction": 0.18,
        "damping": 0.8,
        "thickness": 4.0e-4,
    },
    "knit": {
        "areal_density": 0.30,
        "youngs_modulus": 8.5e2,
        "bending_stiffness": 5.5e-4,
        "poisson_ratio": 0.38,
        "friction": 0.52,
        "damping": 3.2,
        "thickness": 1.5e-3,
    },
    "default": {
        "areal_density": 0.25,
        "youngs_modulus": 2.0e3,
        "bending_stiffness": 1.0e-3,
        "poisson_ratio": 0.30,
        "friction": 0.40,
        "damping": 2.0,
        "thickness": 1.0e-3,
    },
}


def _preset_name(image: RGBImage, seg: Segmentation) -> str:
    mask = seg.mask > 0
    if not mask.any():
        return "default"
    px = np.asarray(image, dtype=np.float64)[mask]
    if px.size == 0:
        return "default"
    mean = px.mean(axis=0)
    brightness = float(mean.mean())
    sat = float((mean.max() - mean.min()) / max(mean.max(), 1.0))
    # Coarse deterministic heuristic: dark+high-contrast → denim; bright+smooth
    # → silk; mid-tone → cotton; satin sheen (high saturation) → knit.
    if brightness < 95.0 and sat > 0.20:
        return "denim"
    if brightness > 190.0 and sat < 0.16:
        return "silk"
    if sat > 0.42:
        return "knit"
    if seg.garment_class == GarmentClass.ONE_PIECES:
        return "cotton"
    return "cotton"


class MaterialPrior:
    """Colour/texture-driven dictionary prior. Never CALIBRATED."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.last_preset: str = "default"

    def prior(self, image: RGBImage, seg: Segmentation) -> MaterialParams:
        name = _preset_name(image, seg)
        self.last_preset = name
        p = dict(FABRIC_PRESETS[name])
        # Broad prior variance: this is an uncalibrated guess.
        variance = {
            "areal_density": (0.25 * p["areal_density"]) ** 2,
            "youngs_modulus": (0.55 * p["youngs_modulus"]) ** 2,
            "bending_stiffness": (0.55 * p["bending_stiffness"]) ** 2,
            "poisson_ratio": 0.08**2,
            "friction": 0.15**2,
            "damping": (0.35 * max(p["damping"], 0.4)) ** 2,
            "thickness": (0.25 * p["thickness"]) ** 2,
        }
        return MaterialParams(
            **p,
            state=MaterialState.AI_PRIOR,
            variance=variance,
            scene_tag="",
            calibrated_at=None,
        )


def material_prior_level(params: MaterialParams) -> float:
    """Confidence that the *prior tier* is plausible, never calibration truth."""
    # A tight prior support and a common fabric tier raise this modestly.
    support = weighted_geometric_mean(
        np.array([1.0 / (1.0 + np.sqrt(max(params.variance.get(k, 1.0), 0.0))) for k in params.variance] or [0.5]),
        np.ones(len(params.variance) or 1),
    ) if params.variance else 0.35
    return float(np.clip(0.35 + 0.35 * support, 0.0, 0.75))


__all__ = ["MaterialPrior", "FABRIC_PRESETS", "material_prior_level"]
