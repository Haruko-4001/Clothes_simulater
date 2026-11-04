"""Degradation routing. Never hard-guess out-of-domain input (SPEC §2.5 / §10.4).

IMPL: src/tryon/policy/degrade.py
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import Config
from ..types import FramingCompliance, FramingVerdict


class DegradeAction(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    ASYNC_3D = "ASYNC_3D"
    AI_COMPLETION = "AI_COMPLETION"
    REJECT = "REJECT"


@dataclass
class DegradeDecision:
    action: DegradeAction
    reasons: list[str]
    guide_text: str
    allow_2d_keyframe: bool
    allow_full_image: bool


def route_framing(compliance: FramingCompliance, cfg: Config) -> DegradeDecision:
    verdict = compliance.verdict
    if verdict == FramingVerdict.OK:
        return DegradeDecision(
            DegradeAction.FULL,
            [],
            compliance.guide_text,
            allow_2d_keyframe=True,
            allow_full_image=True,
        )
    if verdict == FramingVerdict.PARTIAL:
        return DegradeDecision(
            DegradeAction.PARTIAL,
            ["partial_crop"],
            compliance.guide_text,
            allow_2d_keyframe=False,
            allow_full_image=False,
        )
    if verdict == FramingVerdict.GUIDE:
        reasons = ["side_or_back"] if compliance.facing_estimate in {"side", "back"} else ["guide"]
        return DegradeDecision(
            DegradeAction.ASYNC_3D,
            reasons,
            compliance.guide_text,
            allow_2d_keyframe=False,
            allow_full_image=False,
        )
    return DegradeDecision(
        DegradeAction.REJECT,
        ["extreme_occlusion_or_domain"],
        compliance.guide_text,
        allow_2d_keyframe=False,
        allow_full_image=False,
    )


def route_asset_confidence(
    back_confidence: float,
    cfg: Config,
) -> list[DegradeAction]:
    actions = []
    if float(back_confidence) < float(cfg.require("asset.back_confidence_threshold")):
        actions.append(DegradeAction.AI_COMPLETION)
    return actions


def requires_human_review(overall_confidence: float, cfg: Config) -> bool:
    return bool(float(overall_confidence) < float(cfg.require("policy.require_human_review_below")))


__all__ = [
    "DegradeAction",
    "DegradeDecision",
    "route_framing",
    "route_asset_confidence",
    "requires_human_review",
]
