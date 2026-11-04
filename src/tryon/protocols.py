"""Protocols for every swappable external dependency (SPEC §4 / §8).

Callers depend only on these protocols; ``Real*`` and ``Stub*`` implement them.
All protocols are ``@runtime_checkable``.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import (
    BodyState,
    ContactEvent,
    DrapeReport,
    Frame,
    FramingCompliance,
    GarmentAsset,
    GateVerdict,
    GeometryDraft,
    HandState,
    IdentificationResult,
    KeyframeRequest,
    KeyframeResult,
    LossWeights,
    MaterialParams,
    ObservationBatch,
    PhysicsState,
    RGBImage,
    RenderOutput,
    Segmentation,
    SimReadyMesh,
    SystemStatus,
    TriggerContext,
    CalibHistory,
)


@runtime_checkable
class Segmenter(Protocol):
    def run(self, image: RGBImage) -> Segmentation: ...


@runtime_checkable
class GeometryEstimator(Protocol):
    def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft: ...


@runtime_checkable
class MeshRepairer(Protocol):
    def repair(self, draft: GeometryDraft) -> SimReadyMesh: ...


@runtime_checkable
class DrapeAligner(Protocol):
    def align(self, mesh: SimReadyMesh, seg: Segmentation) -> DrapeReport: ...


@runtime_checkable
class MaterialPrior(Protocol):
    def prior(self, image: RGBImage, seg: Segmentation) -> MaterialParams: ...


@runtime_checkable
class AssetBuilder(Protocol):
    def build(self, image: RGBImage, *, timeout_s: float = 300.0) -> GarmentAsset: ...


@runtime_checkable
class Camera(Protocol):
    K: np.ndarray
    T_CW: np.ndarray

    def capture(self) -> Frame: ...


@runtime_checkable
class BodyTracker(Protocol):
    def track(self, frame: Frame, history: deque | None = None) -> BodyState: ...


@runtime_checkable
class FramingGate(Protocol):
    def check(self, frame: Frame, body: BodyState) -> FramingCompliance: ...


@runtime_checkable
class HandTracker(Protocol):
    def track(self, frame: Frame) -> HandState: ...


@runtime_checkable
class ContactExtractor(Protocol):
    def extract(
        self,
        hands: HandState,
        body: BodyState,
        phys: PhysicsState,
    ) -> list[ContactEvent]: ...


@runtime_checkable
class PhysicsSolver(Protocol):
    def step(
        self,
        state: PhysicsState,
        asset: GarmentAsset,
        contacts: list[ContactEvent],
        dt: float,
    ) -> PhysicsState: ...


@runtime_checkable
class Renderer(Protocol):
    def render(
        self,
        frame: Frame,
        asset: GarmentAsset,
        state: PhysicsState,
        camera: Camera,
    ) -> RenderOutput: ...


@runtime_checkable
class KeyframeTrigger(Protocol):
    def should_fire(self, ctx: TriggerContext) -> bool: ...


@runtime_checkable
class KeyframeTryOn(Protocol):
    def generate(self, req: KeyframeRequest) -> KeyframeResult: ...


@runtime_checkable
class Compositor(Protocol):
    def compose(
        self,
        frame: Frame,
        rendered: RenderOutput,
        status: SystemStatus,
        keyframe: KeyframeResult | None = None,
    ) -> RGBImage: ...


@runtime_checkable
class HUD(Protocol):
    def draw(self, frame: RGBImage, status: SystemStatus) -> RGBImage: ...


@runtime_checkable
class Display(Protocol):
    def show(self, frame: RGBImage) -> None: ...


@runtime_checkable
class TrajectoryCollector(Protocol):
    def observe(self, ev: ContactEvent, state: PhysicsState) -> None: ...

    def is_excited(self) -> bool: ...


@runtime_checkable
class SystemIdentifier(Protocol):
    def update(self, batch: ObservationBatch) -> IdentificationResult: ...


@runtime_checkable
class CalibrationGate(Protocol):
    def evaluate(self, res: IdentificationResult, hist: CalibHistory) -> GateVerdict: ...


@runtime_checkable
class LicenseGate(Protocol):
    def assert_startup_allowed(self, cfg: Any) -> None: ...

    def check(self, backend: str, *, mode: str, weights_license: str = "") -> None: ...


PROTOCOL_KEYS = (
    "Segmenter",
    "GeometryEstimator",
    "MeshRepairer",
    "DrapeAligner",
    "MaterialPrior",
    "AssetBuilder",
    "Camera",
    "BodyTracker",
    "FramingGate",
    "HandTracker",
    "ContactExtractor",
    "PhysicsSolver",
    "Renderer",
    "KeyframeTrigger",
    "KeyframeTryOn",
    "Compositor",
    "HUD",
    "Display",
    "TrajectoryCollector",
    "SystemIdentifier",
    "CalibrationGate",
    "LicenseGate",
)

__all__ = list(PROTOCOL_KEYS) + ["PROTOCOL_KEYS"]
