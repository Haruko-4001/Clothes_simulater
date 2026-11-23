"""Factory: ``build(cfg) -> dict[str, object]`` keyed by Protocol name.

IMPL: src/tryon/factory.py
Mock mode uses stubs. Real mode probes optional dependencies; missing ones
fall back to the stub with ``E_ADAPTER_FALLBACK`` (SPEC §8).
"""
from __future__ import annotations

from typing import Any

from .config import Config
from .asset.asset_builder import AssetBuilder
from .asset.drape import DrapeAligner
from .asset.material_prior import MaterialPrior
from .asset.mesh_repair import MeshRepairer
from .asset.registry import AssetRegistry
from .asset.segment import Segmenter
from .asset.geometry import GeometryEstimator, SilhouetteGeometryEstimator
from .asset.stubs import StubGeometryEstimator, StubSegmenter
from .calibration.collector import TrajectoryCollector
from .calibration.gates import CalibrationGate
from .calibration.identifier import SystemIdentifier
from .display import Compositor, HeadlessDisplay, HUD, WindowDisplay
from .keyframe.trigger import KeyframeTrigger
from .keyframe.tryon import FashnTryOn, StubKeyframeTryOn
from .logging import log_event
from .perception.body import BodyTracker, MediaPipeBodyTracker
from .perception.camera import RealCamera
from .perception.contact import ContactExtractor
from .perception.framing import FramingGate
from .perception.hands import HandTracker, MediaPipeHandTracker
from .perception.stubs import StubBodyTracker, StubCamera, StubHandTracker
from .physics.solver import DiffClothSolver, PhysicsSolver
from .physics.warp_solver import WarpSolver
from .policy.license_gate import LicenseGate
from .render.cv2_render import OpenCVMeshRenderer
from .render.gaussian import GaussianSplatRenderer
from .render.mesh_render import PILMeshRenderer


def _try(primary, stub, label: str, *args, **kwargs):
    try:
        return primary(*args, **kwargs)
    except ImportError as exc:
        log_event("E_ADAPTER_FALLBACK", f"{label}: {exc}")
    except NotImplementedError as exc:
        log_event("E_ADAPTER_FALLBACK", f"{label}: {exc}")
    return stub(*args, **kwargs)


def build(cfg: Config) -> dict[str, Any]:
    mode = str(cfg.get("mode", "mock"))
    gate = LicenseGate(cfg)
    gate.assert_startup_allowed(cfg)

    # Pipeline A ---------------------------------------------------------
    if mode in {"real", "hybrid"}:
        segmenter = _try(Segmenter, StubSegmenter, "Segmenter", cfg)
        geometry = _try(GeometryEstimator, StubGeometryEstimator, "GeometryEstimator", cfg)
    elif str(cfg.get("asset.geometry", "stub")) == "silhouette":
        # Pure-numpy image-driven lofting: usable in mock (no weights/network).
        segmenter = StubSegmenter(cfg)
        geometry = _try(
            SilhouetteGeometryEstimator, StubGeometryEstimator, "GeometryEstimator", cfg
        )
    else:
        segmenter = StubSegmenter(cfg)
        geometry = StubGeometryEstimator(cfg)
    repairer = MeshRepairer(cfg)
    drape = DrapeAligner(cfg)
    prior = MaterialPrior(cfg)
    registry = AssetRegistry(cfg)
    asset_builder = AssetBuilder(cfg, segmenter, geometry, repairer, drape, prior, registry)

    # Perception ---------------------------------------------------------
    if mode in {"real", "hybrid"}:
        camera = _try(RealCamera, StubCamera, "Camera", cfg)
        body_backend = str(cfg.get("perception.body_backend", "mediapipe"))
        hand_backend = str(cfg.get("perception.hand_backend", "mediapipe"))
        if hand_backend == "mediapipe":
            hand_tracker = _try(MediaPipeHandTracker, StubHandTracker, "HandTracker", cfg)
        else:
            hand_tracker = _try(HandTracker, StubHandTracker, "HandTracker", cfg)
        if body_backend == "mediapipe":
            body_tracker = _try(MediaPipeBodyTracker, StubBodyTracker, "BodyTracker", cfg)
        else:
            body_tracker = _try(BodyTracker, StubBodyTracker, "BodyTracker", cfg)
    else:
        camera = StubCamera(cfg)
        body_tracker = StubBodyTracker(cfg)
        hand_tracker = StubHandTracker(cfg)
    framing = FramingGate(cfg)
    contact = ContactExtractor(cfg)

    # Physics ------------------------------------------------------------
    physics_backend = str(cfg.get("physics.solver", "stub_xpbd"))
    if physics_backend == "warp":
        # Warp runs on CUDA *or* CPU with no weights and no network, so it is
        # selectable in every mode (mock included).
        physics = _try(WarpSolver, PhysicsSolver, "PhysicsSolver", cfg)
    elif mode in {"real", "hybrid"} and physics_backend == "diffcloth":
        physics = _try(DiffClothSolver, PhysicsSolver, "PhysicsSolver", cfg)
    else:
        physics = PhysicsSolver(cfg)

    # Rendering ----------------------------------------------------------
    backend = str(cfg.get("render.backend", "stub"))
    if backend == "opencv":
        # Local rasterisation speed-up; no weights/network ⇒ any mode.
        renderer = _try(OpenCVMeshRenderer, PILMeshRenderer, "Renderer", cfg)
    elif mode in {"real", "hybrid"} and backend in {"3dgs", "gaussian"}:
        renderer = _try(GaussianSplatRenderer, PILMeshRenderer, "Renderer", cfg)
    else:
        renderer = PILMeshRenderer(cfg)

    # Keyframe -----------------------------------------------------------
    trigger = KeyframeTrigger(cfg)
    kf_backend = str(cfg.get("keyframe.backend", "stub"))
    if mode in {"real", "hybrid"} and kf_backend in {"fashn_vton_1_5", "fashn"}:
        tryon = _try(FashnTryOn, StubKeyframeTryOn, "KeyframeTryOn", cfg)
    else:
        tryon = StubKeyframeTryOn(cfg)
    compositor = Compositor(cfg)
    hud = HUD(cfg)
    display = WindowDisplay(cfg) if mode == "real" else HeadlessDisplay(cfg)

    # Calibration --------------------------------------------------------
    collector = TrajectoryCollector(cfg)
    identifier = SystemIdentifier(cfg)
    calib_gate = CalibrationGate(cfg)

    return {
        "Segmenter": segmenter,
        "GeometryEstimator": geometry,
        "MeshRepairer": repairer,
        "DrapeAligner": drape,
        "MaterialPrior": prior,
        "AssetBuilder": asset_builder,
        "Camera": camera,
        "BodyTracker": body_tracker,
        "FramingGate": framing,
        "HandTracker": hand_tracker,
        "ContactExtractor": contact,
        "PhysicsSolver": physics,
        "Renderer": renderer,
        "KeyframeTrigger": trigger,
        "KeyframeTryOn": tryon,
        "Compositor": compositor,
        "HUD": hud,
        "Display": display,
        "TrajectoryCollector": collector,
        "SystemIdentifier": identifier,
        "CalibrationGate": calib_gate,
        "LicenseGate": gate,
        "AssetRegistry": registry,
    }


__all__ = ["build"]
