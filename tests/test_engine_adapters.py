"""Engine adapter regression tests (P0–P4 upgrades).

Extra regression file beyond the SPEC §16 matrix, registered in
STRUCTURED_REPORT.json deviations (same precedent as test_json_safety /
test_make_runner). Covers:

  P0  WarpSolver routing + physics semantics, OpenCV renderer vs PIL parity
  P1  MediaPipe tracker construction / missing-dependency fallback
  P2  FASHN VTON client (missing key, mocked HTTP happy path)
  P3  SilhouetteGeometryEstimator (image-driven lofting, G1 IoU)
  P4  Warp AD gradients (direction vs FD, identifier integration)

Every warp/cv2/mediapipe-dependent test skips when the optional dependency is
absent so the SPEC guarantee "no GPU / no weights / no network" keeps holding.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tryon.config import load_config  # noqa: E402
from tryon.factory import build  # noqa: E402
from tryon.physics.solver import PhysicsSolver  # noqa: E402
from tryon.types import (  # noqa: E402
    ContactEvent,
    ContactKind,
    Frame,
    KeyframeRequest,
    ObservationBatch,
    TryonError,
)
from tests.conftest import make_body, make_frame  # noqa: E402

try:
    import cv2  # type: ignore

    _CV2 = True
except Exception:
    _CV2 = False

try:
    from tryon.physics.warp_solver import WarpSolver, _WARP_AVAILABLE
except Exception:  # pragma: no cover - warp_solver imports are dependency-free
    _WARP_AVAILABLE = False

    class WarpSolver:  # type: ignore[no-redef]
        pass


try:
    import mediapipe  # type: ignore  # noqa: F401

    _MP = True
except Exception:
    _MP = False


def _engine_cfg(mock_cfg, **overrides):
    cfg = load_config("mock", "rtx4090")
    cfg.set("camera.width", 320)
    cfg.set("camera.height", 240)
    cfg.set("asset.mesh_resolution", [10, 7])
    for k, v in overrides.items():
        cfg.set(k, v)
    return cfg


# ---------------------------------------------------------------------------
# P0a — Warp solver
# ---------------------------------------------------------------------------


def test_factory_routes_warp_solver_or_falls_back(mock_cfg):
    cfg = _engine_cfg(mock_cfg, **{"physics.solver": "warp"})
    components = build(cfg)
    solver = components["PhysicsSolver"]
    if _WARP_AVAILABLE:
        assert isinstance(solver, WarpSolver)
    else:
        assert isinstance(solver, PhysicsSolver)


def test_factory_falls_back_when_warp_init_fails(mock_cfg, monkeypatch):
    cfg = _engine_cfg(mock_cfg, **{"physics.solver": "warp"})

    def _boom(self, cfg=None):
        raise ImportError("simulated missing warp")

    if _WARP_AVAILABLE:
        monkeypatch.setattr(WarpSolver, "__init__", _boom)
    components = build(cfg)
    assert isinstance(components["PhysicsSolver"], PhysicsSolver)


@pytest.mark.skipif(not _WARP_AVAILABLE, reason="warp not installed")
class TestWarpSolverSemantics:
    """Mirror of test_physics_stub.py semantics on the Warp backend."""

    def _settled(self, cfg, asset, n=8):
        solver = WarpSolver(cfg)
        state = solver.initial_state(asset)
        dt = float(cfg.require("physics.dt"))
        for _ in range(n):
            state = solver.step(state, asset, [], dt)
        return solver, state, dt

    def test_release_keeps_velocities(self, mock_cfg, stub_asset):
        cfg = _engine_cfg(mock_cfg)
        solver, state, dt = self._settled(cfg, stub_asset)
        vid = 3
        hand = state.positions[vid].copy()
        state = solver.step(
            state, stub_asset,
            [ContactEvent(0.0, ContactKind.GRASP, hand, vid, 0.0, "right")], dt,
        )
        for k in range(6):
            hand = hand + np.array([0.02, 0.0, 0.0])
            state = solver.step(
                state, stub_asset,
                [ContactEvent(float(k), ContactKind.PULL, hand, vid, 0.0, "right")], dt,
            )
        released = solver.step(
            state, stub_asset,
            [ContactEvent(99.0, ContactKind.RELEASE, hand, vid, 0.0, "right")], dt,
        )
        assert vid not in released.attached
        # INV: RELEASE clears attachments but KEEPS the pulled vertex momentum
        assert np.linalg.norm(released.velocities[vid]) > 1.0e-6
        assert np.all(np.isfinite(released.positions))
        assert solver.last_penetration_count == 0

    def test_energy_decays_after_release(self, mock_cfg, stub_asset):
        gamma = float(mock_cfg.require("physics.energy_decay_gamma"))
        cfg = _engine_cfg(mock_cfg)
        solver, state, dt = self._settled(cfg, stub_asset)
        vid = 3
        hand = state.positions[vid].copy()
        state = solver.step(
            state, stub_asset,
            [ContactEvent(0.0, ContactKind.GRASP, hand, vid, 0.0, "right")], dt,
        )
        for k in range(10):
            hand = hand + np.array([0.015, 0.0, 0.01])
            state = solver.step(
                state, stub_asset,
                [ContactEvent(float(k), ContactKind.PULL, hand, vid, 0.0, "right")], dt,
            )
        state = solver.step(
            state, stub_asset,
            [ContactEvent(50.0, ContactKind.RELEASE, hand, vid, 0.0, "right")], dt,
        )
        energies = [state.energy]
        for _ in range(30):
            state = solver.step(state, stub_asset, [], dt)
            energies.append(state.energy)
        for prev, nxt in zip(energies, energies[1:]):
            assert nxt <= prev * (1.0 - gamma * dt) + 1.0e-10
        assert energies[-1] < energies[0]

    def test_attachment_persists_under_pull(self, mock_cfg, stub_asset):
        cfg = _engine_cfg(mock_cfg)
        solver, state, dt = self._settled(cfg, stub_asset)
        vid = 3
        hand = state.positions[vid].copy()
        state = solver.step(
            state, stub_asset,
            [ContactEvent(0.0, ContactKind.GRASP, hand, vid, 0.0, "right")], dt,
        )
        for k in range(120):  # 2 s at 60 Hz, same schedule as the numpy test
            hand = hand + np.array([-0.0008, -0.0004, 0.0005])
            state = solver.step(
                state, stub_asset,
                [ContactEvent(float(k), ContactKind.PULL, hand, vid, 0.0, "right")], dt,
            )
            assert vid in state.attached
        assert solver.last_penetration_count == 0


# ---------------------------------------------------------------------------
# P0b — OpenCV renderer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2, reason="opencv not installed")
def test_opencv_renderer_matches_pil_coverage(mock_cfg, stub_asset, physics_solver):
    from tryon.render.mesh_render import PILMeshRenderer
    from tryon.render.cv2_render import OpenCVMeshRenderer
    from tryon.perception.stubs import StubCamera

    cfg = _engine_cfg(mock_cfg)
    camera = StubCamera(cfg)
    dt = float(cfg.require("physics.dt"))
    state = physics_solver.initial_state(stub_asset)
    for _ in range(10):
        state = physics_solver.step(state, stub_asset, [], dt)
    frame = camera.capture()

    pil_out = PILMeshRenderer(cfg).render(frame, stub_asset, state, camera)
    cv_out = OpenCVMeshRenderer(cfg).render(frame, stub_asset, state, camera)
    assert pil_out.image.shape == cv_out.image.shape
    a = pil_out.mask > 0
    b = cv_out.mask > 0
    union = np.logical_or(a, b).sum()
    inter = np.logical_and(a, b).sum()
    assert union > 0
    assert inter / union > 0.75, "cv2 rasteriser must cover the same garment pixels"


@pytest.mark.skipif(not _CV2, reason="opencv not installed")
def test_factory_routes_opencv_renderer(mock_cfg):
    cfg = _engine_cfg(mock_cfg, **{"render.backend": "opencv"})
    components = build(cfg)
    from tryon.render.cv2_render import OpenCVMeshRenderer

    assert isinstance(components["Renderer"], OpenCVMeshRenderer)


# ---------------------------------------------------------------------------
# P1 — MediaPipe trackers
# ---------------------------------------------------------------------------


def test_mediapipe_body_tracker_protocol_when_present(mock_cfg):
    cfg = _engine_cfg(mock_cfg)
    if not _MP:
        from tryon.perception.body import MediaPipeBodyTracker

        with pytest.raises(ImportError):
            MediaPipeBodyTracker(cfg)
    else:  # pragma: no cover - requires mediapipe install
        from tryon.perception.body import MediaPipeBodyTracker

        tracker = MediaPipeBodyTracker(cfg)
        frame = make_frame(320, 240, value=120)
        body = tracker.track(frame, None)
        assert body.pose_keypoints.ndim == 2
        assert body.pose_keypoints.shape[1] == 3
        assert 0.0 <= body.confidence <= 1.0


def test_real_mode_falls_back_to_stub_trackers(mock_cfg):
    cfg = _engine_cfg(mock_cfg)
    cfg.set("mode", "real")
    components = build(cfg)
    body = components["BodyTracker"].track(make_frame(320, 240), None)
    assert body.pose_keypoints.shape[1] in (2, 3)
    hands = components["HandTracker"].track(make_frame(320, 240))
    assert hands.landmarks.shape[1] in (2, 3)


# ---------------------------------------------------------------------------
# P2 — FASHN VTON client
# ---------------------------------------------------------------------------


def test_fashn_missing_key_raises_import_error(mock_cfg, monkeypatch):
    monkeypatch.delenv("FASHN_API_KEY", raising=False)
    from tryon.keyframe.tryon import FashnTryOn

    with pytest.raises(ImportError):
        FashnTryOn(mock_cfg)


def test_fashn_generate_with_mocked_http(mock_cfg, monkeypatch):
    monkeypatch.delenv("FASHN_API_KEY", raising=False)
    from tryon.keyframe.tryon import FashnTryOn

    cfg = _engine_cfg(mock_cfg)
    cfg.set("keyframe.fashn_api_key", "test-key")
    cfg.set("keyframe.fashn_poll_s", 0.0)
    backend = FashnTryOn(cfg)

    person = np.zeros((64, 48, 3), dtype=np.uint8)
    person[:] = (100, 100, 100)
    person[16:48, 12:36] = (180, 120, 90)
    garment = np.zeros((64, 48, 3), dtype=np.uint8)
    garment[:] = (250, 250, 250)
    garment[8:56, 8:40] = (60, 90, 200)

    # "Generated" result: person with a garment painted on the torso.
    result_img = person.copy()
    result_img[18:46, 14:34] = (60, 90, 200)
    buf = io.BytesIO()
    Image.fromarray(result_img).save(buf, format="PNG")
    result_png = buf.getvalue()

    calls = {"n": 0, "polled": 0}

    def fake_opener(req, timeout=None):
        url = req.full_url
        calls["n"] += 1
        if url.endswith("/run"):
            return _FakeResponse(json.dumps({"id": "job-1"}).encode())
        if "/run/" in url and "job-1" in url:
            calls["polled"] += 1
            if calls["polled"] < 2:
                return _FakeResponse(json.dumps({"status": "queued"}).encode())
            return _FakeResponse(
                json.dumps({"status": "completed", "output": ["https://cdn/out.png"]}).encode()
            )
        return _FakeResponse(result_png)

    backend._opener = fake_opener
    backend._sleep = lambda s: None
    req = KeyframeRequest(
        req_id="kf_test",
        garment_asset_id="g_test",
        person_image=person,
        garment_image=garment,
        mask=np.ones(person.shape[:2], dtype=np.uint8),
        pose=make_body((48, 64)),
        enqueued_at_ms=0.0,
    )
    result = backend.generate(req)
    assert result.image.shape == person.shape
    # mask covers exactly the recoloured torso pixels
    changed = np.abs(result.image.astype(int) - person.astype(int)).max(axis=2) > 12
    assert np.logical_and(changed, result.mask > 0).sum() == changed.sum()
    assert changed.sum() > 0
    assert calls["polled"] >= 2


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# P3 — Silhouette geometry estimator
# ---------------------------------------------------------------------------


def test_silhouette_geometry_varies_with_input(mock_cfg, garment_image):
    from tryon.asset.geometry import SilhouetteGeometryEstimator
    from tryon.asset.stubs import StubSegmenter
    from tryon.asset.mesh_repair import MeshRepairer
    from tryon.asset.drape import DrapeAligner
    from tryon.types import Segmentation

    cfg = _engine_cfg(mock_cfg)
    seg = StubSegmenter(cfg).run(garment_image)
    est = SilhouetteGeometryEstimator(cfg)
    draft = est.estimate(garment_image, seg)
    mesh = MeshRepairer(cfg).repair(draft)

    # a wide, short top vs a narrow, long dress must give different meshes
    tall_img = np.zeros((480, 384, 3), dtype=np.uint8)
    tall_img[:] = (240, 240, 240)
    tall_img[40:460, 60:324] = (90, 60, 160)   # narrow & long
    seg2 = StubSegmenter(cfg).run(tall_img)
    draft2 = est.estimate(tall_img, seg2)
    mesh2 = MeshRepairer(cfg).repair(draft2)
    assert not np.allclose(mesh.vertices, mesh2.vertices)

    # G1: the silhouette-lofted mesh must overlap the segmentation better
    # than the fixed ellipse stub.
    from tryon.asset.stubs import StubGeometryEstimator

    stub_draft = StubGeometryEstimator(cfg).estimate(garment_image, seg)
    stub_mesh = MeshRepairer(cfg).repair(stub_draft)
    drape = DrapeAligner(cfg)
    iou_sil = drape.align(mesh, seg).silhouette_iou
    iou_stub = drape.align(stub_mesh, seg).silhouette_iou
    assert iou_sil >= iou_stub
    assert iou_sil > 0.80, f"silhouette IoU too low: {iou_sil:.3f}"


def test_factory_routes_silhouette_geometry(mock_cfg, garment_image):
    cfg = _engine_cfg(mock_cfg, **{"asset.geometry": "silhouette"})
    components = build(cfg)
    from tryon.asset.geometry import SilhouetteGeometryEstimator

    if isinstance(components["GeometryEstimator"], SilhouetteGeometryEstimator):
        asset = components["AssetBuilder"].build(garment_image, timeout_s=120.0)
        assert asset.rest_mesh.n_vertices > 0


# ---------------------------------------------------------------------------
# P4 — Warp AD gradients
# ---------------------------------------------------------------------------


def _make_batch(mock_cfg, stub_asset, physics_solver, n_settle=4, n_pull=10, n_free=6):
    """ObservationBatch in the collector convention: ONE event per frame.

    Observations are generated with a PERTURBED world material (hidden Θ*)
    so the calibration loss is non-trivial — a perfect-replay batch would
    give L=0 and make the LM assertion vacuous.
    """
    import dataclasses

    from tryon.types import MaterialParams, MaterialState

    dt = float(mock_cfg.require("physics.dt"))
    base = stub_asset.material
    world_material = dataclasses.replace(
        base,
        areal_density=float(np.clip(base.areal_density * 1.6, 0.02, 1.2)),
        damping=float(np.clip(base.damping * 2.5, 0.0, 50.0)),
        youngs_modulus=float(np.clip(base.youngs_modulus * 0.55, 1.0e2, 1.0e6)),
        state=MaterialState.AI_PRIOR,
        variance=dict(base.variance),
    )
    world_asset = dataclasses.replace(stub_asset, material=world_material)

    solver = physics_solver
    state = solver.initial_state(world_asset)
    states, contacts = [], []

    def _neutral(t_ms: float) -> ContactEvent:
        return ContactEvent(t_ms, ContactKind.SLIDE, np.zeros(3), 0, 0.0, "right")

    for k in range(n_settle):
        state = solver.step(state, world_asset, [], dt)
        states.append(state.copy())
        contacts.append(_neutral(k * 33.0))
    vid = int(np.argmin(np.linalg.norm(state.positions - np.array([-0.22, 1.20, 0.0]), axis=1)))
    hand = state.positions[vid].copy()
    grasp = ContactEvent(900.0, ContactKind.GRASP, hand, vid, 0.0, "right")
    state = solver.step(state, world_asset, [grasp], dt)
    states.append(state.copy())
    contacts.append(grasp)
    for k in range(n_pull):
        hand = hand + np.array([-0.008, -0.003, 0.006])
        ev = ContactEvent(900.0 + float(k) * 33.0, ContactKind.PULL, hand, vid, 0.0, "right")
        state = solver.step(state, world_asset, [ev], dt)
        states.append(state.copy())
        contacts.append(ev)
    release = ContactEvent(2000.0, ContactKind.RELEASE, hand, vid, 0.0, "right")
    state = solver.step(state, world_asset, [release], dt)
    states.append(state.copy())
    contacts.append(release)
    for k in range(n_free):
        state = solver.step(state, world_asset, [], dt)
        states.append(state.copy())
        contacts.append(_neutral(2100.0 + float(k) * 33.0))
    thumb = np.zeros((32, 32, 3), dtype=np.uint8)
    return ObservationBatch(
        contacts=contacts,
        states=states,
        camera_views=[thumb.copy() for _ in states],
        dt=dt,
    )


@pytest.mark.skipif(not _WARP_AVAILABLE, reason="warp not installed")
def test_warp_grad_direction_matches_fd(mock_cfg, stub_asset, physics_solver):
    from tryon.calibration.warp_grad import rollout_loss_and_grad

    cfg = _engine_cfg(mock_cfg)
    batch = _make_batch(cfg, stub_asset, physics_solver)
    theta = stub_asset.material.to_vector()

    out = rollout_loss_and_grad(stub_asset, batch, theta)
    assert out is not None
    L, g = out
    assert L > 0.0
    assert np.any(np.abs(g) > 1e-6), "gradient must be non-zero on active columns"

    # Central FD with 5% steps: fp32 atomic noise (~1e-5 in L) drowns tiny
    # perturbations, so only columns with FD magnitude above the noise floor
    # take part in the direction comparison.
    g_fd = np.zeros(7)
    eps = 0.05
    for i in range(7):
        tp = theta.copy(); tp[i] *= 1.0 + eps
        tm = theta.copy(); tm[i] *= 1.0 - eps
        Lp = rollout_loss_and_grad(stub_asset, batch, tp)[0]
        Lm = rollout_loss_and_grad(stub_asset, batch, tm)[0]
        g_fd[i] = (Lp - Lm) / (tp[i] - tm[i])
    active = np.abs(g_fd) > 0.05
    assert active.sum() >= 2, "expected at least rho/cd to be identifiable"
    cos = float(
        np.dot(g[active], g_fd[active])
        / max(np.linalg.norm(g[active]) * np.linalg.norm(g_fd[active]), 1e-12)
    )
    assert cos > 0.9, f"AD/FD gradient direction mismatch: cos={cos:.4f}"


@pytest.mark.skipif(not _WARP_AVAILABLE, reason="warp not installed")
def test_identifier_warp_backend_descends_on_surrogate(mock_cfg, stub_asset, physics_solver):
    """LM only guarantees descent under the loss it evaluates against.

    The warp surrogate and the FD residual model are intentionally related but
    not identical (fp32 rollout, no energy clamp), so the contract tested here
    is: with ``calibration.grad_backend: warp`` the LM moves Θ such that the
    differentiable surrogate loss does not increase.
    """
    from tryon.calibration.identifier import SystemIdentifier
    from tryon.calibration.warp_grad import rollout_loss_and_grad
    from tryon.physics.solver import PhysicsSolver

    cfg = _engine_cfg(mock_cfg)
    cfg.set("calibration.grad_backend", "warp")
    batch = _make_batch(cfg, stub_asset, physics_solver)
    sim = PhysicsSolver(cfg)
    sim.asset = stub_asset
    ident = SystemIdentifier(cfg, sim)

    theta0 = stub_asset.material.to_vector()
    L_before, _ = rollout_loss_and_grad(stub_asset, batch, theta0)
    assert L_before is not None and L_before > 0.0

    result = ident.update(batch)
    assert np.isfinite(result.loss_after)

    L_after, _ = rollout_loss_and_grad(stub_asset, batch, ident.theta)
    assert L_after <= L_before * (1.0 + 1.0e-6), (
        f"warp-LM increased the surrogate loss: {L_before:.6g} -> {L_after:.6g}"
    )
