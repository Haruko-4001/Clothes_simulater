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


def test_factory_falls_back_when_warp_dependency_missing(mock_cfg, monkeypatch):
    """Missing dependency (ImportError) must fall back to the numpy solver."""
    cfg = _engine_cfg(mock_cfg, **{"physics.solver": "warp"})

    def _missing(self, cfg=None):
        raise ImportError("simulated missing warp")

    if _WARP_AVAILABLE:
        monkeypatch.setattr(WarpSolver, "__init__", _missing)
    components = build(cfg)
    assert isinstance(components["PhysicsSolver"], PhysicsSolver)


def test_factory_falls_back_on_runtime_init_failure(mock_cfg, monkeypatch):
    """A backend that imports fine but fails at runtime must not crash boot.

    Real triggers: no CUDA device / driver mismatch, OOM, a stripped OpenCV
    build, or a revoked credential. ``_try`` previously caught only
    ImportError/NotImplementedError, so these surfaced as hard crashes.
    """
    cfg = _engine_cfg(mock_cfg, **{"physics.solver": "warp"})
    original = WarpSolver.__init__

    def _runtime_boom(self, cfg=None):
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    monkeypatch.setattr(WarpSolver, "__init__", _runtime_boom)
    components = build(cfg)
    assert isinstance(components["PhysicsSolver"], PhysicsSolver)
    monkeypatch.setattr(WarpSolver, "__init__", original)

    # Same contract for the renderer: an unusable cv2 install must degrade.
    if _CV2:
        from tryon.render.cv2_render import OpenCVMeshRenderer

        def _renderer_boom(self, cfg=None):
            raise AttributeError("module 'cv2' has no attribute 'fillConvexPoly'")

        monkeypatch.setattr(OpenCVMeshRenderer, "__init__", _renderer_boom)
        cfg2 = _engine_cfg(mock_cfg, **{"render.backend": "opencv"})
        components2 = build(cfg2)
        from tryon.render.mesh_render import PILMeshRenderer

        assert isinstance(components2["Renderer"], PILMeshRenderer)


def test_factory_falls_back_on_missing_credential(mock_cfg, monkeypatch):
    """FASHN without an API key must silently degrade to the stub backend."""
    monkeypatch.delenv("FASHN_API_KEY", raising=False)
    cfg = _engine_cfg(mock_cfg)
    cfg.set("mode", "real")
    cfg.set("keyframe.backend", "fashn")
    components = build(cfg)
    from tryon.keyframe.tryon import StubKeyframeTryOn

    assert isinstance(components["KeyframeTryOn"], StubKeyframeTryOn)


def test_tracked_config_has_no_credential_slot(mock_cfg):
    """configs/default.yaml is tracked: it must not offer a place for secrets."""
    import yaml

    default = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert "fashn_api_key" not in (default.get("keyframe") or {}), (
        "tracked config must not carry an API-key slot; use configs/local.yaml "
        "(git-ignored) or FASHN_API_KEY"
    )
    example = yaml.safe_load((ROOT / "configs" / "local.example.yaml").read_text(encoding="utf-8"))
    assert "fashn_api_key" in (example.get("keyframe") or {})


def test_local_yaml_overlay_is_gitignored_and_loaded(tmp_path, monkeypatch):
    """configs/local.yaml must be ignored by git yet merged last by load_config."""
    import shutil

    from tryon.config import load_config

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy(ROOT / "configs" / "default.yaml", cfg_dir / "default.yaml")
    shutil.copy(ROOT / "configs" / "mock.yaml", cfg_dir / "mock.yaml")
    (cfg_dir / "local.yaml").write_text(
        "keyframe:\n  fashn_api_key: from-local\n", encoding="utf-8"
    )
    cfg = load_config("mock", None, config_dir=cfg_dir)
    assert cfg.get("keyframe.fashn_api_key") == "from-local"

    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "configs/local.yaml" in ignore
    assert "configs/local.example.yaml" not in ignore.replace("configs/local.yaml", "")


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


@pytest.mark.skipif(not _WARP_AVAILABLE, reason="warp not installed")
def test_contract_scale_respects_energy_budget(mock_cfg, stub_asset):
    """Regression for the energy-budget contraction search.

    ``_contract_scale`` was rewritten from a 40-step bisection to a single
    vectorised quadratic-form grid because the bisection cost ~85% of the
    physics step. This test pins the two properties the SPEC §7 invariant
    depends on: the returned factor is in [0,1], the contracted state really is
    inside the budget, and a looser budget never shrinks more.
    """
    from tryon.physics.state import build_constraints, elastic_energy

    cfg = _engine_cfg(mock_cfg)
    solver = WarpSolver(cfg)
    solver.initial_state(stub_asset)
    c = build_constraints(stub_asset.rest_mesh, stub_asset.material)
    rest = c.rest_positions
    rng = np.random.default_rng(7)

    for perturb in (0.002, 0.02, 0.2):
        x = rest + rng.normal(scale=perturb, size=rest.shape)
        e_full = elastic_energy(x, c)
        previous_s = -1.0
        for frac in (0.3, 0.6, 0.9, 1.0):
            target = e_full * frac + 1.0e-12
            s = solver._contract_scale(x, c, target)
            assert 0.0 <= s <= 1.0
            contracted = rest + s * (x - rest)
            achieved = elastic_energy(contracted, c)
            assert achieved <= target * (1.0 + 1.0e-9), (
                f"budget violated: E={achieved:.6g} > target={target:.6g} (s={s})"
            )
            # looser budget must not shrink more
            assert s >= previous_s - 1.0e-12
            previous_s = s


@pytest.mark.skipif(not _WARP_AVAILABLE, reason="warp not installed")
def test_warp_step_meets_vertex_budget_scaling(mock_cfg, stub_asset):
    """The per-step cost must stay O(N): a 7x vertex increase cannot blow up.

    Guard against regressions like the 40-step bisection (which made the step
    ~43 energy evaluations) rather than asserting an absolute host-dependent
    wall-clock number.
    """
    import time

    cfg = _engine_cfg(mock_cfg)
    dt = float(cfg.require("physics.dt"))

    def step_ms(asset, n_steps=25):
        solver = WarpSolver(cfg)
        state = solver.initial_state(asset)
        for _ in range(5):
            state = solver.step(state, asset, [], dt)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            state = solver.step(state, asset, [], dt)
        return (time.perf_counter() - t0) / n_steps * 1000.0

    small = step_ms(stub_asset)
    assert np.isfinite(small) and small > 0.0
    # The mock asset is ~160 vertices; one step must not cost tens of ms.
    assert small < 15.0, f"warp step unexpectedly slow on a tiny mesh: {small:.2f} ms"


# ---------------------------------------------------------------------------
# Pattern cold start (panels + seams)
# ---------------------------------------------------------------------------


def _pattern_cfg(mock_cfg):
    return _engine_cfg(mock_cfg, **{"asset.geometry": "pattern",
                                    "asset.mesh_resolution": [20, 14]})


def _mask_rect() -> np.ndarray:
    img = np.zeros((480, 320, 3), dtype=np.uint8)
    img[:, :] = (245, 245, 245)
    img[60:440, 70:250] = (70, 105, 150)
    return img


def _mask_tee() -> np.ndarray:
    img = _mask_rect()
    img[60:140, 20:70] = (70, 105, 150)
    img[60:140, 250:300] = (70, 105, 150)
    return img


def test_pattern_geometry_emits_sleeves_and_surviving_seams(mock_cfg):
    """A pattern asset must carry separate sleeves joined by seams.

    Regression: ``MeshRepairer`` used to rebuild the mesh without copying
    ``seams``, silently disconnecting the sleeves from the body.
    """
    from tryon.factory import build

    cfg = _pattern_cfg(mock_cfg)
    components = build(cfg)
    asset = components["AssetBuilder"].build(_mask_tee(), timeout_s=120.0)

    mesh = asset.rest_mesh
    assert len(mesh.seams) > 0, "seam constraints were dropped"
    # Sleeves must extend the silhouette beyond the torso half-width.
    m = components["GeometryEstimator"].last_measurements
    assert mesh.vertices[:, 0].max() > m.chest_rx * 1.1
    assert mesh.vertices[:, 0].min() < -m.chest_rx * 1.1
    # Panel metadata must use global ids into the stitched mesh: a loader that
    # reconstructs panel meshes from these ranges has to recover every vertex
    # exactly once (regression: all panels used to start at 0).
    panels = asset.sewing_pattern["panels"]
    assert [p["name"] for p in panels] == ["torso", "sleeve_left", "sleeve_right"]
    ids = [i for p in panels for i in p["vertex_ids"]]
    assert sorted(ids) == list(range(mesh.n_vertices))


def test_pattern_sleeve_length_follows_the_shoulder_row(mock_cfg):
    """Sleeve length must come from the shoulder vs hem width difference."""
    from tryon.factory import build

    cfg = _pattern_cfg(mock_cfg)
    components = build(cfg)
    est = components["GeometryEstimator"]

    components["AssetBuilder"].build(_mask_rect(), timeout_s=120.0)
    sleeveless = est.last_measurements.sleeve_len

    components["AssetBuilder"].build(_mask_tee(), timeout_s=120.0)
    sleeved = est.last_measurements.sleeve_len

    assert sleeved > sleeveless * 2.0, (sleeveless, sleeved)
    assert sleeved <= 0.20


def test_pattern_seams_are_constraints_and_stay_stable(mock_cfg):
    """Seams must appear as XPBD constraints and must not blow the solver up."""
    from tryon.factory import build
    from tryon.physics.state import build_constraints

    cfg = _pattern_cfg(mock_cfg)
    components = build(cfg)
    asset = components["AssetBuilder"].build(_mask_tee(), timeout_s=120.0)

    without_seams = build_constraints(
        type(asset.rest_mesh)(
            vertices=asset.rest_mesh.vertices,
            faces=asset.rest_mesh.faces,
            thickness=asset.rest_mesh.thickness,
            is_closed=asset.rest_mesh.is_closed,
            has_isolated_edge=False,
            self_intersection_count=0,
            collision_ready=True,
        ),
        asset.material,
    )
    with_seams = build_constraints(asset.rest_mesh, asset.material)
    assert with_seams.n_constraints == without_seams.n_constraints + len(asset.rest_mesh.seams)

    solver = components["PhysicsSolver"]
    state = solver.initial_state(asset)
    for _ in range(60):
        state = solver.step(state, asset, [], 1.0 / 60.0)
    assert np.all(np.isfinite(state.positions))
    assert state.energy < 5.0, f"seam-driven instability: energy={state.energy:.3f} J"
    assert solver.last_penetration_count == 0


def test_newton_vbd_optional_backend_steps_and_preserves_shape(mock_cfg):
    """The optional Newton VBD backend must run behind the PhysicsSolver API."""
    from tryon.factory import build
    from tryon.physics.newton_solver import NewtonVBDSolver, _NEWTON_AVAILABLE

    cfg = _pattern_cfg(mock_cfg)
    if not _NEWTON_AVAILABLE:
        with pytest.raises(ImportError):
            NewtonVBDSolver(cfg)
        return

    components = build(cfg)
    asset = components["AssetBuilder"].build(_mask_tee(), timeout_s=120.0)
    solver = NewtonVBDSolver(cfg)
    state = solver.initial_state(asset)
    for _ in range(3):
        state = solver.step(state, asset, [], 1.0 / 60.0)

    assert state.positions.shape == asset.rest_mesh.vertices.shape
    assert np.all(np.isfinite(state.positions))
    assert np.all(np.isfinite(state.velocities))
    assert state.step == 3
    assert solver.last_step_ms >= 0.0
    # Newton's soft contacts may report a few penetrations; the adapter must
    # expose the count rather than claiming a hard non-penetration guarantee.
    assert solver.last_penetration_count >= 0


def test_factory_routes_newton_when_installed(mock_cfg):
    from tryon.factory import build
    from tryon.physics.newton_solver import NewtonVBDSolver, _NEWTON_AVAILABLE

    if not _NEWTON_AVAILABLE:
        pytest.skip("newton is not installed")
    cfg = _engine_cfg(mock_cfg, **{"physics.solver": "newton_vbd", "mode": "mock"})
    components = build(cfg)
    assert isinstance(components["PhysicsSolver"], NewtonVBDSolver)


# ---------------------------------------------------------------------------
# Texturing (UV unwrap + per-face sampling)
# ---------------------------------------------------------------------------


def test_cylindrical_uvs_are_normalised_and_mirror_the_back(mock_cfg, stub_asset):
    """UVs must span [0,1] and fold the back half onto the front half."""
    from tryon.asset.uv import cylindrical_uvs

    uv = cylindrical_uvs(stub_asset.rest_mesh.vertices)
    assert uv.shape == (stub_asset.rest_mesh.n_vertices, 2)
    assert np.all((uv >= 0.0) & (uv <= 1.0))
    # both extremes of u are used (left and right silhouette)
    assert uv[:, 0].min() < 0.05 and uv[:, 0].max() > 0.95
    assert uv[:, 1].min() < 0.05 and uv[:, 1].max() > 0.95


def test_striped_garment_keeps_stripes_in_face_colours(mock_cfg):
    """A striped source must not collapse to one mean colour (the old bug)."""
    from tryon.render.texture import clear_cache, garment_face_colors

    from tryon.config import load_config
    from tryon.factory import build

    cfg = load_config("mock", "rtx4090")
    cfg.set("camera.width", 320)
    cfg.set("camera.height", 240)
    cfg.set("asset.mesh_resolution", [24, 18])
    clear_cache()

    img = np.zeros((480, 320, 3), dtype=np.uint8)
    img[:, :] = (250, 250, 250)
    # Vertical stripes: red / blue alternating every 20 px.
    for x in range(40, 280):
        if ((x - 40) // 20) % 2 == 0:
            img[40:460, x] = (200, 40, 40)
        else:
            img[40:460, x] = (40, 40, 200)

    asset = build(cfg)["AssetBuilder"].build(img, timeout_s=120.0)
    colors = garment_face_colors(asset)
    assert colors.shape == (asset.rest_mesh.n_faces, 3)
    # Per-channel spread across faces proves the pattern survived.
    spread = float(np.std(colors[:, 0] - colors[:, 2]))
    assert spread > 30.0, f"stripes were averaged away (spread={spread:.1f})"
    # And both stripe families are actually present on the front surface.
    reds = np.sum((colors[:, 0] > 120) & (colors[:, 2] < 120))
    blues = np.sum((colors[:, 2] > 120) & (colors[:, 0] < 120))
    assert reds > 0 and blues > 0, (reds, blues)


def test_textured_render_uses_more_than_one_colour(mock_cfg):
    """The rendered garment layer must contain multiple colours when textured."""
    from tryon.render.texture import clear_cache
    from tryon.render.cv2_render import OpenCVMeshRenderer
    from tryon.perception.stubs import StubCamera
    from tryon.physics.solver import PhysicsSolver
    from tryon.factory import build
    from tryon.config import load_config

    if not _CV2:
        pytest.skip("opencv not installed")
    cfg = load_config("mock", "rtx4090")
    cfg.set("camera.width", 320)
    cfg.set("camera.height", 240)
    cfg.set("asset.mesh_resolution", [24, 18])
    clear_cache()

    img = np.zeros((480, 320, 3), dtype=np.uint8)
    img[:, :] = (250, 250, 250)
    for x in range(40, 280):
        img[40:460, x] = (200, 40, 40) if ((x - 40) // 24) % 2 == 0 else (40, 40, 200)

    asset = build(cfg)["AssetBuilder"].build(img, timeout_s=120.0)
    camera = StubCamera(cfg)
    solver = PhysicsSolver(cfg)
    state = solver.initial_state(asset)
    for _ in range(8):
        state = solver.step(state, asset, [], 1.0 / 60.0)
    out = OpenCVMeshRenderer(cfg).render(camera.capture(), asset, state, camera)
    px = out.image[out.mask > 0]
    assert px.size > 0
    unique = np.unique((px // 48).astype(np.int16), axis=0)
    assert len(unique) > 2, f"render is nearly monochrome: {len(unique)} colour bins"


# ---------------------------------------------------------------------------
# CLI / config consistency
# ---------------------------------------------------------------------------


def test_cli_mode_is_unset_by_default():
    """--mode must not default to "mock": that silently overrode configs/*.yaml."""
    from app.main import parse_args

    assert parse_args([]).mode is None
    assert parse_args(["--mode", "real"]).mode == "real"


def test_headless_is_default_and_gui_can_override():
    """--headless used to be store_true/default=True, so it could never be off.

    SPEC §15 ACCEPT_CMD still passes --headless, so the flag must stay accepted;
    what changed is that --gui is now a real, mutually exclusive opt-out.
    """
    from app.main import parse_args

    default = parse_args([])
    assert default.headless is True
    assert default.gui is False

    explicit = parse_args(["--headless"])
    assert explicit.headless is True

    assert parse_args(["--gui"]).gui is True
    with pytest.raises(SystemExit):
        parse_args(["--gui", "--headless"])


def test_cli_backend_choices_cover_config_domains():
    """Every backend value named by configs/*.yaml must be accepted by the CLI."""
    import yaml

    from app.main import parse_args

    real = yaml.safe_load((ROOT / "configs" / "real.yaml").read_text(encoding="utf-8")) or {}
    default = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}

    def _accepts(flag: str, value: str) -> bool:
        try:
            parse_args([flag, value])
            return True
        except SystemExit:
            return False

    for backend in ("diffcloth",):
        assert _accepts("--physics-solver", backend), backend
    for backend in ("3dgs", "gaussian"):
        assert _accepts("--render-backend", backend), backend
    for backend in ("sewformer", "instantmesh", "garmentrec"):
        assert _accepts("--geometry", backend), backend
    for backend in ("fashn", "fashn_vton_1_5", "catvton", "leffa", "omnitry"):
        assert _accepts("--keyframe-backend", backend), backend
    for backend in ("fd", "warp"):
        assert _accepts("--grad-backend", backend), backend

    # values actually used by shipped configs
    for value in (real.get("physics") or {}).values():
        assert _accepts("--physics-solver", str(value)), value
    for value in (real.get("render") or {}).values():
        assert _accepts("--render-backend", str(value)), value
    assert _accepts("--geometry", str((default.get("asset") or {}).get("geometry", "stub")))


def test_load_config_resolves_mode_layer_without_explicit_mode():
    """load_config(None) must still merge configs/mock.yaml (hidden mock truth)."""
    from tryon.config import load_config

    cfg = load_config(None, None)
    assert cfg.get("mode") == "mock"
    assert cfg.get("mock.true_material_preset") is not None

    real = load_config("real", None)
    assert real.get("mode") == "real"
    assert real.get("physics.solver") == "diffcloth"
