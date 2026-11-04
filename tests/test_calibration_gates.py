from types import SimpleNamespace

import numpy as np
import pytest

from tryon.calibration.collector import TrajectoryCollector
from tryon.calibration.gates import CalibrationGate
from tryon.calibration.identifier import SystemIdentifier
from tryon.factory import build
from tryon.types import (
    CalibHistory,
    ContactEvent,
    ContactKind,
    GateVerdict,
    GeometryProvenance,
    IdentificationResult,
    MaterialParams,
    ObservationBatch,
    PhysicsState,
)


def _state(energy=0.0) -> PhysicsState:
    return PhysicsState(
        positions=np.zeros((4, 3)),
        velocities=np.zeros((4, 3)),
        attached={},
        lam=np.zeros(1),
        step=0,
        energy=energy,
    )


def _result(loss_before=10.0, loss_after=1.0, cond=100.0, predicts=True) -> IdentificationResult:
    return IdentificationResult(
        params=MaterialParams(),
        loss_before=loss_before,
        loss_after=loss_after,
        param_variance={"youngs_modulus": 1.0},
        fisher_cond=cond,
        predicts_next_frame=predicts,
    )


def test_light_shake_is_not_excited(mock_cfg):
    c = TrajectoryCollector(mock_cfg)
    for i in range(12):
        ev = ContactEvent(
            t_ms=i * 33.0,
            kind=ContactKind.PULL,
            world_pos=np.array([i * 0.0005, 0.0, 0.0]),
            vertex_id=0,
            force_proxy=0.0,
            hand_id="right",
        )
        c.observe(ev, _state())
    assert c.is_excited() is False
    assert c.last_mask == []
    assert c.last_reason == "light_shake"


def test_single_axis_pull_is_excited(mock_cfg):
    c = TrajectoryCollector(mock_cfg)
    for i in range(10):
        ev = ContactEvent(
            t_ms=i * 33.0,
            kind=ContactKind.PULL,
            world_pos=np.array([i * 0.04, 0.0, 0.0]),
            vertex_id=0,
            force_proxy=0.0,
            hand_id="right",
        )
        c.observe(ev, _state(energy=1.0 - 0.01 * i))
    assert c.is_excited() is True
    assert "youngs_modulus" in c.last_mask
    assert "damping" in c.last_mask


def _hist(variances, n=2) -> CalibHistory:
    k = len(variances)
    return CalibHistory([1.0] * k, list(variances), [10.0] * k, n_interactions=n)


def test_commit_requires_all_three_gates(mock_cfg):
    gate = CalibrationGate(mock_cfg)
    hist = _hist([5.0, 4.0, 3.0, 2.0, 1.0])
    assert gate.evaluate(_result(), hist) == GateVerdict.COMMIT


def test_predicts_false_blocks_commit(mock_cfg):
    gate = CalibrationGate(mock_cfg)
    hist = _hist([5.0, 4.0, 3.0, 2.0, 1.0])
    assert gate.evaluate(_result(predicts=False), hist) == GateVerdict.KEEP_PRIOR


def test_high_condition_blocks_commit(mock_cfg):
    gate = CalibrationGate(mock_cfg)
    hist = _hist([5.0, 4.0, 3.0, 2.0, 1.0])
    cond = float(mock_cfg.require("calibration.kappa_max")) * 2.0
    assert gate.evaluate(_result(cond=cond), hist) == GateVerdict.KEEP_PRIOR


def test_insufficient_loss_drop_blocks_commit(mock_cfg):
    gate = CalibrationGate(mock_cfg)
    hist = _hist([5.0, 4.0, 3.0, 2.0, 1.0])
    assert gate.evaluate(_result(loss_before=10.0, loss_after=9.9), hist) == GateVerdict.KEEP_PRIOR


def test_non_convergence_degrades_after_limit(mock_cfg):
    gate = CalibrationGate(mock_cfg)
    limit = int(mock_cfg.require("calibration.max_interactions_before_degrade"))
    hist = _hist([2.0, 2.0, 2.0, 2.0, 2.0], n=limit + 1)
    assert gate.evaluate(_result(predicts=False), hist) == GateVerdict.DEGRADE


def test_ablation_masks_are_honoured(mock_cfg, stub_asset):
    names = ["youngs_modulus", "damping", "bending_stiffness", "areal_density"]
    cfg = mock_cfg.clone()
    cfg.set("calibration.params", names + ["friction"])
    J = np.ones((12, 7), dtype=np.float64)
    ident = SystemIdentifier(cfg)
    active = ident._active_mask(J, {}, 1e-6)
    assert bool(active[4]) is True
    active_no_friction = ident._active_mask(J, {"disable_friction_opt": True}, 1e-6)
    assert bool(active_no_friction[4]) is False

    cfg2 = mock_cfg.clone()
    cfg2.set("calibration.ablation.disable_material_opt", True)
    sim = SimpleNamespace(asset=stub_asset)
    ident2 = SystemIdentifier(cfg2, sim)
    batch = ObservationBatch(
        contacts=[ContactEvent(0.0, ContactKind.SLIDE, np.zeros(3), 0, 0.0, "right")],
        states=[_state()],
        camera_views=[np.zeros((4, 4, 3), dtype=np.uint8)],
    )
    res = ident2.update(batch)
    assert np.allclose(res.params.to_vector(), stub_asset.material.to_vector())
    assert res.predicts_next_frame is False


def test_disable_back_prior_ablation_changes_provenance(mock_cfg, garment_image):
    cfg = mock_cfg.clone()
    cfg.set("calibration.ablation.disable_back_prior", True)
    asset = build(cfg)["AssetBuilder"].build(garment_image, timeout_s=120.0)
    assert asset.provenance["back"] == GeometryProvenance.PRIOR_PATTERN
    assert asset.confidence.back_geometry < float(
        cfg.require("asset.back_confidence_threshold")
    )
