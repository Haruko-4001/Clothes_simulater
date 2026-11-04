import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tryon.asset.asset_builder import AppearanceMesh
from tryon.asset.mesh_repair import MeshRepairer
from tryon.asset.registry import AssetRegistry
from tryon.display.hud import HUD
from tryon.types import (
    FramingCompliance,
    FramingVerdict,
    GeometryDraft,
    GeometryProvenance,
    MaterialParams,
    MaterialState,
    SimReadyMesh,
    SystemStatus,
    TimingBook,
    TryonError,
    GarmentAsset,
)


def test_asset_mesh_is_closed_manifold_and_thick(stub_asset):
    m = stub_asset.rest_mesh
    assert isinstance(m, SimReadyMesh)
    assert m.is_closed is True
    assert m.has_isolated_edge is False
    assert m.self_intersection_count == 0
    assert m.thickness > 0.0
    assert m.n_vertices > 0 and m.n_faces > 0


def test_g1_and_back_provenance_and_ai_completion(stub_asset):
    asset = stub_asset
    g1 = asset.sewing_pattern["g1"]
    assert g1["pass"] is True
    assert g1["iou"] >= 0.90
    assert asset.provenance["back"] == GeometryProvenance.PRIOR_BACK
    assert asset.confidence.back_geometry < 0.65

    framing = FramingCompliance(
        verdict=FramingVerdict.OK,
        visible_keypoint_ratio=1.0,
        bbox_clipped=False,
        facing_estimate="frontal",
        occlusion_ratio=0.0,
        framing_confidence=0.9,
        guide_text="取景合规",
    )
    status = SystemStatus(
        asset_state="MATERIAL_PRIOR",
        material_state=asset.material.state,
        confidence=asset.confidence,
        timing=TimingBook(),
        framing=framing,
    )
    from tryon.config import load_config

    hud = HUD(load_config("mock"))
    img = hud.draw(np.zeros((240, 320, 3), dtype=np.uint8), status)
    assert img.shape == (240, 320, 3)
    assert "AI 补全" in hud.text
    assert "AI_PRIOR" in hud.text
    assert "OK" in hud.text
    assert "取景合规" in hud.text


def test_appearance_mesh_is_separate_and_cannot_enter_physics(stub_asset, physics_solver, mock_cfg):
    assert stub_asset.appearance_mesh is not stub_asset.rest_mesh
    assert isinstance(stub_asset.appearance_mesh, AppearanceMesh)
    assert np.allclose(stub_asset.appearance_mesh.vertices, stub_asset.rest_mesh.vertices)

    state = physics_solver.initial_state(stub_asset)
    fake = SimpleNamespace(
        rest_mesh=stub_asset.appearance_mesh,
        material=stub_asset.material,
        asset_id=stub_asset.asset_id,
        collision_body=stub_asset.collision_body,
    )
    with pytest.raises(TypeError):
        physics_solver.step(state, fake, [], float(mock_cfg.require("physics.dt")))


def test_material_prior_is_ai_prior_and_in_domain(stub_asset):
    m = stub_asset.material
    assert m.state == MaterialState.AI_PRIOR
    assert m.calibrated_at is None
    lo = np.array([0.02, 1e2, 1e-6, 0.0, 0.0, 0.0, 1e-4])
    hi = np.array([1.20, 1e6, 1e-1, 0.49, 1.5, 50.0, 5e-3])
    assert np.all(m.to_vector() >= lo)
    assert np.all(m.to_vector() <= hi)


def test_mesh_repair_rejects_invalid_thickness(stub_asset):
    raw = SimpleNamespace(
        vertices=stub_asset.rest_mesh.vertices,
        faces=stub_asset.rest_mesh.faces,
        thickness=0.0,
    )
    draft = GeometryDraft(
        panels=[],
        stitches=[(0, 1)],
        rest_mesh=raw,
        back_appearance=None,
        class_prior=stub_asset.garment_class,
        provenance={"back": GeometryProvenance.PRIOR_BACK},
    )
    with pytest.raises(TryonError) as exc:
        MeshRepairer().repair(draft)
    assert exc.value.code == "E_ASSET_GEOM_FAIL"


def test_registry_commit_increments_version_once_and_sets_ttl(stub_asset, mock_cfg, tmp_path):
    reg = AssetRegistry(mock_cfg, path=tmp_path / "registry.json")
    reg.register(stub_asset)
    params = MaterialParams(
        state=MaterialState.CALIBRATING,
        scene_tag="",
        calibrated_at=None,
    )
    committed = reg.commit(stub_asset, params, scene_tag="scene_a", now_s=100.0, ttl_s=60.0)
    assert committed.version == stub_asset.version + 1
    assert committed.material.state == MaterialState.CALIBRATED
    assert committed.material.calibrated_at == 100.0
    assert committed.material.scene_tag == "scene_a"
    assert reg.version_of(stub_asset.asset_id) == committed.version
    assert reg.is_expired(stub_asset.asset_id, now_s=200.0) is True
    assert reg.is_expired(stub_asset.asset_id, now_s=120.0) is False
    with pytest.raises(ValueError):
        reg.register(dataclasses.replace(committed, version=0))
