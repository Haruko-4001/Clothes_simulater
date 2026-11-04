import numpy as np
import pytest

from tryon.types import (
    BodyState,
    CalibHistory,
    Confidence,
    ContactEvent,
    ContactKind,
    FramingCompliance,
    FramingVerdict,
    GarmentClass,
    GeometryDraft,
    GeometryProvenance,
    HandState,
    KeyframeRequest,
    KeyframeResult,
    LossWeights,
    MaterialParams,
    MaterialState,
    ObservationBatch,
    PhysicsState,
    Residuals,
    Segmentation,
    SystemStatus,
    TimingBook,
    TriggerContext,
    overall_confidence,
    weighted_geometric_mean,
)
from collections import deque


def _conf() -> Confidence:
    return Confidence(
        segmentation=0.9,
        visible_geometry=0.9,
        back_geometry=0.4,
        topology=1.0,
        material=0.5,
        overall=0.7,
    )


def _framing() -> FramingCompliance:
    return FramingCompliance(
        verdict=FramingVerdict.OK,
        visible_keypoint_ratio=1.0,
        bbox_clipped=False,
        facing_estimate="frontal",
        occlusion_ratio=0.0,
        framing_confidence=0.9,
        guide_text="取景合规",
    )


def test_timing_book_five_independent_fields():
    t = TimingBook()
    t.camera_capture_fps = 30.0
    t.display_fps = 29.5
    t.physics_hz = 60.0
    t.keyframe_fps = 0.5
    t.generation_throughput_fps = 3.0
    assert t.camera_capture_fps == 30.0
    assert t.display_fps == 29.5
    assert t.physics_hz == 60.0
    assert t.keyframe_fps == 0.5
    assert t.generation_throughput_fps == 3.0
    # Changing one field must never change another.
    before = t.as_dict()
    t.display_fps = 1.0
    for key, value in before.items():
        if key != "display_fps":
            assert getattr(t, key) == value


def test_material_params_domain_and_state_invariant():
    m = MaterialParams()
    assert m.state == MaterialState.AI_PRIOR
    assert m.calibrated_at is None
    with pytest.raises(ValueError):
        MaterialParams(areal_density=10.0)
    with pytest.raises(ValueError):
        MaterialParams(youngs_modulus=-1.0)
    with pytest.raises(ValueError):
        MaterialParams(state=MaterialState.AI_PRIOR, calibrated_at=1.0)
    calibrated = MaterialParams(
        state=MaterialState.CALIBRATED,
        calibrated_at=1.0,
        scene_tag="scene",
    )
    assert calibrated.calibrated_at == 1.0
    assert calibrated.is_expired(2.0) is False


def test_confidence_weighted_geometric_mean():
    c = _conf()
    weights = {
        "segmentation": 1.0,
        "visible_geometry": 1.5,
        "back_geometry": 1.0,
        "topology": 1.0,
        "material": 1.0,
    }
    overall = overall_confidence(c, weights)
    assert 0.0 < overall <= 1.0
    assert overall == pytest.approx(
        weighted_geometric_mean(
            [0.9, 0.9, 0.4, 1.0, 0.5], [1.0, 1.5, 1.0, 1.0, 1.0]
        )
    )


def test_all_referenced_types_instantiable(stub_asset):
    h, w = 32, 24
    seg = Segmentation(
        mask=np.zeros((h, w), dtype=np.uint8),
        garment_class=GarmentClass.TOPS,
        bbox=(0, 0, w, h),
        confidence=0.9,
    )
    body = BodyState(
        smpl_params={"beta": np.zeros(10), "theta": np.zeros(3), "transl": np.zeros(3)},
        pose_keypoints=np.zeros((17, 3)),
        bbox=(0, 0, w, h),
        confidence=0.8,
    )
    hand = HandState(landmarks=np.zeros((21, 3)), mesh=None, handedness="right", occluded=False)
    draft = GeometryDraft(
        panels=[],
        stitches=[(0, 1)],
        rest_mesh=stub_asset.rest_mesh,
        back_appearance=None,
        class_prior=GarmentClass.TOPS,
        provenance={"back": GeometryProvenance.PRIOR_BACK},
    )
    assert draft.stitches
    state = PhysicsState(
        positions=np.zeros((4, 3)),
        velocities=np.zeros((4, 3)),
        attached={},
        lam=np.zeros(1),
        step=0,
        energy=0.0,
    )
    contact = ContactEvent(0.0, ContactKind.SLIDE, np.zeros(3), None, 0.0, "right")
    batch = ObservationBatch(
        contacts=[contact],
        states=[state],
        camera_views=[np.zeros((h, w, 3), dtype=np.uint8)],
        K=np.eye(3),
        T_CW=np.eye(4),
        dt=1 / 60,
    )
    assert len(batch) == 1
    weights = LossWeights()
    res = Residuals(
        reproj=np.zeros((1, 8)),
        velocity=np.zeros((1, 12)),
        silhouette=np.zeros(1),
        penetration=np.zeros(1),
    )
    assert res.dim == 22
    hist = CalibHistory([1.0], [1.0], [1.0])
    hist.append(0.9, 0.8, 0.7)
    assert len(hist.losses) == 2
    ctx = TriggerContext(
        pose_history=deque([body]),
        framing=_framing(),
        garment_id=stub_asset.asset_id,
        elapsed_since_kf_ms=0.0,
        bbox_crossed=False,
        state="TRACKING",
    )
    assert ctx.garment_id
    status = SystemStatus(
        asset_state="MATERIAL_PRIOR",
        material_state=MaterialState.AI_PRIOR,
        confidence=_conf(),
        timing=TimingBook(),
        framing=_framing(),
    )
    assert status.material_state == MaterialState.AI_PRIOR
    req = KeyframeRequest(
        req_id="r",
        garment_asset_id=stub_asset.asset_id,
        person_image=np.zeros((h, w, 3), dtype=np.uint8),
        garment_image=np.zeros((h, w, 3), dtype=np.uint8),
        mask=np.zeros((h, w), dtype=np.uint8),
        pose=body,
        enqueued_at_ms=0.0,
    )
    out = KeyframeResult(
        req_id=req.req_id,
        image=req.person_image,
        mask=req.mask,
        finished_at_ms=1.0,
    )
    assert out.req_id == "r"
