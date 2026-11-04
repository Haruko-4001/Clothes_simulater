from collections import deque

import numpy as np

from tryon.display.compositor import Compositor
from tryon.perception.framing import FramingGate
from tryon.perception.stubs import StubBodyTracker
from tryon.keyframe.trigger import KeyframeTrigger
from tryon.types import (
    BodyState,
    Confidence,
    Frame,
    FramingVerdict,
    RenderOutput,
    SystemStatus,
    TimingBook,
    TriggerContext,
)


W, H = 320, 240


def _frame() -> Frame:
    return Frame(frame_id=1, pts_ms=0.0, image=np.full((H, W, 3), 80, dtype=np.uint8))


def _body(
    *,
    facing="frontal",
    occlusion=0.02,
    cropped=False,
    **flags,
) -> BodyState:
    kp = np.array(
        [
            [160, 20], [160, 45], [128, 50], [112, 90], [106, 130],
            [192, 50], [208, 90], [214, 130], [144, 140], [144, 180],
            [144, 225], [176, 140], [176, 180], [176, 225], [160, 140],
            [160, 110], [160, 75],
        ],
        dtype=np.float64,
    )
    if cropped:
        kp[8:, 1] = H * 1.15
    return BodyState(
        smpl_params={
            "beta": np.zeros(10),
            "theta": np.zeros(3),
            "transl": np.zeros(3),
            "facing": facing,
            "occlusion": occlusion,
            "keypoints_2d": kp,
            **flags,
        },
        pose_keypoints=kp,
        bbox=(int(kp[:, 0].min()), int(kp[:, 1].min()), int(kp[:, 0].max()), int(kp[:, 1].max())),
        confidence=0.92,
        fitted_from_frames=30,
    )


def _status(framing, visible_bbox=None) -> SystemStatus:
    return SystemStatus(
        asset_state="MATERIAL_PRIOR",
        material_state="AI_PRIOR",
        confidence=Confidence(0.9, 0.9, 0.4, 1.0, 0.5, 0.7),
        timing=TimingBook(),
        framing=framing,
        visible_bbox=visible_bbox,
    )


def test_full_frontal_is_ok(mock_cfg):
    gate = FramingGate(mock_cfg)
    c = gate.check(_frame(), _body())
    assert c.verdict == FramingVerdict.OK
    assert c.guide_text != ""
    assert c.visible_keypoint_ratio > 0.9
    assert c.bbox_clipped is False


def test_knee_crop_partial_and_no_out_of_frame_content(mock_cfg):
    gate = FramingGate(mock_cfg)
    c = gate.check(_frame(), _body(cropped=True))
    assert c.verdict == FramingVerdict.PARTIAL
    assert c.guide_text != ""

    rgb = _frame().image
    rendered = RenderOutput(
        image=np.full((H, W, 3), 255, dtype=np.uint8),
        mask=np.ones((H, W), dtype=np.uint8),
        coverage_bbox=(0, 0, W, H),
    )
    bbox = (20, 20, 100, 100)
    out = Compositor(mock_cfg).compose(
        _frame(), rendered, _status(c, visible_bbox=bbox)
    )
    outside = np.ones((H, W), dtype=bool)
    outside[bbox[1] : bbox[3], bbox[0] : bbox[2]] = False
    assert np.array_equal(out[outside], rgb[outside])
    assert np.all(out[bbox[1] : bbox[3], bbox[0] : bbox[2]] == 255)


def test_side_back_requires_async_and_blocks_2d_keyframe(mock_cfg):
    gate = FramingGate(mock_cfg)
    c = gate.check(_frame(), _body(facing="back"))
    assert c.verdict == FramingVerdict.GUIDE
    assert c.guide_text != ""
    trigger = KeyframeTrigger(mock_cfg)
    hist = deque([_body() for _ in range(30)])
    ctx = TriggerContext(
        pose_history=hist,
        framing=c,
        garment_id="g",
        elapsed_since_kf_ms=1e9,
        bbox_crossed=False,
        state="TRACKING",
    )
    assert trigger.should_fire(ctx) is False


def test_extreme_occlusion_and_multi_layer_reject(mock_cfg):
    gate = FramingGate(mock_cfg)
    c = gate.check(_frame(), _body(occlusion=0.9))
    assert c.verdict == FramingVerdict.REJECT
    assert c.guide_text != ""
    c2 = gate.check(_frame(), _body(transparent=True))
    assert c2.verdict == FramingVerdict.REJECT
    c3 = gate.check(_frame(), _body(multi_layer=True))
    assert c3.verdict == FramingVerdict.REJECT


def test_multi_frame_accumulation_raises_confidence(mock_cfg):
    tracker = StubBodyTracker(mock_cfg)
    frame = _frame()
    history: deque[BodyState] = deque(maxlen=30)
    confidences = []
    fitted = []
    for i in range(20):
        body = tracker.track(frame, history)
        history.append(body)
        confidences.append(body.confidence)
        fitted.append(body.fitted_from_frames)
    assert confidences[0] < confidences[5] < confidences[-1]
    assert fitted[0] == 1
    assert fitted[-1] == 20
    assert all(b >= a for a, b in zip(confidences, confidences[1:]))
