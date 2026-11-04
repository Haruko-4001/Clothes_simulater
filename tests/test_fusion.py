import numpy as np
import pytest

from tryon.keyframe.fuse import FusionBlender, alpha_sequence, blend, is_expired


def test_alpha_sequence_is_exactly_six_frames_and_decreasing(mock_cfg):
    alphas = alpha_sequence(mock_cfg)
    assert len(alphas) == int(mock_cfg.require("keyframe.fuse_frames")) == 6
    assert alphas[0] == pytest.approx(float(mock_cfg.require("keyframe.alpha0")))
    assert np.all(np.diff(alphas) < 0)


def test_fusion_blender_steps_six_frames(mock_cfg):
    f = FusionBlender(mock_cfg)
    f.begin(1000.0)
    seen = []
    for i in range(6):
        alpha, active = f.step(1000.0 + i * 33.333)
        assert active is True
        seen.append(alpha)
    alpha, active = f.step(1300.0)
    assert active is False
    assert alpha == 0.0
    assert f.count == 6
    assert np.all(np.diff(np.asarray(seen)) < 0)


def test_pose_expiry_threshold(mock_cfg):
    delta = float(mock_cfg.require("keyframe.delta_pose"))
    a = np.zeros(3)
    b = np.full(3, delta * 0.5)
    c = np.full(3, delta * 2.0)
    assert is_expired(b, a, delta) is False
    assert is_expired(c, a, delta) is True


def test_blend_respects_protected_mask():
    render = np.zeros((4, 4, 3), dtype=np.uint8)
    kf = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    protected = np.zeros((4, 4), dtype=np.uint8)
    protected[0, 0] = 1
    out = blend(render, kf, 0.5, mask, protected_mask=protected)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[1, 1].tolist() == [100, 100, 100]
