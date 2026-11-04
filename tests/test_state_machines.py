import time

import numpy as np

from tryon.interaction.state_machine import (
    AssetState,
    AssetStateMachine,
    InteractionState,
    InteractionStateMachine,
    KeyframeState,
    KeyframeStateMachine,
)
from tryon.keyframe.tryon import KeyframeService
from tryon.types import (
    BodyState,
    ContactKind,
    GateVerdict,
    KeyframeRequest,
    KeyframeResult,
)


def _prepare(mock_cfg):
    sm = AssetStateMachine(mock_cfg)
    sm.g1(True)
    sm.confidence_review(0.9)
    return sm


def test_asset_state_machine_happy_path_and_versions(mock_cfg):
    sm = _prepare(mock_cfg)
    assert sm.state == AssetState.MATERIAL_PRIOR
    assert sm.excitation(True, 10.0) == AssetState.CALIBRATING
    assert sm.calibration(GateVerdict.COMMIT, scene_tag="s", ttl_s=60.0) == AssetState.CALIBRATED
    assert sm.version == 1
    assert sm.scene_tag == "s"
    assert sm.ttl_s == 60.0


def test_asset_state_machine_failure_and_review_paths(mock_cfg):
    sm = AssetStateMachine(mock_cfg)
    assert sm.g1(False) == AssetState.DEGRADED
    assert sm.needs_human_review is True

    sm2 = AssetStateMachine(mock_cfg)
    assert sm2.g1(True) == AssetState.GEOMETRY_READY
    assert sm2.confidence_review(0.1) == AssetState.NEEDS_REVIEW
    assert sm2.manual_review(True) == AssetState.GEOMETRY_READY

    sm3 = _prepare(mock_cfg)
    sm3.excitation(True, 10.0)
    assert sm3.interaction_update(51, False) == AssetState.DEGRADED


def test_interaction_state_machine_release_semantics(mock_cfg):
    sm = InteractionStateMachine(mock_cfg)
    assert sm.state == InteractionState.IDLE
    assert sm.on_event(ContactKind.GRASP) == InteractionState.GRASPED
    assert sm.on_event(ContactKind.PULL) == InteractionState.PULLING
    assert sm.on_event(ContactKind.RELEASE) == InteractionState.REBOUNDING
    assert sm.attached_cleared is True
    assert sm.update_energy(0.0) == InteractionState.IDLE
    assert sm.energy_invariant(2.0, 1.0, 1 / 60, 0.15)


def test_keyframe_state_machine_all_states_and_hysteresis(mock_cfg):
    sm = KeyframeStateMachine(mock_cfg)
    eps_in = float(mock_cfg.require("keyframe.epsilon_in"))
    eps_out = float(mock_cfg.require("keyframe.epsilon_out"))
    assert eps_in < eps_out
    m = sm.m
    for _ in range(m):
        assert sm.observe(eps_in * 0.1) is False
    assert sm.state == KeyframeState.STANDBY
    fired = sm.observe(
        eps_out + 1.0,
        elapsed_ms=float(mock_cfg.require("keyframe.min_interval_ms")) + 1,
    )
    assert fired is False  # hysteresis returns to TRACKING
    assert sm.state == KeyframeState.TRACKING

    for _ in range(m):
        sm.observe(eps_in * 0.1)
    assert sm.state == KeyframeState.STANDBY
    fired = sm.observe(eps_in * 0.1, elapsed_ms=1e9)
    assert fired is True
    assert sm.state == KeyframeState.GENERATING
    assert sm.result_ready(False) == KeyframeState.FUSING
    assert sm.fuse_done() == KeyframeState.TRACKING


class _SlowBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, req: KeyframeRequest) -> KeyframeResult:
        self.calls += 1
        time.sleep(0.25)
        return KeyframeResult(
            req_id=req.req_id,
            image=req.person_image,
            mask=req.mask,
            finished_at_ms=time.time() * 1000.0,
        )


def test_keyframe_queue_bounded_at_two(mock_cfg):
    backend = _SlowBackend()
    svc = KeyframeService(mock_cfg, backend)
    body = BodyState(
        smpl_params={"beta": np.zeros(10), "theta": np.zeros(3), "transl": np.zeros(3)},
        pose_keypoints=np.zeros((17, 3)),
        bbox=(0, 0, 8, 8),
        confidence=0.9,
    )
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    results = []
    for i in range(4):
        req = KeyframeRequest(
            req_id=f"r{i}",
            garment_asset_id="g",
            person_image=img,
            garment_image=img,
            mask=np.zeros((8, 8), dtype=np.uint8),
            pose=body,
            enqueued_at_ms=time.time() * 1000.0,
        )
        results.append(svc.submit(req))
        assert svc.queue_size() <= int(mock_cfg.require("keyframe.max_queue"))
    assert any(not ok for ok in results)
    svc.close()
