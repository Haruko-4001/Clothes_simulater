"""Shared fixtures: mock_cfg / stub_asset / synth_trajectory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tryon.config import load_config  # noqa: E402
from tryon.factory import build  # noqa: E402
from tryon.physics.solver import PhysicsSolver  # noqa: E402
from tryon.types import ContactEvent, ContactKind, Frame, HandState, BodyState  # noqa: E402


@pytest.fixture(scope="session")
def mock_cfg():
    cfg = load_config("mock", "rtx4090")
    cfg.set("camera.width", 320)
    cfg.set("camera.height", 240)
    cfg.set("asset.mesh_resolution", [10, 7])
    return cfg


@pytest.fixture(scope="session")
def garment_image(mock_cfg):
    from app.main import load_garment_image

    return load_garment_image(ROOT / "assets" / "samples" / "garment.png", mock_cfg)


@pytest.fixture(scope="session")
def stub_asset(mock_cfg, garment_image):
    components = build(mock_cfg)
    builder = components["AssetBuilder"]
    return builder.build(garment_image, timeout_s=120.0)


@pytest.fixture()
def physics_solver(mock_cfg):
    return PhysicsSolver(mock_cfg)


@pytest.fixture()
def synth_trajectory(stub_asset, mock_cfg, physics_solver):
    """Settle → grasp → pull → release → rebound states for physics tests."""
    asset = stub_asset
    solver = physics_solver
    dt = float(mock_cfg.require("physics.dt"))
    state = solver.initial_state(asset)
    states: list = []
    for _ in range(8):
        state = solver.step(state, asset, [], dt)
        states.append(state.copy())
    pos = state.positions
    vid = int(np.argmin(np.linalg.norm(pos - np.array([-0.22, 1.20, 0.0]), axis=1)))
    hand = pos[vid].copy()
    state = solver.step(
        state, asset,
        [ContactEvent(0.0, ContactKind.GRASP, hand, vid, 0.0, "right")], dt,
    )
    states.append(state.copy())
    for k in range(18):
        hand = hand + np.array([-0.008, -0.003, 0.006])
        state = solver.step(
            state, asset,
            [ContactEvent(float(k), ContactKind.PULL, hand, vid, 0.0, "right")], dt,
        )
        states.append(state.copy())
    pulled = state.copy()
    state = solver.step(
        state, asset,
        [ContactEvent(30.0, ContactKind.RELEASE, hand, vid, 0.0, "right")], dt,
    )
    released = state.copy()
    states.append(released)
    rebound = []
    for _ in range(25):
        state = solver.step(state, asset, [], dt)
        rebound.append(state.copy())
        states.append(state.copy())
    return {
        "asset": asset,
        "solver": solver,
        "dt": dt,
        "vertex_id": vid,
        "settled": states[:8],
        "pulled": pulled,
        "released": released,
        "rebound": rebound,
        "states": states,
    }


def make_frame(width: int = 320, height: int = 240, value: int = 90) -> Frame:
    img = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(frame_id=1, pts_ms=0.0, image=img)


def make_body(
    frame_size: tuple[int, int] = (320, 240),
    *,
    facing: str = "frontal",
    occlusion: float = 0.02,
    cropped: bool = False,
    **flags,
) -> BodyState:
    w, h = frame_size
    base = np.array(
        [
            [w * 0.50, h * 0.12], [w * 0.50, h * 0.22],
            [w * 0.40, h * 0.25], [w * 0.35, h * 0.40], [w * 0.33, h * 0.55],
            [w * 0.60, h * 0.25], [w * 0.65, h * 0.40], [w * 0.67, h * 0.55],
            [w * 0.45, h * 0.58], [w * 0.45, h * 0.75], [w * 0.45, h * 0.94],
            [w * 0.55, h * 0.58], [w * 0.55, h * 0.75], [w * 0.55, h * 0.94],
            [w * 0.50, h * 0.58], [w * 0.50, h * 0.45], [w * 0.50, h * 0.32],
        ],
        dtype=np.float64,
    )
    if cropped:
        base[8:, 1] = h * 1.10
    x0, y0 = base[:, 0].min(), base[:, 1].min()
    x1, y1 = base[:, 0].max(), base[:, 1].max()
    return BodyState(
        smpl_params={
            "beta": np.zeros(10),
            "theta": np.zeros(3),
            "transl": np.zeros(3),
            "facing": facing,
            "occlusion": occlusion,
            "keypoints_2d": base,
            **flags,
        },
        pose_keypoints=base,
        bbox=(int(x0), int(y0), int(x1), int(y1)),
        confidence=0.9,
        fitted_from_frames=30,
    )


def make_hand(occluded: bool = False) -> HandState:
    return HandState(
        landmarks=np.zeros((21, 3)),
        mesh=None,
        handedness="right",
        occluded=occluded,
    )
