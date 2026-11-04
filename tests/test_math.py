import numpy as np
import pytest

from tryon.math import (
    backproject,
    camera_to_world,
    compose_T,
    compose_Ts,
    is_SE3,
    look_at,
    project,
    quat_from_axis_angle,
    quat_to_R,
    transform,
)
from tryon.math.projection import make_intrinsics
from tryon.types import TryonError


def test_quat_to_R_orthonormal_det_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        q = quat_from_axis_angle(axis, rng.uniform(-np.pi, np.pi))
        R = quat_to_R(q)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-9
        assert is_SE3(compose_T(R, rng.normal(size=3)))


def test_quat_to_R_rejects_non_unit():
    with pytest.raises(TryonError) as exc:
        quat_to_R(np.array([2.0, 0.0, 0.0, 0.0]))
    assert exc.value.code == "E_MATH_QUAT"
    with pytest.raises(TryonError):
        quat_to_R(np.array([0.0, 0.0, 0.0, 0.0]))


def test_project_backproject_roundtrip():
    K = make_intrinsics(320, 240, 50.0)
    T_CW, _ = look_at(
        np.array([0.0, 1.0, 3.0]), np.array([0.0, 0.9, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    p_W = np.array([[0.0, 1.2, 0.0], [0.2, 0.9, 0.1], [-0.15, 1.5, -0.2]])
    uv, depth = project(K, T_CW, p_W)
    assert np.all(depth > 0)
    p_C = backproject(K, uv, depth)
    p_W_back = camera_to_world(T_CW, p_C)
    assert np.allclose(p_W_back, p_W, atol=1e-9)


def test_project_raises_on_non_positive_depth():
    K = make_intrinsics(320, 240)
    T_CW = np.eye(4)
    with pytest.raises(TryonError) as exc:
        project(K, T_CW, np.array([[0.0, 0.0, -1.0]]))
    assert exc.value.code == "E_MATH_DEPTH"


def test_se3_composition_associativity_and_transform():
    q1 = quat_from_axis_angle([0, 0, 1], 0.3)
    q2 = quat_from_axis_angle([1, 0, 0], -0.7)
    T1 = compose_T(quat_to_R(q1), np.array([0.1, 0.2, 0.3]))
    T2 = compose_T(quat_to_R(q2), np.array([-0.2, 0.0, 0.4]))
    T3 = compose_T(np.eye(3), np.array([1.0, 2.0, 3.0]))
    left = compose_Ts(T1, T2, T3)
    right = T1 @ T2 @ T3
    assert np.allclose(left, right)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]])
    assert np.allclose(
        transform(left, pts),
        transform(T1, transform(T2, transform(T3, pts))),
    )
    assert is_SE3(left)


def test_look_at_is_se3_and_depth_positive():
    T_CW, R_c2w = look_at(
        np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert is_SE3(T_CW)
    assert abs(float(np.linalg.det(R_c2w)) - 1.0) < 1e-9
    K = make_intrinsics(320, 240)
    _, depth = project(K, T_CW, np.array([[0.0, 1.0, 0.0], [0.1, 0.5, -1.0]]))
    assert np.all(depth > 0)
