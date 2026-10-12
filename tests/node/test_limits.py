import numpy as np
import pytest

from franka_teleop.config import JointGuardConfig, TeleopConfig, WorkspaceConfig
from franka_teleop.node.limits import (
    limit_joints,
    limit_singularity,
    limit_workspace,
    min_singular_value,
    predict_joint_velocity,
    shape_twist,
)

DOWN = np.diag([1.0, -1.0, -1.0])  # ready pose: tool z points down


def test_shape_twist_scaling_operator_yaw_and_tool_frame():
    tc = TeleopConfig(max_linear_speed=0.2, max_angular_speed=0.5)
    t = shape_twist((1, 0, 0), (0, 0, 1), 0.5, "base", DOWN, tc)
    assert np.allclose(t, [0.1, 0, 0, 0, 0, 0.25])
    tc180 = TeleopConfig(max_linear_speed=0.2, max_angular_speed=0.5, operator_yaw_deg=180)
    t = shape_twist((1, 1, 0), (0, 0, 1), 1.0, "base", DOWN, tc180)
    assert np.allclose(t, [-0.2, -0.2, 0, 0, 0, 0.5])  # yaw about z is unchanged by yaw
    t = shape_twist((0, 0, 1), (1, 0, 0), 1.0, "tool", DOWN, tc)
    assert np.allclose(t, [0, 0, -0.2, 0.5, 0, 0])  # "up" = approach direction = base -z
    t = shape_twist((5, -5, 0), (0, 0, 0), 1.0, "base", DOWN, tc)
    assert np.allclose(t[:2], [0.2, -0.2])  # intent is clipped to [-1, 1]


WS = WorkspaceConfig(lower=[0.3, -0.4, 0.05], upper=[0.7, 0.4, 0.6], slow_zone=0.05)


@pytest.mark.parametrize(
    ("pos", "twist", "expected", "limited"),
    [
        ((0.5, 0.0, 0.3), (0.15, -0.15, 0.15), (0.15, -0.15, 0.15), False),  # centre: free
        ((0.7, 0.0, 0.3), (0.15, 0, 0), (0.0, 0, 0), True),  # on the +x wall moving out
        ((0.7, 0.0, 0.3), (-0.15, 0, 0), (-0.15, 0, 0), False),  # on the wall moving in
        ((0.68, 0.0, 0.3), (0.15, 0, 0), (0.06, 0, 0), True),  # slow zone: 3/s * 0.02 m
        ((0.75, 0.0, 0.3), (0.15, 0, 0), (0.0, 0, 0), True),  # outside: blocked
        ((0.75, 0.0, 0.3), (0.0, 0, 0), (0.0, 0, 0), False),  # outside: never pulled back
        ((0.5, 0.0, 0.06), (0, 0, -0.15), (0, 0, -0.03), True),  # near the table
    ],
)
def test_limit_workspace(pos, twist, expected, limited):
    out, lim = limit_workspace(np.r_[twist, 0.1, 0.2, 0.3], np.array(pos), WS, 0.15)
    assert np.allclose(out[:3], expected) and lim == limited
    assert np.allclose(out[3:], [0.1, 0.2, 0.3])  # angular untouched


def test_workspace_closed_loop_never_leaves_box():
    rng = np.random.default_rng(0)
    p, dt = np.array([0.5, 0.0, 0.3]), 0.01
    lower, upper = np.array(WS.lower), np.array(WS.upper)
    for _ in range(5000):
        twist = np.r_[rng.uniform(-0.15, 0.15, 3), 0, 0, 0]
        out, _ = limit_workspace(twist, p, WS, 0.15)
        p = p + out[:3] * dt
        assert np.all(p >= lower - 1e-9) and np.all(p <= upper + 1e-9)


GUARD = JointGuardConfig(margin=0.2, clearance=0.03, max_joint_speed=1.0, damping=0.01)
LOWER, UPPER = -np.ones(7) * 2.0, np.ones(7) * 2.0


def random_jacobian(rng):
    return rng.normal(size=(6, 7)) * 0.5


def test_predict_joint_velocity_realizes_twist():
    rng = np.random.default_rng(1)
    for _ in range(50):
        J, twist = random_jacobian(rng), rng.normal(size=6) * 0.1
        assert np.allclose(J @ predict_joint_velocity(twist, J, 1e-4), twist, atol=1e-5)


def test_limit_joints_slows_towards_limit_but_not_away():
    J = np.hstack([np.eye(6), np.zeros((6, 1))])
    twist = np.array([0.3, 0, 0, 0, 0, 0])  # moves joint 0 positively at 0.3 rad/s
    q = np.zeros(7)
    out, flags = limit_joints(twist, q, J, LOWER, UPPER, GUARD)
    assert np.allclose(out, twist) and flags == ()
    q[0] = 1.92  # 0.08 from the limit: allowed 1.0 * (0.08 - 0.03) / 0.2 = 0.25 rad/s < 0.3
    out, flags = limit_joints(twist, q, J, LOWER, UPPER, GUARD)
    assert np.allclose(out[0], 0.25, atol=1e-3) and flags == ("joint",)
    q[0] = 2.0 - GUARD.clearance  # stops `clearance` before the limit
    out, flags = limit_joints(twist, q, J, LOWER, UPPER, GUARD)
    assert np.allclose(out, 0) and flags == ("joint",)
    out, flags = limit_joints(-twist, q, J, LOWER, UPPER, GUARD)  # backing out is free
    assert np.allclose(out, -twist) and flags == ()


def test_limit_joints_uses_measured_speed_when_robot_ik_is_faster():
    J = np.hstack([np.eye(6), np.zeros((6, 1))])
    twist = np.array([0.1, 0, 0, 0, 0, 0])  # predicted dq0 = 0.1
    q = np.zeros(7)
    q[0] = 1.92  # allowed 0.25 rad/s: prediction alone is fine
    out, flags = limit_joints(twist, q, J, LOWER, UPPER, GUARD)
    assert np.allclose(out, twist)
    measured = np.zeros(7)
    measured[0] = 0.5  # but the robot is actually turning joint 0 at 0.5 rad/s
    out, flags = limit_joints(twist, q, J, LOWER, UPPER, GUARD, dq_measured=measured)
    assert np.allclose(out[0], 0.1 * 0.25 / 0.5) and flags == ("joint",)
    measured[0] = -0.5  # measured motion away from the limit does not tighten the guard
    out, _ = limit_joints(twist, q, J, LOWER, UPPER, GUARD, dq_measured=measured)
    assert np.allclose(out, twist)


def planar_arm_jacobian(q):
    """6x7 Jacobian of a planar 2-link arm (joints 0, 1; unit links) padded with a healthy
    5-DoF block, so sigma depends on the 'elbow' angle q[1] like a real elbow singularity."""
    J = np.zeros((6, 7))
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    J[0, :2] = [-s1 - s12, -s12]
    J[1, :2] = [c1 + c12, c12]
    J[2:, 2:] = np.eye(4, 5)
    J[5, 6] = 1.0
    return J


def test_singularity_guard_blocks_straightening_but_allows_bending():
    cfg = JointGuardConfig(sigma_slow=0.06, sigma_stop=0.035, rot_weight=1.0, damping=0.01)
    reach_out = np.array([1.0, 0, 0, 0, 0, 0]) * 0.05  # pulls the elbow straight (q1 -> 0)
    q = np.zeros(7)
    q[1] = 0.5  # bent: healthy
    J = planar_arm_jacobian(q)
    assert min_singular_value(J, 1.0) > cfg.sigma_slow
    out, limited = limit_singularity(reach_out, q, J, planar_arm_jacobian, cfg)
    assert np.allclose(out, reach_out) and not limited
    q[1] = 0.05  # nearly straight
    J = planar_arm_jacobian(q)
    sigma = min_singular_value(J, 1.0)
    assert sigma < cfg.sigma_stop
    out, limited = limit_singularity(reach_out, q, J, planar_arm_jacobian, cfg)
    assert np.allclose(out, 0) and limited
    out, limited = limit_singularity(-reach_out, q, J, planar_arm_jacobian, cfg)  # pull back in
    assert np.allclose(out, -reach_out) and not limited


def test_singularity_guard_closed_loop_keeps_sigma_above_stop():
    cfg = JointGuardConfig(sigma_slow=0.06, sigma_stop=0.035, rot_weight=1.0, damping=0.01)
    q, dt = np.zeros(7), 0.01
    q[0], q[1] = 0.3, 0.8
    for _ in range(3000):  # keep reaching out along the arm
        J = planar_arm_jacobian(q)
        direction = J[:2, :2] @ np.array([0.0, -1.0])  # straightening the elbow
        twist = np.r_[direction / np.linalg.norm(direction) * 0.1, 0, 0, 0, 0]
        out, _ = limit_singularity(twist, q, J, planar_arm_jacobian, cfg)
        q = q + predict_joint_velocity(out, J, cfg.damping) * dt
        assert min_singular_value(planar_arm_jacobian(q), 1.0) > cfg.sigma_stop - 0.005


def test_limit_joints_caps_joint_speed_near_singularity():
    J = np.hstack([np.eye(6), np.zeros((6, 1))])
    J[0, 0] = 0.05  # joint 0 barely moves the EE in x: near-singular direction
    out, flags = limit_joints(np.array([0.1, 0, 0, 0, 0, 0]), np.zeros(7), J, LOWER, UPPER, GUARD)
    dq = predict_joint_velocity(out, J, GUARD.damping)
    assert "singularity" in flags and np.abs(dq).max() <= GUARD.max_joint_speed + 1e-9


def test_joint_guard_closed_loop_never_crosses_limits():
    rng = np.random.default_rng(2)
    dt = 0.01
    for _ in range(20):
        J = random_jacobian(rng)
        q = rng.uniform(-1.5, 1.5, 7)
        for _ in range(1500):
            twist = rng.normal(size=6) * 0.2
            out, _ = limit_joints(twist, q, J, LOWER, UPPER, GUARD)
            q = q + predict_joint_velocity(out, J, GUARD.damping) * dt
            assert np.all(q <= UPPER + 0.01) and np.all(q >= LOWER - 0.01)


def test_hand_keypoint_jacobian_is_v_plus_w_cross_r():
    from franka_teleop.geometry import mat_from_quat
    from franka_teleop.node.limits import HAND_POINTS, hand_keypoints

    rng = np.random.default_rng(3)
    for _ in range(20):
        T = np.eye(4)
        T[:3, :3] = mat_from_quat(rng.normal(size=4))
        T[:3, 3] = rng.normal(size=3)
        J, dq = rng.normal(size=(6, 7)), rng.normal(size=7)
        v, w = (J @ dq)[:3], (J @ dq)[3:]
        for (p, Jp), local in zip(hand_keypoints(T, J), HAND_POINTS, strict=True):
            r = T[:3, :3] @ local
            assert np.allclose(p, T[:3, 3] + r)
            assert np.allclose(Jp @ dq, v + np.cross(w, r))


def test_limit_floor_ramps_descent_to_zero_and_never_blocks_lifting():
    from franka_teleop.node.limits import limit_floor

    J = np.zeros((3, 7))
    J[2, 0] = 1.0  # joint 0 velocity moves the keypoint vertically
    twist, dq = np.ones(6) * 0.1, np.r_[-0.1, np.zeros(6)]  # keypoint descends at 0.1 m/s
    point = lambda z: [(np.array([0.4, 0.0, z]), J, 0.07)]  # noqa: E731
    out, lim = limit_floor(twist, dq, point(0.5), 0.0, 0.05, 0.15)
    assert np.allclose(out, twist) and not lim  # far above the floor
    out, lim = limit_floor(twist, dq, point(0.07 + 0.02), 0.0, 0.05, 0.15)
    assert lim and np.allclose(out, twist * (3.0 * 0.02 / 0.1))  # gain 3/s * 2 cm
    out, lim = limit_floor(twist, dq, point(0.06), 0.0, 0.05, 0.15)
    assert lim and np.allclose(out, 0)  # inside the clearance: no further descent
    out, lim = limit_floor(twist, -dq, point(0.06), 0.0, 0.05, 0.15)
    assert not lim and np.allclose(out, twist)  # lifting is always allowed


def test_rolling_the_hand_over_near_the_table_stops_before_contact():
    """Closed loop: a hand 15 cm above the table keeps rolling (as in the soak failure).
    The hand corners must stay above floor + clearance."""
    from franka_teleop.geometry import exp_so3
    from franka_teleop.node.limits import hand_keypoints, limit_floor

    J = np.hstack([np.eye(6), np.zeros((6, 1))])  # dq[:6] is the twist
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # pointing down
    T[:3, 3] = [0.45, 0.0, 0.15]
    dt, lowest = 0.01, np.inf
    for _ in range(1000):
        twist = np.array([0, 0, 0, 0.6, 0, 0])
        keypoints = [(p, Jp, 0.01) for p, Jp in hand_keypoints(T, J)]
        out, _ = limit_floor(twist, np.r_[twist, 0.0], keypoints, 0.0, 0.05, 0.15)
        T[:3, :3] = exp_so3(out[3:] * dt) @ T[:3, :3]
        lowest = min(lowest, min(p[2] for p, _ in hand_keypoints(T, J)))
    assert lowest > 0.01 - 1e-3, lowest
