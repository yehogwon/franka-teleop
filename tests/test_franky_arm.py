"""FrankyArm against franky-sim over the real libfranka protocol."""

import time

import numpy as np
import pytest

from franka_teleop.arm.base import ArmError
from franka_teleop.arm.franky_arm import FrankyArm
from franka_teleop.config import READY_Q

from .conftest import make_test_config

pytestmark = pytest.mark.sim


@pytest.fixture
def arm(franky_sim):
    a = FrankyArm(make_test_config(franky_sim.host).robot)
    yield a
    a.close()


def wait_until(pred, timeout, period=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(period)
    return False


def test_read_reports_ready_pose(arm):
    s = arm.read()
    assert s.q.shape == (7,) and s.jacobian.shape == (6, 7) and s.O_T_EE.shape == (4, 4)
    assert np.allclose(s.q, READY_Q, atol=2e-3)
    assert np.allclose(s.ee_pos, [0.307, 0.0, 0.487], atol=2e-3)
    assert np.allclose(s.ee_rot, np.diag([1.0, -1.0, -1.0]), atol=1e-2)  # pointing down
    assert s.errors == () and not s.in_control and s.robot_mode == "Idle"


def test_jacobian_maps_joint_velocity_to_ee_twist(arm):
    """J(q) @ dq reproduces the commanded twist while the robot tracks it."""
    arm.send_twist(np.array([0.03, 0, 0, 0, 0, 0]), hold_ms=300)
    time.sleep(0.25)
    s = arm.read()
    assert s.in_control
    assert np.allclose(s.jacobian @ s.dq, [0.03, 0, 0, 0, 0, 0], atol=4e-3)


def test_twist_moves_then_robot_stops_on_its_own(arm):
    p0 = arm.read().ee_pos.copy()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        arm.send_twist(np.array([0.0, 0.04, 0.0, 0, 0, 0]), hold_ms=100)
        time.sleep(0.01)
    assert wait_until(lambda: not arm.read().in_control, 1.0)
    d = arm.read().ee_pos - p0
    assert 0.03 < d[1] < 0.045 and abs(d[0]) < 3e-3 and abs(d[2]) < 3e-3


def test_angular_twist_rotates_about_the_ee_point(arm):
    s0 = arm.read()
    arm.send_twist(np.array([0, 0, 0, 0, 0, 0.3]), hold_ms=500)
    assert wait_until(lambda: arm.motion_finished(), 3.0)
    s1 = arm.read()
    dR = s1.ee_rot @ s0.ee_rot.T
    assert np.arctan2(dR[1, 0], dR[0, 0]) > 0.1  # +z yaw in the base frame
    assert np.linalg.norm(s1.ee_pos - s0.ee_pos) < 3e-3  # rotation about the EE, not the base


def test_stop_is_clean_and_type_switch_is_reported(arm):
    arm.send_twist(np.array([0.03, 0, 0, 0, 0, 0]), hold_ms=2000)
    time.sleep(0.1)
    with pytest.raises(ArmError, match="motion type"):
        arm.move_joints(READY_Q)
    arm.stop()  # must swallow the expected "preempted" error
    assert arm.motion_finished()
    assert arm.read().errors == ()


def test_home_reaches_target_after_settling(arm):
    target = np.array([0.2, -0.5, 0.1, -2.0, 0.1, 1.8, 0.6])
    arm.move_joints(target)
    assert wait_until(arm.motion_finished, 10.0)
    time.sleep(1.0)  # franky-sim's joint PD lags ~65 mrad at "finished", ~2 mrad after 1 s
    assert np.abs(arm.read().q - target).max() < 0.01


def test_gripper_close_and_open(arm):
    arm.set_gripper(True)
    assert wait_until(lambda: not arm.read().gripper_busy, 5.0)
    assert arm.read().gripper_width < 0.01
    arm.set_gripper(False)
    assert wait_until(lambda: not arm.read().gripper_busy, 5.0)
    assert arm.read().gripper_width > 0.07


def test_recover_succeeds(arm):
    assert arm.recover()


def test_halt_decelerates_velocity_and_joint_motions(arm):
    arm.send_twist(np.array([0.05, 0, 0, 0, 0, 0]), hold_ms=5000)
    time.sleep(0.3)
    arm.halt()
    assert wait_until(arm.motion_finished, 1.0)
    s = arm.read()
    assert s.errors == () and np.abs(s.dq).max() < 0.02
    arm.move_joints([0.3, -0.5, 0.0, -2.2, 0.0, 1.8, 0.785])
    time.sleep(0.3)
    arm.halt()
    assert wait_until(arm.motion_finished, 2.0)
    q_stop = arm.read().q
    time.sleep(0.5)
    assert np.abs(arm.read().q - q_stop).max() < 0.02  # stays put, did not continue to the goal
    arm.halt()  # idle: no-op
