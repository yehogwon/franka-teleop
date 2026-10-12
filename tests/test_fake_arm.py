import numpy as np
import pytest

from franka_teleop.arm.base import ArmError
from franka_teleop.arm.fake_arm import FakeArm


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_velocity_hold_then_ramp_to_rest():
    clk = Clock()
    arm = FakeArm(clk)
    arm.send_twist(np.array([0.1, 0, 0, 0, 0, 0]), hold_ms=100)
    x0 = arm.read().ee_pos[0]
    clk.t = 0.09
    assert arm.read().in_control
    clk.t = 0.5
    s = arm.read()
    assert not s.in_control and s.ee_pos[0] > x0


def test_motion_type_switch_and_error_latch():
    clk = Clock()
    arm = FakeArm(clk)
    arm.send_twist(np.ones(6) * 0.01, hold_ms=100)
    with pytest.raises(ArmError):
        arm.move_joints(np.zeros(7))
    arm.stop()
    arm.move_joints(np.zeros(7))
    with pytest.raises(ArmError):
        arm.send_twist(np.zeros(6), hold_ms=100)
    clk.t = 1.0
    assert arm.motion_finished() and np.allclose(arm.read().q, 0)
    arm.inject_error()
    assert arm.read().errors == ("cartesian_reflex",)
    with pytest.raises(ArmError):
        arm.send_twist(np.zeros(6), hold_ms=100)
    assert arm.recover() and arm.read().errors == ()
