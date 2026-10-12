"""TeleopNode state machine against the kinematic FakeArm with a manual clock."""

import time

import numpy as np
import pytest

from franka_teleop.arm.fake_arm import FakeArm
from franka_teleop.config import load_config
from franka_teleop.node.teleop import ERROR, HOMING, IDLE, LOCKED, TELEOP, TeleopNode
from franka_teleop.protocol import ACTIONS, TeleopCommand

from ..conftest import CONFIGS

DT = 0.01


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Operator:
    """Builds a TeleopCommand stream the way ft-joystick does."""

    def __init__(self, session=11, source="script"):
        self.session, self.source, self.seq = session, source, 0
        self.counters = dict.fromkeys(ACTIONS, 0)

    def cmd(self, deadman=True, linear=(0, 0, 0), angular=(0, 0, 0), press=()):
        for a in press:
            self.counters[a] += 1
        self.seq += 1
        return TeleopCommand(
            session=self.session,
            seq=self.seq,
            t_send=time.time(),
            source=self.source,
            deadman=deadman,
            linear=linear,
            angular=angular,
            **self.counters,
        )


class Rig:
    def __init__(self, config="sim.toml", **overrides):
        self.clock = Clock()
        self.arm = FakeArm(self.clock)
        self.cfg = load_config(CONFIGS / config, **overrides)
        self.node = TeleopNode(self.arm, self.cfg, clock=self.clock)
        self.op = Operator()

    def step(self, cmd=None):
        self.clock.t += DT
        return self.node.tick(cmd)

    def run(self, seconds, **cmd_kw):
        status = None
        for _ in range(round(seconds / DT)):
            status = self.step(self.op.cmd(**cmd_kw))
        return status

    def silent(self, seconds):
        status = None
        for _ in range(round(seconds / DT)):
            status = self.step(None)
        return status

    def twists(self):
        return [c[1] for c in self.arm.calls if c[0] == "send_twist"]

    def count(self, name):
        return sum(1 for c in self.arm.calls if c[0] == name)

    def engage(self):
        """Release, then hold the deadman: the normal way to start."""
        self.run(0.05, deadman=False)
        self.run(0.05, deadman=True)


def test_deadman_must_be_released_once_per_session():
    rig = Rig()
    rig.run(0.5, deadman=True, linear=(1, 0, 0))  # holding from the very first message
    assert rig.twists() == [] and rig.node.mode == IDLE
    rig.run(0.05, deadman=False)
    s = rig.run(0.2, deadman=True, linear=(1, 0, 0))
    assert s.mode == TELEOP and rig.twists()
    expected = 0.15 * rig.cfg.teleop.speed_levels[rig.cfg.teleop.initial_speed_level]
    assert np.allclose(rig.twists()[-1], [expected, 0, 0, 0, 0, 0])
    assert rig.arm.read().ee_pos[0] > 0.307


def test_releasing_deadman_halts_and_sends_nothing_more():
    rig = Rig()
    rig.engage()
    rig.run(0.3, deadman=True, linear=(0, 1, 0))
    n = len(rig.twists())
    s = rig.run(0.5, deadman=False, linear=(0, 1, 0))
    assert s.mode == IDLE and rig.count("halt") == 1 and len(rig.twists()) == n
    assert np.allclose(s.twist, 0)


def test_stick_release_sends_one_zero_twist_then_lets_hold_expire():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    n = len(rig.twists())
    rig.run(0.5, deadman=True)
    assert len(rig.twists()) == n + 1 and not rig.twists()[-1].any()
    assert not rig.arm.read().in_control


def test_stale_stream_halts_and_long_outage_requires_rearm():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    timeout_ticks = int(rig.cfg.teleop.cmd_timeout_ms / 1000 / DT)
    ticks = 0
    while not rig.count("halt"):
        s = rig.step(None)  # the last fresh command keeps streaming until it goes stale
        ticks += 1
        assert ticks <= timeout_ticks + 1
    assert ticks >= timeout_ticks
    n = len(rig.twists())
    s = rig.silent(0.2)
    assert len(rig.twists()) == n and s.mode == IDLE and s.message == "operator stream stale"
    # short outage: motion resumes while the deadman is held
    rig.silent(0.3)
    s = rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert s.mode == TELEOP and len(rig.twists()) > n and s.message == "operator stream resumed"
    # long outage: must release first
    rig.silent(1.5)
    m = len(rig.twists())
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) == m
    rig.engage()
    rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) > m


def test_disallowed_source_is_ignored_on_real_config():
    rig = Rig("real.toml")
    rig.op = Operator(source="keyboard")
    rig.engage()
    s = rig.run(0.3, deadman=True, linear=(1, 0, 0))
    assert rig.twists() == [] and "not allowed" in s.message and s.session == 0


def test_gripper_toggle_needs_deadman_and_fires_once_per_press():
    rig = Rig()
    rig.run(0.05, deadman=False, press=("gripper",))
    assert rig.count("set_gripper") == 0
    rig.run(0.05, deadman=False)
    rig.step(rig.op.cmd(deadman=False, press=("gripper",)))
    assert rig.count("set_gripper") == 0 and "hold the deadman" in rig.node.message
    rig.run(0.05, deadman=True)
    rig.step(rig.op.cmd(deadman=True, press=("gripper",)))
    rig.run(0.2, deadman=True)  # holding the counter value does not re-trigger
    assert [c for c in rig.arm.calls if c[0] == "set_gripper"] == [("set_gripper", True)]
    rig.step(rig.op.cmd(deadman=True, press=("gripper",)))
    assert rig.arm.calls[-1] == ("set_gripper", False)


def test_home_waits_for_rest_then_moves_and_disarms():
    rig = Rig()
    rig.engage()
    rig.run(0.3, deadman=True, linear=(1, 0, 0))
    rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0), press=("home",)))
    assert rig.node.mode == HOMING and rig.count("halt") == 1 and rig.count("move_joints") == 0
    s = rig.run(1.5, deadman=True)
    assert rig.count("move_joints") == 1
    assert np.allclose(rig.arm.read().q, rig.cfg.robot.home_q)
    assert s.mode == IDLE and "home reached" in s.message
    n = len(rig.twists())
    rig.run(0.2, deadman=True, linear=(1, 0, 0))  # disarmed after homing
    assert len(rig.twists()) == n


def test_releasing_deadman_aborts_homing():
    rig = Rig()
    rig.engage()
    rig.step(rig.op.cmd(deadman=True, press=("home",)))
    rig.run(0.1, deadman=True)
    assert rig.count("move_joints") == 1
    s = rig.run(0.05, deadman=False)
    assert s.mode == IDLE and "aborted" in s.message and rig.count("halt") == 1


def test_reengaging_while_a_joint_stop_decelerates_waits_instead_of_faulting():
    """Found by the soak test: abort homing (a joint stop motion starts decelerating), re-engage
    at once and push a stick. A velocity motion must not be sent until the joint motion ended,
    or franky raises 'motion type change while moving'."""
    rig = Rig()
    rig.engage()
    rig.step(rig.op.cmd(deadman=True, press=("home",)))
    rig.run(0.1, deadman=True)
    assert rig.count("move_joints") == 1
    rig.run(0.02, deadman=False)  # abort: joint stop motion now decelerating for 0.2 s
    s = rig.run(0.05, deadman=True, linear=(1, 0, 0))
    assert s.mode != "error" and rig.twists() == [] and "waiting" in s.message
    s = rig.run(0.3, deadman=True, linear=(1, 0, 0))
    assert s.mode == TELEOP and rig.twists()


def test_unlocking_during_a_joint_stop_waits_too():
    rig = Rig()
    rig.engage()
    rig.step(rig.op.cmd(deadman=True, press=("home",)))
    rig.run(0.1, deadman=True)
    rig.step(rig.op.cmd(deadman=True, press=("stop",)))  # joint stop motion, locked
    rig.step(rig.op.cmd(deadman=False, press=("recover",)))
    s = rig.run(0.05, deadman=True, linear=(0, 1, 0))
    assert s.mode != "error" and rig.twists() == []
    s = rig.run(0.3, deadman=True, linear=(0, 1, 0))
    assert s.mode == TELEOP and rig.twists()


def test_robot_error_latches_until_recover_and_requires_rearm():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    rig.arm.inject_error("cartesian_reflex")
    s = rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert s.mode == ERROR and "cartesian_reflex" in s.message
    n = len(rig.twists())
    rig.run(0.3, deadman=False)
    rig.run(0.3, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) == n and rig.node.mode == ERROR  # releasing does not clear errors
    rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0), press=("recover",)))
    assert rig.node.mode == IDLE and rig.count("recover") == 1
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) == n  # still holding since recovery: not armed
    rig.engage()
    rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) > n


def test_failed_recovery_stays_in_error():
    rig = Rig()
    rig.arm.inject_error("joint_reflex")
    rig.arm.recover_ok = False
    rig.run(0.1, deadman=False)
    s = rig.step(rig.op.cmd(deadman=False, press=("recover",)))
    assert s.mode == ERROR and "recovery failed" in s.message


def test_arm_exception_enters_error():
    rig = Rig()
    rig.engage()
    rig.arm.move_joints(np.zeros(7))  # a joint motion is running: velocity commands fail
    s = rig.run(0.05, deadman=True, linear=(1, 0, 0))
    assert s.mode == ERROR and "motion type" in s.message


def test_stop_locks_until_recover():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    s = rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0), press=("stop",)))
    assert s.mode == LOCKED and rig.count("halt") == 1
    n = len(rig.twists())
    rig.run(0.1, deadman=False)
    rig.run(0.3, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) == n
    rig.step(rig.op.cmd(deadman=False, press=("recover",)))
    assert rig.node.mode == IDLE and rig.count("recover") == 0  # unlocking needs no recovery
    rig.engage()
    rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) > n


def test_speed_levels_clamp_and_frame_toggles():
    rig = Rig()
    rig.engage()
    levels = rig.cfg.teleop.speed_levels
    for _ in range(10):
        rig.step(rig.op.cmd(press=("speed_up",)))
        rig.step(rig.op.cmd())
    assert rig.node.speed_level == len(levels) - 1
    s = rig.run(0.05, deadman=True, linear=(1, 0, 0))
    assert np.isclose(s.twist[0], 0.15 * levels[-1]) and s.speed_scale == levels[-1]
    for _ in range(10):
        rig.step(rig.op.cmd(press=("speed_down",)))
        rig.step(rig.op.cmd())
    assert rig.node.speed_level == 0
    rig.step(rig.op.cmd(press=("frame",)))
    s = rig.run(0.05, deadman=True, linear=(0, 0, 1))
    assert s.frame == "tool" and s.twist[2] < 0  # tool "up" = approach = base -z at the ready pose


def test_workspace_is_enforced_in_closed_loop():
    rig = Rig()
    rig.engage()
    upper_x = rig.cfg.workspace.upper[0]
    for _ in range(1200):  # 12 s at 0.075 m/s: well past the +x wall
        s = rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0)))
        assert rig.arm.read().ee_pos[0] <= upper_x + 1e-3
    assert "workspace" in s.limiters and abs(s.ee_pos[0] - upper_x) < 0.01


def test_stale_and_duplicate_seq_are_ignored():
    rig = Rig()
    rig.engage()
    good = rig.op.cmd(deadman=True, linear=(1, 0, 0))
    rig.step(good)
    n = len(rig.twists())
    rig.step(good)  # duplicate seq
    old = TeleopCommand(
        session=good.session,
        seq=good.seq - 5,
        t_send=0,
        source="script",
        deadman=True,
        home=99,
    )
    s = rig.step(old)
    assert rig.node.mode != HOMING and s.last_cmd_seq == good.seq
    assert len(rig.twists()) == n + 2  # the node keeps streaming the last fresh command


def test_new_session_does_not_replay_counters():
    rig = Rig()
    rig.engage()
    rig.op = Operator(session=99)
    rig.op.counters = dict.fromkeys(ACTIONS, 5)  # a restarted client with nonzero counters
    s = rig.run(0.1, deadman=True)
    assert rig.count("set_gripper") == 0 and s.mode == IDLE and s.session == 99


@pytest.mark.parametrize("field", ["seq", "session", "last_cmd_seq", "cmd_age_ms"])
def test_status_reports_link_health(field):
    rig = Rig()
    s = rig.step(None)
    assert s.session == 0 and s.last_cmd_seq == -1 and s.cmd_age_ms == float("inf")
    c = rig.op.cmd(deadman=False)
    s = rig.step(c)
    assert s.session == c.session and s.last_cmd_seq == c.seq and s.cmd_age_ms == 0.0
    s = rig.step(None)
    assert np.isclose(s.cmd_age_ms, DT * 1000)
    assert getattr(s, field) is not None


def test_shutdown_halts_motion_and_closes_arm():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))

    def advancing_clock():
        rig.clock.t += 0.01
        return rig.clock.t

    rig.node.clock = advancing_clock
    rig.node.shutdown()
    assert rig.count("halt") == 1
    assert rig.arm.calls[-1] == ("close", {"in_control": False})


@pytest.mark.parametrize("scenario", ["deadman_release", "aborted_homing", "stop_button"])
def test_shutdown_waits_for_a_braking_stop_motion(scenario):
    """Found in review: shutdown right after a halt used to close() while the stop motion was
    still braking, and FrankyArm.close() then aborted it with stop()."""
    rig = Rig()
    rig.engage()
    if scenario == "aborted_homing":
        rig.step(rig.op.cmd(deadman=True, press=("home",)))
        rig.run(0.1, deadman=True)
        rig.step(rig.op.cmd(deadman=False))  # joint stop motion braking for 0.2 s
    elif scenario == "stop_button":
        rig.run(0.3, deadman=True, linear=(1, 0, 0))
        rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0), press=("stop",)))
    else:
        rig.run(0.3, deadman=True, linear=(1, 0, 0))
        rig.step(rig.op.cmd(deadman=False))
    assert rig.arm.read().in_control  # still braking

    def advancing_clock():
        rig.clock.t += 0.01
        return rig.clock.t

    rig.node.clock = advancing_clock
    rig.arm.clock = advancing_clock
    rig.node.shutdown()
    assert rig.arm.calls[-1] == ("close", {"in_control": False})
    assert rig.count("stop") == 0


def test_non_finite_intent_is_dropped():
    rig = Rig()
    rig.engage()
    rig.run(0.1, deadman=True, linear=(1, 0, 0))
    n = len(rig.twists())
    s = rig.step(rig.op.cmd(deadman=True, linear=(float("nan"), 0, 0)))
    assert "non-finite" in s.message and all(np.isfinite(t).all() for t in rig.twists())
    s = rig.run(0.3, deadman=True, linear=(float("inf"), 0, 0))
    assert s.mode == IDLE and rig.count("halt") == 1  # dropped commands -> stale -> halt
    assert len(rig.twists()) <= n + int(rig.cfg.teleop.cmd_timeout_ms / 10) + 1


def test_stop_wins_over_recover_in_the_same_message():
    rig = Rig()
    rig.engage()
    rig.step(rig.op.cmd(deadman=False, press=("stop",)))
    assert rig.node.mode == LOCKED
    s = rig.step(rig.op.cmd(deadman=False, press=("recover", "stop")))
    assert s.mode == LOCKED


def test_frame_toggle_is_refused_while_moving():
    rig = Rig()
    rig.engage()
    rig.run(0.1, deadman=True, linear=(0, 0, 1))
    s = rig.step(rig.op.cmd(deadman=True, linear=(0, 0, 1), press=("frame",)))
    assert s.frame == "base" and "release the sticks" in s.message
    rig.run(0.1, deadman=True)  # sticks released: one zero twist sent
    s = rig.step(rig.op.cmd(deadman=True, press=("frame",)))
    assert s.frame == "tool"


def test_single_planner_rejection_is_retried_transparently():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    rig.arm.planner_fail_next = True
    s = rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0)))
    assert "retrying" in s.message and rig.node.planner_rejections == 1
    n = len(rig.twists())
    s = rig.run(0.1, deadman=True, linear=(1, 0, 0))
    assert s.mode == TELEOP and len(rig.twists()) > n  # motion resumed without a release
    assert rig.count("halt") == 0 and rig.count("recover") == 0


def test_repeated_planner_rejections_halt_and_require_release():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    for _ in range(3):
        rig.arm.planner_fail_next = True
        s = rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0)))
        rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0)))
    assert s.mode == IDLE and "repeatedly" in s.message and rig.count("halt") == 1
    n = len(rig.twists())
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    assert len(rig.twists()) == n  # disarmed
    rig.engage()
    assert rig.run(0.1, deadman=True, linear=(1, 0, 0)).mode == TELEOP


def test_planner_rejection_with_robot_error_latches_error():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    rig.arm.planner_fail_next = True
    rig.arm.errors = ("cartesian_motion_generator_velocity_discontinuity",)
    s = rig.step(rig.op.cmd(deadman=True, linear=(1, 0, 0)))
    assert s.mode == ERROR


def test_near_zero_twist_components_are_sent_as_zero():
    rig = Rig()
    rig.engage()
    rig.run(0.2, deadman=True, linear=(1, 0, 0))
    s = rig.step(rig.op.cmd(deadman=True, linear=(0.004, 0, 0), angular=(0, 0.005, 1)))
    # 0.004 * 0.075 m/s = 0.3 mm/s and 0.005 * 0.3 rad/s = 1.5 mrad/s fall in the deadband
    assert s.twist[0] == 0.0 and s.twist[4] == 0.0 and s.twist[5] > 0
