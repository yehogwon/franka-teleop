"""ft-node logic + FrankyArm + franky-sim over the real libfranka protocol, with commands and
status over real ZMQ sockets. Measures what the safety design promises: stop latencies for each
failure mode, and workspace / joint-limit compliance in closed loop."""

from __future__ import annotations

import itertools
import threading
import time

import numpy as np
import pytest

from franka_teleop import transport
from franka_teleop.arm.franky_arm import FrankyArm
from franka_teleop.node.server import NodeServer
from franka_teleop.protocol import ACTIONS, NodeStatus, TeleopCommand

from ..conftest import make_test_config

pytestmark = pytest.mark.sim

RATE = 100.0


class Operator(threading.Thread):
    """A joystick stand-in: publishes TeleopCommands at 100 Hz from a mutable intent."""

    def __init__(self, cfg, session=4242):
        super().__init__(daemon=True)
        self.pub = transport.publisher(cfg.net.bind(cfg.net.cmd_port))
        self.session, self.seq = session, 0
        self.deadman, self.linear, self.angular = False, (0, 0, 0), (0, 0, 0)
        self.counters = dict.fromkeys(ACTIONS, 0)
        self.alive = True
        self.running = True

    def press(self, action):
        self.counters[action] += 1

    def set(self, deadman=True, linear=(0, 0, 0), angular=(0, 0, 0)):
        self.deadman, self.linear, self.angular = deadman, linear, angular

    def run(self):
        while self.running:
            if self.alive:
                self.seq += 1
                transport.send(
                    self.pub,
                    TeleopCommand(
                        session=self.session,
                        seq=self.seq,
                        t_send=time.time(),
                        source="script",
                        deadman=self.deadman,
                        linear=self.linear,
                        angular=self.angular,
                        **self.counters,
                    ),
                )
            time.sleep(1 / RATE)
        self.pub.close()


class Rig:
    def __init__(self, host, **overrides):
        self.cfg = make_test_config(host, **overrides)
        self.op = Operator(self.cfg)
        self.op.start()
        self.arm = FrankyArm(self.cfg.robot)
        self.server = NodeServer(self.cfg, self.arm)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.status_sub = transport.subscriber(self.cfg.net.connect(self.cfg.net.status_port))
        assert transport.recv_wait(self.status_sub, NodeStatus, 10.0) is not None, "no NodeStatus"

    def status(self) -> NodeStatus:
        s = transport.recv_wait(self.status_sub, NodeStatus, 1.0)
        assert s is not None
        return s

    def engage(self):
        self.op.set(deadman=False)
        time.sleep(0.1)
        self.op.set(deadman=True)
        time.sleep(0.1)

    def drive(self, seconds, **intent):
        self.op.set(deadman=True, **intent)
        time.sleep(seconds)

    def wait_rest(self, timeout=2.0, speed=1e-3):
        """Seconds until the EE speed (from J dq) drops below `speed` m/s."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s = self.arm_state()
            if np.linalg.norm((s.jacobian @ s.dq)[:3]) < speed:
                return time.monotonic() - t0
            time.sleep(0.002)
        return None

    def arm_state(self):
        if self.thread.is_alive():
            return self.server.node.state  # never call franky from two threads at once
        return self.arm.read()

    def close(self):
        self.op.running = False
        self.server.running = False
        self.thread.join(5)
        self.server.close()
        self.status_sub.close()
        self.op.join(2)


@pytest.fixture
def rig(franky_sim_proc):
    r = Rig(franky_sim_proc)
    yield r
    r.close()


def test_forward_motion_matches_commanded_speed(rig, record_property):
    p0 = rig.status().ee_pos
    rig.engage()
    rig.drive(0.5, linear=(1, 0, 0))
    samples = []
    for _ in range(50):  # steady state: robot-side EE velocity, independent of sim speed
        time.sleep(0.01)
        a = rig.arm_state()
        samples.append(a.jacobian @ a.dq)
    s = rig.status()
    assert s.mode == "teleop" and s.limiters == ()
    v = 0.15 * rig.cfg.teleop.speed_levels[rig.cfg.teleop.initial_speed_level]
    mean = np.mean(samples, axis=0)
    record_property("steady_vx", float(mean[0]))
    assert abs(mean[0] - v) < 0.05 * v, mean
    assert np.abs(mean[1:3]).max() < 0.03 * v and np.abs(mean[3:]).max() < 0.01
    rig.op.set(deadman=False)
    assert rig.wait_rest() is not None
    s = rig.status()
    d = np.subtract(s.ee_pos, p0)
    assert d[0] > 0.04 and abs(d[1]) < 3e-3 and abs(d[2]) < 3e-3
    assert s.robot_errors == "" and s.mode == "idle"


def measure_stop(rig, trigger):
    rig.engage()
    rig.drive(0.6, linear=(1, 0, 0))
    trigger()
    latency = rig.wait_rest(timeout=3.0)
    assert latency is not None
    return latency


def test_stop_latency_deadman_release(rig, record_property):
    latency = measure_stop(rig, lambda: rig.op.set(deadman=False))
    record_property("deadman_release_stop_s", latency)
    assert latency < 0.25, latency


def test_stop_latency_operator_dies(rig, record_property):
    def kill():
        rig.op.alive = False  # stream stops mid-motion, deadman "held"

    latency = measure_stop(rig, kill)
    record_property("operator_death_stop_s", latency)
    timeout = rig.cfg.teleop.cmd_timeout_ms / 1000
    assert timeout < latency < timeout + 0.25, latency
    assert rig.status().mode == "idle"


def test_stop_latency_node_dies(rig, record_property):
    """If ft-node stops ticking, franky's hold_ms makes the robot stop on its own."""

    def freeze():
        rig.server.running = False  # no more ticks, no halt: only the robot-side hold remains
        rig.thread.join(1.0)

    latency = measure_stop(rig, freeze)
    record_property("node_death_stop_s", latency)
    assert latency < rig.cfg.teleop.hold_ms / 1000 + 0.3, latency


def move_to(rig, target, tol=0.005, timeout=10.0):
    """Drive the EE to `target` with a proportional joystick."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        err = np.subtract(target, rig.status().ee_pos)
        if np.linalg.norm(err) < tol:
            return
        rig.op.set(deadman=True, linear=tuple(float(x) for x in np.clip(err * 10, -1, 1)))
        time.sleep(0.01)
    s = rig.status()
    raise AssertionError(
        f"could not reach {target}: at {np.round(s.ee_pos, 3)}, mode {s.mode}, limiters "
        f"{s.limiters}, q {np.round(s.q, 2)}, msg {s.message!r}"
    )


# Each wall is approached from a start where it is comfortably reachable: the table (-z) right
# next to the base would need the elbow folded onto its joint limit.
@pytest.mark.parametrize(
    ("axis", "sign", "start"),
    [(0, 1, (0.45, 0.0, 0.25)), (2, 1, None), (2, -1, (0.5, 0.0, 0.3)), (1, -1, None)],
)
def test_workspace_walls_hold_in_closed_loop(rig, axis, sign, start, record_property):
    ws = rig.cfg.workspace
    for _ in range(3):  # top speed level
        rig.op.press("speed_up")
        time.sleep(0.03)
    rig.engage()
    if start is not None:
        move_to(rig, start)
    intent = [0, 0, 0]
    intent[axis] = sign
    wall = ws.upper[axis] if sign > 0 else ws.lower[axis]
    worst = -np.inf
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6.0:
        rig.op.set(deadman=True, linear=tuple(intent))
        worst = max(worst, sign * (rig.status().ee_pos[axis] - wall))
        time.sleep(0.01)
    s = rig.status()
    record_property(f"overshoot_axis{axis}_{sign:+d}_mm", worst * 1e3)
    assert worst < 2e-3, f"overshoot {worst * 1e3:.2f} mm"
    gap = abs(s.ee_pos[axis] - wall)
    assert gap < 0.01, f"stopped {gap * 1e3:.1f} mm short of the wall, limiters {s.limiters}"
    assert s.limiters == ("workspace",) and s.robot_errors == ""


def test_singularity_guard_stops_before_tracking_breaks(rig, record_property):
    """Reaching out and up straightens the elbow. Without the guard, franky-sim's tracking
    collapses there (the hand drifted down 3 cm/s with no command)."""
    from franka_teleop.node.limits import min_singular_value

    g = rig.cfg.joint_guard
    for _ in range(3):
        rig.op.press("speed_up")
        time.sleep(0.03)
    rig.engage()
    worst_err, min_sigma, saw = 0.0, np.inf, False
    t0 = time.monotonic()
    while time.monotonic() - t0 < 8.0:
        rig.op.set(deadman=True, linear=(1, 0, 0.6))
        time.sleep(0.01)
        s, a = rig.status(), rig.arm_state()
        saw |= "singularity" in s.limiters
        min_sigma = min(min_sigma, min_singular_value(a.jacobian, g.rot_weight))
        if time.monotonic() - t0 > 0.5:  # tracking error once past the initial ramp
            worst_err = max(worst_err, np.linalg.norm((a.jacobian @ a.dq)[:3] - s.twist[:3]))
    record_property("min_sigma", min_sigma)
    record_property("worst_tracking_error_mps", worst_err)
    assert saw and min_sigma > g.sigma_stop - 0.005, min_sigma
    assert worst_err < 0.02, worst_err
    assert rig.status().robot_errors == ""
    # backing out is never blocked
    p = rig.status().ee_pos
    rig.drive(0.5, linear=(-1, 0, -0.6))
    assert rig.status().ee_pos[0] < p[0] - 0.02


@pytest.mark.parametrize("yaw", [1, -1])
def test_joint_guard_stops_wrist_before_its_limit(rig, yaw, record_property):
    for _ in range(3):  # top speed level: 0.6 rad/s
        rig.op.press("speed_up")
        time.sleep(0.03)
    rig.engage()
    lo, hi = rig.cfg.robot.joint_lower[6], rig.cfg.robot.joint_upper[6]
    closest = np.inf
    t0 = time.monotonic()
    saw_guard = False
    # With the hand pointing down, base-frame yaw is mostly joint 7 (axis = -z): CCW -> q7 falls.
    while time.monotonic() - t0 < 12.0:
        rig.op.set(deadman=True, angular=(0, 0, yaw))
        s = rig.status()
        closest = min(closest, hi - s.q[6], s.q[6] - lo)
        saw_guard |= "joint" in s.limiters
        time.sleep(0.01)
    record_property(f"joint7_min_margin_rad_yaw{yaw:+d}", closest)
    assert saw_guard and closest > 0.0, closest
    assert rig.status().robot_errors == ""


def test_home_and_gripper(rig):
    rig.engage()
    rig.drive(0.8, linear=(0.5, 0.5, -0.5), angular=(0.3, 0, 0.5))
    rig.op.press("home")
    t0 = time.monotonic()
    while rig.status().mode != "homing":
        assert time.monotonic() - t0 < 1.0
    while rig.status().mode == "homing":
        assert time.monotonic() - t0 < 15.0
    s = rig.status()
    assert "home reached" in s.message
    time.sleep(1.0)  # franky-sim's joint PD settles ~1 s after the motion reports done
    assert np.abs(np.subtract(rig.status().q, rig.cfg.robot.home_q)).max() < 0.01
    rig.engage()
    rig.op.press("gripper")
    time.sleep(2.5)
    assert rig.status().gripper_width < 0.01 and rig.status().gripper_closed
    rig.op.press("gripper")
    time.sleep(2.5)
    assert rig.status().gripper_width > 0.07 and not rig.status().gripper_closed


def test_tool_frame_rotation_pivots_about_the_fingertips(rig):
    rig.engage()
    rig.op.press("frame")
    time.sleep(0.05)
    s0 = rig.status()
    assert s0.frame == "tool"
    rig.drive(1.5, angular=(1, 0, 0))  # roll about the tool x axis
    rig.op.set(deadman=False)
    rig.wait_rest()
    s1 = rig.status()
    assert np.linalg.norm(np.subtract(s1.ee_pos, s0.ee_pos)) < 5e-3
    assert np.abs(np.subtract(s1.ee_quat, s0.ee_quat)).max() > 0.05  # it did rotate


def test_status_stream_rate_and_echo(rig):
    rig.engage()
    seen, t0 = [], time.monotonic()
    while time.monotonic() - t0 < 1.0:
        s = transport.recv_wait(rig.status_sub, NodeStatus, 0.1)
        if s is not None:
            seen.append(s)
    assert len(seen) > 50  # subscriber CONFLATE may skip some of the 100 Hz statuses
    assert all(a.seq < b.seq for a, b in itertools.pairwise(seen))
    assert seen[-1].last_cmd_seq > 0 and seen[-1].session == 4242
    assert seen[-1].cmd_age_ms < 50 and rig.server.node.loop_overruns == 0


def test_floor_guard_keeps_wrist_and_hand_off_the_table(rig, record_property):
    """Deterministic version of a soak failure: low over the table, keep rolling the hand over.
    Without the floor guard the wrist hit the table (franky-sim then shoved joint 6 past its
    limit; a real arm would have taken a collision reflex)."""
    from franka_teleop.node.limits import hand_keypoints

    ws = rig.cfg.workspace
    for _ in range(3):
        rig.op.press("speed_up")
        time.sleep(0.03)
    rig.engage()
    move_to(rig, (0.45, 0.0, 0.15))
    lowest, saw = np.inf, False
    t0 = time.monotonic()
    while time.monotonic() - t0 < 8.0:
        rig.op.set(deadman=True, angular=(1, 0, 0))
        time.sleep(0.01)
        a, s = rig.arm_state(), rig.status()
        saw |= "floor" in s.limiters
        links = [p[2] - ws.link_clearance for p, _ in rig.arm.keypoints_at(a.q)]
        hand = [p[2] - ws.hand_clearance for p, _ in hand_keypoints(a.O_T_EE, a.jacobian)]
        lowest = min(lowest, *links, *hand)
    record_property("floor_min_margin_mm", (lowest - ws.floor) * 1e3)
    assert saw and lowest - ws.floor > -3e-3, f"{(lowest - ws.floor) * 1e3:.1f} mm"
    assert rig.status().robot_errors == ""
