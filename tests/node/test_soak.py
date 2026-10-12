"""Soak: a randomized operator drives ft-node + franky-sim for minutes while every NodeStatus is
checked against the safety invariants. FT_SOAK_SECONDS sets the duration (default 60)."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from franka_teleop import transport
from franka_teleop.node.limits import min_singular_value
from franka_teleop.protocol import NodeStatus

from .test_node_sim import Rig

pytestmark = [pytest.mark.sim, pytest.mark.slow]

SOAK_S = float(os.environ.get("FT_SOAK_SECONDS", "60"))


class Recorder(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.sub = transport.subscriber(cfg.net.connect(cfg.net.status_port), conflate=False)
        self.statuses: list[NodeStatus] = []
        self.running = True

    def run(self):
        while self.running:
            s = transport.recv_wait(self.sub, NodeStatus, 0.1)
            if s is not None:
                self.statuses.append(s)
        self.sub.close()


def test_randomized_operator_soak(franky_sim_proc, record_property):
    rig = Rig(franky_sim_proc)
    rec = Recorder(rig.cfg)
    rec.start()
    rng = np.random.default_rng(int(os.environ.get("FT_SOAK_SEED", "7")))
    cfg, g = rig.cfg, rig.cfg.joint_guard
    lower, upper = np.array(cfg.workspace.lower), np.array(cfg.workspace.upper)
    jl, ju = np.array(cfg.robot.joint_lower), np.array(cfg.robot.joint_upper)
    lin, ang = np.zeros(3), np.zeros(3)
    deadman = False
    events = {"homes": 0, "stops": 0, "gripper": 0, "frames": 0}
    min_sigma, worst_ws, worst_joint = np.inf, -np.inf, np.inf
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < SOAK_S:
            # Ornstein-Uhlenbeck sticks: smooth, occasionally saturated
            lin += -0.5 * lin * 0.02 + rng.normal(0, 0.25, 3)
            ang += -0.5 * ang * 0.02 + rng.normal(0, 0.15, 3)
            lin, ang = np.clip(lin, -1, 1), np.clip(ang, -1, 1)
            r = rng.random()
            if r < 0.01:
                deadman = not deadman
            elif r < 0.013:
                rig.op.press(rng.choice(["speed_up", "speed_down"]))
            elif r < 0.015:
                rig.op.press("frame")
                events["frames"] += 1
            elif r < 0.017:
                rig.op.press("gripper")
                events["gripper"] += 1
            elif r < 0.0175:
                rig.op.press("home")
                events["homes"] += 1
            elif r < 0.018:
                rig.op.press("stop")
                events["stops"] += 1
            if rig.status().mode == "locked" and rng.random() < 0.05:
                rig.op.press("recover")
            rig.op.set(
                deadman=deadman,
                linear=tuple(float(x) for x in lin),
                angular=tuple(float(x) for x in ang),
            )
            a = rig.arm_state()
            if rig.server.node.mode == "teleop":
                min_sigma = min(min_sigma, min_singular_value(a.jacobian, g.rot_weight))
            time.sleep(0.02)
    finally:
        rig.op.set(deadman=False)
        time.sleep(0.3)
        rec.running = False
        rec.join(2)
        overruns = rig.server.node.loop_overruns
        rig.close()

    statuses = rec.statuses
    assert len(statuses) > SOAK_S * 50
    teleop = [s for s in statuses if s.mode == "teleop"]
    for s in teleop:
        p = np.array(s.ee_pos)
        worst_ws = max(worst_ws, float(np.max(np.r_[lower - p, p - upper])))
        worst_joint = min(worst_joint, float(np.min(np.r_[np.array(s.q) - jl, ju - np.array(s.q)])))
    modes = {m: sum(s.mode == m for s in statuses) for m in ("idle", "teleop", "homing", "locked")}
    limiters = {
        lim: sum(lim in s.limiters for s in statuses)
        for lim in ("workspace", "singularity", "joint")
    }
    for k, v in {**events, **modes, **limiters}.items():
        record_property(k, v)
    record_property("worst_workspace_violation_mm", worst_ws * 1e3)
    record_property("min_joint_margin_rad", worst_joint)
    record_property("min_sigma_teleop", min_sigma)
    record_property("loop_overruns", overruns)
    print(f"\nsoak {SOAK_S:.0f}s: events {events} modes {modes} limiters {limiters}")
    print(
        f"worst workspace violation {worst_ws * 1e3:.2f} mm, min joint margin {worst_joint:.4f}"
        f" rad, min sigma {min_sigma:.4f}, overruns {overruns}"
    )

    assert all(s.robot_errors == "" for s in statuses), "robot error during soak"
    assert all(s.mode != "error" for s in statuses), [
        s.message for s in statuses if s.mode == "error"
    ][:3]
    assert modes["teleop"] > 0.2 * len(statuses)  # the operator really drove the arm
    assert worst_ws < 2e-3, f"left the workspace by {worst_ws * 1e3:.2f} mm"
    assert worst_joint > 0.0, f"crossed a joint limit by {-worst_joint:.4f} rad"
    assert min_sigma > g.sigma_stop - 0.005, min_sigma
