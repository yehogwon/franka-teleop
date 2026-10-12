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
    if os.environ.get("FT_SOAK_DUMP"):  # for offline analysis of a failure
        np.savez(
            os.environ["FT_SOAK_DUMP"],
            q=np.array([s.q for s in statuses]),
            dq=np.array([s.dq for s in statuses]),
            twist=np.array([s.twist for s in statuses]),
            mode=np.array([s.mode for s in statuses]),
            limiters=np.array(["|".join(s.limiters) for s in statuses]),
            message=np.array([s.message for s in statuses]),
            t=np.array([s.t for s in statuses]),
        )
    assert len(statuses) > SOAK_S * 50
    worst_i = 0
    for i, s in enumerate(statuses):
        if s.mode != "teleop":
            continue
        p = np.array(s.ee_pos)
        worst_ws = max(worst_ws, float(np.max(np.r_[lower - p, p - upper])))
        margin = float(np.min(np.r_[np.array(s.q) - jl, ju - np.array(s.q)]))
        if margin < worst_joint:
            worst_joint, worst_i = margin, i

    def context(i):  # the statuses leading to the closest joint-limit approach
        rows = []
        for s in statuses[max(0, i - 40) : i + 3 : 4]:
            q = np.array(s.q)
            m = np.r_[q - jl, ju - q]
            j = int(np.argmin(m)) % 7
            rows.append(
                f"seq {s.seq} {s.mode:6s} j{j + 1} margin {m.min():+.4f} q{j + 1} {q[j]:+.4f} "
                f"dq{j + 1} {s.dq[j]:+.3f} twist {np.round(s.twist, 3)} lim {s.limiters}"
            )
        return "\n".join(rows)

    modes = {m: sum(s.mode == m for s in statuses) for m in ("idle", "teleop", "homing", "locked")}
    limiters = {
        lim: sum(lim in s.limiters for s in statuses)
        for lim in ("workspace", "floor", "singularity", "joint")
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
    assert worst_joint > 0.0, (
        f"crossed a joint limit by {-worst_joint:.4f} rad:\n{context(worst_i)}"
    )
    assert min_sigma > g.sigma_stop - 0.005, min_sigma
