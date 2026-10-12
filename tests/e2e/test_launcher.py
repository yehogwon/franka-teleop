"""ft-teleop end to end: every process is the installed console script, as deployed.

The real-robot path (sim.enabled = false: preflight -> node -> joystick) is exercised against a
franky-sim process on a loopback address, i.e. the exact code that will drive the arm, talking
the libfranka protocol to an emulated robot."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from franka_teleop import transport
from franka_teleop.protocol import NodeStatus

from ..conftest import REPO, free_loopback_host, make_test_config

pytestmark = [pytest.mark.e2e, pytest.mark.sim]

BIN = Path(sys.executable).parent
SMOKE = REPO / "configs" / "scripts" / "smoke.toml"
DEMO = REPO / "configs" / "scripts" / "demo.toml"


class StatusRecorder(threading.Thread):
    """Collects every NodeStatus published during a run."""

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

    def stop(self):
        self.running = False
        self.join(2)


def processes_with(marker: str) -> list[str]:
    """Command lines of live processes mentioning `marker` (e.g. the effective config dir)."""
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if marker in cmd and "pytest" not in cmd:
            found.append(cmd)
    return found


def launch(config_path, *extra, timeout=60):
    return subprocess.run(
        [str(BIN / "ft-teleop"), "--config", str(config_path), "--viewer", "none", *extra],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture
def real_path_config(franky_sim_proc, config_file):
    """A real-robot config (sim.enabled = false) pointed at an emulated robot. Scripted input is
    allowed explicitly, since the default real config only accepts a gamepad."""
    cfg = make_test_config(
        franky_sim_proc, sim__enabled=False, teleop__allowed_sources=["gamepad", "script"]
    )
    return cfg, config_file(cfg)


def test_real_robot_path_preflight_node_joystick(real_path_config):
    cfg, path = real_path_config
    rec = StatusRecorder(cfg)
    rec.start()
    try:
        r = launch(path, "--input", "script", "--script", str(SMOKE))
    finally:
        rec.stop()
    assert r.returncode == 0, r.stderr[-3000:]
    assert "preflight against the robot" in r.stderr and "ft-node ready" in r.stderr
    assert "ft-sim" not in r.stderr  # no simulator is started for a robot config
    moving = [s for s in rec.statuses if s.mode == "teleop"]
    assert moving, "node never entered teleop"
    x0, x1 = rec.statuses[0].ee_pos[0], rec.statuses[-1].ee_pos[0]
    assert x1 - x0 > 0.04, (x0, x1)
    assert all(s.robot_errors == "" for s in rec.statuses)
    assert not rec.statuses[-1].sim and rec.statuses[-1].mode == "idle"


def test_real_robot_path_rejects_keyboard_and_script_by_default(franky_sim_proc, config_file):
    cfg = make_test_config(franky_sim_proc, sim__enabled=False)  # allowed_sources = gamepad
    rec = StatusRecorder(cfg)
    rec.start()
    try:
        r = launch(config_file(cfg), "--input", "script", "--script", str(SMOKE))
    finally:
        rec.stop()
    assert r.returncode == 0, r.stderr[-3000:]
    assert all(s.mode != "teleop" for s in rec.statuses)
    assert any("not allowed" in s.message for s in rec.statuses)
    assert abs(rec.statuses[-1].ee_pos[0] - rec.statuses[0].ee_pos[0]) < 1e-3


def test_preflight_failure_aborts_before_connecting(config_file):
    cfg = make_test_config(free_loopback_host(), sim__enabled=False)  # nothing listens there
    t0 = time.monotonic()
    r = launch(config_file(cfg), "--input", "script", "--script", str(SMOKE))
    assert r.returncode == 1 and "preflight failed" in r.stderr
    assert "ft-node ready" not in r.stderr and time.monotonic() - t0 < 30


def _start_launcher(path, script):
    proc = subprocess.Popen(
        [
            str(BIN / "ft-teleop"),
            "--config",
            str(path),
            "--viewer",
            "none",
            "--input",
            "script",
            "--script",
            str(script),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _wait_for_node(cfg, timeout=30.0) -> NodeStatus:
    sub = transport.subscriber(cfg.net.connect(cfg.net.status_port))
    try:
        s = transport.recv_wait(sub, NodeStatus, timeout)
        assert s is not None, "ft-node never published"
        return s
    finally:
        sub.close()


def test_sigterm_stops_every_process(real_path_config):
    cfg, path = real_path_config
    proc = _start_launcher(path, DEMO)
    marker = str(Path(path).parent)
    try:
        _wait_for_node(cfg)
        time.sleep(1.5)  # mid-script, arm moving
        proc.send_signal(signal.SIGTERM)
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, proc.stderr.read()[-3000:]
    time.sleep(0.5)
    assert processes_with(marker) == [] and processes_with("ft-teleop-") == []


def test_node_crash_tears_down_session(real_path_config):
    cfg, path = real_path_config
    proc = _start_launcher(path, DEMO)
    try:
        _wait_for_node(cfg)
        nodes = [
            int(pid)
            for pid in os.listdir("/proc")
            if pid.isdigit()
            and "ft-node" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
            and "ft-teleop-" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        ]
        assert len(nodes) == 1
        os.kill(nodes[0], signal.SIGKILL)  # a crash: no halt, no cleanup
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
    err = proc.stderr.read()
    assert proc.returncode == 1 and "node exited unexpectedly" in err
    assert processes_with("ft-teleop-") == []
    # The node died mid-motion without halting: the arm must have come to rest by itself.
    from franka_teleop.arm.franky_arm import FrankyArm

    arm = FrankyArm(cfg.robot)
    try:
        s = arm.read()
        assert np.abs(s.dq).max() < 1e-3 and s.errors == () and not s.in_control
    finally:
        arm.close()


def test_sim_session_runs_the_demo_script(franky_sim_proc, config_file):
    pytest.importorskip("franka_teleop.sim.server")
    cfg = make_test_config(free_loopback_host())
    rec = StatusRecorder(cfg)
    rec.start()
    try:
        r = launch(config_file(cfg), "--input", "script", "--script", str(DEMO), timeout=120)
    finally:
        rec.stop()
    assert r.returncode == 0, r.stderr[-3000:]
    assert "ft-sim ready" in r.stderr and "ft-node ready" in r.stderr
    widths = [s.gripper_width for s in rec.statuses if not np.isnan(s.gripper_width)]
    assert min(widths) < 0.01 and widths[-1] > 0.07  # demo closes, then opens the gripper
    assert all(s.robot_errors == "" for s in rec.statuses)
    assert processes_with("ft-teleop-") == []
