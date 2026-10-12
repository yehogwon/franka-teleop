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


def _proc_table() -> dict[int, tuple[int, str]]:
    """pid -> (ppid, cmdline) for every live process."""
    table = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:  # exited while scanning
            continue
        ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        table[int(pid)] = (ppid, cmd.replace(b"\0", b" ").decode(errors="ignore"))
    return table


def descendants(root: int) -> dict[int, str]:
    """Live descendants of `root` (pid -> cmdline). Other test runs on the machine (e.g. in a
    parallel worktree) start identically named processes, so never match by name alone."""
    table = _proc_table()
    found, frontier = {}, [root]
    while frontier:
        parent = frontier.pop()
        for pid, (ppid, cmd) in table.items():
            if ppid == parent and pid not in found:
                found[pid] = cmd
                frontier.append(pid)
    return found


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return state[state.rindex(")") + 2] != "Z"


class Launcher:
    """ft-teleop as a subprocess; remembers every descendant it ever had."""

    def __init__(self, config_path, *extra):
        self.proc = subprocess.Popen(
            [str(BIN / "ft-teleop"), "--config", str(config_path), "--viewer", "none", *extra],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.seen: dict[int, str] = {}
        self.stderr = ""
        self._err_reader = threading.Thread(target=self._read_err, daemon=True)
        self._err_reader.start()

    def _read_err(self):
        self.stderr = self.proc.stderr.read()

    def poll_children(self):
        self.seen.update(descendants(self.proc.pid))

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.proc.poll() is None:
            self.poll_children()
            if time.monotonic() > deadline:
                self.proc.kill()
                raise TimeoutError("ft-teleop did not exit")
            time.sleep(0.1)
        self._err_reader.join(5)
        return self.proc.returncode

    def child(self, name: str) -> int:
        self.poll_children()
        pids = [pid for pid, cmd in self.seen.items() if f"/{name} " in cmd and alive(pid)]
        assert len(pids) == 1, (name, self.seen)
        return pids[0]

    def leftovers(self) -> dict[int, str]:
        return {pid: cmd for pid, cmd in self.seen.items() if alive(pid)}


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
    run = Launcher(path, "--input", "script", "--script", str(DEMO))
    _wait_for_node(cfg)
    time.sleep(1.5)  # mid-script, arm moving
    run.poll_children()
    assert {"ft-node", "ft-joystick"} <= {
        name for cmd in run.seen.values() for name in ("ft-node", "ft-joystick") if name in cmd
    }
    run.proc.send_signal(signal.SIGTERM)
    assert run.wait(30) == 0, run.stderr[-3000:]
    assert run.leftovers() == {}


def test_node_crash_tears_down_session(real_path_config):
    cfg, path = real_path_config
    run = Launcher(path, "--input", "script", "--script", str(DEMO))
    _wait_for_node(cfg)
    time.sleep(0.5)
    os.kill(run.child("ft-node"), signal.SIGKILL)  # a crash: no halt, no cleanup
    assert run.wait(30) == 1 and "node exited unexpectedly" in run.stderr
    assert run.leftovers() == {}
    # The node died mid-motion without halting: the arm must have come to rest by itself.
    from franka_teleop.arm.franky_arm import FrankyArm

    arm = FrankyArm(cfg.robot)
    try:
        deadline = time.monotonic() + 1.0
        while True:
            s = arm.read()
            if np.abs(s.dq).max() < 1e-3 or time.monotonic() > deadline:
                break
            time.sleep(0.02)
        assert np.abs(s.dq).max() < 1e-3, f"arm still moving: dq={np.round(s.dq, 4)}"
        assert s.errors == () and not s.in_control
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
    assert r.leftovers() == {}
