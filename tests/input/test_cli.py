"""ft-joystick / ft-check-joystick as subprocesses (the console scripts `uv run` would start)."""

import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from franka_teleop import transport

from ..conftest import CONFIGS, REPO, make_test_config
from .conftest import cmd_sub, drain

pytestmark = pytest.mark.e2e

BIN = Path(sys.executable).parent


def spawn(name, *args, **kw):
    return subprocess.Popen(
        [str(BIN / name), *map(str, args)],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kw,
    )


def run(name, *args, timeout=30):
    return subprocess.run(
        [str(BIN / name), *map(str, args)],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def collect_until_exit(proc, sub, timeout=30):
    msgs = []
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        assert time.monotonic() < deadline, "process did not exit"
        msgs += drain(sub, timeout=0.05)
    time.sleep(0.1)
    return msgs + drain(sub)


@pytest.fixture
def cfg_and_file(config_file):
    cfg = make_test_config()
    return cfg, config_file(cfg)


def test_ft_joystick_script_smoke(cfg_and_file):
    cfg, path = cfg_and_file
    sub = cmd_sub(cfg)
    try:
        proc = spawn(
            "ft-joystick",
            "--config",
            path,
            "--device",
            "script",
            "--script",
            CONFIGS / "scripts" / "smoke.toml",
            "--exit-when-done",
            "--quiet",
        )
        msgs = collect_until_exit(proc, sub)
        _out, err = proc.communicate(timeout=10)
    finally:
        sub.close()
    assert proc.returncode == 0, err
    assert "Traceback" not in err
    assert len(msgs) > 150  # ~2.6 s script + 0.3 s tail at 100 Hz, minus the slow join
    assert {m.source for m in msgs} == {"script"} and len({m.session for m in msgs}) == 1
    seqs = [m.seq for m in msgs]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    moving = [m for m in msgs if m.deadman]
    first = msgs.index(moving[0])
    assert first > 0 and not any(m.deadman for m in msgs[:first])  # released before moving
    assert all(m.linear == (0.5, 0.0, 0.0) and m.angular == (0, 0, 0) for m in moving)
    assert moving[-1].t_send - moving[0].t_send == pytest.approx(2.0, abs=0.2)
    tail = msgs[msgs.index(moving[-1]) + 1 :]
    assert tail[-1].t_send - tail[0].t_send > 0.4  # trailing neutral step + shutdown
    assert all(not m.deadman and m.linear == (0, 0, 0) for m in tail)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_ft_joystick_signal_sends_neutral_tail_and_exits_0(cfg_and_file, sig):
    cfg, path = cfg_and_file
    sub = cmd_sub(cfg)
    try:
        # demo.toml without --exit-when-done would run ~13 s, then idle forever.
        proc = spawn("ft-joystick", "--config", path, "--script", CONFIGS / "scripts" / "demo.toml")
        msgs = []
        deadline = time.monotonic() + 20
        while not any(m.deadman for m in msgs):
            assert time.monotonic() < deadline and proc.poll() is None
            msgs += drain(sub, timeout=0.05)
        proc.send_signal(sig)
        msgs += collect_until_exit(proc, sub, timeout=5)
        _out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        sub.close()
    assert proc.returncode == 0, err
    assert "Traceback" not in err
    last_moving = max(i for i, m in enumerate(msgs) if m.deadman)
    tail = msgs[last_moving + 1 :]
    assert tail[-1].t_send - tail[0].t_send >= 0.25
    assert all(not m.deadman and m.linear == (0, 0, 0) and m.angular == (0, 0, 0) for m in tail)
    assert "waiting for ft-node" in err  # HUD (not a TTY: plain lines on stderr)


def test_ft_joystick_keyboard_needs_a_tty(cfg_and_file):
    _, path = cfg_and_file
    r = run("ft-joystick", "--config", path, "--device", "keyboard", "--quiet")
    assert r.returncode == 2
    assert "needs a terminal" in r.stderr and "Traceback" not in r.stderr


def test_ft_joystick_rejects_bad_script(cfg_and_file, tmp_path):
    _, path = cfg_and_file
    bad = tmp_path / "bad.toml"
    bad.write_text("[[step]]\nduration = 1\npress = ['jump']\n")
    r = run("ft-joystick", "--config", path, "--script", bad, "--quiet")
    assert r.returncode == 2
    assert "unknown actions ['jump']" in r.stderr and "Traceback" not in r.stderr


def test_ft_joystick_reports_busy_port(cfg_and_file):
    cfg, path = cfg_and_file
    squatter = transport.publisher(cfg.net.bind(cfg.net.cmd_port))
    try:
        r = run("ft-joystick", "--config", path, "--script", CONFIGS / "scripts" / "smoke.toml")
    finally:
        squatter.close()
    assert r.returncode == 1
    assert "another ft-joystick running" in r.stderr and "Traceback" not in r.stderr


def test_ft_check_joystick_without_gamepad():
    if glob.glob("/dev/input/by-id/*joystick*") or glob.glob("/dev/input/js*"):
        pytest.skip("a joystick is connected to this machine")
    r = run("ft-check-joystick", "--once", timeout=60)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "No gamepad found" in r.stdout
    for hint in ("ls -l /dev/input/by-id/", "'input' group", "SDL_GAMECONTROLLERCONFIG"):
        assert hint in r.stdout
    assert "ALSA" not in r.stderr and "pygame" not in r.stdout  # no audio init, no banner


def test_console_scripts_are_installed():
    for name in ("ft-joystick", "ft-check-joystick"):
        assert os.access(BIN / name, os.X_OK), name
