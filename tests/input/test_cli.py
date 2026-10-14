"""ft-joystick / ft-check-joystick as subprocesses (the console scripts `uv run` would start)."""

import glob
import os
import pty
import select
import signal
import subprocess
import sys
import termios
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


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT])
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


class PtyJoystick:
    """ft-joystick --device keyboard on a pseudo-terminal, like an operator's terminal."""

    def __init__(self, path, *args):
        self.master, self.slave = pty.openpty()
        self.before = termios.tcgetattr(self.slave)
        self.proc = subprocess.Popen(
            [str(BIN / "ft-joystick"), "--config", str(path), "--device", "keyboard", *args],
            cwd=REPO,
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
        )

    def type(self, keys: bytes) -> None:
        os.write(self.master, keys)

    def read_output(self) -> bytes:
        out = b""
        while select.select([self.master], [], [], 0.05)[0]:
            try:
                out += os.read(self.master, 65536)
            except OSError:
                break
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(5)
        os.close(self.master)
        os.close(self.slave)


def pump(sub, seconds, term=None):
    """Commands received during ``seconds`` (reading the terminal's output if ``term``)."""
    msgs, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        msgs += drain(sub, timeout=0.05)
        if term is not None:
            term.read_output()
    return msgs


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGQUIT])
def test_ft_joystick_keyboard_hangup_releases_deadman_and_restores_tty(cfg_and_file, sig):
    """Review r2: closing the terminal (SIGHUP) or Ctrl-\\ (SIGQUIT) with the keyboard deadman
    latched ON: neutral, deadman-released tail, exit 0, terminal settings restored."""
    cfg, path = cfg_and_file
    sub = cmd_sub(cfg)
    term = PtyJoystick(path, "--quiet")
    try:
        msgs = pump(sub, 1.5, term)
        assert msgs and not any(m.deadman for m in msgs)
        term.type(b" ")  # deadman ON (latched)
        msgs += pump(sub, 0.5, term)
        term.type(b"w")
        msgs += pump(sub, 0.2, term)
        assert msgs[-1].deadman and msgs[-1].linear == (1, 0, 0)
        term.proc.send_signal(sig)
        assert term.proc.wait(5) == 0
        time.sleep(0.2)
        msgs += drain(sub)
        last_on = max(i for i, m in enumerate(msgs) if m.deadman)
        tail = msgs[last_on + 1 :]
        assert tail and tail[-1].t_send - tail[0].t_send >= 0.25
        assert all(not m.deadman and m.linear == (0, 0, 0) for m in tail)
        after = termios.tcgetattr(term.slave)
        assert after[0] == term.before[0] and after[3] == term.before[3]  # iflag, lflag
    finally:
        term.close()
        sub.close()


def test_ft_joystick_keeps_streaming_after_ctrl_s(cfg_and_file):
    """Review r8: Ctrl-S (XOFF) must not freeze the command loop through blocking HUD writes."""
    cfg, path = cfg_and_file
    sub = cmd_sub(cfg)
    term = PtyJoystick(path)  # with the HUD
    try:
        pump(sub, 1.5, term)
        term.type(b" ")  # deadman ON
        before = pump(sub, 0.5, term)
        assert len(before) > 30 and before[-1].deadman
        term.type(b"\x13")  # Ctrl-S; then nobody reads the terminal for a while
        time.sleep(0.5)
        frozen = pump(sub, 2.0)
        assert len(frozen) > 0.8 * 2.0 * cfg.input.rate_hz
        assert all(m.deadman for m in frozen)  # \x13 is an ignored key, not a deadman toggle
        term.type(b"\x11")  # Ctrl-Q
        after = pump(sub, 0.5, term)
        assert len(after) > 30
        term.proc.send_signal(signal.SIGTERM)
        assert term.proc.wait(5) == 0
    finally:
        term.close()
        sub.close()


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


def _wait_http(url, timeout=30.0):
    import urllib.request

    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            assert time.monotonic() < deadline, f"{url} did not come up"
            time.sleep(0.1)


def test_ft_joystick_webpad_streams_gamepad_commands(cfg_and_file):
    """The web joystick's commands are a gamepad's: same source, same mapping, same stream."""
    from .test_webpad import FORWARD, RELEASED, Page

    cfg, path = cfg_and_file
    port = cfg.input.webpad_port
    sub = cmd_sub(cfg)
    proc = spawn("ft-joystick", "--config", path, "--device", "webpad")
    try:
        _wait_http(f"http://127.0.0.1:{port}/healthz")
        page = Page(port)
        msgs = []

        def stream(state, seconds):
            nonlocal msgs
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                page.send(state)
                msgs += drain(sub, timeout=0.04)

        stream(RELEASED, 0.5)
        stream(FORWARD, 0.5)
        page.close()
        t_end = time.monotonic() + 0.4
        while time.monotonic() < t_end:
            msgs += drain(sub, timeout=0.05)
        proc.send_signal(signal.SIGTERM)
        msgs += collect_until_exit(proc, sub, timeout=10)
        _out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        sub.close()
    assert proc.returncode == 0, err
    assert "Traceback" not in err
    assert f"web joystick: open http://localhost:{port}/" in err
    assert {m.source for m in msgs} == {"gamepad"} and len({m.session for m in msgs}) == 1
    moving = [m for m in msgs if m.deadman]
    assert len(moving) > 20 and all(m.linear == (1.0, 0.0, 0.0) for m in moving)
    first = msgs.index(moving[0])
    assert first > 0 and not any(m.deadman for m in msgs[:first])
    after = msgs[msgs.index(moving[-1]) + 1 :]
    assert len(after) > 20 and not any(m.deadman for m in after)  # page closed: pad unplugged


def test_ft_joystick_webpad_needs_opt_in_on_a_robot_config(config_file):
    robot = make_test_config(sim__enabled=False)
    r = run("ft-joystick", "--config", config_file(robot), "--device", "webpad", "--quiet")
    assert r.returncode == 2
    assert 'add "webpad" to teleop.allowed_sources' in r.stderr
    assert "Traceback" not in r.stderr

    opted_in = make_test_config(sim__enabled=False, teleop__allowed_sources=["gamepad", "webpad"])
    r = run(
        "ft-joystick", "--config", config_file(opted_in), "--device", "webpad",
        "--duration", "0.5", "--quiet",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "web joystick: open" in r.stderr
