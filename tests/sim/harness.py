"""Run ft-* console scripts as subprocesses (shared by the sim and viz tests)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import zmq

from franka_teleop.config import Config, write_config
from franka_teleop.protocol import SceneInfo, SceneRequest, decode, encode
from franka_teleop.sim.server import READY_PREFIX, SUMMARY_PREFIX
from franka_teleop.transport import context


def script(name: str) -> str:
    """Path of a console script of this venv (tests the real entry point)."""
    return str(Path(sys.executable).with_name(name))


class Proc:
    """A subprocess whose combined stdout/stderr is drained by a thread (so it never blocks on
    a full pipe) and kept as a list of lines."""

    def __init__(self, args: list[str], env: dict[str, str] | None = None):
        self.args = args
        self.lines: list[str] = []
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})},
        )
        self._drainer = threading.Thread(target=self._drain, daemon=True)
        self._drainer.start()

    def _drain(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def wait_for_line(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in list(self.lines):
                if line.startswith(prefix):
                    return line
            if self.proc.poll() is not None:
                self._drainer.join(timeout=2.0)
                break
            time.sleep(0.02)
        for line in self.lines:
            if line.startswith(prefix):
                return line
        raise AssertionError(
            f"no line starting with {prefix!r} from {self.args} "
            f"(exit code {self.proc.poll()}):\n{self.output}"
        )

    def wait(self, timeout: float) -> int:
        code = self.proc.wait(timeout=timeout)
        self._drainer.join(timeout=2.0)
        return code

    def terminate(self, timeout: float = 5.0) -> int:
        """SIGTERM; returns the exit code (kills and fails if it does not exit in time)."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            return self.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.wait(5.0)
            msg = f"{self.args} did not exit within {timeout} s of SIGTERM"
            raise AssertionError(msg) from None

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


class FtSim(Proc):
    def __init__(self, cfg: Config, cfg_path: Path, *extra: str):
        self.cfg = cfg
        write_config(cfg, cfg_path)
        super().__init__([script("ft-sim"), "--config", str(cfg_path), *extra])

    def wait_ready(self, timeout: float = 60.0) -> str:
        return self.wait_for_line(READY_PREFIX, timeout)

    def summary(self) -> dict[str, float]:
        return parse_stats(self.wait_for_line(SUMMARY_PREFIX, 1.0))


def parse_stats(line: str) -> dict[str, float]:
    """'ft-sim stats: a=1 b=2.5' -> {'a': 1.0, 'b': 2.5}"""
    _, _, pairs = line.partition(": ")
    return {k: float(v) for k, v in (kv.split("=") for kv in pairs.split())}


def request_scene_info(cfg: Config, timeout: float = 5.0) -> SceneInfo:
    sock = context().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(cfg.net.connect(cfg.net.scene_rpc_port))
    try:
        sock.send(encode(SceneRequest()))
        assert sock.poll(int(timeout * 1000), zmq.POLLIN), "no SceneInfo reply"
        return decode(sock.recv(), SceneInfo)
    finally:
        sock.close()
