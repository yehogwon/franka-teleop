"""Shared fixtures. Each franky-sim instance binds its own 127.x.y.z loopback address and every
ZMQ endpoint gets an ephemeral port, so tests (and parallel worktrees) never collide."""

from __future__ import annotations

import random
import socket
from pathlib import Path

import pytest

from franka_teleop.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def free_loopback_host() -> str:
    """A random 127.x.y.z where franky-sim's fixed ports (1337 TCP/UDP, 1338 TCP/UDP) are free."""
    for _ in range(200):
        host = f"127.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(2, 254)}"
        try:
            for port in (1337, 1338):
                for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                    with socket.socket(socket.AF_INET, kind) as s:
                        s.bind((host, port))
            return host
        except OSError:
            continue
    raise RuntimeError("no free loopback address")


def make_test_config(host: str | None = None, **overrides) -> Config:
    """configs/sim.toml with a private robot host, ephemeral ports and realtime=ignore."""
    ports = {
        f"net__{name}": free_port()
        for name in ("cmd_port", "status_port", "scene_port", "scene_rpc_port")
    }
    defaults = {
        "robot__host": host or free_loopback_host(),
        "robot__realtime": "ignore",
        "viewer__web_port": free_port(),
        "input__webpad_port": free_port(),
        **ports,
    }
    return load_config(CONFIGS / "sim.toml", **(defaults | overrides))


class SimHandle:
    def __init__(self, host: str, sim, robot, server):
        self.host, self.sim, self.robot, self.server = host, sim, robot, server


@pytest.fixture
def franky_sim():
    """A running franky-sim (tabletop scene, ready pose) on a private loopback address."""
    from franky_sim import SimulationServer

    from franka_teleop.config import READY_Q
    from franka_teleop.sim.scene import make_simulator

    host = free_loopback_host()
    sim, robot = make_simulator("tabletop", READY_Q)
    server = SimulationServer(sim, hostname_candidates=[host])
    server.init()
    server.run_async()
    try:
        yield SimHandle(host, sim, robot, server)
    finally:
        server.cleanup()
        sim.cleanup()


@pytest.fixture
def franky_sim_proc():
    """franky-sim in a separate process (like deployment). Yields its loopback host."""
    import subprocess
    import sys

    host = free_loopback_host()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "franky_sim_proc.py"), host],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.strip() == "ready", f"franky-sim failed to start: {line!r}"
        yield host
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def test_config():
    return make_test_config


@pytest.fixture
def config_file(tmp_path):
    """Write a Config to a TOML file and return its path (for --config of ft-* subprocesses)."""
    from franka_teleop.config import write_config

    def _write(cfg: Config, name: str = "config.toml"):
        return write_config(cfg, tmp_path / name)

    return _write
