"""ft-check-robot: against franky-sim over the real libfranka protocol, a franky-sim that
refuses the client's version like a real robot does, an unreachable host, and the helpers."""

import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from franka_teleop.config import JOINT_LOWER, JOINT_UPPER, READY_Q
from franka_teleop.tools import check_robot as cr
from franka_teleop.tools import check_rt

from ..conftest import free_loopback_host, make_test_config

INDEX_ROOT = "https://timschneider42.github.io/franky/whl/by-robot-server-version/"


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    monkeypatch.delenv("FT_REALTIME", raising=False)
    monkeypatch.delenv("FT_CONFIG", raising=False)


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("FT_REALTIME", "FT_CONFIG")}


def run_json(capsys, argv) -> tuple[int, dict, dict]:
    code = cr.main([*argv, "--json"])
    doc = json.loads(capsys.readouterr().out)
    return code, doc, {r["name"]: r for r in doc["results"]}


# -- against franky-sim ------------------------------------------------------------------------


@pytest.mark.sim
def test_sim_robot_is_ready(franky_sim, config_file, capsys):
    path = config_file(make_test_config(franky_sim.host))
    code, doc, by_name = run_json(capsys, ["--config", str(path)])
    assert code == 0 and doc["ok"], doc
    assert by_name["connection"]["status"] == "PASS"
    assert by_name["connection"]["data"]["server_version"] == cr.SIM_SERVER_VERSION
    mode = by_name["robot_mode"]
    assert mode["status"] == "PASS" and mode["data"]["mode"] == "Idle"
    assert by_name["errors"]["status"] == "PASS" and by_name["errors"]["data"]["current"] == []
    joints = by_name["joints"]
    assert joints["status"] == "PASS" and np.allclose(joints["data"]["q"], READY_Q, atol=2e-3)
    ws = by_name["workspace"]
    assert ws["status"] == "PASS" and np.allclose(ws["data"]["ee"], [0.307, 0.0, 0.487], atol=3e-3)
    assert by_name["state_rate"]["data"]["fresh"] > 100  # WARN allowed: sim is sleep()-paced
    gripper = by_name["gripper"]
    assert gripper["status"] == "PASS"
    assert gripper["data"]["max_width"] == pytest.approx(0.08, abs=1e-3)
    assert 0.0 < gripper["data"]["width"] <= gripper["data"]["max_width"]
    assert "realtime: robot.realtime = ignore (simulator)" in doc["notes"]


@pytest.mark.sim
@pytest.mark.e2e
def test_console_script_with_config_file(franky_sim, config_file):
    path = config_file(make_test_config(franky_sim.host))
    exe = Path(sys.executable).with_name("ft-check-robot")
    proc = subprocess.run(
        [str(exe), "--config", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env(),
    )
    out = proc.stdout
    assert proc.returncode == 0, out + proc.stderr
    assert f"PASS  connection: connected to {franky_sim.host} (franky-control" in out
    assert "PASS  robot_mode: Idle" in out
    assert "PASS  errors: no active errors" in out
    assert "PASS  joints: all 7 inside limits" in out
    assert re.search(r"PASS  workspace: EE at \[\+0\.30\d, [+-]0\.000, \+0\.48\d\] m inside", out)
    assert re.search(r"PASS  gripper: connected, width \d+\.\d mm of 80\.0 mm", out)
    assert re.search(r"(PASS|WARN)  state_rate: \d+ Hz over 200 reads", out)
    assert out.rstrip().splitlines()[-1].startswith("ft-check-robot: OK")


@pytest.mark.sim
@pytest.mark.skipif(check_rt.try_sched_fifo()[0], reason="this machine may use SCHED_FIFO")
def test_realtime_enforce_on_a_non_rt_box_points_to_ft_check_rt(franky_sim, config_file, capsys):
    cfg = make_test_config(franky_sim.host)
    cfg.robot.realtime = "enforce"
    code, doc, by_name = run_json(capsys, ["--config", str(config_file(cfg))])
    assert code == 1
    assert by_name["realtime"]["status"] == "FAIL" and "ft-check-rt" in by_name["realtime"]["hint"]
    assert "continuing the read-only checks with realtime = ignore" in doc["notes"]
    # the read-only checks still ran
    assert by_name["connection"]["status"] == "PASS" and by_name["joints"]["status"] == "PASS"


# -- a robot that speaks another server version ------------------------------------------------

ROBOT_SERVER_VERSION = 9


@pytest.fixture
def old_robot(monkeypatch):
    """franky-sim whose robot server speaks version 9 and, like a real robot (unlike
    franky-sim), refuses clients of another version."""
    from franky_sim import SimulationServer
    from franky_sim import franka_server as fs
    from franky_sim.franka_robot_server import FrankaRobotServer

    from franka_teleop.sim.scene import make_simulator

    original = fs.ConnectCommand.handle

    def handle(self, server, command_type_struct="I"):
        if self.version == server.library_version:
            return original(self, server, command_type_struct)
        # libfranka's Connect response is packed {uint8 status; uint16 version}.
        header = fs.MessageHeader(self.command_type, self.command_id, 8)
        body = struct.pack(
            "<BH5x", fs.ConnectStatus.kIncompatibleLibraryVersion, server.library_version
        )
        self.client_socket.sendall(header.to_bytes(command_type_struct) + body)

    monkeypatch.setattr(fs.ConnectCommand, "handle", handle)
    monkeypatch.setattr(
        FrankaRobotServer, "library_version", property(lambda self: ROBOT_SERVER_VERSION)
    )
    host = free_loopback_host()
    sim, _ = make_simulator("empty", READY_Q)
    server = SimulationServer(sim, hostname_candidates=[host])
    server.init()
    server.run_async()
    try:
        yield host
    finally:
        server.cleanup()
        sim.cleanup()


@pytest.mark.sim
def test_incompatible_version_prints_uv_guidance(old_robot, config_file, capsys):
    path = config_file(make_test_config(old_robot))
    code = cr.main(["--config", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert (
        "FAIL  server_version: the robot speaks robot server version 9 (robot system >= 5.7.2); "
        "the installed franky-control 2.0.0 (libfranka 0.21.2) speaks 10"
    ) in out
    assert "fix: install the franky wheel built for server version 9 (libfranka 0.17.0)" in out
    assert f'url = "{INDEX_ROOT}9/"' in out
    assert 'name = "franky-server-9"' in out and "explicit = true" in out
    assert 'franky-control = { index = "franky-server-9" }' in out
    assert "uv lock --upgrade-package franky-control && uv sync" in out
    assert "2.0.0+libfranka.0.17.0" in out
    assert "franky-sim emulates server version 10" in out
    assert " gripper: " not in out  # nothing else is attempted without a connection


@pytest.mark.sim
def test_incompatible_version_json(old_robot, config_file, capsys):
    path = config_file(make_test_config(old_robot))
    code, doc, by_name = run_json(capsys, ["--config", str(path)])
    assert code == 1 and not doc["ok"]
    data = by_name["server_version"]["data"]
    assert data["robot_server_version"] == 9 and data["franky_server_version"] == 10
    assert f"{INDEX_ROOT}9/" in by_name["server_version"]["hint"]


# -- unreachable -------------------------------------------------------------------------------


def test_unreachable_host_fails_fast_with_network_hint(config_file, capsys):
    host = free_loopback_host()  # nothing listens there
    path = config_file(make_test_config(host))
    t0 = time.monotonic()
    code = cr.main(["--config", str(path)])
    out = capsys.readouterr().out
    assert code == 1 and time.monotonic() - t0 < 30
    assert f"FAIL  connection: {host} refused the connection" in out
    assert "activate" in out and "FCI" in out and "static IP" in out and "cable" in out


# -- helpers -----------------------------------------------------------------------------------


def test_parse_versions():
    new = (
        "libfranka: Incompatible library version (server version: 9, library version: 10). "
        "Please check https://frankarobotics.github.io for system updates"
    )
    assert cr.parse_versions(new) == (9, 10)
    old = "libfranka: Incompatible library version. \nServer version: 4\nLibrary version: 5"
    assert cr.parse_versions(old) == (4, 5)
    assert cr.parse_versions("something else") == (None, None)


def test_installed_franky_speaks_10():
    version, libfranka, server = cr.installed_franky()
    assert version.startswith("2.0.0") and libfranka == "0.21.2" and server == 10


@pytest.mark.parametrize("server", sorted(cr.FRANKY_WHEEL_INDEXES))
def test_guidance_for_every_published_index(server):
    text = cr.version_guidance(server)
    assert f'url = "{INDEX_ROOT}{server}/"' in text
    assert f"2.0.0+libfranka.{cr.FRANKY_WHEEL_INDEXES[server]}" in text
    assert "--prerelease" in text  # "no --prerelease flag needed"


def test_guidance_without_index_or_version():
    newer = cr.version_guidance(11)
    assert "no franky wheel index for server version 11" in newer and f"If {INDEX_ROOT}" in newer
    assert f'url = "{INDEX_ROOT}11/"' in newer  # usable once that index is published
    unknown = cr.version_guidance(None)
    assert cr.FRANKA_COMPAT in unknown and cr.FRANKY_FAQ in unknown
    assert "unknown" in cr.version_mismatch_detail(None, 10)


def test_check_joints():
    assert cr.check_joints(READY_Q, JOINT_LOWER, JOINT_UPPER).status == "PASS"
    near = list(READY_Q)
    near[3] = JOINT_UPPER[3] - 0.05
    r = cr.check_joints(near, JOINT_LOWER, JOINT_UPPER)
    assert r.status == "WARN" and "joint 4 0.050 rad upper" in r.detail
    panda_ok = list(READY_Q)
    panda_ok[5] = 0.3  # valid on a Panda, below the FR3/Panda intersection
    r = cr.check_joints(panda_ok, JOINT_LOWER, JOINT_UPPER)
    assert r.status == "FAIL" and "joint 6" in r.detail and "lower" in r.detail


def test_check_workspace():
    lower, upper = [0.25, -0.45, 0.03], [0.80, 0.45, 0.75]
    inside = cr.check_workspace([0.307, 0.0, 0.487], lower, upper)
    assert inside.status == "PASS" and "x lower" in inside.detail
    low = cr.check_workspace([0.5, 0.0, 0.01], lower, upper)
    assert low.status == "WARN" and "2.0 cm outside" in low.detail
    assert "z 0.010 < 0.03" in low.detail
    assert "only allow motion back" in low.hint


def test_check_mode_and_errors():
    assert cr.check_mode("Idle").status == "PASS"
    for mode in ("Reflex", "UserStopped", "Guiding", "Other", "Move"):
        assert cr.check_mode(mode).status == "WARN"
    assert cr.check_errors(()).status == "PASS"
    r = cr.check_errors(("joint_reflex",), ("cartesian_reflex",))
    assert r.status == "WARN" and "joint_reflex" in r.detail and "cartesian_reflex" in r.detail


def test_check_state_rate():
    good = {"reads": 200, "rate_hz": 999.0, "fresh": 199, "max_gap_ms": 1.2}
    assert cr.check_state_rate(good, sim=False).status == "PASS"
    slow = {"reads": 200, "rate_hz": 500.0, "fresh": 199, "max_gap_ms": 9.0}
    real = cr.check_state_rate(slow, sim=False)
    assert real.status == "WARN" and "UDP" in real.hint
    assert "expected in simulation" in cr.check_state_rate(slow, sim=True).hint
