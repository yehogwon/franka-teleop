"""ft-sim as a subprocess: scene RPC, scene stream, a libfranka client, shutdown and refusals."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest

from franka_teleop.arm.franky_arm import FrankyArm
from franka_teleop.protocol import SceneState, decode
from franka_teleop.sim.scene import ROBOT_PREFIX, load_model
from franka_teleop.sim.server import STATS_PREFIX, TickSeries
from franka_teleop.transport import recv_latest, recv_wait, subscriber

from ..conftest import make_test_config
from .harness import FtSim, Proc, parse_stats, request_scene_info, script

pytestmark = [pytest.mark.sim, pytest.mark.e2e]


@pytest.fixture(scope="module")
def ft_sim(tmp_path_factory):
    sim = FtSim(make_test_config(), tmp_path_factory.mktemp("ft_sim") / "config.toml")
    try:
        sim.wait_ready()
        yield sim
    finally:
        sim.kill()


def hand_position(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return data.body(ROBOT_PREFIX + "hand").xpos.copy()


def latest_qpos(sock, timeout: float = 2.0) -> np.ndarray:
    msg = recv_wait(sock, SceneState, timeout)
    assert msg is not None, "no SceneState"
    newer, _ = recv_latest(sock, SceneState)
    msg = newer or msg
    return np.frombuffer(msg.qpos, dtype="<f8")


def test_ready_line_names_the_robot_host(ft_sim):
    assert ft_sim.cfg.robot.host in ft_sim.wait_ready()


def test_scene_info_xml_loads_and_matches_nq(ft_sim):
    info = request_scene_info(ft_sim.cfg)
    model = load_model(info.xml)
    assert info.nq == model.nq == 9 + 3 * 7  # robot + three free cubes
    assert info.robot_prefix == ROBOT_PREFIX
    assert info.robot_host == ft_sim.cfg.robot.host  # viewers check it against their config
    # A second request on a fresh socket works too (the REP loop did not wedge).
    assert request_scene_info(ft_sim.cfg).nq == info.nq


def test_scene_state_streams_at_viewer_fps(ft_sim):
    cfg = ft_sim.cfg
    sock = subscriber(cfg.net.connect(cfg.net.scene_port), conflate=False)
    try:
        first = recv_wait(sock, SceneState, 5.0)
        assert first is not None
        assert first.nq == 30 and len(first.qpos) == 8 * first.nq
        t0, msgs = time.monotonic(), []
        while time.monotonic() - t0 < 2.0:
            if sock.poll(100):
                msgs.append(sock.recv())
        rate = len(msgs) / 2.0
        assert abs(rate - cfg.viewer.fps) < 0.25 * cfg.viewer.fps, rate
        seqs = [decode(m, SceneState).seq for m in msgs]
        assert seqs == sorted(seqs) and seqs[0] > first.seq
    finally:
        sock.close()


def test_twist_moves_the_streamed_hand(ft_sim):
    cfg = ft_sim.cfg
    model = load_model(request_scene_info(cfg).xml)
    sock = subscriber(cfg.net.connect(cfg.net.scene_port))
    arm = FrankyArm(cfg.robot)
    try:
        p0 = hand_position(model, latest_qpos(sock))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            arm.send_twist(np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]), hold_ms=100)
            time.sleep(0.01)
        deadline = time.monotonic() + 2.0
        while arm.read().in_control and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)
        d = hand_position(model, latest_qpos(sock)) - p0
        assert 0.03 < d[1] < 0.045, d
        assert abs(d[0]) < 3e-3 and abs(d[2]) < 3e-3, d
    finally:
        arm.close()
        sock.close()


def test_second_sim_on_the_same_host_fails_clearly(ft_sim, tmp_path):
    other = FtSim(make_test_config(ft_sim.cfg.robot.host), tmp_path / "config.toml")
    try:
        assert other.wait(30.0) == 1
    finally:
        other.kill()
    assert f"cannot serve the robot on {ft_sim.cfg.robot.host}" in other.output
    assert "ft-sim ready:" not in other.output


def test_scene_port_in_use_fails_clearly(ft_sim, tmp_path):
    cfg = make_test_config()
    cfg.net.scene_port = ft_sim.cfg.net.scene_port
    other = FtSim(cfg, tmp_path / "config.toml")
    try:
        assert other.wait(30.0) == 1
    finally:
        other.kill()
    assert "cannot bind the scene ports" in other.output


def test_refuses_a_real_robot_config(config_file):
    path = config_file(make_test_config(sim__enabled=False))
    p = Proc([script("ft-sim"), "--config", str(path)])
    assert p.wait(30.0) == 2
    assert "sim.enabled = false" in p.output


def test_sigterm_exits_cleanly_and_loop_timing_stays_healthy(tmp_path):
    """Publisher active (a subscriber connected) and a libfranka client streaming twists."""
    cfg = make_test_config()
    sim = FtSim(cfg, tmp_path / "config.toml", "--verbose")
    sock = arm = None
    try:
        sim.wait_ready()
        sock = subscriber(cfg.net.connect(cfg.net.scene_port), conflate=False)
        arm = FrankyArm(cfg.robot)
        t0 = time.monotonic()
        sign = 1.0
        while time.monotonic() - t0 < 6.0:  # long enough for one periodic stats line
            arm.send_twist(np.array([0.0, 0.03 * sign, 0.0, 0.0, 0.0, 0.0]), hold_ms=100)
            recv_latest(sock, SceneState)
            if time.monotonic() - t0 > 3.0:
                sign = -1.0
            time.sleep(0.01)
        arm.close()
        arm = None
        t_term = time.monotonic()
        assert sim.terminate(timeout=5.0) == 0, sim.output
        assert time.monotonic() - t_term < 5.0
    finally:
        if arm is not None:
            arm.close()
        if sock is not None:
            sock.close()
        sim.kill()
    window = parse_stats(sim.wait_for_line(STATS_PREFIX, 1.0))
    summary = sim.summary()
    print(f"\nft-sim timing: window {window}\nft-sim timing: summary {summary}")
    for stats in (window, summary):
        assert stats["tick_mean_ms"] < 1.2, stats
        assert stats["rtf"] > 0.9, stats


def test_realtime_factor_slows_the_simulation(tmp_path):
    sim = FtSim(make_test_config(), tmp_path / "config.toml", "--realtime-factor", "0.5")
    try:
        sim.wait_ready()
        time.sleep(2.0)
        assert sim.terminate() == 0
    finally:
        sim.kill()
    s = sim.summary()
    assert 0.45 < s["rtf"] < 0.55 and 1.9 < s["tick_mean_ms"] < 2.1, s


def test_tick_series_summary_and_ring():
    s = TickSeries()
    for i in range(TickSeries.RING + 100):
        s.record(0.001 if i % 200 else 0.005)
    mean, p99, mx = s.summary()
    assert mx == pytest.approx(0.005)
    assert mean == pytest.approx(0.00102, rel=1e-3)
    assert 0.001 <= p99 <= 0.00102  # upper edge of a 10 µs bin
    recent = s.recent(s.n - 10)
    assert len(recent) == 10
    assert len(s.recent(0)) == TickSeries.RING
