"""Viewer state sources: ft-sim scene (RPC + stream) and ft-node status (robot only)."""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import pytest
import zmq

from franka_teleop.config import READY_Q
from franka_teleop.protocol import NodeStatus, ProtocolError, SceneInfo
from franka_teleop.sim.scene import ROBOT_PREFIX, scene_xml
from franka_teleop.sim.server import SceneRpc
from franka_teleop.transport import context, publisher, send
from franka_teleop.viz.overlay import STALE_AFTER_S, state_text, viewer_rows
from franka_teleop.viz.source import (
    SceneMismatch,
    SceneSource,
    SourceError,
    StatusSource,
    decode_scene_reply,
    open_source,
    request_scene_info,
)

from ..conftest import make_test_config


def node_status(q, width=math.nan, seq=0) -> NodeStatus:
    return NodeStatus(
        seq=seq,
        t=time.time(),
        mode="teleop",
        sim=True,
        q=tuple(float(x) for x in q),
        dq=(0.0,) * 7,
        ee_pos=(0.3, 0.0, 0.5),
        ee_quat=(1.0, 0.0, 0.0, 0.0),
        gripper_width=width,
        speed_level=1,
        speed_scale=0.5,
        frame="base",
    )


def publish_until(sock, msg, pred, timeout=3.0) -> bool:
    """PUB/SUB drops messages until the subscription has propagated: resend until seen."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        send(sock, msg)
        time.sleep(0.02)
        if pred():
            return True
    return False


@pytest.mark.sim
@pytest.mark.e2e
def test_scene_source_follows_ft_sim(viz_sim):
    src = open_source(viz_sim.cfg, "scene", wait=5.0)
    try:
        assert isinstance(src, SceneSource) and src.model.nq == 30
        assert not src.has_state
        assert src.wait_for_state(3.0)
        assert np.allclose(src.qpos[src.arm_addr], READY_Q, atol=5e-3)
        # Cubes rest on the ground (free joints: xyz + quaternion).
        jid = mujoco.mj_name2id(src.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint")
        adr = src.model.jnt_qposadr[jid]
        assert src.qpos[adr + 2] == pytest.approx(0.025, abs=2e-3)
        # poll() is non-blocking and keeps returning the newest state.
        t0 = time.perf_counter()
        src.poll()
        assert time.perf_counter() - t0 < 0.01
    finally:
        src.close()


@pytest.mark.sim
@pytest.mark.e2e
def test_auto_prefers_ft_sim(viz_sim):
    src = open_source(viz_sim.cfg, "auto")
    try:
        assert src.kind == "scene"
    finally:
        src.close()


def test_auto_falls_back_to_status_without_ft_sim():
    cfg = make_test_config()
    t0 = time.monotonic()
    src = open_source(cfg, "auto")
    try:
        assert isinstance(src, StatusSource)
        assert time.monotonic() - t0 < 3.0
        assert src.model.nq == 9  # robot only
    finally:
        src.close()


@pytest.fixture
def fake_ft_sim():
    """Factory: an in-process ft-sim scene RPC on ``cfg``'s port, reporting ``robot_host``."""
    servers = []

    def start(cfg, robot_host: str) -> SceneRpc:
        rep = context().socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(cfg.net.bind(cfg.net.scene_rpc_port))
        info = SceneInfo(
            xml=scene_xml("empty"), nq=9, robot_prefix=ROBOT_PREFIX, robot_host=robot_host
        )
        rpc = SceneRpc(rep, info)
        rpc.start()
        servers.append(rpc)
        return rpc

    yield start
    for rpc in servers:
        rpc.stop()


def test_auto_uses_status_with_a_real_robot_config(fake_ft_sim):
    """Review r5: a leftover ft-sim on the shared ports must not stand in for the real arm."""
    sim_cfg = make_test_config()
    rpc = fake_ft_sim(sim_cfg, sim_cfg.robot.host)
    real_cfg = make_test_config(host="172.16.0.2", sim__enabled=False)
    real_cfg.net = sim_cfg.net  # same ports, as configs/sim.toml and real.toml
    t0 = time.monotonic()
    src = open_source(real_cfg, "auto")
    try:
        assert src.kind == "status" and "ft-node status" in src.describe()
        assert rpc.served == 0  # ft-sim is not even asked ...
        assert time.monotonic() - t0 < 2.0  # ... so there is no probe timeout either
    finally:
        src.close()


@pytest.mark.parametrize("reported", ["", "same"])
def test_auto_uses_ft_sim_simulating_this_robot_host(fake_ft_sim, reported):
    cfg = make_test_config()
    fake_ft_sim(cfg, cfg.robot.host if reported == "same" else reported)  # "": older ft-sim
    src = open_source(cfg, "auto")
    try:
        assert isinstance(src, SceneSource)
    finally:
        src.close()


def test_scene_of_another_robot_host_is_rejected(fake_ft_sim, capsys):
    cfg = make_test_config()
    fake_ft_sim(cfg, "127.9.9.9")
    src = open_source(cfg, "auto")
    try:
        assert src.kind == "status"
    finally:
        src.close()
    assert "simulates robot.host 127.9.9.9" in capsys.readouterr().err
    t0 = time.monotonic()
    with pytest.raises(SceneMismatch, match=f"127.9.9.9.*robot.host is {cfg.robot.host}"):
        open_source(cfg, "scene", wait=5.0)
    assert time.monotonic() - t0 < 2.0  # does not wait for a stale sim to change


def test_error_reply_from_ft_sim_is_reported():
    rpc = SceneRpc(None, SceneInfo(xml="", nq=0, robot_prefix=""))
    reply = rpc.reply_to([b"a", b"b"])
    with pytest.raises(ProtocolError, match=r"ft-sim rejected the scene request: .*2 frames"):
        decode_scene_reply(reply)


def test_scene_source_gives_up_after_wait():
    cfg = make_test_config()
    t0 = time.monotonic()
    with pytest.raises(SourceError, match="ft-sim did not answer"):
        open_source(cfg, "scene", wait=1.5)
    assert 1.4 < time.monotonic() - t0 < 4.0


def test_request_scene_info_retries_on_fresh_sockets():
    cfg = make_test_config()
    assert request_scene_info(cfg, 0.2) is None
    assert request_scene_info(cfg, 0.2) is None  # a wedged REQ socket would raise here


def test_status_source_maps_joints_and_fingers():
    cfg = make_test_config()
    pub = publisher(cfg.net.bind(cfg.net.status_port))
    src = StatusSource(cfg)
    try:
        # Before ft-node reports: the configured start pose, not "received".
        assert not src.has_state
        assert np.allclose(src.qpos[src.arm_addr], cfg.sim.initial_q)

        q = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 0.4])
        assert publish_until(pub, node_status(q, width=0.06), src.poll)
        assert src.has_state and src.status.mode == "teleop"
        assert np.allclose(src.qpos[src.arm_addr], q)
        assert np.allclose(src.qpos[src.finger_addr], 0.03)

        # The fingers really open by `width` in the model.
        data = mujoco.MjData(src.model)
        data.qpos[:] = src.qpos
        mujoco.mj_kinematics(src.model, data)
        left = data.body(ROBOT_PREFIX + "left_finger").xpos
        right = data.body(ROBOT_PREFIX + "right_finger").xpos
        assert np.linalg.norm(left - right) == pytest.approx(0.06, abs=1e-6)

        # Unknown width (no gripper): fingers keep their last value; joints still follow.
        q2 = q + 0.1
        assert publish_until(
            pub, node_status(q2, seq=1), lambda: src.poll() and src.status.seq == 1
        )
        assert np.allclose(src.qpos[src.arm_addr], q2)
        assert np.allclose(src.qpos[src.finger_addr], 0.03)
    finally:
        src.close()
        pub.close()


def test_status_source_ignores_nan_joints():
    """Review r6: ft-node publishes NaN joints while it has no arm state."""
    cfg = make_test_config()
    src = StatusSource(cfg)
    try:
        nan_q = node_status([math.nan] * 7)
        assert not src._poll_state(nan_q)
        assert np.allclose(src.qpos[src.arm_addr], cfg.sim.initial_q) and not src.has_state
        q = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 0.4])
        assert src._poll_state(node_status(q))
        assert not src._poll_state(node_status([*q[:6], math.inf]))
        assert np.allclose(src.qpos[src.arm_addr], q)  # the last valid pose is kept
        data = mujoco.MjData(src.model)
        data.qpos[:] = src.qpos
        mujoco.mj_kinematics(src.model, data)
        assert np.isfinite(data.geom_xpos).all()
    finally:
        src.close()


def test_state_and_status_go_stale():
    """Viewers flag a state (and a NodeStatus) older than STALE_AFTER_S."""
    cfg = make_test_config()
    pub = publisher(cfg.net.bind(cfg.net.status_port))
    src = StatusSource(cfg)
    try:
        rows = dict(viewer_rows(src))
        assert rows["source"] == "status (waiting for state)"
        assert rows["node"].startswith("no status")
        assert math.isinf(src.state_age()) and math.isinf(src.status_age())
        assert publish_until(pub, node_status(READY_Q), src.poll)
        now = time.monotonic()
        rows = viewer_rows(src, now)
        assert rows[0] == ("source", "status") and rows[1][0] == "mode"  # live, no flag
        later = now + STALE_AFTER_S + 1.0  # ages just over 2 s
        rows = viewer_rows(src, later)
        assert rows[0] == ("source", "status (STALE: no update for 2 s)")
        assert rows[1] == ("node", "STALE: no status for 2 s (last known below)")
        assert ("mode", "teleop [sim]") in rows
        assert state_text(True, STALE_AFTER_S - 0.1) == "live"
    finally:
        src.close()
        pub.close()


@pytest.mark.sim
@pytest.mark.e2e
def test_scene_source_also_reports_node_status(viz_sim):
    cfg = viz_sim.cfg
    pub = publisher(cfg.net.bind(cfg.net.status_port))
    src = open_source(cfg, "scene", wait=5.0)

    def got_status() -> bool:
        src.poll()
        return src.status is not None

    try:
        assert publish_until(pub, node_status(READY_Q, width=0.08), got_status)
        assert src.status.gripper_width == 0.08
    finally:
        src.close()
        pub.close()
