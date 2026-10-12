import math
import time

import msgspec
import pytest

from franka_teleop import protocol, transport
from franka_teleop.protocol import (
    ACTIONS,
    PROTOCOL_VERSION,
    NodeStatus,
    ProtocolError,
    SceneInfo,
    SceneState,
    TeleopCommand,
    decode,
    encode,
)

from .conftest import free_port


def cmd(seq=0, **kw):
    return TeleopCommand(
        session=7, seq=seq, t_send=time.time(), source="script", deadman=False, **kw
    )


def status(seq=0):
    return NodeStatus(
        seq=seq,
        t=time.time(),
        mode="idle",
        sim=True,
        q=(0.0,) * 7,
        dq=(0.0,) * 7,
        ee_pos=(0.3, 0.0, 0.5),
        ee_quat=(1.0, 0.0, 0.0, 0.0),
        speed_level=1,
        speed_scale=0.5,
        frame="base",
    )


@pytest.mark.parametrize(
    "msg",
    [
        cmd(3, linear=(0.5, -1.0, 0.0), gripper=2, home=1),
        status(9),
        SceneState(seq=1, t_sim=0.5, qpos=bytes(72), nq=9),
        SceneInfo(xml="<mujoco/>", nq=9, robot_prefix="robot0_"),
    ],
)
def test_roundtrip(msg):
    # Compare bytes: NodeStatus defaults contain NaN, and NaN != NaN.
    assert encode(decode(encode(msg), type(msg))) == encode(msg)


def test_defaults_are_safe():
    s = status()
    assert math.isinf(s.cmd_age_ms) and math.isnan(s.gripper_width) and s.session == 0
    c = cmd()
    assert not c.deadman and c.linear == (0, 0, 0) and all(v == 0 for v in c.counters().values())
    assert tuple(c.counters()) == ACTIONS


def test_version_mismatch_is_rejected():
    data = msgspec.msgpack.encode({**msgspec.to_builtins(cmd()), "v": PROTOCOL_VERSION + 1})
    with pytest.raises(ProtocolError, match="protocol version"):
        decode(data, TeleopCommand)


def test_garbage_and_wrong_type_are_rejected():
    with pytest.raises(ProtocolError):
        decode(b"\x00\x01garbage", TeleopCommand)
    with pytest.raises(ProtocolError):
        decode(encode(status()), TeleopCommand)


def test_pubsub_conflate_delivers_only_newest():
    port = free_port()
    pub = transport.publisher(f"tcp://127.0.0.1:{port}")
    sub = transport.subscriber(f"tcp://127.0.0.1:{port}")
    try:
        time.sleep(0.2)  # slow joiner
        for i in range(50):
            assert transport.send(pub, cmd(i))
        time.sleep(0.1)
        msg, bad = transport.recv_latest(sub, TeleopCommand)
        assert msg is not None and msg.seq == 49 and bad == 0
        assert transport.recv_latest(sub, TeleopCommand) == (None, 0)
        pub.send(b"not msgpack")
        time.sleep(0.05)
        assert transport.recv_latest(sub, TeleopCommand) == (None, 1)
        transport.send(pub, cmd(50))
        assert transport.recv_wait(sub, TeleopCommand, 1.0).seq == 50
        assert transport.recv_wait(sub, TeleopCommand, 0.05) is None
    finally:
        pub.close()
        sub.close()


def test_protocol_module_documents_topology():
    assert "ft-joystick" in protocol.__doc__ and "ft-node" in protocol.__doc__
