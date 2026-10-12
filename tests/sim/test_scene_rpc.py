"""ft-sim's scene RPC thread in-process: every request is answered and nothing kills it."""

from __future__ import annotations

import msgspec
import pytest
import zmq

from franka_teleop.protocol import PROTOCOL_VERSION, SceneInfo, SceneRequest, decode, encode
from franka_teleop.sim.server import SceneRpc
from franka_teleop.transport import context

from ..conftest import free_port

INFO = SceneInfo(xml="<mujoco/>", nq=0, robot_prefix="robot0_", robot_host="127.0.0.9")


@pytest.fixture
def rpc():
    endpoint = f"tcp://127.0.0.1:{free_port()}"
    rep = context().socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(endpoint)
    server = SceneRpc(rep, INFO)
    server.start()
    yield server, endpoint
    server.stop()
    assert not server.is_alive()


def ask(endpoint: str, frames: list[bytes], timeout: float = 2.0) -> bytes | None:
    sock = context().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    try:
        sock.send_multipart(frames)
        return sock.recv() if sock.poll(int(timeout * 1000)) else None
    finally:
        sock.close()


def error_of(reply: bytes) -> str:
    msg = msgspec.msgpack.decode(reply)
    assert msg["v"] == PROTOCOL_VERSION
    return msg["error"]


def test_scene_request_gets_the_scene_info(rpc):
    server, endpoint = rpc
    assert decode(ask(endpoint, [encode(SceneRequest())]), SceneInfo) == INFO
    assert server.served == 1


def test_bad_requests_get_an_error_reply_and_the_thread_survives(rpc, capsys):
    """Review r7: a two-frame request used to kill the thread (REP send in the wrong state)."""
    server, endpoint = rpc
    good = encode(SceneRequest())
    bad_version = msgspec.msgpack.encode({"v": PROTOCOL_VERSION + 1})
    for frames, why in [
        ([good, b"x"], "got 2 frames"),
        ([b"", good, b""], "got 3 frames"),
        ([b"\xc1garbage"], "cannot decode SceneRequest"),
        ([bad_version], f"protocol version {PROTOCOL_VERSION + 1}"),
    ]:
        reply = ask(endpoint, frames)
        assert reply is not None, frames
        assert why in error_of(reply)
        assert server.is_alive()
        assert decode(ask(endpoint, [good]), SceneInfo) == INFO  # still serving
    assert server.rejected == 4 and server.served == 4
    assert "bad scene request" in capsys.readouterr().err


def test_old_scene_info_without_robot_host_still_decodes():
    """SceneInfo.robot_host is optional: an ft-sim from before it is still understood."""
    old = msgspec.msgpack.encode(
        {"v": PROTOCOL_VERSION, "xml": "<mujoco/>", "nq": 0, "robot_prefix": "robot0_"}
    )
    assert decode(old, SceneInfo).robot_host == ""
