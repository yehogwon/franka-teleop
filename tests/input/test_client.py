"""JoystickClient in-process: sources -> ZMQ -> a test SUB (and a fake node publishing status)."""

import io
import math
import threading
import time
from contextlib import contextmanager, suppress

import pytest
import zmq

from franka_teleop import transport
from franka_teleop.config import InputConfig
from franka_teleop.input.client import SHUTDOWN_S, JoystickClient, new_session
from franka_teleop.input.mapping import Intent
from franka_teleop.input.script import Script, ScriptSource
from franka_teleop.protocol import ACTIONS, NodeStatus, ProtocolError, TeleopCommand, decode

from ..conftest import make_test_config


class FakeSource:
    def __init__(self, intent=None, name="script"):
        self.name = name
        self.intent = intent or Intent()
        self.done = False
        self.closed = False
        self.rumbles = []
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.intent

    def rumble(self, low, high, ms):
        self.rumbles.append((low, high, ms))
        return True

    def close(self):
        self.closed = True


def status(**kw):
    defaults = dict(
        seq=0,
        t=time.time(),
        mode="idle",
        sim=True,
        q=(0.0,) * 7,
        dq=(0.0,) * 7,
        ee_pos=(0.307, 0.0, 0.487),
        ee_quat=(1.0, 0.0, 0.0, 0.0),
        speed_level=1,
        speed_scale=0.5,
        frame="base",
    )
    return NodeStatus(**(defaults | kw))


def cmd_sub(cfg):
    """A non-conflating SUB on the command port: sees every command (after the slow join)."""
    return transport.subscriber(cfg.net.connect(cfg.net.cmd_port), conflate=False)


def drain(sock, timeout=0.0):
    """Every queued TeleopCommand (waiting up to ``timeout`` for the first one)."""
    out = []
    if timeout and not sock.poll(int(timeout * 1000)):
        return out
    while True:
        try:
            data = sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return out
        with suppress(ProtocolError):
            out.append(decode(data, TeleopCommand))


@contextmanager
def running(client, **run_kw):
    t = threading.Thread(target=client.run, kwargs=run_kw, daemon=True)
    t.start()
    try:
        yield t
    finally:
        client.stop()
        t.join(timeout=10)
        assert not t.is_alive()


def collect_while(thread, sub):
    msgs = []
    while thread.is_alive():
        msgs += drain(sub, timeout=0.05)
    time.sleep(0.05)
    return msgs + drain(sub)


@pytest.fixture
def cfg():
    return make_test_config()


@pytest.fixture
def sockets():
    socks = []
    yield socks
    for s in socks:
        s.close()


def test_session_ids_are_random_nonzero_31_bit():
    ids = {new_session() for _ in range(200)}
    assert len(ids) == 200 and all(0 < i < 2**31 for i in ids)


def test_heartbeat_seq_and_rate(cfg, sockets):
    sub = cmd_sub(cfg)
    sockets.append(sub)
    src = FakeSource()  # idle: deadman released
    client = JoystickClient(cfg, src, hud=False)
    with running(client, max_seconds=1.5) as t:
        msgs = collect_while(t, sub)
    assert len(msgs) > 100
    seqs = [m.seq for m in msgs]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))  # monotonic, no gaps
    assert {m.session for m in msgs} == {client.session} and 0 < client.session < 2**31
    assert all(m.source == "script" and not m.deadman for m in msgs)  # heartbeat while idle
    assert all(m.linear == (0, 0, 0) and m.angular == (0, 0, 0) for m in msgs)
    rate = (len(msgs) - 1) / (msgs[-1].t_send - msgs[0].t_send)
    assert rate == pytest.approx(cfg.input.rate_hz, rel=0.2)
    expected = cfg.input.rate_hz * (1.5 + SHUTDOWN_S)
    assert client.sent == pytest.approx(expected, rel=0.2)
    assert abs(msgs[-1].t_send - time.time()) < 5
    assert src.closed


def test_rate_follows_config(sockets):
    cfg = make_test_config(input__rate_hz=40.0)
    sub = cmd_sub(cfg)
    sockets.append(sub)
    client = JoystickClient(cfg, FakeSource(), hud=False)
    with running(client, max_seconds=1.0) as t:
        msgs = collect_while(t, sub)
    rate = (len(msgs) - 1) / (msgs[-1].t_send - msgs[0].t_send)
    assert rate == pytest.approx(40.0, rel=0.2)


def test_commands_carry_intent_and_counters(cfg, sockets):
    counters = {a: i + 1 for i, a in enumerate(ACTIONS)}
    src = FakeSource(
        Intent(deadman=True, linear=(0.5, -0.25, 1.0), angular=(0, 0.1, -1), counters=counters)
    )
    client = JoystickClient(cfg, src, hud=False)
    try:
        cmd = client.step()
        assert cmd.deadman and cmd.linear == (0.5, -0.25, 1.0) and cmd.angular == (0, 0.1, -1)
        assert cmd.counters() == counters
        assert all(type(v) is float for v in (*cmd.linear, *cmd.angular))
        assert client.step().seq == cmd.seq + 1
    finally:
        client.close()


def test_motion_without_deadman_is_zeroed(cfg):
    src = FakeSource(Intent(deadman=False, linear=(1, 0, 0), angular=(0, 0, 1)))
    client = JoystickClient(cfg, src, hud=False)
    try:
        cmd = client.step()
        assert not cmd.deadman and cmd.linear == (0, 0, 0) and cmd.angular == (0, 0, 0)
    finally:
        client.close()


def test_stop_sends_neutral_tail_keeping_counters(cfg, sockets):
    sub = cmd_sub(cfg)
    sockets.append(sub)
    counters = dict.fromkeys(ACTIONS, 0) | {"gripper": 3, "home": 1}
    src = FakeSource(Intent(deadman=True, linear=(0.5, 0, 0), counters=counters))
    client = JoystickClient(cfg, src, hud=False)
    with running(client) as t:
        time.sleep(0.5)
        client.stop()
        msgs = collect_while(t, sub)
    moving = [i for i, m in enumerate(msgs) if m.deadman]
    assert moving and msgs[moving[0]].linear == (0.5, 0, 0)
    tail = msgs[moving[-1] + 1 :]
    assert len(tail) >= 0.8 * SHUTDOWN_S * cfg.input.rate_hz
    assert tail[-1].t_send - tail[0].t_send >= 0.8 * SHUTDOWN_S
    for m in tail:
        assert not m.deadman and m.linear == (0, 0, 0) and m.angular == (0, 0, 0)
        assert m.counters() == counters  # counters never go backwards
        assert m.session == client.session
    assert src.closed


def test_exit_when_done(cfg, sockets):
    sub = cmd_sub(cfg)
    sockets.append(sub)
    script = Script.loads("[[step]]\nduration = 0.3\nlinear = [0.4, 0, 0]\npress = ['gripper']")
    client = JoystickClient(cfg, ScriptSource(script), exit_when_done=True, hud=False)
    t0 = time.monotonic()
    t = threading.Thread(target=client.run, daemon=True)
    t.start()
    msgs = collect_while(t, sub)
    t.join(5)
    elapsed = time.monotonic() - t0
    assert script.duration + SHUTDOWN_S - 0.1 < elapsed < script.duration + SHUTDOWN_S + 1.0
    assert not msgs[0].deadman  # leading neutral step: deadman released first
    assert any(m.deadman and m.linear[0] == 0.4 for m in msgs)
    assert not msgs[-1].deadman and msgs[-1].linear == (0, 0, 0)
    assert msgs[-1].gripper == 1


def test_without_exit_when_done_keeps_heartbeat_after_script(cfg):
    script = Script.loads("[[step]]\nduration = 0.05")
    src = ScriptSource(script)
    client = JoystickClient(cfg, src, hud=False)
    with running(client, max_seconds=1.0) as t:
        t.join(0.9)
        assert t.is_alive() and src.done
        assert not client.last_cmd.deadman


@pytest.fixture
def fake_node(cfg, sockets):
    """A NodeStatus PUB bound like ft-node's, plus a SUB that sees the client's commands."""
    pub = transport.publisher(cfg.net.bind(cfg.net.status_port))
    sub = transport.subscriber(cfg.net.connect(cfg.net.cmd_port))
    sockets += [pub, sub]
    return pub, sub


def test_rtt_from_echoed_seq(cfg, fake_node):
    pub, sub = fake_node
    client = JoystickClient(cfg, FakeSource(), hud=False)
    with running(client):
        deadline = time.monotonic() + 3.0
        while client.rtt_s is None and time.monotonic() < deadline:
            cmd = transport.recv_wait(sub, TeleopCommand, 0.1)
            if cmd is not None:
                transport.send(
                    pub, status(mode="teleop", last_cmd_seq=cmd.seq, session=cmd.session)
                )
        assert client.rtt_s is not None
        assert 0 < client.rtt_s < 0.2
        assert client.last_status.mode == "teleop" and client.node_alive()
        assert "rtt " in client.hud_line() and "ms" in client.hud_line()


def test_echo_from_another_session_is_ignored(cfg, fake_node):
    pub, _sub = fake_node
    client = JoystickClient(cfg, FakeSource(), hud=False)
    try:
        deadline = time.monotonic() + 3.0
        while client.last_status is None and time.monotonic() < deadline:
            client.step()
            transport.send(pub, status(last_cmd_seq=client.seq - 1, session=client.session + 1))
            time.sleep(0.01)
        assert client.last_status is not None and client.rtt_s is None
        assert "another session in control" in client.hud_line()
    finally:
        client.close()


def step_until(client, pub, msg, pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        transport.send(pub, msg)
        client.step()
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_rumble_on_transition_into_error(cfg, fake_node):
    pub, _ = fake_node
    src = FakeSource(name="gamepad")
    client = JoystickClient(cfg, src, hud=False)
    try:
        step_until(client, pub, status(mode="idle"), lambda: client.last_status is not None)
        assert src.rumbles == []
        step_until(client, pub, status(mode="error", message="reflex"), lambda: src.rumbles)
        for _ in range(10):  # staying in error does not rumble again
            transport.send(pub, status(mode="error", message="reflex"))
            client.step()
            time.sleep(0.005)
        assert len(src.rumbles) == 1
        assert "ERROR" in client.hud_line() and "reflex" in client.hud_line()
        step_until(client, pub, status(mode="idle"), lambda: client.last_status.mode == "idle")
        step_until(client, pub, status(mode="error"), lambda: len(src.rumbles) == 2)
    finally:
        client.close()


def test_rumble_is_optional(cfg, fake_node):
    pub, _ = fake_node

    class NoRumble(FakeSource):
        rumble = None

    src = NoRumble()
    client = JoystickClient(cfg, src, hud=False)
    try:
        step_until(client, pub, status(mode="error"), lambda: client.last_status is not None)
    finally:
        client.close()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_hud_line_contents_and_staleness(cfg, fake_node):
    pub, _ = fake_node
    clock = Clock()
    src = FakeSource(Intent(deadman=True))
    client = JoystickClient(cfg, src, hud=False, clock=clock)
    try:
        client.step()
        line = client.hud_line()
        assert "waiting for ft-node" in line and "DEADMAN" in line
        st = status(
            mode="teleop",
            speed_level=2,
            speed_scale=0.75,
            frame="tool",
            ee_pos=(0.4, -0.05, 0.3),
            gripper_width=0.0412,
            gripper_closed=True,
            limiters=("workspace",),
            message="homing done",
        )
        step_until(client, pub, st, lambda: client.last_status is not None)
        line = client.hud_line()
        for part in (
            "teleop",
            "speed 3/4 x0.75",
            "tool",
            "ee +0.400 -0.050 +0.300",
            "grip closed 41mm",
            "rtt --",
            "limit workspace",
            "homing done",
        ):
            assert part in line, (part, line)
        clock.t += 1.5  # no status for > 1 s
        assert "waiting for ft-node" in client.hud_line()
        nan_grip = status(gripper_width=math.nan)
        step_until(client, pub, nan_grip, lambda: client.last_status.gripper_width != 0.0412)
        assert "grip open |" in client.hud_line()
    finally:
        client.close()


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_hud_refreshes_single_line_at_5hz(cfg):
    clock = Clock()
    out = FakeTTY()
    client = JoystickClient(cfg, FakeSource(), hud=True, hud_stream=out, clock=clock)
    try:
        for _ in range(100):  # 1 s at 100 Hz
            client.step()
            clock.t += 0.01
        text = out.getvalue()
        assert "\n" not in text
        assert 4 <= text.count("\r") <= 6
    finally:
        client.close()
    assert out.getvalue().endswith("\n")


def test_hud_off_writes_nothing(cfg):
    out = FakeTTY()
    client = JoystickClient(cfg, FakeSource(), hud=False, hud_stream=out)
    client.step()
    client.close()
    assert out.getvalue() == ""


def test_second_client_cannot_bind_the_command_port(cfg):
    client = JoystickClient(cfg, FakeSource(), hud=False)
    try:
        with pytest.raises(zmq.ZMQError):
            JoystickClient(cfg, FakeSource(), hud=False)
    finally:
        client.close()
    JoystickClient(cfg, FakeSource(), hud=False).close()  # port free again


def test_virtual_gamepad_end_to_end(cfg, virtual_gamepad, gamepad_reader, sockets):
    """Virtual pad -> GamepadReader -> mapping -> JoystickClient -> ZMQ -> SUB."""
    from franka_teleop.input.gamepad import GamepadSource

    pad = virtual_gamepad()
    sub = cmd_sub(cfg)
    sockets.append(sub)
    src = GamepadSource(InputConfig(), reader=gamepad_reader(pad.index))
    client = JoystickClient(cfg, src, hud=False)

    def tick(n=1):
        for _ in range(n):
            client.step()
            time.sleep(0.002)
        return drain(sub, timeout=0.2)

    try:
        deadline = time.monotonic() + 3.0
        while not tick() and time.monotonic() < deadline:  # slow joiner
            pass
        msgs = tick(3)
        assert msgs and all(m.source == "gamepad" and not m.deadman for m in msgs)

        pad.press("leftshoulder")
        pad.set_axis("lefty", -1.0)  # forward
        pad.set_axis("righttrigger", 1.0)  # up
        pad.set_axis("rightx", -1.0)  # yaw left
        last = tick(3)[-1]
        assert last.deadman
        assert last.linear == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)
        assert last.angular == pytest.approx((0.0, 0.0, 1.0), abs=1e-3)

        pad.press("a")
        tick()
        pad.release("a")
        tick()
        pad.press("a")
        last = tick(2)[-1]
        assert last.gripper == 2 and last.home == 0

        pad.release("leftshoulder")
        last = tick(2)[-1]
        assert not last.deadman and last.linear == (0, 0, 0)

        pad.press("leftshoulder")
        tick()
        pad.detach()  # unplugged mid-motion
        last = tick(2)[-1]
        assert not last.deadman and last.linear == (0, 0, 0) and last.gripper == 2
        assert "NO GAMEPAD" in client.hud_line()
    finally:
        client.close()
