"""Web joystick: browser pad state -> SDL virtual pad -> the regular gamepad reader and mapping.

A page is played by a Python WebSocket client that speaks the page's protocol (JSON pad state);
tests/e2e/test_webpad_browser.py drives the real page in Chromium.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pygame
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Request
from websockets.sync.client import connect

from franka_teleop.config import InputConfig
from franka_teleop.input.client import make_source
from franka_teleop.input.webpad import STALE_S, WebPadSource, parse_state, same_origin

from ..conftest import free_port, make_test_config

FORWARD = {"axes": {"lefty": -1.0}, "buttons": {"leftshoulder": True}}
RELEASED = {"axes": {}, "buttons": {}}
LB_ONLY = {"axes": {}, "buttons": {"leftshoulder": True}}


class Page:
    """A pad page: connects like the browser does (Origin = the page's own origin)."""

    def __init__(self, port: int):
        self.port = port
        self.ws = connect(f"ws://127.0.0.1:{port}/ws", origin=f"http://127.0.0.1:{port}")
        self.role = None
        self.recv_role()

    def recv_role(self, timeout: float = 2.0) -> dict:
        while True:
            msg = json.loads(self.ws.recv(timeout))
            if msg["type"] == "role":
                self.role = msg["role"]
                return msg

    def recv(self, timeout: float = 2.0) -> dict:
        return json.loads(self.ws.recv(timeout))

    def send(self, state: dict) -> None:
        self.ws.send(json.dumps({"type": "state", **state}))

    def take(self) -> None:
        self.ws.send(json.dumps({"type": "take"}))

    def close(self) -> None:
        self.ws.close()


@pytest.fixture
def webpad(sdl):
    """Factory for WebPadSources on free ports, closed after the test."""
    sources = []

    def make(**kw):
        src = WebPadSource(InputConfig(webpad_port=0, **kw))
        sources.append(src)
        return src

    yield make
    for src in sources:
        src.close()
    pygame.event.pump()
    pygame.event.get()


def hold(src: WebPadSource, page: Page, state: dict, seconds: float = 0.1):
    """Keep sending ``state`` (as the page's heartbeat does) while ft-joystick's loop reads."""
    t_end = time.monotonic() + seconds
    intent = None
    while time.monotonic() < t_end:
        page.send(state)
        time.sleep(0.01)
        intent = src.read()
    return intent


# -- pure helpers ------------------------------------------------------------------------------


def test_parse_state_clamps_and_ignores_unknown_names():
    s = parse_state(
        {
            "axes": {"leftx": 3.0, "lefty": -2, "lefttrigger": -0.5, "righttrigger": 0.25, "z": 9},
            "buttons": {"a": True, "leftshoulder": False, "turbo": True},
        }
    )
    assert s.axes == {"leftx": 1.0, "lefty": -1.0, "lefttrigger": 0.0, "righttrigger": 0.25}
    assert s.buttons == {"a": True, "leftshoulder": False}


@pytest.mark.parametrize(
    "msg",
    [
        {"axes": {"leftx": float("nan")}},
        {"axes": {"leftx": float("inf")}},
        {"axes": {"leftx": "1"}},
        {"axes": {"leftx": True}},
        {"buttons": {"a": 1}},
        {"axes": [1, 2]},
    ],
)
def test_parse_state_rejects_bad_values(msg):
    with pytest.raises(ValueError):
        parse_state(msg)


def _request(host: str | None, origin: str | None) -> Request:
    headers = Headers()
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    return Request("/ws", headers)


@pytest.mark.parametrize(
    ("host", "origin", "ok"),
    [
        ("127.0.0.1:8081", "http://127.0.0.1:8081", True),
        ("localhost:8081", "http://localhost:8081", True),
        ("[::1]:8081", "http://[::1]:8081", True),
        ("192.168.1.20:8081", "http://192.168.1.20:8081", True),
        ("127.0.0.1:8081", "http://evil.example", False),  # another site in the same browser
        ("127.0.0.1:8081", "http://127.0.0.1:8080", False),  # even the viewer's own origin
        ("evil.example:8081", "http://evil.example:8081", False),  # DNS rebinding
        ("127.0.0.1:8081", None, False),
        (None, "http://127.0.0.1:8081", False),
        ("127.0.0.1:8081", "null", False),
    ],
)
def test_same_origin(host, origin, ok):
    assert same_origin(_request(host, origin)) is ok


# -- server + SDL virtual pad + gamepad reader ---------------------------------------------------


def test_page_drives_the_gamepad_mapping(webpad):
    src = webpad()
    assert src.name == "gamepad"  # ft-node sees a gamepad
    assert not src.read().deadman and not src.reader.connected  # no page: pad unplugged
    assert "no page connected" in src.status() and src.server.url in src.status()

    page = Page(src.server.port)
    assert page.role == "controller"
    hold(src, page, RELEASED)  # LB seen released once: the mapper arms the deadman
    intent = hold(src, page, FORWARD)
    assert intent.deadman and intent.linear == (1.0, 0.0, 0.0)
    assert src.reader.connected and src.reader.kind == "controller"

    intent = hold(src, page, {"axes": {"righttrigger": 1.0}, "buttons": {"leftshoulder": True}})
    assert intent.linear == (0.0, 0.0, 1.0)
    intent = hold(src, page, {"axes": {"rightx": -1.0}, "buttons": {"leftshoulder": True}})
    assert intent.angular == (0.0, 0.0, 1.0)  # right stick left: yaw left

    # Buttons are rising edges, like on a physical pad: holding A toggles the gripper once.
    intent = hold(src, page, {"axes": {}, "buttons": {"leftshoulder": True, "a": True}}, 0.2)
    assert intent.counters["gripper"] == 1
    intent = hold(src, page, RELEASED)
    assert not intent.deadman and intent.counters["gripper"] == 1


def test_silent_page_releases_the_pad_and_resumes(webpad):
    src = webpad()
    page = Page(src.server.port)
    hold(src, page, RELEASED)
    assert hold(src, page, FORWARD).deadman
    time.sleep(STALE_S + 0.1)  # frozen tab: nothing arrives
    intent = src.read()
    assert not intent.deadman and intent.linear == (0.0, 0.0, 0.0)
    assert "not responding" in src.status()
    assert src.reader.connected  # released, not unplugged
    assert hold(src, page, FORWARD).deadman  # the heartbeat is back: a fresh LB press


def test_closing_the_page_unplugs_the_pad_and_rearm_is_required(webpad):
    src = webpad()
    page = Page(src.server.port)
    hold(src, page, RELEASED)
    assert hold(src, page, FORWARD).deadman
    page.close()
    time.sleep(0.05)
    intent = src.read()
    assert not intent.deadman and not src.reader.connected

    # A page that comes back with LB already held does not move the arm: the reader re-adopts
    # the re-plugged pad and the mapper wants LB released first, as for a physical pad.
    page = Page(src.server.port)
    intent = hold(src, page, FORWARD, 0.2)
    assert src.reader.connected and not intent.deadman
    assert "release it to arm the deadman" in src.status()
    hold(src, page, RELEASED)
    assert hold(src, page, FORWARD).deadman


def test_second_page_watches_until_it_takes_over(webpad):
    src = webpad()
    first = Page(src.server.port)
    second = Page(src.server.port)
    assert second.role == "spectator"
    assert first.recv_role()["pages"] == 2
    hold(src, first, RELEASED)
    assert hold(src, first, FORWARD).deadman

    # The spectator's state is ignored.
    second.send({"axes": {"leftx": 1.0}, "buttons": {"leftshoulder": True}})
    intent = hold(src, first, FORWARD)
    assert intent.linear == (1.0, 0.0, 0.0)

    # Take over: a re-plugged pad, so even a held LB on the new page needs a release first.
    second.take()
    assert second.recv_role()["role"] == "controller"
    assert first.recv_role()["role"] == "spectator"
    intent = hold(src, second, FORWARD, 0.2)
    assert not intent.deadman
    hold(src, second, RELEASED)
    assert hold(src, second, FORWARD).deadman
    first.send(RELEASED)  # ignored now
    assert hold(src, second, FORWARD).deadman


def test_garbled_message_releases_the_pad(webpad):
    src = webpad()
    page = Page(src.server.port)
    hold(src, page, RELEASED)
    assert hold(src, page, FORWARD).deadman
    page.send({"axes": {"lefty": "fast"}, "buttons": {"leftshoulder": True}})
    time.sleep(0.05)
    assert not src.read().deadman
    page.ws.send("not json")
    time.sleep(0.05)
    assert not src.read().deadman
    assert src.server.bad_messages == 2


def test_rumble_reaches_the_controller_page(webpad):
    src = webpad()
    assert not src.rumble(1.0, 1.0, 400)  # nobody to tell
    page = Page(src.server.port)
    assert src.rumble(1.0, 0.5, 400)
    assert page.recv() == {"type": "rumble", "low": 1.0, "high": 0.5, "ms": 400}


def test_reader_never_adopts_another_pad(webpad, virtual_gamepad):
    """A physical pad on the laptop (here: an unmapped virtual joystick, plugged in first) is
    not what the web pad's reader reads."""
    other = virtual_gamepad("joystick")
    other.set_button("leftshoulder", True)
    other.set_axis("lefty", -1.0)
    src = webpad()
    assert src.reader.device_index is None  # unplugged: no page yet
    for _ in range(5):
        intent = src.read()
        assert not intent.deadman and not src.reader.connected
    page = Page(src.server.port)
    intent = hold(src, page, RELEASED)
    assert src.reader.connected and src.reader.kind == "controller"
    assert src.reader.device_index != other.index
    assert not intent.deadman


def test_http_routes(webpad):
    src = webpad()
    base = f"http://127.0.0.1:{src.server.port}"
    with urllib.request.urlopen(base + "/") as r:
        body = r.read().decode()
        assert r.status == 200 and r.headers["Content-Type"].startswith("text/html")
        # Asks for its own browser process: embedded in the viewer's panel, the pad must not
        # share a main thread with the 3D view, whose frames delayed its heartbeat by up to
        # 0.8 s with software rendering (the pad then read as released, stop-and-go motion).
        assert r.headers["Origin-Agent-Cluster"] == "?1"
    assert 'aria-label="virtual gamepad"' in body
    with urllib.request.urlopen(base + "/healthz") as r:
        assert r.read() == b"ok\n"
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/nope")
    assert e.value.code == 404
    with pytest.raises(InvalidStatus) as e:
        connect(f"ws://127.0.0.1:{src.server.port}/ws", origin="http://evil.example")
    assert e.value.response.status_code == 403


def test_busy_port_is_a_clear_error(webpad, sdl):
    src = webpad()
    cfg = make_test_config(input__webpad_port=src.server.port)
    with pytest.raises(RuntimeError, match="cannot serve the web joystick"):
        make_source("webpad", cfg)


def test_free_port_helper_is_usable_for_webpad(sdl):
    port = free_port()
    src = WebPadSource(InputConfig(webpad_port=port))
    try:
        assert src.server.port == port
    finally:
        src.close()
