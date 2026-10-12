"""The real pygame/SDL reader against SDL virtual gamepads (see conftest.py)."""

import logging

import pytest

from franka_teleop.config import InputConfig
from franka_teleop.input.gamepad import GamepadSource, list_controllers
from franka_teleop.input.mapping import BUTTONS, STICK_AXES, GamepadMapper

TOL = 1e-4  # int16 quantization


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def pad(virtual_gamepad):
    return virtual_gamepad()


def test_virtual_controller_uses_gamecontroller_api(pad, gamepad_reader):
    r = gamepad_reader(pad.index)
    assert r.connected and r.kind == "controller"
    snap = r.read()
    assert snap.connected and snap.name == "Virtual Controller"
    assert r.mapping()["leftshoulder"] == "b9"


def test_rest_state_is_neutral(pad, gamepad_reader):
    snap = gamepad_reader(pad.index).read()
    assert all(snap.axis(a) == 0.0 for a in STICK_AXES)
    assert snap.axis("lefttrigger") == 0.0 and snap.axis("righttrigger") == 0.0
    assert not any(snap.button(b) for b in BUTTONS)


@pytest.mark.parametrize("axis", STICK_AXES)
@pytest.mark.parametrize("value", [-1.0, -0.5, 0.25, 1.0])
def test_stick_axes_are_normalized(pad, gamepad_reader, axis, value):
    r = gamepad_reader(pad.index)
    pad.set_axis(axis, value)
    snap = r.read()
    assert snap.axis(axis) == pytest.approx(value, abs=TOL)
    assert all(snap.axis(a) == 0.0 for a in STICK_AXES if a != axis)


def test_stick_extremes_are_clamped(pad, gamepad_reader):
    r = gamepad_reader(pad.index)
    pad.set_raw_axis(0, -32768)
    assert r.read().axis("leftx") == -1.0
    pad.set_raw_axis(0, 32767)
    assert r.read().axis("leftx") == 1.0


@pytest.mark.parametrize("trigger", ["lefttrigger", "righttrigger"])
def test_trigger_range_is_zero_to_one(pad, gamepad_reader, trigger):
    r = gamepad_reader(pad.index)
    seen = []
    for v in (0.0, 0.25, 0.5, 1.0):
        pad.set_axis(trigger, v)
        seen.append(r.read().axis(trigger))
    assert seen == pytest.approx([0.0, 0.25, 0.5, 1.0], abs=TOL)
    # SDL's GameController trigger range is 0..32767; raw minimum -> exactly 0, never negative.
    pad.set_raw_axis(4 if trigger == "lefttrigger" else 5, -32768)
    assert r.read().axis(trigger) == 0.0


@pytest.mark.parametrize("button", BUTTONS)
def test_every_button(pad, gamepad_reader, button):
    r = gamepad_reader(pad.index)
    pad.press(button)
    snap = r.read()
    assert snap.button(button)
    assert not any(snap.button(b) for b in BUTTONS if b != button)
    pad.release(button)
    assert not r.read().button(button)


# Through reader + mapping: (pad input, expected linear, expected angular)
SIGN_CASES = [
    ({"lefty": -1.0}, (1, 0, 0), (0, 0, 0)),  # left stick up: forward
    ({"lefty": 1.0}, (-1, 0, 0), (0, 0, 0)),
    ({"leftx": -1.0}, (0, 1, 0), (0, 0, 0)),  # left stick left: left
    ({"leftx": 1.0}, (0, -1, 0), (0, 0, 0)),
    ({"righttrigger": 1.0}, (0, 0, 1), (0, 0, 0)),  # RT: up
    ({"lefttrigger": 1.0}, (0, 0, -1), (0, 0, 0)),  # LT: down
    ({"rightx": -1.0}, (0, 0, 0), (0, 0, 1)),  # right stick left: yaw CCW
    ({"rightx": 1.0}, (0, 0, 0), (0, 0, -1)),
    ({"righty": -1.0}, (0, 0, 0), (0, 1, 0)),  # right stick up: +pitch
    ({"righty": 1.0}, (0, 0, 0), (0, -1, 0)),
    ({"dpright": True}, (0, 0, 0), (1, 0, 0)),  # d-pad right: +roll
    ({"dpleft": True}, (0, 0, 0), (-1, 0, 0)),
]


@pytest.mark.parametrize(("state", "linear", "angular"), SIGN_CASES)
def test_signs_through_reader_and_mapping(pad, gamepad_reader, state, linear, angular):
    src = GamepadSource(InputConfig(), reader=gamepad_reader(pad.index))
    src.read()
    for name, v in state.items():
        pad.set_button(name, v) if isinstance(v, bool) else pad.set_axis(name, v)
    pad.press("leftshoulder")
    out = src.read()
    assert out.deadman
    assert out.linear == pytest.approx(linear, abs=1e-3)
    assert out.angular == pytest.approx(angular, abs=1e-3)


def test_counters_through_reader(pad, gamepad_reader):
    src = GamepadSource(InputConfig(), reader=gamepad_reader(pad.index))
    src.read()
    expected = {
        "a": "gripper",
        "b": "stop",
        "x": "frame",
        "y": "home",
        "start": "recover",
        "dpup": "speed_up",
        "dpdown": "speed_down",
    }
    for button, action in expected.items():
        for _ in range(2):
            pad.press(button)
            src.read()
            pad.release(button)
            src.read()
        assert src.read().counters[action] == 2, button
    assert sum(src.read().counters.values()) == 2 * len(expected)


def test_disconnect_and_reconnect(pad, gamepad_reader):
    clock = Clock()
    r = gamepad_reader(pad.index, reconnect_interval_s=1.0, clock=clock)
    src = GamepadSource(InputConfig(), reader=r)
    pad.press("leftshoulder", "y")
    pad.set_axis("lefty", -1.0)
    assert src.read().deadman
    pad.detach()  # unplugged with the deadman and Y held
    out = src.read()
    assert not r.connected and not src.snapshot.connected
    assert not out.deadman and out.linear == (0, 0, 0) and out.angular == (0, 0, 0)
    assert "NO GAMEPAD" in src.status()
    assert not src.read().deadman  # stays disconnected

    pad.attach()
    pad.press("leftshoulder", "y")
    clock.t += 1.1  # next reconnect attempt
    out = src.read()
    assert r.connected and r.kind == "controller"
    assert out.deadman
    assert out.counters["home"] == 0  # Y held through the re-plug does not fire
    pad.release("y")
    src.read()
    pad.press("y")
    assert src.read().counters["home"] == 1


def test_reconnect_is_throttled(pad, gamepad_reader, monkeypatch):
    clock = Clock()
    r = gamepad_reader(pad.index, reconnect_interval_s=1.0, clock=clock)
    pad.detach()
    r.read()
    assert not r.connected
    import pygame

    attempts = []
    real = pygame.joystick.get_count
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: attempts.append(1) or real())
    for _ in range(20):
        clock.t += 0.01
        assert not r.read().connected
    assert len(attempts) == 0  # no attempt within the interval
    clock.t += 1.0
    r.read()
    assert len(attempts) == 1


def test_missing_gamepad_reads_disconnected(gamepad_reader):
    r = gamepad_reader(7)
    assert not r.connected and r.kind is None
    snap = r.read()
    assert not snap.connected
    out = GamepadMapper().update(snap)
    assert not out.deadman and out.linear == (0, 0, 0)
    assert r.rumble(1.0, 1.0, 100) is False


def test_rumble_is_safe_without_support(pad, gamepad_reader):
    r = gamepad_reader(pad.index)
    assert r.rumble(0.5, 0.5, 50) is False  # virtual pads cannot rumble; must not raise


def test_list_controllers(pad):
    infos = list_controllers()
    mine = [i for i in infos if i.index == pad.index]
    assert len(mine) == 1
    info = mine[0]
    assert info.name == "Virtual Controller" and info.is_controller
    assert len(info.guid) == 32 and info.mapping["a"] == "b0"


# -- raw joystick fallback ---------------------------------------------------------------------


@pytest.fixture
def raw_pad(virtual_gamepad):
    return virtual_gamepad("joystick")


def test_fallback_for_unmapped_pads_warns(raw_pad, gamepad_reader, caplog):
    with caplog.at_level(logging.WARNING, logger="franka_teleop.input.gamepad"):
        r = gamepad_reader(raw_pad.index)
    assert r.kind == "joystick" and r.mapping() is None
    assert "no SDL GameController mapping" in caplog.text
    assert "SDL_GAMECONTROLLERCONFIG" in caplog.text


def test_fallback_axes_buttons_and_hat(raw_pad, gamepad_reader):
    r = gamepad_reader(raw_pad.index)
    snap = r.read()
    assert snap.connected and all(v == 0.0 for v in snap.axes.values())
    for axis in STICK_AXES:
        raw_pad.set_axis(axis, -1.0)
        assert r.read().axis(axis) == pytest.approx(-1.0, abs=TOL)
        raw_pad.set_axis(axis, 0.0)
    for trig in ("lefttrigger", "righttrigger"):
        raw_pad.set_axis(trig, 1.0)
        assert r.read().axis(trig) == pytest.approx(1.0, abs=TOL)
        raw_pad.set_axis(trig, 0.5)
        assert r.read().axis(trig) == pytest.approx(0.5, abs=TOL)
        raw_pad.set_axis(trig, 0.0)
        assert r.read().axis(trig) == 0.0
    for button in ("a", "b", "x", "y", "start", "back", "leftshoulder", "rightshoulder"):
        raw_pad.press(button)
        assert r.read().button(button)
        raw_pad.release(button)
    for (x, y), pressed in {
        (0, 1): "dpup",
        (0, -1): "dpdown",
        (-1, 0): "dpleft",
        (1, 0): "dpright",
    }.items():
        raw_pad.set_hat(x, y)
        snap = r.read()
        assert snap.button(pressed)
        assert sum(snap.button(b) for b in ("dpup", "dpdown", "dpleft", "dpright")) == 1


def test_fallback_trigger_reads_zero_until_seen_at_rest(raw_pad, gamepad_reader):
    raw_pad.set_raw_axis(2, 0)  # some drivers report 0 (= "half pressed") before the first event
    r = gamepad_reader(raw_pad.index)
    assert r.read().axis("lefttrigger") == 0.0
    raw_pad.set_axis("lefttrigger", 0.0)  # -1: at rest -> armed
    assert r.read().axis("lefttrigger") == 0.0
    raw_pad.set_raw_axis(2, 0)
    assert r.read().axis("lefttrigger") == pytest.approx(0.5, abs=TOL)


def test_fallback_disconnect(raw_pad, gamepad_reader):
    r = gamepad_reader(raw_pad.index)
    raw_pad.press("leftshoulder")
    assert r.read().button("leftshoulder")
    raw_pad.detach()
    snap = r.read()
    assert not snap.connected and not r.connected
