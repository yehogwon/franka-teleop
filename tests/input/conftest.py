"""SDL virtual gamepads, so the real pygame reader can be tested without hardware or a display.

The devices themselves live in :mod:`franka_teleop.input.sdl_virtual` (the web joystick uses
them too).
"""

from __future__ import annotations

from contextlib import suppress

import pytest
import zmq

from franka_teleop import transport
from franka_teleop.input.sdl_virtual import (  # noqa: F401  (re-exported for the tests)
    CONTROLLER_AXIS_INDEX,
    CONTROLLER_BUTTON_INDEX,
    SDLUnavailable,
    VirtualGamepad,
    load_sdl,
)
from franka_teleop.protocol import ProtocolError, TeleopCommand, decode


@pytest.fixture(scope="session")
def sdl():
    from franka_teleop.input import gamepad

    gamepad.init()
    try:
        return load_sdl()
    except SDLUnavailable as e:
        pytest.skip(str(e))


@pytest.fixture
def virtual_gamepad(sdl):
    """Factory: ``pad = virtual_gamepad()`` or ``virtual_gamepad("joystick")``. Detached after
    the test, and the pygame event queue drained so the next test starts clean."""
    import pygame

    pads: list[VirtualGamepad] = []

    def attach(kind: str = "controller") -> VirtualGamepad:
        pad = VirtualGamepad(sdl, kind)
        pads.append(pad)
        pygame.event.pump()
        return pad

    yield attach
    for pad in pads:
        pad.detach()
    pygame.event.pump()
    pygame.event.get()


@pytest.fixture
def gamepad_reader():
    """Factory for GamepadReaders that are closed after the test."""
    from franka_teleop.input.gamepad import GamepadReader

    readers = []

    def make(*args, **kw):
        r = GamepadReader(*args, **kw)
        readers.append(r)
        return r

    yield make
    for r in readers:
        r.close()


# -- command stream helpers ---------------------------------------------------------------


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
