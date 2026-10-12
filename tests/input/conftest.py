"""SDL virtual gamepads, so the real pygame reader can be tested without hardware or a display.

pygame bundles libSDL2 (2.28). Its virtual-joystick API (SDL_JoystickAttachVirtual) adds a
device that SDL, and therefore pygame in the *same process*, treats like a plugged-in pad.
"""

from __future__ import annotations

import ctypes
import glob
import os

import pytest

from franka_teleop.input.mapping import BUTTONS, STICK_AXES, TRIGGER_AXES

SDL_JOYSTICK_TYPE_UNKNOWN = 0
SDL_JOYSTICK_TYPE_GAMECONTROLLER = 1

# SDL's automatic GameController mapping of a virtual controller with 6 axes / 15 buttons:
# leftx=a0 lefty=a1 rightx=a2 righty=a3 lefttrigger=a4 righttrigger=a5, buttons b0..b14 in
# mapping.BUTTONS order (a b x y back guide start leftstick rightstick leftshoulder
# rightshoulder dpup dpdown dpleft dpright).
CONTROLLER_AXIS_INDEX = {n: i for i, n in enumerate(STICK_AXES + TRIGGER_AXES)}
CONTROLLER_BUTTON_INDEX = {n: i for i, n in enumerate(BUTTONS)}


def _load_sdl() -> ctypes.CDLL:
    import pygame

    pattern = os.path.join(os.path.dirname(pygame.__file__), "..", "pygame.libs", "libSDL2-*.so*")
    libs = sorted(glob.glob(pattern))
    if not libs:
        pytest.skip("pygame's bundled libSDL2 not found")
    sdl = ctypes.CDLL(libs[0])
    sdl.SDL_JoystickAttachVirtual.argtypes = [ctypes.c_int] * 4
    sdl.SDL_JoystickAttachVirtual.restype = ctypes.c_int
    sdl.SDL_JoystickDetachVirtual.argtypes = [ctypes.c_int]
    sdl.SDL_JoystickOpen.argtypes = [ctypes.c_int]
    sdl.SDL_JoystickOpen.restype = ctypes.c_void_p
    sdl.SDL_JoystickClose.argtypes = [ctypes.c_void_p]
    sdl.SDL_JoystickSetVirtualAxis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16]
    sdl.SDL_JoystickSetVirtualButton.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
    sdl.SDL_JoystickSetVirtualHat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
    sdl.SDL_GetError.restype = ctypes.c_char_p
    return sdl


class VirtualGamepad:
    """A virtual pad. ``kind="controller"``: SDL GameController (auto-mapped, Xbox layout).
    ``kind="joystick"``: an unmapped Xbox-like joystick (6 axes, 11 buttons, 1 hat) that
    exercises the reader's raw fallback (gamepad.FALLBACK_AXES / FALLBACK_BUTTONS)."""

    def __init__(self, sdl: ctypes.CDLL, kind: str = "controller"):
        from franka_teleop.input import gamepad

        self.sdl = sdl
        self.kind = kind
        if kind == "controller":
            self.axis_index = CONTROLLER_AXIS_INDEX
            self.button_index = CONTROLLER_BUTTON_INDEX
            self._spec = (SDL_JOYSTICK_TYPE_GAMECONTROLLER, 6, 15, 0)
        elif kind == "joystick":
            self.axis_index = gamepad.FALLBACK_AXES
            self.button_index = gamepad.FALLBACK_BUTTONS
            self._spec = (SDL_JOYSTICK_TYPE_UNKNOWN, 6, 11, 1)
        else:
            raise ValueError(kind)
        self.index = -1
        self.handle = None
        self.attach()

    @property
    def attached(self) -> bool:
        return self.handle is not None

    def attach(self) -> None:
        self.index = self.sdl.SDL_JoystickAttachVirtual(*self._spec)
        if self.index < 0:
            raise RuntimeError(f"SDL_JoystickAttachVirtual: {self.sdl.SDL_GetError().decode()}")
        self.handle = self.sdl.SDL_JoystickOpen(self.index)
        if not self.handle:
            raise RuntimeError(f"SDL_JoystickOpen: {self.sdl.SDL_GetError().decode()}")
        self.reset()

    def detach(self) -> None:
        """Simulate unplugging the pad."""
        if self.handle is None:
            return
        self.sdl.SDL_JoystickClose(self.handle)
        self.handle = None
        self.sdl.SDL_JoystickDetachVirtual(self.index)

    def reset(self) -> None:
        """Sticks centred, triggers released, buttons up, hat centred."""
        for name in STICK_AXES:
            self.set_axis(name, 0.0)
        for name in TRIGGER_AXES:
            self.set_axis(name, 0.0)
        for name in self.button_index:
            self.set_button(name, False)
        if self.kind == "joystick":
            self.set_hat(0, 0)

    def set_raw_axis(self, index: int, raw: int) -> None:
        self.sdl.SDL_JoystickSetVirtualAxis(self.handle, index, raw)

    def set_axis(self, name: str, value: float) -> None:
        """Sticks: value in [-1, 1] (SDL signs: +y is down). Triggers: value in [0, 1]."""
        if name in TRIGGER_AXES:
            # The raw axis spans the full int16 range; SDL maps it to a 0..32767 trigger.
            raw = round(-32768 + max(0.0, min(1.0, value)) * 65535)
        else:
            raw = round(max(-1.0, min(1.0, value)) * 32767)
        self.set_raw_axis(self.axis_index[name], raw)

    def set_button(self, name: str, down: bool = True) -> None:
        self.sdl.SDL_JoystickSetVirtualButton(self.handle, self.button_index[name], int(down))

    def press(self, *names: str) -> None:
        for n in names:
            self.set_button(n, True)

    def release(self, *names: str) -> None:
        for n in names:
            self.set_button(n, False)

    def set_hat(self, x: int, y: int) -> None:
        """Hat 0 of a ``joystick`` pad; x=+1 right, y=+1 up (pygame convention)."""
        up, right, down, left = 1, 2, 4, 8  # SDL_HAT_* bits
        value = (up * (y > 0)) | (right * (x > 0)) | (down * (y < 0)) | (left * (x < 0))
        self.sdl.SDL_JoystickSetVirtualHat(self.handle, 0, value)


@pytest.fixture(scope="session")
def sdl():
    from franka_teleop.input import gamepad

    gamepad.init()
    return _load_sdl()


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
