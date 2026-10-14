"""SDL virtual gamepads: devices that SDL, and therefore pygame in the *same process*, treats
exactly like a plugged-in pad.

pygame bundles libSDL2 (2.28). Its virtual-joystick API (``SDL_JoystickAttachVirtual``) adds a
device whose axes and buttons are set from code. The web joystick (:mod:`.webpad`) drives one,
so browser input reaches ft-node through the same reader, mapping and command stream as a
physical pad; the tests use them to exercise the reader without hardware.

Call SDL functions on one thread only (the one that reads the pad).
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os

from .mapping import BUTTONS, STICK_AXES, TRIGGER_AXES

SDL_JOYSTICK_TYPE_UNKNOWN = 0
SDL_JOYSTICK_TYPE_GAMECONTROLLER = 1

# SDL's automatic GameController mapping of a virtual controller with 6 axes / 15 buttons:
# leftx=a0 lefty=a1 rightx=a2 righty=a3 lefttrigger=a4 righttrigger=a5, buttons b0..b14 in
# mapping.BUTTONS order (a b x y back guide start leftstick rightstick leftshoulder
# rightshoulder dpup dpdown dpleft dpright).
CONTROLLER_AXIS_INDEX = {n: i for i, n in enumerate(STICK_AXES + TRIGGER_AXES)}
CONTROLLER_BUTTON_INDEX = {n: i for i, n in enumerate(BUTTONS)}


class SDLUnavailable(RuntimeError):
    pass


@functools.cache
def load_sdl() -> ctypes.CDLL:
    """pygame's bundled libSDL2, with the virtual-joystick functions declared. The library is
    the one pygame already loaded, so devices attached here are the ones pygame sees."""
    import pygame

    pattern = os.path.join(os.path.dirname(pygame.__file__), "..", "pygame.libs", "libSDL2-*.so*")
    libs = sorted(glob.glob(pattern))
    if not libs:
        raise SDLUnavailable(f"pygame's bundled libSDL2 not found ({pattern})")
    sdl = ctypes.CDLL(libs[0])
    sdl.SDL_JoystickAttachVirtual.argtypes = [ctypes.c_int] * 4
    sdl.SDL_JoystickAttachVirtual.restype = ctypes.c_int
    sdl.SDL_JoystickDetachVirtual.argtypes = [ctypes.c_int]
    sdl.SDL_JoystickOpen.argtypes = [ctypes.c_int]
    sdl.SDL_JoystickOpen.restype = ctypes.c_void_p
    sdl.SDL_JoystickClose.argtypes = [ctypes.c_void_p]
    sdl.SDL_JoystickInstanceID.argtypes = [ctypes.c_void_p]
    sdl.SDL_JoystickInstanceID.restype = ctypes.c_int32
    sdl.SDL_JoystickGetDeviceInstanceID.argtypes = [ctypes.c_int]
    sdl.SDL_JoystickGetDeviceInstanceID.restype = ctypes.c_int32
    sdl.SDL_NumJoysticks.restype = ctypes.c_int
    sdl.SDL_JoystickSetVirtualAxis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16]
    sdl.SDL_JoystickSetVirtualButton.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
    sdl.SDL_JoystickSetVirtualHat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
    sdl.SDL_GetError.restype = ctypes.c_char_p
    return sdl


class VirtualGamepad:
    """A virtual pad. ``kind="controller"``: SDL GameController (auto-mapped, Xbox layout).
    ``kind="joystick"``: an unmapped Xbox-like joystick (6 axes, 11 buttons, 1 hat) that
    exercises the reader's raw fallback (gamepad.FALLBACK_AXES / FALLBACK_BUTTONS)."""

    def __init__(self, sdl: ctypes.CDLL | None = None, kind: str = "controller"):
        from . import gamepad

        gamepad.init()
        self.sdl = sdl if sdl is not None else load_sdl()
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
        self.handle = None
        self.instance_id = -1
        """SDL instance id of the current attachment (a re-attach gets a new one)."""
        self.attach()

    @property
    def attached(self) -> bool:
        return self.handle is not None

    @property
    def index(self) -> int:
        """Current SDL device index (-1 if detached). SDL renumbers devices when one with a
        lower index is removed, so this is looked up by instance id each time."""
        if self.handle is None:
            return -1
        for i in range(self.sdl.SDL_NumJoysticks()):
            if self.sdl.SDL_JoystickGetDeviceInstanceID(i) == self.instance_id:
                return i
        return -1

    def attach(self) -> None:
        index = self.sdl.SDL_JoystickAttachVirtual(*self._spec)
        if index < 0:
            raise RuntimeError(f"SDL_JoystickAttachVirtual: {self.sdl.SDL_GetError().decode()}")
        self.handle = self.sdl.SDL_JoystickOpen(index)
        if not self.handle:
            raise RuntimeError(f"SDL_JoystickOpen: {self.sdl.SDL_GetError().decode()}")
        self.instance_id = self.sdl.SDL_JoystickInstanceID(self.handle)
        self.reset()

    def detach(self) -> None:
        """Simulate unplugging the pad."""
        if self.handle is None:
            return
        index = self.index
        self.sdl.SDL_JoystickClose(self.handle)
        self.handle = None
        if index >= 0:
            self.sdl.SDL_JoystickDetachVirtual(index)

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
