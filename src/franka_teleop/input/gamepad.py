"""Gamepad reader on pygame / SDL2.

Pads SDL has a GameController mapping for (Xbox, PlayStation, Switch Pro, most Logitech, ...;
see ``SDL_GAMECONTROLLERCONFIG`` for others) are read through ``pygame._sdl2.controller``, which
reports every pad in the same Xbox-style layout. Anything else falls back to the raw joystick
API with an Xbox-like index table and a warning.

Values reported by SDL 2.28 (verified with an SDL virtual controller, see tests/input):
GameController sticks are int16 (-32768..32767) and triggers 0..32767 (0 at rest); raw joystick
axes are floats in [-1, 1], with Xbox-style triggers at -1 at rest.

Only the display (dummy driver, needed for the event queue), joystick and controller modules are
initialized; ``pygame.init()`` would also open audio and print ALSA errors on machines without a
sound card. The reader owns the pygame event queue: :meth:`GamepadReader.read` drains it.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Deliver joystick events even when some window of this process does not have focus.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame
from pygame._sdl2 import controller as sdl_controller

from ..config import InputConfig
from .mapping import (
    TRIGGER_AXES,
    GamepadMapper,
    GamepadSnapshot,
    Intent,
    clamp,
)

log = logging.getLogger(__name__)

AXIS_MAX = 32767.0
RECONNECT_INTERVAL_S = 1.0

CONTROLLER_AXES = {
    "leftx": pygame.CONTROLLER_AXIS_LEFTX,
    "lefty": pygame.CONTROLLER_AXIS_LEFTY,
    "rightx": pygame.CONTROLLER_AXIS_RIGHTX,
    "righty": pygame.CONTROLLER_AXIS_RIGHTY,
    "lefttrigger": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
    "righttrigger": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
}
CONTROLLER_BUTTONS = {
    "a": pygame.CONTROLLER_BUTTON_A,
    "b": pygame.CONTROLLER_BUTTON_B,
    "x": pygame.CONTROLLER_BUTTON_X,
    "y": pygame.CONTROLLER_BUTTON_Y,
    "back": pygame.CONTROLLER_BUTTON_BACK,
    "guide": pygame.CONTROLLER_BUTTON_GUIDE,
    "start": pygame.CONTROLLER_BUTTON_START,
    "leftstick": pygame.CONTROLLER_BUTTON_LEFTSTICK,
    "rightstick": pygame.CONTROLLER_BUTTON_RIGHTSTICK,
    "leftshoulder": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
    "rightshoulder": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
    "dpup": pygame.CONTROLLER_BUTTON_DPAD_UP,
    "dpdown": pygame.CONTROLLER_BUTTON_DPAD_DOWN,
    "dpleft": pygame.CONTROLLER_BUTTON_DPAD_LEFT,
    "dpright": pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
}

# Raw-joystick fallback: the layout SDL's Linux backend reports for Xbox-style pads on the
# xpad/xone/xpadneo drivers (evdev ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ; buttons sorted
# by evdev code; d-pad as hat 0). Other pads may differ: check with ft-check-joystick.
FALLBACK_AXES = {
    "leftx": 0,
    "lefty": 1,
    "lefttrigger": 2,
    "rightx": 3,
    "righty": 4,
    "righttrigger": 5,
}
FALLBACK_BUTTONS = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "leftshoulder": 4,
    "rightshoulder": 5,
    "back": 6,
    "start": 7,
    "guide": 8,
    "leftstick": 9,
    "rightstick": 10,
}
FALLBACK_TRIGGER_REST = -0.9
"""A raw trigger reads 0 until it has been seen below this (at rest, -1). Some drivers report 0
until the first event; without this a resting trigger would read as half pressed."""

_DEVICE_REMOVED = (pygame.JOYDEVICEREMOVED, pygame.CONTROLLERDEVICEREMOVED)
_DEVICE_ADDED = (pygame.JOYDEVICEADDED, pygame.CONTROLLERDEVICEADDED)


def init() -> None:
    """Initialize the pygame modules the reader needs (idempotent). Not ``pygame.init()``."""
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not sdl_controller.get_init():
        sdl_controller.init()


@dataclass(frozen=True)
class ControllerInfo:
    index: int
    name: str
    guid: str
    is_controller: bool
    """SDL has a GameController mapping for it (standard layout)."""
    mapping: dict[str, str] | None


def list_controllers() -> list[ControllerInfo]:
    init()
    pygame.event.pump()
    out = []
    for i in range(pygame.joystick.get_count()):
        joy = pygame.joystick.Joystick(i)
        is_ctl = bool(sdl_controller.is_controller(i))
        mapping = None
        if is_ctl:
            mapping = sdl_controller.Controller(i).get_mapping()
        out.append(ControllerInfo(i, joy.get_name(), joy.get_guid(), is_ctl, mapping))
    return out


@dataclass(frozen=True)
class _Device:
    """An SDL joystick device as probed: its current index and what identifies it."""

    index: int
    guid: str
    name: str
    instance_id: int
    """Unique per connection: SDL never reuses it, not even for the same pad re-plugged."""


class GamepadReader:
    """Reads one pad into normalized :class:`GamepadSnapshot` s.

    The pad is the one found at SDL device ``index`` on the first successful open. From then on
    the reader follows that pad, not the index: SDL renumbers devices when one is unplugged, so
    after a dropout a different pad may sit at the same index.

    A missing or unplugged pad reads as ``connected=False``; the reader retries every
    ``reconnect_interval_s`` (and at once when SDL reports a new device) and only adopts

    - the same SDL device (instance id) if it is still present (e.g. after a read error), or
    - exactly one device with the pad's GUID and name that was not present while the pad was
      connected. SDL never reuses instance ids, so a re-plugged pad comes back with a new one,
      while every device that coexisted with the pad is known not to be it.

    If several devices qualify it refuses to guess and logs an error. SDL cannot tell two pads
    of the same model apart, so if an identical pad (and only that) is plugged in during the
    dropout, it is adopted; the mapper's deadman re-arm then still needs a deliberate release
    and press of LB on it.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        reconnect_interval_s: float = RECONNECT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ):
        init()
        self.log = logger if logger is not None else log
        self.index = index
        """SDL device index the pad is first looked for at (input.gamepad_index)."""
        self.device_index: int | None = None
        """SDL device index of the open pad (may differ from ``index`` after a reconnect)."""
        self.reconnect_interval_s = reconnect_interval_s
        self.clock = clock
        self.name = ""
        self.guid = ""
        self.kind: str | None = None
        """"controller" (SDL GameController), "joystick" (raw fallback) or None."""
        self._ctl: sdl_controller.Controller | None = None
        self._joy: pygame.joystick.JoystickType | None = None
        self._instance_id: int | None = None
        self._identity: tuple[str, str] | None = None
        """(GUID, joystick name) of the pad, fixed at the first open."""
        self._last_instance_id: int | None = None
        self._foreign: set[int] = set()
        """Instance ids of devices seen while the pad was connected: never adopted."""
        self._matches = 0
        """Devices that qualified as the pad at the last reconnect attempt."""
        self._next_attempt = -math.inf
        self._trigger_armed: dict[str, bool] = {}
        pygame.event.pump()
        self._try_open()

    @property
    def connected(self) -> bool:
        return self.kind is not None

    def mapping(self) -> dict[str, str] | None:
        return self._ctl.get_mapping() if self._ctl is not None else None

    def status(self) -> str:
        """Why there is no input, for the HUD ("" while connected)."""
        if self.connected:
            return ""
        if self._identity is None:
            return f"NO GAMEPAD (index {self.index})"
        if self._matches > 1:
            return (
                f"NO GAMEPAD: {self._matches} pads look like your {self.name!r}; cannot tell "
                "which is yours (unplug the others)"
            )
        return f"NO GAMEPAD: waiting for your {self.name!r} to come back (other pads ignored)"

    def _probe_all(self) -> list[_Device]:
        """Every SDL joystick device now present (one ``get_count`` call)."""
        devices = []
        for i in range(pygame.joystick.get_count()):
            try:
                joy = pygame.joystick.Joystick(i)
            except pygame.error:
                continue
            try:
                devices.append(_Device(i, joy.get_guid(), joy.get_name(), joy.get_instance_id()))
            except pygame.error:
                pass
            finally:
                # pygame returns its existing object for a device that is already open: for a
                # raw-joystick pad that is the reader's own, and quit() would close it.
                if joy is not self._joy:
                    with contextlib.suppress(pygame.error):
                        joy.quit()
        return devices

    def _note_foreign(self) -> None:
        """A device appeared while the pad is connected: it is not the pad, now or later."""
        self._foreign.update(
            d.instance_id for d in self._probe_all() if d.instance_id != self._instance_id
        )

    def _choose(self, devices: list[_Device]) -> _Device | None:
        if self._identity is None:  # first open: whatever is at the configured index
            return next((d for d in devices if d.index == self.index), None)
        for d in devices:
            if d.instance_id == self._last_instance_id:
                return d
        matches = [
            d
            for d in devices
            if (d.guid, d.name) == self._identity and d.instance_id not in self._foreign
        ]
        if len(matches) > 1 and len(matches) != self._matches:
            self.log.error(
                "%d gamepads look like the one in use (%r, GUID %s), at indices %s; not "
                "adopting any of them. Unplug the others.",
                len(matches),
                self.name,
                self.guid,
                [d.index for d in matches],
            )
        self._matches = len(matches)
        return matches[0] if len(matches) == 1 else None

    def _try_open(self) -> bool:
        self._next_attempt = self.clock() + self.reconnect_interval_s
        devices = self._probe_all()
        dev = self._choose(devices)
        if dev is None:
            return False
        try:
            if sdl_controller.is_controller(dev.index):
                ctl = sdl_controller.Controller(dev.index)
                self._instance_id = ctl.as_joystick().get_instance_id()
                self._ctl, self.kind, self.name = ctl, "controller", ctl.name
            else:
                joy = pygame.joystick.Joystick(dev.index)
                self._instance_id = joy.get_instance_id()
                self._joy, self.kind, self.name = joy, "joystick", joy.get_name()
                self._trigger_armed = dict.fromkeys(TRIGGER_AXES, False)
                self.log.warning(
                    "%r (GUID %s) has no SDL GameController mapping; using a raw Xbox-style "
                    "layout. Check every axis with ft-check-joystick, or provide a mapping via "
                    "SDL_GAMECONTROLLERCONFIG.",
                    self.name,
                    dev.guid,
                )
        except pygame.error as e:
            self.log.warning("cannot open gamepad %d: %s", dev.index, e)
            self._close_device()
            return False
        self.device_index = dev.index
        self._matches = 1
        if self._identity is None:
            self._identity = (dev.guid, dev.name)
            self.guid = dev.guid
            self._foreign = {d.instance_id for d in devices if d.instance_id != dev.instance_id}
            self.log.info("gamepad %d connected: %s (%s)", dev.index, self.name, self.kind)
        else:
            self.log.info("gamepad %r is back (now index %d)", self.name, dev.index)
        return True

    def _close_device(self) -> None:
        for dev in (self._ctl, self._joy):
            if dev is not None:
                with contextlib.suppress(pygame.error):
                    dev.quit()
        self._ctl = self._joy = None
        self.kind = None
        self._instance_id = None
        self.device_index = None

    def _disconnected(self) -> GamepadSnapshot:
        if self.connected:
            self.log.warning(
                "gamepad %r disconnected; waiting for it to come back (other pads are ignored)",
                self.name,
            )
            self._last_instance_id = self._instance_id
            self._close_device()
            self._next_attempt = self.clock() + self.reconnect_interval_s
        return GamepadSnapshot.disconnected(self.name)

    def read(self) -> GamepadSnapshot:
        """Pump SDL events, handle hot-plugging, and return the current state."""
        removed = added = changed = False
        for ev in pygame.event.get():
            if ev.type in _DEVICE_REMOVED:
                changed = True
                removed |= self.connected and getattr(ev, "instance_id", None) == self._instance_id
            elif ev.type in _DEVICE_ADDED:
                changed = added = True
        if added and self.connected:
            self._note_foreign()
        elif changed and not self.connected:
            self._next_attempt = -math.inf  # the device list changed: look again now
        if removed or (self._ctl is not None and not self._ctl.attached()):
            return self._disconnected()
        if not self.connected and (self.clock() < self._next_attempt or not self._try_open()):
            return GamepadSnapshot.disconnected(self.name)
        try:
            return self._read_controller() if self._ctl is not None else self._read_joystick()
        except pygame.error:
            return self._disconnected()

    def _read_controller(self) -> GamepadSnapshot:
        c = self._ctl
        axes = {}
        for name, axis in CONTROLLER_AXES.items():
            v = c.get_axis(axis) / AXIS_MAX
            axes[name] = clamp(v, 0.0, 1.0) if name in TRIGGER_AXES else clamp(v)
        buttons = {name: bool(c.get_button(b)) for name, b in CONTROLLER_BUTTONS.items()}
        return GamepadSnapshot(axes=axes, buttons=buttons, connected=True, name=self.name)

    def _read_joystick(self) -> GamepadSnapshot:
        j = self._joy
        naxes, nbuttons = j.get_numaxes(), j.get_numbuttons()
        axes = {}
        for name, i in FALLBACK_AXES.items():
            if name in TRIGGER_AXES:
                v = j.get_axis(i) if i < naxes else -1.0
                if v < FALLBACK_TRIGGER_REST:
                    self._trigger_armed[name] = True
                axes[name] = (clamp(v) + 1.0) / 2.0 if self._trigger_armed[name] else 0.0
            else:
                axes[name] = clamp(j.get_axis(i)) if i < naxes else 0.0
        buttons = {
            name: bool(j.get_button(i)) for name, i in FALLBACK_BUTTONS.items() if i < nbuttons
        }
        if j.get_numhats() > 0:
            hx, hy = j.get_hat(0)
            buttons |= {"dpleft": hx < 0, "dpright": hx > 0, "dpup": hy > 0, "dpdown": hy < 0}
        return GamepadSnapshot(axes=axes, buttons=buttons, connected=True, name=self.name)

    def rumble(self, low: float, high: float, duration_ms: int) -> bool:
        """Rumble if the pad supports it. Returns False (and does nothing) otherwise."""
        dev = self._ctl if self._ctl is not None else self._joy
        if dev is None:
            return False
        try:
            return bool(dev.rumble(clamp(low, 0, 1), clamp(high, 0, 1), int(duration_ms)))
        except pygame.error:
            return False

    def close(self) -> None:
        self._close_device()


class GamepadSource:
    """Input source: :class:`GamepadReader` + :class:`GamepadMapper`."""

    name = "gamepad"
    done = False

    def __init__(
        self,
        cfg: InputConfig | None = None,
        *,
        reader: GamepadReader | None = None,
        mapper: GamepadMapper | None = None,
    ):
        cfg = cfg if cfg is not None else InputConfig()
        self.reader = reader if reader is not None else GamepadReader(cfg.gamepad_index)
        self.mapper = mapper if mapper is not None else GamepadMapper.from_config(cfg)
        self.snapshot = GamepadSnapshot.disconnected()

    def read(self) -> Intent:
        self.snapshot = self.reader.read()
        return self.mapper.update(self.snapshot)

    def rumble(self, low: float, high: float, duration_ms: int) -> bool:
        return self.reader.rumble(low, high, duration_ms)

    def status(self) -> str:
        if not self.snapshot.connected:
            return self.reader.status() or "NO GAMEPAD"
        if self.mapper.deadman_blocked:
            return "LB held since the pad connected: release it to arm the deadman"
        return ""

    def close(self) -> None:
        self.reader.close()
