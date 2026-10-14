"""Terminal keyboard input, usable over SSH (no X server, no /dev/input access needed).

Split in two: :class:`KeyboardState` is pure (fed ``(key, timestamp)`` events, queried at a time
``t``) and unit-tested; :class:`TerminalKeys` is a thin cbreak-mode stdin reader.

Terminals deliver key presses and autorepeats but no key releases, so "held" is inferred from
timing (see :data:`HOLD_FIRST_S`, :data:`HOLD_REPEAT_S`) and the deadman is a latched toggle.
Prefer a gamepad on the real robot; by default the node accepts keyboard input only in sim.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable, Iterable

from .mapping import Intent, apply_invert, zero_counters

HOLD_FIRST_S = 0.6
"""A key counts as held for this long after a fresh press. The first autorepeat arrives only
after the autorepeat delay (X11/Wayland/console defaults are 250-660 ms, usually 500 ms), so a
held key must survive that gap. Cost: a quick tap moves for this long."""

HOLD_REPEAT_S = 0.12
"""...and for this long after each autorepeat event. Typical repeat rates are 25-33 Hz
(30-40 ms apart); 0.12 s tolerates a late or dropped repeat and still stops promptly after the
key is released."""

# key -> (intent axis index: 0-2 linear (forward, left, up), 3-5 angular (roll, pitch, yaw), sign)
MOTION_KEYS: dict[str, tuple[int, float]] = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "a": (1, 1.0),
    "d": (1, -1.0),
    "r": (2, 1.0),
    "f": (2, -1.0),
    "o": (3, 1.0),  # roll right, like d-pad right
    "u": (3, -1.0),
    "i": (4, 1.0),  # pitch, like right stick up
    "k": (4, -1.0),
    "j": (5, 1.0),  # yaw left (CCW from above), like right stick left
    "l": (5, -1.0),
}
ACTION_KEYS: dict[str, str] = {
    "g": "gripper",
    "h": "home",
    "b": "stop",
    "c": "recover",
    "t": "frame",
    "]": "speed_up",
    "[": "speed_down",
}
DEADMAN_KEY = " "
QUIT_KEY = "q"

KEY_HELP = """\
keyboard: type in THIS terminal (keys in a viewer window only move its camera)
  space = deadman on/off (latched)   q = quit (Ctrl-C works too)
  w/s forward/back   a/d left/right   r/f up/down
  j/l yaw left/right   i/k pitch up/down   u/o roll left/right
  g gripper   h home   b stop   c recover   t base/tool frame   ]/[ speed up/down
  Directions are for the base frame. In tool frame they follow the tool's axes: at the ready
  pose a (left) moves the hand to base -y and r (up) moves it down, along the approach."""

_OPPOSITE = {
    k: next(o for o, (ax2, s2) in MOTION_KEYS.items() if ax2 == ax and s2 == -s)
    for k, (ax, s) in MOTION_KEYS.items()
}


class KeyboardState:
    """Key events in, intent out. Pure: time is passed in, never read.

    - Motion keys are held while within their hold window; each axis is +1, -1 or 0. Pressing a
      key releases its opposite at once, so reversing does not wait for the old window to end.
    - Action keys fire on *fresh* presses only: an event that arrives while the key is still
      within its hold window is autorepeat and is ignored (otherwise holding ``g`` would toggle
      the gripper ~30 times a second). Two deliberate taps of the same key must therefore be
      more than :data:`HOLD_FIRST_S` apart.
    - The deadman is asymmetric. Switching it ON needs a fresh space press, like an action key.
      Switching it OFF does not: any space event while it is ON, autorepeat or a quick second
      tap included, turns it off at once. So a held, auto-repeating or hammered space fails
      safe: it can switch the deadman off but never flaps it back on (that needs a fresh press,
      i.e. space released for its hold window first).
    - Switching the deadman off also forgets held motion keys.
    - After ``q`` (quit) every further key is ignored and the deadman stays off.
    """

    def __init__(self, hold_first_s: float = HOLD_FIRST_S, hold_repeat_s: float = HOLD_REPEAT_S):
        self.hold_first_s = hold_first_s
        self.hold_repeat_s = hold_repeat_s
        self.deadman = False
        self.quit = False
        self.counters = zero_counters()
        self._until: dict[str, float] = {}

    def _event(self, key: str, t: float) -> bool:
        """Extend ``key``'s hold window; True if this is a fresh press rather than autorepeat."""
        until = self._until.get(key)
        fresh = until is None or t > until
        self._until[key] = t + self.hold_first_s if fresh else max(until, t + self.hold_repeat_s)
        return fresh

    def _deadman_off(self) -> None:
        self.deadman = False
        for k in MOTION_KEYS:
            self._until.pop(k, None)

    def feed(self, key: str, t: float) -> None:
        if self.quit:
            return
        if key.isalpha():
            key = key.lower()  # Shift / Caps Lock
        if key in MOTION_KEYS:
            self._until.pop(_OPPOSITE[key], None)
            self._event(key, t)
        elif key in ACTION_KEYS:
            if self._event(key, t):
                self.counters[ACTION_KEYS[key]] += 1
        elif key == DEADMAN_KEY:
            # Always extend the window (also when switching off), so that autorepeat or
            # hammering after an OFF does not count as a fresh press that turns it back on.
            fresh = self._event(key, t)
            if self.deadman:
                self._deadman_off()
            elif fresh:
                self.deadman = True
        elif key == QUIT_KEY:
            self.quit = True
            self._deadman_off()

    def held(self, key: str, t: float) -> bool:
        until = self._until.get(key)
        return until is not None and t <= until

    def intent(self, t: float) -> Intent:
        v = [0.0] * 6
        for key, (axis, sign) in MOTION_KEYS.items():
            if self.held(key, t):
                v[axis] += sign
        return Intent(
            deadman=self.deadman,
            linear=(v[0], v[1], v[2]),
            angular=(v[3], v[4], v[5]),
            counters=dict(self.counters),
        )


def split_keys(text: str) -> list[str]:
    """Split terminal input into single-character keys, dropping escape sequences.

    Arrow and function keys arrive as ``ESC [ ... final`` or ``ESC O x``; without this their
    ``[`` would read as "speed down". Alt+key (``ESC key``) is dropped too.
    """
    keys: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\x1b":
            keys.append(ch)
            i += 1
            continue
        i += 1
        if i < n and text[i] == "[":  # CSI: parameters, then a final byte in @..~
            i += 1
            while i < n and not ("@" <= text[i] <= "~"):
                i += 1
            i += 1
        elif i < n:  # SS3 (ESC O x) or Alt+key (ESC x)
            i += 2 if text[i] == "O" else 1
    return keys


class TerminalKeys:
    """Non-blocking key reader for a terminal in cbreak mode (no echo, no line buffering).

    cbreak keeps ISIG, so Ctrl-C still raises SIGINT. Output flow control (IXON/IXOFF) is
    switched off: with it, Ctrl-S (XOFF) stops the terminal's output, and a program writing a
    status line to it blocks until Ctrl-Q. Ctrl-S / Ctrl-Q now arrive as ordinary (ignored) keys.

    :meth:`restore` puts the saved terminal settings back; it runs on :meth:`close` (call it
    from a ``finally``) and at interpreter exit. SIGTERM and SIGHUP must be turned into a normal
    exit (ft-joystick does) for either to run.
    """

    def __init__(self, fd: int | None = None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        if not os.isatty(self.fd):
            raise RuntimeError("keyboard input needs a terminal, but stdin is not a TTY")
        self.hung_up = False
        """The terminal is gone (EOF or EIO on read): no more keys will come."""
        self._saved: list | None = termios.tcgetattr(self.fd)
        atexit.register(self.restore)
        mode = termios.tcgetattr(self.fd)
        tty.cfmakecbreak(mode)
        mode[tty.IFLAG] &= ~(termios.IXON | termios.IXOFF)
        termios.tcsetattr(self.fd, termios.TCSANOW, mode)

    def read(self) -> str:
        """Everything typed since the last call ("" if nothing)."""
        chunks: list[bytes] = []
        while not self.hung_up and select.select([self.fd], [], [], 0)[0]:
            try:
                data = os.read(self.fd, 1024)
            except OSError:  # EIO: the terminal was hung up
                data = b""
            if not data:  # readable but nothing to read: hung up
                self.hung_up = True
                break
            chunks.append(data)
        return b"".join(chunks).decode(errors="ignore")

    def restore(self) -> None:
        if self._saved is None:
            return
        saved, self._saved = self._saved, None
        atexit.unregister(self.restore)
        # TCSAFLUSH: drop unread input, so autorepeated keys do not spill into the shell.
        # termios.error: the terminal is already gone (e.g. SSH session closed).
        with contextlib.suppress(termios.error):
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, saved)

    close = restore


class KeyboardSource:
    """Input source reading the terminal; ``done`` once ``q`` was pressed or the terminal is
    gone (which also releases a latched deadman)."""

    name = "keyboard"

    def __init__(
        self,
        invert: Iterable[str] = (),
        *,
        terminal: TerminalKeys | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.invert = tuple(invert)
        self.clock = clock
        self.state = KeyboardState()
        self.terminal = terminal if terminal is not None else TerminalKeys()

    def read(self) -> Intent:
        now = self.clock()
        for key in split_keys(self.terminal.read()):
            self.state.feed(key, now)
            if self.state.quit:
                break  # keys typed (or pasted) after q are not applied
        if getattr(self.terminal, "hung_up", False):
            self.state.feed(QUIT_KEY, now)  # nobody can switch a latched deadman off any more
        return apply_invert(self.state.intent(now), self.invert)

    @property
    def done(self) -> bool:
        return self.state.quit

    def status(self) -> str:
        if self.state.deadman:
            return "space = off"
        # Viewers have their own key bindings (viser moves its camera with W/A/S/D/Q/E), so
        # an operator typing into the viewer window sees the camera move and the arm not.
        return "deadman off: press space here, not in the viewer"

    def close(self) -> None:
        self.terminal.close()
