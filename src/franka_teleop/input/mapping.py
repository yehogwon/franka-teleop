"""Gamepad state -> operator intent. Pure Python (no pygame), so it is fully unit-testable.

Names follow SDL's GameController API (Xbox layout), which pygame exposes for every pad SDL
has a mapping for (Xbox, PlayStation, Logitech, ...):

- sticks ``leftx lefty rightx righty`` in [-1, 1]. SDL convention: +x is right, +y is **down**
  (pushing a stick away from you gives a *negative* y).
- triggers ``lefttrigger righttrigger`` in [0, 1], 0 at rest.
- buttons ``a b x y back guide start leftstick rightstick leftshoulder rightshoulder
  dpup dpdown dpleft dpright``.

Default layout (operator frame, see :class:`franka_teleop.protocol.TeleopCommand`)::

    left stick       forward/back, left/right       LB (hold)   deadman
    RT / LT          up / down                      A           gripper toggle
    right stick      yaw (x), pitch (y)             B           stop
    d-pad left/right roll                           X           base/tool frame
    d-pad up/down    speed level +/-                Y           home
                                                    Start       recover from error
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from ..protocol import ACTIONS

STICK_AXES = ("leftx", "lefty", "rightx", "righty")
TRIGGER_AXES = ("lefttrigger", "righttrigger")
AXES = STICK_AXES + TRIGGER_AXES
BUTTONS = (
    "a",
    "b",
    "x",
    "y",
    "back",
    "guide",
    "start",
    "leftstick",
    "rightstick",
    "leftshoulder",
    "rightshoulder",
    "dpup",
    "dpdown",
    "dpleft",
    "dpright",
)

DEADMAN_BUTTON = "leftshoulder"
BUTTON_ACTIONS: Mapping[str, str] = {
    "a": "gripper",
    "b": "stop",
    "x": "frame",
    "y": "home",
    "start": "recover",
    "dpup": "speed_up",
    "dpdown": "speed_down",
}

LAYOUT_HELP = """\
  hold LB = deadman        left stick = forward/back, left/right    RT/LT = up/down
  right stick = yaw (x), pitch (y)    d-pad left/right = roll    d-pad up/down = speed +/-
  A gripper   B stop   X base/tool frame   Y home   Start recover"""

# Names accepted by input.invert, in (linear..., angular...) order.
INTENT_AXES = ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z")

Vec3 = tuple[float, float, float]
ZERO3: Vec3 = (0.0, 0.0, 0.0)


def zero_counters() -> dict[str, int]:
    return dict.fromkeys(ACTIONS, 0)


@dataclass(frozen=True)
class GamepadSnapshot:
    """Normalized gamepad state at one instant. Missing axes/buttons read as 0/released."""

    axes: Mapping[str, float] = field(default_factory=dict)
    buttons: Mapping[str, bool] = field(default_factory=dict)
    connected: bool = True
    name: str = ""

    def axis(self, name: str) -> float:
        return float(self.axes.get(name, 0.0))

    def button(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))

    @classmethod
    def disconnected(cls, name: str = "") -> GamepadSnapshot:
        return cls(connected=False, name=name)


@dataclass(frozen=True)
class Intent:
    """What every input source produces: the payload of a TeleopCommand.

    ``linear``/``angular`` are normalized to [-1, 1] in the operator frame; ``counters`` holds
    one monotonically increasing counter per name in :data:`franka_teleop.protocol.ACTIONS`.
    """

    deadman: bool = False
    linear: Vec3 = ZERO3
    angular: Vec3 = ZERO3
    counters: Mapping[str, int] = field(default_factory=zero_counters)

    def neutral(self) -> Intent:
        """Deadman released, no motion, counters kept (they must never go backwards)."""
        return Intent(deadman=False, counters=dict(self.counters))


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """Circular deadzone with rescale: |v| in [deadzone, 1] maps to [0, 1], direction kept.

    A radial (not per-axis) deadzone keeps diagonals smooth and does not snap a slightly
    off-axis push onto the axis. Magnitudes above 1 (square stick gates) are clamped to 1.
    """
    m = math.hypot(x, y)
    if m <= deadzone or m == 0.0:
        return 0.0, 0.0
    scaled = min(1.0, (m - deadzone) / (1.0 - deadzone)) if deadzone < 1.0 else 0.0
    k = scaled / m
    return x * k, y * k


def scalar_deadzone(v: float, deadzone: float) -> float:
    """Deadzone with rescale for a trigger in [0, 1]: [deadzone, 1] maps to [0, 1]."""
    v = clamp(v, 0.0, 1.0)
    if v <= deadzone or deadzone >= 1.0:
        return 0.0
    return (v - deadzone) / (1.0 - deadzone)


def expo(x: float, e: float) -> float:
    """``(1-e)·x + e·x³``, clamped to [-1, 1]: fine control near zero, full speed at the edge.

    Odd, monotonic for e in [0, 1], and maps 0 -> 0 and ±1 -> ±1.
    """
    x = clamp(x)
    return clamp((1.0 - e) * x + e * x * x * x)


def apply_invert(intent: Intent, invert: Iterable[str]) -> Intent:
    """Negate the named intent axes (see :data:`INTENT_AXES`)."""
    invert = set(invert)
    if not invert:
        return intent
    unknown = invert - set(INTENT_AXES)
    if unknown:
        raise ValueError(f"unknown intent axes {sorted(unknown)}; expected {INTENT_AXES}")
    v = [*intent.linear, *intent.angular]
    v = [-x if name in invert else x for name, x in zip(INTENT_AXES, v, strict=True)]
    # Avoid -0.0 in the output (harmless, but noisy in logs and test diffs).
    v = [x + 0.0 for x in v]
    return replace(intent, linear=(v[0], v[1], v[2]), angular=(v[3], v[4], v[5]))


class GamepadMapper:
    """Stateful mapping from :class:`GamepadSnapshot` to :class:`Intent`.

    State is the previous button set (for rising edges) and the action counters. The first
    connected snapshot (at startup and after every reconnect) only seeds the previous button
    set: a button that is already held when the pad (re)appears does not fire, so re-plugging
    a pad with Y pressed does not send the arm home.
    """

    def __init__(self, deadzone: float = 0.12, expo: float = 0.5, invert: Iterable[str] = ()):
        if not 0.0 <= deadzone < 1.0:
            raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")
        if not 0.0 <= expo <= 1.0:
            raise ValueError(f"expo must be in [0, 1], got {expo}")
        self.deadzone = deadzone
        self.expo = expo
        self.invert = tuple(invert)
        apply_invert(Intent(), self.invert)  # validate names early
        self.counters = zero_counters()
        self._prev: dict[str, bool] | None = None

    @classmethod
    def from_config(cls, cfg) -> GamepadMapper:
        """From an :class:`franka_teleop.config.InputConfig`."""
        return cls(deadzone=cfg.deadzone, expo=cfg.expo, invert=cfg.invert)

    def _stick(self, snap: GamepadSnapshot, xname: str, yname: str) -> tuple[float, float]:
        x, y = radial_deadzone(snap.axis(xname), snap.axis(yname), self.deadzone)
        return expo(x, self.expo), expo(y, self.expo)

    def update(self, snap: GamepadSnapshot) -> Intent:
        if not snap.connected:
            self._prev = None
            return Intent(counters=dict(self.counters))

        pressed = {b: snap.button(b) for b in BUTTONS}
        if self._prev is not None:
            for button, action in BUTTON_ACTIONS.items():
                if pressed[button] and not self._prev[button]:
                    self.counters[action] += 1
        self._prev = pressed

        lx, ly = self._stick(snap, "leftx", "lefty")
        rx, ry = self._stick(snap, "rightx", "righty")
        lt = scalar_deadzone(snap.axis("lefttrigger"), self.deadzone)
        rt = scalar_deadzone(snap.axis("righttrigger"), self.deadzone)
        roll = float(pressed["dpright"]) - float(pressed["dpleft"])
        # "+ 0.0" turns -0.0 into 0.0.
        intent = Intent(
            deadman=pressed[DEADMAN_BUTTON],
            linear=(-ly + 0.0, -lx + 0.0, clamp(rt - lt) + 0.0),
            angular=(roll + 0.0, -ry + 0.0, -rx + 0.0),
            counters=dict(self.counters),
        )
        return apply_invert(intent, self.invert)
