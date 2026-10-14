"""Operator controls as Markdown tables, for viewers (ft-view --mode web).

The keyboard and gamepad rows are built from the same tables the input sources use
(:data:`keyboard.MOTION_KEYS`, :data:`keyboard.ACTION_KEYS`, :data:`mapping.BUTTON_ACTIONS`), so
a remapped key shows up here as well.
"""

from __future__ import annotations

from .keyboard import ACTION_KEYS, DEADMAN_KEY, MOTION_KEYS, QUIT_KEY
from .mapping import BUTTON_ACTIONS, DEADMAN_BUTTON

DEVICES = ("keyboard", "gamepad")
"""Devices with a controls table (the web joystick uses the gamepad's)."""

# (intent axis, sign) -> direction, operator frame (see protocol.TeleopCommand).
MOTION_LABELS: dict[tuple[int, float], str] = {
    (0, 1.0): "forward",
    (0, -1.0): "back",
    (1, 1.0): "left",
    (1, -1.0): "right",
    (2, 1.0): "up",
    (2, -1.0): "down",
    (3, 1.0): "roll right",
    (3, -1.0): "roll left",
    (4, 1.0): "pitch up",
    (4, -1.0): "pitch down",
    (5, 1.0): "yaw left",
    (5, -1.0): "yaw right",
}
ACTION_LABELS: dict[str, str] = {
    "gripper": "gripper close / open (needs deadman)",
    "home": "go home (needs deadman; letting go of it aborts)",
    "stop": "stop and lock",
    "recover": "recover from a robot error / unlock",
    "frame": "toggle base / tool frame",
    "speed_up": "speed level up",
    "speed_down": "speed level down",
}
BUTTON_NAMES: dict[str, str] = {
    "a": "A",
    "b": "B",
    "x": "X",
    "y": "Y",
    "back": "Back",
    "guide": "Guide",
    "start": "Start",
    "leftstick": "left stick click",
    "rightstick": "right stick click",
    "leftshoulder": "LB",
    "rightshoulder": "RB",
    "dpup": "d-pad ↑",
    "dpdown": "d-pad ↓",
    "dpleft": "d-pad ←",
    "dpright": "d-pad →",
}
# Analog axes are mapped in code (mapping.GamepadMapping), not by a table.
GAMEPAD_AXIS_ROWS = (
    ("left stick", "forward / back, left / right"),
    ("RT / LT", "up / down"),
    ("right stick ←→", "yaw"),
    ("right stick ↑↓", "pitch"),
    ("d-pad ← / →", "roll left / right"),
)
FRAME_NOTE = (
    "Directions are for the base frame, for an operator behind the robot looking along +x "
    "(see teleop.operator_yaw_deg). "
    "In tool frame they follow the tool's axes (up = along the approach, out of the fingers)."
)


NBSP = "\u00a0"  # keeps "w / s" on one line in a narrow table column


def _key(k: str) -> str:
    return "**space**" if k == " " else f"**{k}**"


def keyboard_rows() -> list[tuple[str, str]]:
    rows = [(_key(DEADMAN_KEY), "deadman on / off (latched)")]
    by_axis: dict[int, dict[float, str]] = {}
    for key, (axis, sign) in MOTION_KEYS.items():
        by_axis.setdefault(axis, {})[sign] = key
    for axis in sorted(by_axis):
        plus, minus = by_axis[axis][1.0], by_axis[axis][-1.0]
        label = f"{MOTION_LABELS[(axis, 1.0)]} / {MOTION_LABELS[(axis, -1.0)]}"
        rows.append((f"{_key(plus)}{NBSP}/{NBSP}{_key(minus)}", label))
    rows += [(_key(k), ACTION_LABELS[a]) for k, a in ACTION_KEYS.items()]
    rows.append((_key(QUIT_KEY), "quit"))
    return rows


def gamepad_rows() -> list[tuple[str, str]]:
    rows = [(f"{BUTTON_NAMES[DEADMAN_BUTTON]} (hold)", "deadman")]
    rows += list(GAMEPAD_AXIS_ROWS)
    rows += [(BUTTON_NAMES[b], ACTION_LABELS[a]) for b, a in BUTTON_ACTIONS.items()]
    return rows


def _table(head: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"| {head} | action |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def controls_markdown(device: str) -> str:
    if device == "keyboard":
        return (
            "Type in the **terminal running `ft-teleop`** (or `ft-joystick`). Keys pressed in "
            "this page move the camera (W/A/S/D/Q/E), not the robot.\n\n"
            + _table("key", keyboard_rows())
            + f"\n\n{FRAME_NOTE}"
        )
    if device == "gamepad":
        return _table("gamepad", gamepad_rows()) + f"\n\n{FRAME_NOTE}"
    if device == "webpad":
        return (
            "The web joystick is a gamepad: same buttons, same mapping. Hold **LB** (or the space "
            "bar after clicking the pad) while moving a stick.\n\n"
            + _table("gamepad", gamepad_rows())
            + f"\n\n{FRAME_NOTE}"
        )
    raise ValueError(f"no controls for device {device!r}")
