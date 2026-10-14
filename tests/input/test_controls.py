"""Controls tables shown by viewers: generated from the input sources' own key tables."""

from __future__ import annotations

import pytest

from franka_teleop.input.controls import controls_markdown, gamepad_rows, keyboard_rows
from franka_teleop.input.keyboard import ACTION_KEYS, MOTION_KEYS, KeyboardState
from franka_teleop.input.mapping import BUTTON_ACTIONS


def test_keyboard_rows_cover_every_key():
    keys = " ".join(k for k, _ in keyboard_rows())
    for key in [*MOTION_KEYS, *ACTION_KEYS, "q"]:
        assert f"**{key}**" in keys
    assert keyboard_rows()[0] == ("**space**", "deadman on / off (latched)")


@pytest.mark.parametrize(
    ("key", "label", "axis", "sign"),
    [("w", "forward", 0, 1), ("d", "right", 1, -1), ("r", "up", 2, 1), ("j", "yaw left", 5, 1)],
)
def test_keyboard_labels_match_what_the_key_does(key, label, axis, sign):
    """The table says what KeyboardState actually produces for the key."""
    state = KeyboardState()
    state.feed(" ", 0.0)
    state.feed(key, 0.0)
    intent = state.intent(0.0)
    assert (intent.linear + intent.angular)[axis] == sign
    keys, labels = next((k, v) for k, v in keyboard_rows() if f"**{key}**" in k)
    position = 0 if keys.startswith(f"**{key}**") else 1
    assert labels.split(" / ")[position] == label


def test_gamepad_rows_cover_every_button_action():
    rows = dict(gamepad_rows())
    assert rows["LB (hold)"] == "deadman"
    buttons = ("A", "B", "X", "Y", "Start", "d-pad ↑", "d-pad ↓")
    assert sum(k in buttons for k in rows) == len(BUTTON_ACTIONS)
    assert rows["A"].startswith("gripper") and rows["Y"].startswith("go home")


def test_keyboard_controls_say_where_to_type():
    text = controls_markdown("keyboard")
    assert "terminal running `ft-teleop`" in text and "move the camera" in text
    with pytest.raises(ValueError):
        controls_markdown("script")


def test_webpad_controls_are_the_gamepad_table():
    text = controls_markdown("webpad")
    assert "hold **LB** (or the space bar" in text.replace("Hold", "hold")
    assert "| LB (hold) | deadman |" in text and "| A | gripper" in text
