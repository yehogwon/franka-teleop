"""Operator input: gamepad (pygame/SDL), terminal keyboard and TOML scripts -> TeleopCommand.

The gamepad names are imported lazily, so importing this package does not load pygame/SDL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .client import InputSource, JoystickClient, make_source, new_session
from .keyboard import KeyboardSource, KeyboardState
from .mapping import GamepadMapper, GamepadSnapshot, Intent
from .script import Script, ScriptError, ScriptSource, ScriptStep

if TYPE_CHECKING:
    from .gamepad import GamepadReader, GamepadSource, list_controllers

_LAZY = {"GamepadReader", "GamepadSource", "list_controllers"}

__all__ = [
    "GamepadMapper",
    "GamepadReader",
    "GamepadSnapshot",
    "GamepadSource",
    "InputSource",
    "Intent",
    "JoystickClient",
    "KeyboardSource",
    "KeyboardState",
    "Script",
    "ScriptError",
    "ScriptSource",
    "ScriptStep",
    "list_controllers",
    "make_source",
    "new_session",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import gamepad

        return getattr(gamepad, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
