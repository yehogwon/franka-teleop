"""Scripted input: a TOML list of timed steps, for demos, rehearsals and end-to-end tests.

Format (see ``configs/scripts/demo.toml``)::

    description = "optional, shown at start"

    [[step]]
    duration = 2.0              # seconds, required, > 0
    deadman = true              # default true
    linear = [0.5, 0.0, 0.0]    # operator frame (forward, left, up), each in [-1, 1]; default 0
    angular = [0.0, 0.0, 0.0]   # (roll, pitch, yaw), each in [-1, 1]; default 0
    press = ["gripper"]         # actions from protocol.ACTIONS, fired once at step start

A neutral, deadman-released step of :data:`NEUTRAL_PAD_S` is added before the first step (the
node requires every new session to release the deadman once before it moves) and after the
last one (so the run ends stopped). The resulting intent is a pure function of elapsed time.
"""

from __future__ import annotations

import bisect
import math
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol import ACTIONS
from .mapping import ZERO3, Intent, Vec3, zero_counters

NEUTRAL_PAD_S = 0.3
"""Length of the neutral, deadman-released steps added before and after a script."""

_STEP_KEYS = frozenset({"duration", "deadman", "linear", "angular", "press"})
_TOP_KEYS = frozenset({"step", "description"})


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class ScriptStep:
    duration: float
    deadman: bool = True
    linear: Vec3 = ZERO3
    angular: Vec3 = ZERO3
    press: tuple[str, ...] = ()


NEUTRAL_STEP = ScriptStep(duration=NEUTRAL_PAD_S, deadman=False)


class Script:
    """A validated step list (with the neutral padding) and its timeline."""

    def __init__(self, steps: Sequence[ScriptStep], *, description: str = "", name: str = ""):
        if not steps:
            raise ScriptError(f"{name or 'script'}: no steps")
        self.name = name
        self.description = description
        self.steps: tuple[ScriptStep, ...] = (NEUTRAL_STEP, *steps, NEUTRAL_STEP)
        self.starts: list[float] = []
        t = 0.0
        for s in self.steps:
            self.starts.append(t)
            t += s.duration
        self.duration = t

    # -- loading -------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Script:
        path = Path(path)
        try:
            text = path.read_text()
        except OSError as e:
            raise ScriptError(f"cannot read script {path}: {e.strerror}") from e
        return cls.loads(text, name=str(path))

    @classmethod
    def loads(cls, text: str, name: str = "<script>") -> Script:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ScriptError(f"{name}: invalid TOML: {e}") from e
        return cls.from_dict(data, name=name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], name: str = "<script>") -> Script:
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise ScriptError(f"{name}: unknown top-level keys {sorted(unknown)}")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise ScriptError(f"{name}: description must be a string")
        raw = data.get("step")
        if not isinstance(raw, list) or not raw:
            raise ScriptError(f"{name}: needs at least one [[step]]")
        steps = [_parse_step(s, f"{name}: step {i}") for i, s in enumerate(raw, start=1)]
        return cls(steps, description=description, name=name)

    # -- timeline ------------------------------------------------------------------------------

    def step_index(self, t: float) -> int:
        """Index into :attr:`steps` of the step active at elapsed time ``t``."""
        if t < 0:
            return 0
        return min(bisect.bisect_right(self.starts, t) - 1, len(self.steps) - 1)

    def intent_at(self, t: float) -> Intent:
        """Intent at elapsed time ``t``. Counters count presses of every step started by ``t``."""
        i = self.step_index(t)
        counters = zero_counters()
        if t >= 0:
            for s in self.steps[: i + 1]:
                for action in s.press:
                    counters[action] += 1
        step = self.steps[i]
        return Intent(
            deadman=step.deadman, linear=step.linear, angular=step.angular, counters=counters
        )

    def done(self, t: float) -> bool:
        return t >= self.duration


class ScriptSource:
    """Plays a :class:`Script` against a clock; time starts at the first :meth:`read`."""

    name = "script"

    def __init__(self, script: Script, clock: Callable[[], float] = time.monotonic):
        self.script = script
        self.clock = clock
        self._t0: float | None = None

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else self.clock() - self._t0

    def read(self) -> Intent:
        if self._t0 is None:
            self._t0 = self.clock()
        return self.script.intent_at(self.elapsed())

    @property
    def done(self) -> bool:
        return self._t0 is not None and self.script.done(self.elapsed())

    def status(self) -> str:
        t, s = self.elapsed(), self.script
        return f"step {s.step_index(t)}/{len(s.steps) - 1} {t:.1f}/{s.duration:.1f}s"

    def close(self) -> None:
        pass


def _parse_step(raw: Any, where: str) -> ScriptStep:
    if not isinstance(raw, dict):
        raise ScriptError(f"{where}: must be a table")
    unknown = set(raw) - _STEP_KEYS
    if unknown:
        raise ScriptError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(_STEP_KEYS)}")
    if "duration" not in raw:
        raise ScriptError(f"{where}: duration is required")
    duration = raw["duration"]
    if not _is_number(duration) or not math.isfinite(duration) or duration <= 0:
        raise ScriptError(f"{where}: duration must be a positive number, got {duration!r}")
    deadman = raw.get("deadman", True)
    if not isinstance(deadman, bool):
        raise ScriptError(f"{where}: deadman must be true or false, got {deadman!r}")
    press = raw.get("press", [])
    if not isinstance(press, list) or not all(isinstance(p, str) for p in press):
        raise ScriptError(f"{where}: press must be a list of action names")
    bad = [p for p in press if p not in ACTIONS]
    if bad:
        raise ScriptError(f"{where}: unknown actions {bad}; allowed {list(ACTIONS)}")
    return ScriptStep(
        duration=float(duration),
        deadman=deadman,
        linear=_vec3(raw.get("linear", [0.0, 0.0, 0.0]), f"{where}: linear"),
        angular=_vec3(raw.get("angular", [0.0, 0.0, 0.0]), f"{where}: angular"),
        press=tuple(press),
    )


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _vec3(v: Any, where: str) -> Vec3:
    if not isinstance(v, list) or len(v) != 3 or not all(_is_number(x) for x in v):
        raise ScriptError(f"{where} must be a list of 3 numbers, got {v!r}")
    if not all(-1.0 <= x <= 1.0 for x in v):
        raise ScriptError(f"{where} values must be in [-1, 1], got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))
