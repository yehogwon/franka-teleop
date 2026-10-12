import pytest

from franka_teleop.input.script import (
    NEUTRAL_PAD_S,
    Script,
    ScriptError,
    ScriptSource,
    ScriptStep,
)
from franka_teleop.protocol import ACTIONS

from ..conftest import CONFIGS

EXAMPLE = """
description = "test"

[[step]]
duration = 1.0
linear = [0.5, 0, 0]

[[step]]
duration = 0.5
deadman = false
press = ["gripper", "home"]

[[step]]
duration = 2
angular = [0, 0, -1]
press = ["gripper"]
"""


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def test_parse_and_padding():
    s = Script.loads(EXAMPLE)
    assert s.description == "test"
    assert len(s.steps) == 5
    first, *user, last = s.steps
    for pad in (first, last):
        assert pad == ScriptStep(duration=NEUTRAL_PAD_S, deadman=False)
    assert user[0] == ScriptStep(duration=1.0, linear=(0.5, 0.0, 0.0))
    assert user[1] == ScriptStep(duration=0.5, deadman=False, press=("gripper", "home"))
    assert user[2].angular == (0.0, 0.0, -1.0) and user[2].deadman
    assert s.duration == pytest.approx(2 * NEUTRAL_PAD_S + 3.5)
    assert s.starts == pytest.approx([0.0, 0.3, 1.3, 1.8, 3.8])


def test_timeline():
    s = Script.loads(EXAMPLE)
    at = s.intent_at
    # Leading neutral step: deadman released, no motion.
    for t in (-1.0, 0.0, 0.29):
        assert not at(t).deadman and at(t).linear == (0, 0, 0)
    assert at(0.3).deadman and at(0.3).linear == (0.5, 0, 0)
    assert at(1.29).linear == (0.5, 0, 0)
    assert not at(1.3).deadman and at(1.3).linear == (0, 0, 0)
    assert at(1.8).angular == (0, 0, -1) and at(1.8).deadman
    # Trailing neutral step, and beyond the end.
    for t in (3.8, 4.09, 4.1, 100.0):
        assert not at(t).deadman and at(t).linear == (0, 0, 0) and at(t).angular == (0, 0, 0)
    assert not s.done(4.09) and s.done(4.1)


def test_counters_increment_once_at_step_start():
    s = Script.loads(EXAMPLE)
    c = [s.intent_at(t).counters for t in (0.0, 1.29, 1.3, 1.5, 1.79, 1.8, 3.0, 50.0)]
    assert [x["gripper"] for x in c] == [0, 0, 1, 1, 1, 2, 2, 2]
    assert [x["home"] for x in c] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert set(c[0]) == set(ACTIONS) and all(c[-1][a] == 0 for a in ("stop", "frame"))


def test_intent_is_a_pure_function_of_time():
    s = Script.loads(EXAMPLE)
    a = [s.intent_at(t) for t in (2.0, 0.5, 1.5)]
    b = [s.intent_at(t) for t in (1.5, 2.0, 0.5)]
    assert a == [b[1], b[2], b[0]]


def test_source_plays_against_its_clock():
    clock = FakeClock()
    src = ScriptSource(Script.loads(EXAMPLE), clock=clock)
    assert src.name == "script" and not src.done
    clock.t += 10.0  # time before the first read does not count
    assert not src.read().deadman
    clock.t += 0.5
    assert src.read().linear == (0.5, 0, 0)
    assert "step 1/4" in src.status()
    clock.t += 3.59
    assert not src.done
    clock.t += 0.02
    assert src.done
    assert not src.read().deadman and src.read().counters["gripper"] == 2


@pytest.mark.parametrize("name", ["demo.toml", "smoke.toml"])
def test_example_scripts_are_valid(name):
    s = Script.load(CONFIGS / "scripts" / name)
    assert s.steps[0].deadman is False and s.steps[-1].deadman is False
    assert any(step.deadman for step in s.steps)
    assert s.description


def test_demo_script_toggles_gripper_and_returns_to_start():
    s = Script.load(CONFIGS / "scripts" / "demo.toml")
    assert sum("gripper" in step.press for step in s.steps) == 2
    # Square-ish: the commanded linear displacement (deadman held) cancels out.
    disp = [sum(st.linear[i] * st.duration for st in s.steps if st.deadman) for i in range(3)]
    assert disp == pytest.approx([0, 0, 0])


def test_smoke_script_moves_forward_for_about_two_seconds():
    s = Script.load(CONFIGS / "scripts" / "smoke.toml")
    moving = [st for st in s.steps if st.deadman and st.linear[0] > 0]
    assert sum(st.duration for st in moving) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "at least one"),
        ("step = []", "at least one"),
        ("[[step]]\nlinear = [0, 0, 0]", "duration is required"),
        ("[[step]]\nduration = 0", "positive"),
        ("[[step]]\nduration = -1", "positive"),
        ("[[step]]\nduration = 'x'", "positive"),
        ("[[step]]\nduration = nan", "positive"),
        ("[[step]]\nduration = inf", "positive"),
        ("[[step]]\nduration = true", "positive"),
        ("[[step]]\nduration = 1\nlinera = [0, 0, 0]", "unknown keys"),
        ("[[step]]\nduration = 1\ndeadman = 1", "deadman"),
        ("[[step]]\nduration = 1\nlinear = [0, 0]", "list of 3 numbers"),
        ("[[step]]\nduration = 1\nlinear = [0, 0, 'a']", "list of 3 numbers"),
        ("[[step]]\nduration = 1\nangular = [0, 0, 1.5]", r"\[-1, 1\]"),
        ("[[step]]\nduration = 1\nlinear = [-1.01, 0, 0]", r"\[-1, 1\]"),
        ("[[step]]\nduration = 1\npress = ['jump']", "unknown actions"),
        ("[[step]]\nduration = 1\npress = 'gripper'", "list of action names"),
        ("speed = 2\n[[step]]\nduration = 1", "unknown top-level keys"),
        ("description = 3\n[[step]]\nduration = 1", "description"),
        ("[[step]\nduration = 1", "invalid TOML"),
        ("step = 3", "at least one"),
        ("step = [1]", "must be a table"),
    ],
)
def test_validation_errors(text, match):
    with pytest.raises(ScriptError, match=match):
        Script.loads(text)


def test_validation_error_names_file_and_step(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[[step]]\nduration = 1\n\n[[step]]\nduration = 1\npress = ['nope']\n")
    with pytest.raises(ScriptError, match=r"bad\.toml: step 2: unknown actions"):
        Script.load(p)
    with pytest.raises(ScriptError, match="cannot read script"):
        Script.load(tmp_path / "missing.toml")


def test_integer_values_are_accepted():
    s = Script.loads("[[step]]\nduration = 2\nlinear = [1, 0, -1]")
    assert s.steps[1].duration == 2.0 and s.steps[1].linear == (1.0, 0.0, -1.0)
    assert isinstance(s.steps[1].linear[0], float)
