import itertools
import math

import pytest

from franka_teleop.input.mapping import (
    BUTTON_ACTIONS,
    BUTTONS,
    LAYOUT_HELP,
    GamepadMapper,
    GamepadSnapshot,
    Intent,
    apply_invert,
    expo,
    radial_deadzone,
    scalar_deadzone,
)
from franka_teleop.protocol import ACTIONS


def snap(connected=True, **state):
    axes = {k: float(v) for k, v in state.items() if not isinstance(v, bool)}
    buttons = {k: v for k, v in state.items() if isinstance(v, bool)}
    assert set(buttons) <= set(BUTTONS), buttons
    return GamepadSnapshot(axes=axes, buttons=buttons, connected=connected, name="test")


def linear_mapper(**kw):
    """No deadzone and no expo: outputs equal the (sign-converted) inputs."""
    return GamepadMapper(deadzone=kw.pop("deadzone", 0.0), expo=kw.pop("expo", 0.0), **kw)


# -- signs -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "linear", "angular"),
    [
        ({"lefty": -1.0}, (1, 0, 0), (0, 0, 0)),  # stick pushed up -> forward
        ({"lefty": 1.0}, (-1, 0, 0), (0, 0, 0)),
        ({"leftx": -1.0}, (0, 1, 0), (0, 0, 0)),  # stick left -> +y (left)
        ({"leftx": 1.0}, (0, -1, 0), (0, 0, 0)),
        ({"righttrigger": 1.0}, (0, 0, 1), (0, 0, 0)),  # RT -> up
        ({"lefttrigger": 1.0}, (0, 0, -1), (0, 0, 0)),
        ({"rightx": -1.0}, (0, 0, 0), (0, 0, 1)),  # right stick left -> yaw CCW (from above)
        ({"rightx": 1.0}, (0, 0, 0), (0, 0, -1)),
        ({"righty": -1.0}, (0, 0, 0), (0, 1, 0)),  # right stick up -> +pitch
        ({"righty": 1.0}, (0, 0, 0), (0, -1, 0)),
        ({"dpright": True}, (0, 0, 0), (1, 0, 0)),  # roll, digital
        ({"dpleft": True}, (0, 0, 0), (-1, 0, 0)),
        ({"dpleft": True, "dpright": True}, (0, 0, 0), (0, 0, 0)),
    ],
)
@pytest.mark.parametrize("cfg", [{"deadzone": 0.0, "expo": 0.0}, {"deadzone": 0.12, "expo": 0.5}])
def test_axis_signs_at_full_deflection(state, linear, angular, cfg):
    intent = GamepadMapper(**cfg).update(snap(**state))
    assert intent.linear == pytest.approx(linear)
    assert intent.angular == pytest.approx(angular)


def test_rest_is_zero_and_triggers_cancel():
    m = GamepadMapper()
    assert m.update(snap()).linear == (0, 0, 0)
    assert m.update(snap()).angular == (0, 0, 0)
    both = m.update(snap(lefttrigger=1.0, righttrigger=1.0))
    assert both.linear == (0, 0, 0)
    assert not any(math.copysign(1, v) < 0 for v in (*both.linear, *both.angular))  # no -0.0


def test_layout_help_explains_base_and_tool_frame():
    text = " ".join(LAYOUT_HELP.split())
    assert "hold LB = deadman" in text
    assert "Directions are for the base frame" in text
    assert "In tool frame they follow the tool's axes" in text
    assert "base -y" in text and "moves it down, along the approach" in text


def test_deadman_is_left_shoulder():
    m = GamepadMapper()
    assert not m.update(snap()).deadman
    assert m.update(snap(leftshoulder=True)).deadman
    assert not m.update(snap(rightshoulder=True)).deadman


def test_deadman_needs_a_release_after_connect_and_reconnect():
    m = GamepadMapper()
    held = snap(leftshoulder=True, lefty=-1.0)
    assert not m.update(held).deadman and m.deadman_blocked  # held at startup
    assert not m.update(held).deadman
    assert not m.update(snap(lefty=-1.0)).deadman and not m.deadman_blocked  # released: armed
    assert m.update(held).deadman
    m.update(snap(connected=False))  # dropout with LB held ...
    assert not m.deadman_blocked
    out = m.update(held)  # ... and back with LB still held: no motion
    assert not out.deadman and m.deadman_blocked
    assert not m.update(held).deadman
    m.update(snap())
    assert m.update(held).deadman


def test_disconnected_is_deadman_off_and_zero():
    m = GamepadMapper()
    m.update(snap(leftshoulder=True, lefty=-1.0))
    m.update(snap(a=True))
    out = m.update(snap(connected=False, leftshoulder=True, lefty=-1.0, a=True))
    assert not out.deadman and out.linear == (0, 0, 0) and out.angular == (0, 0, 0)
    assert out.counters["gripper"] == 1  # counters are kept, never reset


# -- deadzone / expo ---------------------------------------------------------------------------


def test_radial_deadzone_rescales_and_keeps_direction():
    dz = 0.2
    assert radial_deadzone(0.0, 0.0, dz) == (0.0, 0.0)
    assert radial_deadzone(0.1, 0.1, dz) == (0.0, 0.0)  # |v| = 0.141 < dz
    assert radial_deadzone(0.2, 0.0, dz) == (0.0, 0.0)
    assert radial_deadzone(1.0, 0.0, dz) == pytest.approx((1.0, 0.0))
    assert radial_deadzone(0.6, 0.0, dz) == pytest.approx((0.5, 0.0))
    # Direction is preserved, magnitude rescaled.
    x, y = radial_deadzone(0.3, 0.4, dz)  # |v| = 0.5 -> (0.5-0.2)/0.8 = 0.375
    assert math.hypot(x, y) == pytest.approx(0.375)
    assert math.atan2(y, x) == pytest.approx(math.atan2(0.4, 0.3))
    # Corner of a square gate: magnitude clamped to 1.
    x, y = radial_deadzone(1.0, -1.0, dz)
    assert math.hypot(x, y) == pytest.approx(1.0)
    assert x == pytest.approx(-y)


def test_radial_deadzone_is_radial_not_per_axis():
    # Each component is below the deadzone, but the magnitude is not: the push must count.
    x, y = radial_deadzone(0.15, 0.15, 0.2)
    assert x > 0 and y > 0


def test_deadzone_is_continuous_at_the_edge():
    dz = 0.12
    x, _ = radial_deadzone(dz + 1e-9, 0.0, dz)
    assert 0.0 <= x < 1e-6
    assert 0.0 <= scalar_deadzone(dz + 1e-9, dz) < 1e-6


def test_scalar_deadzone_for_triggers():
    assert scalar_deadzone(0.0, 0.1) == 0.0
    assert scalar_deadzone(0.1, 0.1) == 0.0
    assert scalar_deadzone(0.55, 0.1) == pytest.approx(0.5)
    assert scalar_deadzone(1.0, 0.1) == 1.0
    assert scalar_deadzone(1.5, 0.1) == 1.0  # out-of-range input clamped
    assert scalar_deadzone(-0.5, 0.1) == 0.0


@pytest.mark.parametrize("e", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_expo_bounds_monotonic_odd(e):
    xs = [i / 200 - 1.0 for i in range(401)]
    ys = [expo(x, e) for x in xs]
    assert expo(0.0, e) == 0.0 and expo(1.0, e) == 1.0 and expo(-1.0, e) == -1.0
    assert all(-1.0 <= y <= 1.0 for y in ys)
    assert all(b >= a for a, b in itertools.pairwise(ys))  # monotonic
    assert all(expo(-x, e) == pytest.approx(-expo(x, e)) for x in xs)  # odd
    # Continuous: no jumps bigger than the max slope (1 + 2e) allows.
    assert max(abs(b - a) for a, b in itertools.pairwise(ys)) <= (1 + 2 * e) / 200 + 1e-12
    assert expo(2.0, e) == 1.0 and expo(-2.0, e) == -1.0


def test_expo_softens_small_inputs():
    assert expo(0.5, 0.0) == 0.5
    assert expo(0.5, 0.5) == pytest.approx(0.5 * 0.5 + 0.5 * 0.125)
    assert expo(0.5, 1.0) == pytest.approx(0.125)


def test_stick_response_is_continuous_and_bounded_through_the_mapper():
    m = GamepadMapper(deadzone=0.12, expo=0.5)
    prev = 0.0
    for i in range(1001):
        v = -i / 1000  # stick slowly pushed forward
        fwd = m.update(snap(lefty=v)).linear[0]
        assert 0.0 <= fwd <= 1.0
        assert fwd >= prev and fwd - prev < 0.01
        prev = fwd
    assert prev == pytest.approx(1.0)


# -- counters ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("button", "action"), list(BUTTON_ACTIONS.items()))
def test_counters_increment_on_rising_edge_only(button, action):
    m = GamepadMapper()
    m.update(snap())
    seq = [False, True, True, True, False, False, True, False, True]
    counts = [m.update(snap(**{button: b})).counters[action] for b in seq]
    assert counts == [0, 1, 1, 1, 1, 1, 2, 2, 3]
    other = [a for a in ACTIONS if a != action]
    assert all(m.counters[a] == 0 for a in other)


def test_mapping_table_covers_every_action_once():
    assert sorted(BUTTON_ACTIONS.values()) == sorted(ACTIONS)


def test_held_button_at_startup_and_reconnect_does_not_fire():
    m = GamepadMapper()
    assert m.update(snap(y=True)).counters["home"] == 0  # held at startup
    assert m.update(snap(y=True)).counters["home"] == 0
    assert m.update(snap()).counters["home"] == 0
    assert m.update(snap(y=True)).counters["home"] == 1  # a real press
    m.update(snap(connected=False))
    assert m.update(snap(y=True)).counters["home"] == 1  # still held after re-plug
    m.update(snap())
    assert m.update(snap(y=True)).counters["home"] == 2


def test_counters_are_copied_not_shared():
    m = GamepadMapper()
    m.update(snap())
    first = m.update(snap(a=True))
    m.update(snap())
    m.update(snap(a=True))
    assert first.counters["gripper"] == 1 and m.counters["gripper"] == 2


# -- invert ------------------------------------------------------------------------------------


def test_invert_negates_named_axes():
    m = linear_mapper(invert=["lin_x", "ang_z"])
    out = m.update(snap(lefty=-1.0, leftx=-1.0, rightx=-1.0, righty=-1.0))
    d = math.sqrt(0.5)  # diagonal at full deflection: radial magnitude clamped to 1
    assert out.linear == pytest.approx((-d, d, 0))
    assert out.angular == pytest.approx((0, d, -d))


def test_invert_all_axes():
    i = Intent(linear=(0.1, 0.2, 0.3), angular=(0.4, 0.5, 0.0))
    out = apply_invert(i, ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"])
    assert out.linear == (-0.1, -0.2, -0.3) and out.angular == (-0.4, -0.5, 0.0)
    assert math.copysign(1, out.angular[2]) > 0


def test_invert_rejects_unknown_axes():
    with pytest.raises(ValueError, match="unknown intent axes"):
        GamepadMapper(invert=["pitch"])


def test_from_config():
    from franka_teleop.config import InputConfig

    m = GamepadMapper.from_config(InputConfig(deadzone=0.3, expo=0.0, invert=["lin_y"]))
    assert m.update(snap(leftx=-0.2)).linear == (0, 0, 0)
    assert m.update(snap(leftx=-1.0)).linear == pytest.approx((0, -1, 0))


def test_intent_neutral_keeps_counters():
    i = Intent(deadman=True, linear=(1, 0, 0), counters={a: 3 for a in ACTIONS})
    n = i.neutral()
    assert not n.deadman and n.linear == (0, 0, 0) and n.counters == i.counters
