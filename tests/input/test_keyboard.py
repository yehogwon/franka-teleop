import os
import pty
import select
import termios

import pytest

from franka_teleop.input.keyboard import (
    ACTION_KEYS,
    HOLD_FIRST_S,
    HOLD_REPEAT_S,
    KEY_HELP,
    MOTION_KEYS,
    KeyboardSource,
    KeyboardState,
    TerminalKeys,
    split_keys,
)
from franka_teleop.protocol import ACTIONS


def autorepeat(state, key, t0, t1, delay=0.5, interval=0.033):
    """Feed a key held from t0 to t1 the way a terminal reports it."""
    state.feed(key, t0)
    t = t0 + delay
    while t <= t1:
        state.feed(key, t)
        t += interval


def test_hold_window_constants_are_sane():
    assert 0.5 <= HOLD_FIRST_S <= 0.8  # covers the common 500 ms autorepeat delay
    assert 0.05 <= HOLD_REPEAT_S <= 0.2  # covers 25-33 Hz repeat with margin
    assert HOLD_REPEAT_S < HOLD_FIRST_S


@pytest.mark.parametrize(
    ("key", "linear", "angular"),
    [
        ("w", (1, 0, 0), (0, 0, 0)),
        ("s", (-1, 0, 0), (0, 0, 0)),
        ("a", (0, 1, 0), (0, 0, 0)),
        ("d", (0, -1, 0), (0, 0, 0)),
        ("r", (0, 0, 1), (0, 0, 0)),
        ("f", (0, 0, -1), (0, 0, 0)),
        ("o", (0, 0, 0), (1, 0, 0)),
        ("u", (0, 0, 0), (-1, 0, 0)),
        ("i", (0, 0, 0), (0, 1, 0)),
        ("k", (0, 0, 0), (0, -1, 0)),
        ("j", (0, 0, 0), (0, 0, 1)),
        ("l", (0, 0, 0), (0, 0, -1)),
        ("W", (1, 0, 0), (0, 0, 0)),  # shift / caps lock
    ],
)
def test_motion_keys(key, linear, angular):
    st = KeyboardState()
    st.feed(key, 10.0)
    out = st.intent(10.0)
    assert out.linear == linear and out.angular == angular


def test_single_press_is_held_for_the_first_window():
    st = KeyboardState()
    st.feed("w", 1.0)
    assert st.intent(1.0 + HOLD_FIRST_S - 0.01).linear[0] == 1
    assert st.intent(1.0 + HOLD_FIRST_S + 0.01).linear[0] == 0


def test_autorepeat_keeps_key_held_and_release_stops_quickly():
    st = KeyboardState()
    autorepeat(st, "w", 0.0, 3.0)
    t = 0.0
    while t < 3.0:  # held continuously, including the gap before autorepeat starts
        assert st.intent(t).linear[0] == 1, t
        t += 0.01
    last = max(st._until.values()) - HOLD_REPEAT_S
    assert st.intent(last + HOLD_REPEAT_S - 0.005).linear[0] == 1
    assert st.intent(last + HOLD_REPEAT_S + 0.005).linear[0] == 0  # stops ~0.12 s after release


def test_hold_window_never_shrinks():
    st = KeyboardState()
    st.feed("w", 0.0)
    st.feed("w", 0.1)  # early second event (fast re-tap) must not cut the first window short
    assert st.intent(HOLD_FIRST_S - 0.01).linear[0] == 1


def test_opposite_key_cancels_immediately():
    st = KeyboardState()
    st.feed("w", 0.0)
    st.feed("s", 0.1)
    assert st.intent(0.11).linear[0] == -1
    st.feed("j", 0.2)
    st.feed("l", 0.2)
    assert st.intent(0.2).angular[2] == -1


def test_combined_axes():
    st = KeyboardState()
    for k in "wari":
        st.feed(k, 0.0)
    out = st.intent(0.1)
    assert out.linear == (1, 1, 1) and out.angular == (0, 1, 0)


def test_deadman_toggles_and_is_latched():
    st = KeyboardState()
    assert not st.intent(0.0).deadman
    st.feed(" ", 0.0)
    assert st.intent(0.0).deadman and st.intent(100.0).deadman  # latched, no key-up needed
    st.feed(" ", 1.0)
    assert not st.intent(1.0).deadman


def test_held_space_fails_safe_and_does_not_flap():
    """Holding space: ON at the press, OFF at the first autorepeat, and OFF from then on."""
    st = KeyboardState()
    seen = []
    st.feed(" ", 0.0)
    seen.append(st.deadman)
    t = 0.5
    while t <= 2.0:  # autorepeat at 30 Hz
        st.feed(" ", t)
        seen.append(st.deadman)
        t += 0.033
    assert seen[0] and not any(seen[1:])
    st.feed(" ", 3.0)  # a fresh tap after the key was released switches it on again
    assert st.deadman


def test_deadman_off_is_immediate_within_the_hold_window():
    """Review r3: a quick 'stop' tap right after switching on, or hammering space, must
    switch the deadman off and keep it off."""
    st = KeyboardState()
    st.feed("w", 0.0)
    st.feed(" ", 0.05)  # deadman ON
    for k in range(1, 20):  # 'w' autorepeating
        st.feed("w", 0.05 + k / 30)
    assert st.intent(0.3).deadman and st.intent(0.3).linear == (1, 0, 0)
    st.feed(" ", 0.35)  # 0.3 s after switching on: well inside HOLD_FIRST_S
    out = st.intent(0.36)
    assert not out.deadman and out.linear == (0, 0, 0)
    t = 0.45
    while t < 1.5:  # panicked operator hammers space every 0.1 s
        st.feed(" ", t)
        assert not st.intent(t).deadman, t
        t += 0.1
    assert not st.intent(1.46).deadman
    st.feed(" ", 1.45 + HOLD_FIRST_S + 0.01)  # a deliberate tap after a pause: ON again
    assert st.deadman


def test_deadman_off_forgets_held_motion():
    st = KeyboardState()
    st.feed(" ", 0.0)
    st.feed("w", 1.0)
    st.feed(" ", 1.1)
    out = st.intent(1.2)
    assert not out.deadman and out.linear == (0, 0, 0)


@pytest.mark.parametrize(("key", "action"), list(ACTION_KEYS.items()))
def test_action_keys_count_fresh_presses_not_autorepeat(key, action):
    st = KeyboardState()
    autorepeat(st, key, 0.0, 2.0)  # held for 2 s: ~45 autorepeat events
    assert st.counters[action] == 1
    st.feed(key, 3.0)
    st.feed(key, 3.0 + HOLD_FIRST_S + 0.01)
    assert st.intent(5.0).counters[action] == 3
    assert all(v == 0 for a, v in st.counters.items() if a != action)


def test_key_tables_cover_everything():
    assert sorted(ACTION_KEYS.values()) == sorted(ACTIONS)
    axes = sorted((ax, s) for ax, s in MOTION_KEYS.values())
    assert axes == sorted((ax, s) for ax in range(6) for s in (-1.0, 1.0))


def test_quit_releases_deadman_and_unknown_keys_are_ignored():
    st = KeyboardState()
    st.feed(" ", 0.0)
    for k in "zxyv\n\t\x7f":
        st.feed(k, 0.1)
    assert st.deadman and not st.quit and st.intent(0.1).linear == (0, 0, 0)
    st.feed("q", 0.2)
    assert st.quit and not st.intent(0.2).deadman


def test_keys_after_quit_are_ignored():
    st = KeyboardState()
    st.feed(" ", 0.0)
    st.feed("w", 0.1)
    st.feed("q", 0.2)
    for k in " wg":
        st.feed(k, 0.3)
    out = st.intent(0.3)
    assert not out.deadman and out.linear == (0, 0, 0) and out.counters["gripper"] == 0


def test_keys_after_quit_in_the_same_read_are_ignored():
    """Review r4: 'q' followed by 'w' and space in one terminal read (typed fast or pasted)."""
    term = FakeTerminal()
    now = [0.0]
    src = KeyboardSource(terminal=term, clock=lambda: now[0])
    src.read()
    now[0] = 1.0
    term.pending = "qw "
    out = src.read()
    assert src.done and not out.deadman and out.linear == (0, 0, 0)
    term.pending = " g"  # a deadman latched ON before q is released too
    out = src.read()
    assert not out.deadman and out.counters["gripper"] == 0


def test_key_help_explains_base_and_tool_frame():
    text = " ".join(KEY_HELP.split())
    assert "Directions are for the base frame" in text
    assert "In tool frame they follow the tool's axes" in text
    assert "base -y" in text and "moves it down, along the approach" in text


def test_split_keys_drops_escape_sequences():
    assert split_keys("wasd") == list("wasd")
    assert split_keys("\x1b[A\x1b[B\x1b[C\x1b[D") == []  # arrow keys: no "[" speed-down
    assert split_keys("w\x1b[1;5Cs") == ["w", "s"]  # ctrl+right
    assert split_keys("\x1bOPg") == ["g"]  # F1 (SS3)
    assert split_keys("\x1bg[") == ["["]  # alt+g dropped, plain "[" kept
    assert split_keys("\x1b") == [] and split_keys("\x1b[") == []
    assert split_keys(" ]") == [" ", "]"]


class FakeTerminal:
    def __init__(self):
        self.pending = ""
        self.closed = False

    def read(self):
        out, self.pending = self.pending, ""
        return out

    def close(self):
        self.closed = True


def test_keyboard_source_with_fake_terminal():
    term = FakeTerminal()
    now = [0.0]
    src = KeyboardSource(invert=["lin_y"], terminal=term, clock=lambda: now[0])
    assert src.name == "keyboard"
    term.pending = " wa\x1b[Ag"
    out = src.read()
    assert out.deadman and out.linear == (1, -1, 0) and out.counters["gripper"] == 1
    assert "DEADMAN ON" in src.status()
    now[0] = 1.0
    assert src.read().linear == (0, 0, 0)
    assert not src.done
    term.pending = "q"
    assert not src.read().deadman and src.done
    src.close()
    assert term.closed


def test_terminal_keys_cbreak_and_restore():
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ICANON and before[3] & termios.ECHO
        assert before[0] & termios.IXON  # the pty default: Ctrl-S stops output
        keys = TerminalKeys(slave)
        mode = termios.tcgetattr(slave)
        assert not mode[3] & termios.ICANON and not mode[3] & termios.ECHO
        assert mode[3] & termios.ISIG  # Ctrl-C still raises SIGINT
        assert not mode[0] & (termios.IXON | termios.IXOFF)  # no XON/XOFF flow control
        assert keys.read() == ""
        os.write(master, b"w \x1b[A\x13g\x11")  # Ctrl-S / Ctrl-Q arrive as (ignored) keys
        got = ""
        while len(got) < 8 and select.select([slave], [], [], 2.0)[0]:
            got += keys.read()
        assert split_keys(got) == ["w", " ", "\x13", "g", "\x11"]
        st = KeyboardState()
        for k in split_keys(got):
            st.feed(k, 0.0)
        assert st.deadman and st.counters["gripper"] == 1
        keys.close()
        assert termios.tcgetattr(slave)[:4] == before[:4]  # iflag (IXON) and lflag restored
        keys.close()  # idempotent
    finally:
        os.close(master)
        os.close(slave)


def test_hung_up_terminal_quits_and_releases_the_deadman():
    master, slave = pty.openpty()
    try:
        keys = TerminalKeys(slave)
        src = KeyboardSource(terminal=keys, clock=lambda: 0.0)
        os.write(master, b" w")
        for _ in range(100):
            if src.read().deadman:
                break
            select.select([slave], [], [], 0.02)
        assert src.state.deadman and not src.done
        os.close(master)  # terminal window closed: reads now fail (EIO) or return EOF
        master = None
        out = src.read()
        assert keys.hung_up and src.done
        assert not out.deadman and out.linear == (0, 0, 0)
        keys.close()  # the dead terminal cannot be restored: must not raise
    finally:
        if master is not None:
            os.close(master)
        os.close(slave)


def test_terminal_keys_refuses_non_tty(tmp_path):
    with open(tmp_path / "f", "w") as f, pytest.raises(RuntimeError, match="not a TTY"):
        TerminalKeys(f.fileno())
