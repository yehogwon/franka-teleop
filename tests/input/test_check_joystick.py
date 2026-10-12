"""ft-check-joystick in-process against a virtual pad (subprocess cases are in test_cli.py)."""

import time

from franka_teleop.tools import check_joystick


def test_once_lists_virtual_controller(virtual_gamepad, capsys):
    pad = virtual_gamepad()
    assert check_joystick.main(["--once", "--index", str(pad.index)]) == 0
    out = capsys.readouterr().out
    assert f"[{pad.index}] Virtual Controller" in out
    assert "SDL GameController mapping: yes" in out
    assert "leftshoulder:b9" in out


def test_once_flags_unmapped_pad(virtual_gamepad, capsys):
    pad = virtual_gamepad("joystick")
    assert check_joystick.main(["--once", "--index", str(pad.index)]) == 0
    out = capsys.readouterr().out
    assert "NO (raw fallback layout)" in out and "SDL_GAMECONTROLLERCONFIG" in out


def test_wrong_index(virtual_gamepad, capsys):
    virtual_gamepad()
    assert check_joystick.main(["--once", "--index", "9"]) == 1
    assert "no controller at index 9" in capsys.readouterr().out


def test_live_view_shows_snapshot_and_intent(virtual_gamepad, capsys, monkeypatch):
    pad = virtual_gamepad()
    pad.press("leftshoulder", "a")  # LB held before the tool starts: not armed yet
    pad.set_axis("lefty", -1.0)
    real_sleep, frames = time.sleep, []

    def sleep(s):  # between live-view frames: release LB after frame 3, press it after frame 6
        frames.append(s)
        if len(frames) == 3:
            pad.release("leftshoulder")
        elif len(frames) == 6:
            pad.press("leftshoulder")
        real_sleep(s)

    monkeypatch.setattr(check_joystick.time, "sleep", sleep)
    rc = check_joystick.main(["--index", str(pad.index), "--duration", "0.3", "--rate", "50"])
    assert rc == 0 and len(frames) > 6
    out = capsys.readouterr().out
    assert "left (+0.00,-1.00)" in out
    assert "buttons: a leftshoulder" in out
    held, armed = out.index("(LB held since connect: release it once)"), out.index("deadman ON")
    assert "intent   deadman off" in out[:held] and held < armed
    assert "intent   deadman ON" in out and "fwd +1.00" in out
    assert "hold LB = deadman" in out  # layout legend
    assert "in tool frame they follow the tool's axes" in out.lower()


def test_live_view_shows_who_it_waits_for(virtual_gamepad, capsys, monkeypatch):
    pad = virtual_gamepad()
    real_sleep, frames = time.sleep, []

    def sleep(s):
        frames.append(s)
        if len(frames) == 2:
            pad.detach()
        real_sleep(s)

    monkeypatch.setattr(check_joystick.time, "sleep", sleep)
    assert check_joystick.main(["--index", str(pad.index), "--duration", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "DISCONNECTED (NO GAMEPAD: waiting for your 'Virtual Controller'" in out


def test_no_gamepad_hints(capsys, monkeypatch):
    from franka_teleop.input import gamepad

    monkeypatch.setattr(gamepad, "list_controllers", lambda: [])
    assert check_joystick.main(["--once"]) == 1
    out = capsys.readouterr().out
    assert "No gamepad found" in out
    for hint in ("/dev/input/by-id", "'input' group", "SDL_GAMECONTROLLERCONFIG"):
        assert hint in out
