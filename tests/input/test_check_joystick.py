"""ft-check-joystick in-process against a virtual pad (subprocess cases are in test_cli.py)."""

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


def test_live_view_shows_snapshot_and_intent(virtual_gamepad, capsys):
    pad = virtual_gamepad()
    pad.press("leftshoulder", "a")
    pad.set_axis("lefty", -1.0)
    rc = check_joystick.main(["--index", str(pad.index), "--duration", "0.3", "--rate", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "left (+0.00,-1.00)" in out
    assert "buttons: a leftshoulder" in out
    assert "intent   deadman ON" in out and "fwd +1.00" in out
    assert "hold LB = deadman" in out  # layout legend


def test_no_gamepad_hints(capsys, monkeypatch):
    from franka_teleop.input import gamepad

    monkeypatch.setattr(gamepad, "list_controllers", lambda: [])
    assert check_joystick.main(["--once"]) == 1
    out = capsys.readouterr().out
    assert "No gamepad found" in out
    for hint in ("/dev/input/by-id", "'input' group", "SDL_GAMECONTROLLERCONFIG"):
        assert hint in out
