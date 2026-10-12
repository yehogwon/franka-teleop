"""ft-check-joystick: list game controllers and show live what ft-joystick would send.

Run it before driving the robot: push every stick and press every button, and check that the
intent moves the way the layout says (e.g. left stick up -> ``fwd +1.00``). Flip axes with
``input.invert`` in the config if needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from ..config import ConfigError, add_config_args, config_from_args
from ..protocol import ACTIONS

NO_GAMEPAD_HINTS = """\
No gamepad found. Things to check:
  - Is the pad connected (USB, or paired over Bluetooth)? It should show up in
      ls -l /dev/input/by-id/        (look for *-joystick / *-event-joystick)
  - May you read it? Your user needs access to /dev/input/event*, usually by being in
    the 'input' group:   groups      (to add yourself: sudo usermod -aG input $USER, then
    log out and in again), or via a udev rule for the pad.
  - Once it is found: pads SDL has no GameController mapping for get a guessed Xbox-style
    layout. Give SDL a mapping for them with
      SDL_GAMECONTROLLERCONFIG="<the line for your pad's GUID from SDL_GameControllerDB>"
    (https://github.com/mdqinc/SDL_GameControllerDB)."""


def _fmt_mapping(mapping: dict[str, str] | None) -> str:
    if not mapping:
        return "-"
    return " ".join(f"{k}:{v}" for k, v in mapping.items() if k != "crc")


def _snapshot_lines(snap, intent) -> list[str]:
    a = snap.axis
    pressed = [b for b, down in snap.buttons.items() if down]
    lin, ang = intent.linear, intent.angular
    counters = " ".join(f"{name}={intent.counters[name]}" for name in ACTIONS)
    if not snap.connected:
        head = "pad      DISCONNECTED (waiting for it to come back)"
    else:
        head = (
            f"pad      left ({a('leftx'):+.2f},{a('lefty'):+.2f})  "
            f"right ({a('rightx'):+.2f},{a('righty'):+.2f})  "
            f"LT {a('lefttrigger'):.2f}  RT {a('righttrigger'):.2f}  "
            f"buttons: {' '.join(pressed) or '-'}"
        )
    return [
        head,
        f"intent   deadman {'ON ' if intent.deadman else 'off'}  "
        f"fwd {lin[0]:+.2f} left {lin[1]:+.2f} up {lin[2]:+.2f}  "
        f"roll {ang[0]:+.2f} pitch {ang[1]:+.2f} yaw {ang[2]:+.2f}",
        f"counters {counters}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft-check-joystick",
        description="List gamepads and live-print their normalized state and resulting intent.",
    )
    add_config_args(parser)
    parser.add_argument("--once", action="store_true", help="list controllers and exit")
    parser.add_argument(
        "--index", type=int, default=None, help="pad to watch (default: input.gamepad_index)"
    )
    parser.add_argument("--duration", type=float, help="stop the live view after S seconds")
    parser.add_argument("--rate", type=float, default=20.0, help="live view refresh rate [Hz]")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        print(f"ft-check-joystick: {e}", file=sys.stderr)
        return 2

    from ..input import gamepad
    from ..input.mapping import LAYOUT_HELP, GamepadMapper

    controllers = gamepad.list_controllers()
    if not controllers:
        print(NO_GAMEPAD_HINTS)
        return 1
    print(f"{len(controllers)} controller(s):")
    for c in controllers:
        kind = "yes" if c.is_controller else "NO (raw fallback layout)"
        print(f"  [{c.index}] {c.name}")
        print(f"      GUID {c.guid}   SDL GameController mapping: {kind}")
        if c.mapping:
            print(f"      mapping {_fmt_mapping(c.mapping)}")

    index = cfg.input.gamepad_index if args.index is None else args.index
    chosen = [c for c in controllers if c.index == index]
    if not chosen:
        print(f"no controller at index {index} (set input.gamepad_index or pass --index)")
        return 1
    if not chosen[0].is_controller:
        print(
            "  SDL has no mapping for this pad: ft-joystick will guess an Xbox-style layout.\n"
            "  Verify every axis below, or set SDL_GAMECONTROLLERCONFIG for it."
        )
    if args.once:
        return 0

    reader = gamepad.GamepadReader(index)
    mapper = GamepadMapper.from_config(cfg.input)
    tty = sys.stdout.isatty()
    print(f"\nwatching [{index}] {reader.name} (Ctrl-C to stop). Layout:\n{LAYOUT_HELP}\n")
    period = 1.0 / args.rate
    start = time.monotonic()
    shown = 0
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            snap = reader.read()
            lines = _snapshot_lines(snap, mapper.update(snap))
            if tty:
                width = shutil.get_terminal_size().columns - 1
                up = f"\x1b[{shown}F" if shown else ""
                sys.stdout.write(up + "".join(line[:width] + "\x1b[K\n" for line in lines))
                shown = len(lines)
            else:
                sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
