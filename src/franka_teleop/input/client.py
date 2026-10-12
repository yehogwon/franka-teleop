"""ft-joystick: read an input device and stream TeleopCommands to ft-node.

One :class:`~franka_teleop.protocol.TeleopCommand` is sent every tick at ``input.rate_hz``,
also when idle: the stream doubles as the heartbeat the node's watchdog checks. NodeStatus
from the node drives a one-line HUD and the round-trip time (the node echoes the newest
command's ``seq`` in ``last_cmd_seq``).

On Ctrl-C / SIGTERM / end of script the client sends neutral, deadman-released commands for
:data:`SHUTDOWN_S` before closing, so the node sees an orderly release instead of a timeout.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

import zmq

from .. import transport
from ..config import Config, ConfigError, add_config_args, config_from_args
from ..protocol import ACTIONS, NodeStatus, TeleopCommand
from .mapping import ZERO3, Intent

log = logging.getLogger("ft-joystick")

DEVICES = ("gamepad", "keyboard", "script")
SHUTDOWN_S = 0.3
"""Neutral, deadman-released commands are sent this long before exiting."""
STATUS_TIMEOUT_S = 1.0
"""No NodeStatus for this long: the HUD says "waiting for ft-node"."""
HUD_PERIOD_S = 0.2
HUD_PERIOD_NOTTY_S = 2.0
RTT_HISTORY = 1024
"""Send times kept for matching NodeStatus.last_cmd_seq (about 10 s at 100 Hz)."""
MAX_LAG_TICKS = 5
"""If the loop falls this many ticks behind (e.g. the process was suspended), the schedule is
reset instead of sending the missed ticks in a burst."""
ERROR_RUMBLE = (1.0, 1.0, 400)
"""(low, high, ms): rumble when the node enters "error" (gamepad only)."""


class InputSource(Protocol):
    """What :class:`JoystickClient` reads. Optional extras: ``rumble(low, high, ms) -> bool``
    and ``status() -> str`` (shown in the HUD)."""

    name: str
    """Reported as TeleopCommand.source: "gamepad" | "keyboard" | "script"."""

    def read(self) -> Intent: ...

    @property
    def done(self) -> bool:
        """No more input will come (script finished, keyboard 'q')."""
        ...

    def close(self) -> None: ...


def new_session() -> int:
    """Random, nonzero, 31-bit session id."""
    return random.SystemRandom().randrange(1, 2**31)


class JoystickClient:
    """The ft-joystick loop, usable in-process: ``run()`` blocks until stopped, ``step()`` runs
    a single tick. Sockets are opened in the constructor (the PUB binds the command port)."""

    def __init__(
        self,
        cfg: Config,
        source: InputSource,
        *,
        exit_when_done: bool = False,
        hud: bool = True,
        hud_stream: TextIO | None = None,
        session: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.source = source
        self.exit_when_done = exit_when_done
        self.hud = hud
        self.hud_stream = hud_stream if hud_stream is not None else sys.stderr
        self.session = session if session is not None else new_session()
        self.clock = clock
        self.period = 1.0 / cfg.input.rate_hz

        self.seq = 0
        """seq of the next command."""
        self.sent = 0
        self.last_intent = Intent()
        self.last_cmd: TeleopCommand | None = None
        self.last_status: NodeStatus | None = None
        self.last_status_t = -math.inf
        self.rtt_s: float | None = None
        self.bad_status = 0
        self._send_times: dict[int, float] = {}
        self._last_echo = -1
        self._hud_t = -math.inf
        self._hud_dirty = False
        self._stop = threading.Event()
        self._closed = False

        self.cmd_endpoint = cfg.net.bind(cfg.net.cmd_port)
        self.status_endpoint = cfg.net.connect(cfg.net.status_port)
        self._pub = transport.publisher(self.cmd_endpoint)
        try:
            self._sub = transport.subscriber(self.status_endpoint)
        except zmq.ZMQError:
            self._pub.close()
            raise

    # -- loop ----------------------------------------------------------------------------------

    def stop(self) -> None:
        """Ask :meth:`run` to finish (thread- and signal-safe)."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run(self, max_seconds: float | None = None) -> None:
        """Tick at input.rate_hz until :meth:`stop`, ``max_seconds``, or the source is done
        (with ``exit_when_done``); then send the neutral tail and close."""
        start = next_t = self.clock()
        try:
            while not self._stop.is_set():
                self.step()
                if self.exit_when_done and self.source.done:
                    break
                now = self.clock()
                if max_seconds is not None and now - start >= max_seconds:
                    break
                next_t = self._next_tick(next_t, now)
                self._stop.wait(max(0.0, next_t - now))
        finally:
            self.shutdown()

    def _next_tick(self, next_t: float, now: float) -> float:
        """Drift-compensated schedule: tick k is due at start + k * period."""
        next_t += self.period
        if now - next_t > MAX_LAG_TICKS * self.period:
            next_t = now
        return next_t

    def step(self) -> TeleopCommand:
        """One tick: read the source, send a command, poll status, refresh the HUD."""
        intent = self.source.read()
        self.last_intent = intent
        cmd = self._send(intent)
        self.poll_status()
        self._update_hud()
        return cmd

    def shutdown(self) -> None:
        """Send neutral, deadman-released commands for SHUTDOWN_S, then close."""
        if self._closed:
            return
        try:
            neutral = self.last_intent.neutral()
            start = next_t = self.clock()
            while True:
                self._send(neutral)
                now = self.clock()
                if now - start >= SHUTDOWN_S:
                    break
                next_t = self._next_tick(next_t, now)
                time.sleep(max(0.0, next_t - now))
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.source.close()
        finally:
            if self.hud and self._hud_dirty:
                self.hud_stream.write("\n")
                self.hud_stream.flush()
            # Give the last (neutral) commands a moment to leave the socket.
            self._pub.setsockopt(zmq.LINGER, 200)
            self._pub.close()
            self._sub.close()

    # -- messages ------------------------------------------------------------------------------

    def _send(self, intent: Intent) -> TeleopCommand:
        deadman = bool(intent.deadman)
        # Motion without the deadman is never sent, even though the node would ignore it.
        lin = intent.linear if deadman else ZERO3
        ang = intent.angular if deadman else ZERO3
        cmd = TeleopCommand(
            session=self.session,
            seq=self.seq,
            t_send=time.time(),
            source=self.source.name,
            deadman=deadman,
            linear=(float(lin[0]), float(lin[1]), float(lin[2])),
            angular=(float(ang[0]), float(ang[1]), float(ang[2])),
            **{a: int(intent.counters.get(a, 0)) for a in ACTIONS},
        )
        self._send_times[self.seq] = self.clock()
        self._send_times.pop(self.seq - RTT_HISTORY, None)
        transport.send(self._pub, cmd)
        self.seq += 1
        self.sent += 1
        self.last_cmd = cmd
        return cmd

    def poll_status(self) -> NodeStatus | None:
        """Non-blocking: take the newest NodeStatus, if any arrived since the last call."""
        status, bad = transport.recv_latest(self._sub, NodeStatus)
        if bad:
            if not self.bad_status:
                log.warning("ignoring undecodable NodeStatus (ft-node from another checkout?)")
            self.bad_status += bad
        if status is None:
            return None
        now = self.clock()
        prev = self.last_status
        self.last_status, self.last_status_t = status, now
        echo = status.last_cmd_seq
        if status.session in (0, self.session) and echo > self._last_echo:
            sent_at = self._send_times.get(echo)
            if sent_at is not None:
                self.rtt_s = now - sent_at
                self._last_echo = echo
        if status.mode == "error" and (prev is None or prev.mode != "error"):
            self._on_error(status)
        return status

    def _on_error(self, status: NodeStatus) -> None:
        msg = status.message or status.robot_errors or "no details"
        log.warning("ft-node entered error mode: %s (Start / 'c' recovers)", msg)
        rumble = getattr(self.source, "rumble", None)
        if rumble is not None:
            rumble(*ERROR_RUMBLE)

    # -- HUD -----------------------------------------------------------------------------------

    def node_alive(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return self.last_status is not None and now - self.last_status_t <= STATUS_TIMEOUT_S

    def hud_line(self, now: float | None = None) -> str:
        now = self.clock() if now is None else now
        parts = [self.source.name]
        if self.last_intent.deadman:
            parts.append("DEADMAN")
        src_status = getattr(self.source, "status", None)
        if src_status is not None and (s := src_status()):
            parts.append(s)
        st = self.last_status
        if not self.node_alive(now):
            parts.append(f"waiting for ft-node ({self.status_endpoint})")
            return " | ".join(parts)
        assert st is not None
        n_levels = len(self.cfg.teleop.speed_levels)
        parts.append(st.mode.upper() if st.mode == "error" else st.mode)
        parts.append(f"speed {st.speed_level + 1}/{n_levels} x{st.speed_scale:.2f}")
        parts.append(st.frame)
        x, y, z = st.ee_pos
        parts.append(f"ee {x:+.3f} {y:+.3f} {z:+.3f}")
        grip = "closed" if st.gripper_closed else "open"
        if not math.isnan(st.gripper_width):
            grip += f" {st.gripper_width * 1000:.0f}mm"
        parts.append(f"grip {grip}")
        parts.append(f"rtt {self.rtt_s * 1000:.1f}ms" if self.rtt_s is not None else "rtt --")
        if st.session not in (0, self.session):
            parts.append("another session in control")
        if st.limiters:
            parts.append("limit " + ",".join(st.limiters))
        if st.message:
            parts.append(st.message)
        return " | ".join(parts)

    def _update_hud(self) -> None:
        if not self.hud:
            return
        now = self.clock()
        tty = self.hud_stream.isatty()
        if now - self._hud_t < (HUD_PERIOD_S if tty else HUD_PERIOD_NOTTY_S):
            return
        self._hud_t = now
        line = self.hud_line(now)
        if tty:
            width = shutil.get_terminal_size().columns - 1
            self.hud_stream.write("\r" + line[:width] + "\x1b[K")
            self._hud_dirty = True
        else:
            self.hud_stream.write(line + "\n")
        self.hud_stream.flush()


# -- CLI ---------------------------------------------------------------------------------------


def make_source(device: str, cfg: Config, script: Path | None = None) -> InputSource:
    """Create an input source. Raises ScriptError / RuntimeError with a user-facing message."""
    if device == "script":
        from .script import Script, ScriptSource

        if script is None:
            raise RuntimeError("--device script needs --script PATH (see configs/scripts/)")
        return ScriptSource(Script.load(script))
    if device == "keyboard":
        from .keyboard import KeyboardSource

        return KeyboardSource(invert=cfg.input.invert)
    if device == "gamepad":
        # Imported lazily: script/keyboard runs never touch SDL.
        from .gamepad import GamepadSource

        src = GamepadSource(cfg.input)
        if not src.reader.connected:
            log.warning(
                "no gamepad at index %d yet; waiting for one (run ft-check-joystick to debug)",
                cfg.input.gamepad_index,
            )
        return src
    raise ValueError(f"unknown device {device!r}")


class _HudSafeFormatter(logging.Formatter):
    """Clears the HUD line before a log record, so the two do not interleave."""

    def __init__(self, fmt: str, clear: bool):
        super().__init__(fmt)
        self.clear = clear

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return "\r\x1b[K" + text if self.clear else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft-joystick",
        description="Stream gamepad / keyboard / scripted operator input to ft-node.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--device", choices=DEVICES, default=None, help="input device (default: input.device)"
    )
    parser.add_argument("--script", type=Path, help="TOML script for --device script")
    parser.add_argument("--duration", type=float, help="exit after this many seconds")
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="exit when the script has finished (default: keep sending neutral heartbeats)",
    )
    parser.add_argument("--quiet", action="store_true", help="no status line")
    args = parser.parse_args(argv)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _HudSafeFormatter("ft-joystick: %(message)s", clear=not args.quiet and sys.stderr.isatty())
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        log.error("%s", e)
        return 2
    device = args.device or ("script" if args.script else cfg.input.device)
    if args.script is not None and device != "script":
        parser.error("--script needs --device script")

    from .script import ScriptError

    try:
        source = make_source(device, cfg, args.script)
    except (ScriptError, RuntimeError) as e:
        log.error("%s", e)
        return 2

    if device not in cfg.allowed_sources():
        log.warning(
            "ft-node accepts motion only from %s with this config; %s commands will be ignored",
            sorted(cfg.allowed_sources()),
            device,
        )
    if device == "keyboard":
        from .keyboard import KEY_HELP

        print(KEY_HELP, file=sys.stderr)
    elif device == "script":
        script = source.script  # type: ignore[attr-defined]
        log.info("script %s: %s (%.1f s)", args.script, script.description, script.duration)

    try:
        client = JoystickClient(
            cfg,
            source,
            # 'q' on the keyboard always quits.
            exit_when_done=args.exit_when_done or device == "keyboard",
            hud=not args.quiet,
        )
    except zmq.ZMQError as e:
        source.close()
        log.error(
            "cannot bind %s: %s (is another ft-joystick running?)",
            cfg.net.bind(cfg.net.cmd_port),
            e,
        )
        return 1

    def on_signal(signum, frame):
        client.stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    log.info(
        "session %d: %s -> %s at %g Hz",
        client.session,
        device,
        client.cmd_endpoint,
        1 / client.period,
    )
    client.run(max_seconds=args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
