"""ft-teleop: start a whole teleop session with one command, the same way for sim and robot.

    sim.enabled = true :  ft-sim         -> ft-node -> [ft-view] -> ft-joystick (foreground)
    sim.enabled = false:  ft-check-robot -> ft-node -> [ft-view] -> ft-joystick (foreground)
                          (read-only preflight)

Every child gets the same effective config (file + command-line overrides), written to a
temporary TOML file. The joystick runs in the foreground (keyboard mode needs the terminal);
when it exits, or when any other child dies, everything is shut down in a safe order:
joystick (sends a neutral, released command), node (halts the arm), viewer, simulator.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .config import Config, ConfigError, add_config_args, config_from_args, write_config

READY_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 10.0


def script_path(name: str) -> str:
    """Console script installed next to this interpreter (same venv)."""
    path = Path(sys.executable).parent / name
    return str(path) if path.exists() else name


class Child:
    """A background ft-* process whose output is prefixed and forwarded to stderr."""

    def __init__(self, name: str, argv: list[str], ready_prefix: str | None = None):
        self.name = name
        self.ready = threading.Event()
        self.ready_prefix = ready_prefix
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # Ctrl-C at the terminal reaches only the launcher
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            sys.stderr.write(f"[{self.name}] {line}")
            sys.stderr.flush()
            if self.ready_prefix and line.startswith(self.ready_prefix):
                self.ready.set()

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready.wait(0.1):
                return True
            if self.proc.poll() is not None:
                return False
        return False

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> int | None:
        if self.alive():
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                sys.stderr.write(f"ft-teleop: {self.name} did not stop, killing it\n")
                self.proc.kill()
                self.proc.wait()
        self._reader.join(1.0)
        return self.proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ft-teleop", description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument(
        "--input",
        choices=["gamepad", "keyboard", "script"],
        default=None,
        help="input device (default: input.device from the config)",
    )
    parser.add_argument("--script", default=None, help="TOML script for --input script")
    parser.add_argument(
        "--viewer",
        choices=["auto", "native", "web", "none"],
        default="auto",
        help="auto: native window if $DISPLAY is set, else none",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="real robot: do not run ft-check-robot before connecting",
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="stop the joystick after N seconds"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg: Config = config_from_args(args)
    except (ConfigError, OSError) as e:
        print(f"ft-teleop: {e}", file=sys.stderr)
        return 2
    device = args.input or ("script" if args.script else cfg.input.device)
    if device == "script" and not args.script:
        print("ft-teleop: --input script needs --script PATH", file=sys.stderr)
        return 2
    viewer = args.viewer
    if viewer == "auto":
        viewer = "native" if os.environ.get("DISPLAY") else "none"

    with tempfile.TemporaryDirectory(prefix="ft-teleop-") as tmp:
        cfg_path = str(write_config(cfg, Path(tmp) / "effective.toml"))
        common = ["--config", cfg_path]
        children: list[Child] = []
        stopping = False

        def on_signal(signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, on_signal)

        def shutdown(code: int) -> int:
            for child in reversed(children):  # node before sim: the arm is halted first
                rc = child.stop()
                if rc not in (0, None, -signal.SIGTERM) and code == 0:
                    code = 1
            return code

        try:
            if cfg.sim.enabled:
                sim = Child("sim", [script_path("ft-sim"), *common], ready_prefix="ft-sim ready")
                children.append(sim)
                if not sim.wait_ready(READY_TIMEOUT_S):
                    print("ft-teleop: ft-sim did not start", file=sys.stderr)
                    return shutdown(1)
            elif not args.skip_preflight:
                print(
                    f"ft-teleop: preflight against the robot at {cfg.robot.host}", file=sys.stderr
                )
                rc = subprocess.call([script_path("ft-check-robot"), *common])
                if rc != 0:
                    print(
                        "ft-teleop: preflight failed; fix the issues above (or --skip-preflight)",
                        file=sys.stderr,
                    )
                    return 1

            node = Child("node", [script_path("ft-node"), *common], ready_prefix="ft-node ready")
            children.append(node)
            if not node.wait_ready(READY_TIMEOUT_S):
                print("ft-teleop: ft-node did not start", file=sys.stderr)
                return shutdown(1)

            if viewer != "none":
                children.append(Child("view", [script_path("ft-view"), *common, "--mode", viewer]))

            joy_argv = [script_path("ft-joystick"), *common, "--device", device]
            if args.script:
                joy_argv += ["--script", args.script, "--exit-when-done"]
            if args.duration is not None:
                joy_argv += ["--duration", str(args.duration)]
            # Foreground: inherits the terminal (keyboard mode, HUD). Ctrl-C reaches it too.
            joystick = subprocess.Popen(joy_argv)
            code = 0
            while True:
                try:
                    rc = joystick.wait(0.2)
                    if rc != 0:
                        code = rc if rc > 0 else 1
                    break
                except subprocess.TimeoutExpired:
                    pass
                except KeyboardInterrupt:
                    stopping = True  # the joystick got the same SIGINT and shuts down
                dead = [c for c in children if not c.alive()]
                if dead:
                    names = ", ".join(c.name for c in dead)
                    print(
                        f"\nft-teleop: {names} exited unexpectedly; shutting down", file=sys.stderr
                    )
                    code = 1
                    break
                if stopping:
                    joystick.send_signal(signal.SIGTERM)
            if joystick.poll() is None:
                joystick.send_signal(signal.SIGTERM)
                try:
                    joystick.wait(STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    joystick.kill()
            return shutdown(code)
        except KeyboardInterrupt:
            return shutdown(130)
        except BaseException:
            shutdown(1)
            raise


if __name__ == "__main__":
    sys.exit(main())
