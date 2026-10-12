"""ft-node: connects to the arm (real or franky-sim) and runs TeleopNode at teleop.rate_hz.

ft-joystick --TeleopCommand--> [SUB] ft-node [PUB] --NodeStatus--> ft-joystick, ft-view
                                     |
                              franky / libfranka
                                     |
                              arm  or  franky-sim
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time

import franky

from .. import transport
from ..arm.base import ArmDisconnected
from ..arm.franky_arm import FrankyArm
from ..config import Config, add_config_args, config_from_args
from ..protocol import TeleopCommand
from .teleop import TeleopNode

log = logging.getLogger("ft-node")


def connect_arm(cfg: Config, timeout_s: float) -> FrankyArm:
    """Connect, retrying while the robot (or a just-started franky-sim) is unreachable."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return FrankyArm(cfg.robot)
        except franky.IncompatibleVersionException:
            raise
        except franky.Exception as e:
            if time.monotonic() >= deadline:
                raise
            log.info("waiting for robot at %s: %s", cfg.robot.host, e)
            time.sleep(0.5)


class NodeServer:
    def __init__(self, cfg: Config, arm, clock=time.monotonic):
        self.cfg = cfg
        self.node = TeleopNode(arm, cfg, clock=clock)
        self.cmd_sub = transport.subscriber(cfg.net.connect(cfg.net.cmd_port))
        self.status_pub = transport.publisher(cfg.net.bind(cfg.net.status_port))
        self.running = True
        self.ticks = 0
        self.bad_messages = 0
        self.max_lateness = 0.0

    def run(self, max_seconds: float | None = None) -> None:
        period = 1.0 / self.cfg.teleop.rate_hz
        start = next_t = time.monotonic()
        while self.running:
            if max_seconds is not None and time.monotonic() - start >= max_seconds:
                break
            cmd, bad = transport.recv_latest(self.cmd_sub, TeleopCommand)
            self.bad_messages += bad
            status = self.node.tick(cmd)
            transport.send(self.status_pub, status)
            self.ticks += 1
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                lateness = -delay
                self.max_lateness = max(self.max_lateness, lateness)
                if lateness > period:
                    self.node.loop_overruns += 1
                    next_t = time.monotonic()  # do not try to catch up with a burst

    def close(self) -> None:
        self.node.shutdown()
        self.cmd_sub.close()
        self.status_pub.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ft-node", description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument(
        "--connect-timeout", type=float, default=15.0, help="seconds to wait for the robot"
    )
    parser.add_argument("--duration", type=float, default=None, help="exit after N seconds")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = config_from_args(args)
    target = "franky-sim" if cfg.sim.enabled else "robot"
    try:
        arm = connect_arm(cfg, args.connect_timeout)
    except franky.IncompatibleVersionException as e:
        print(f"ft-node: {e}\nRun `ft-check-robot` for the matching franky build.", file=sys.stderr)
        return 1
    except franky.RealtimeException as e:
        print(f"ft-node: realtime setup failed: {e}\nRun `ft-check-rt`.", file=sys.stderr)
        return 1
    except franky.Exception as e:
        print(f"ft-node: cannot connect to {target} at {cfg.robot.host}: {e}", file=sys.stderr)
        return 1

    server = NodeServer(cfg, arm)

    def on_signal(signum, _frame):
        server.running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(
        f"ft-node ready: {target} at {cfg.robot.host}, {cfg.teleop.rate_hz:.0f} Hz, "
        f"commands from {cfg.net.connect(cfg.net.cmd_port)}, "
        f"status on {cfg.net.bind(cfg.net.status_port)}",
        flush=True,
    )
    code = 0
    try:
        server.run(args.duration)
    except ArmDisconnected as e:
        print(f"ft-node: lost connection to the {target}: {e}", file=sys.stderr)
        code = 1
    finally:
        with contextlib.suppress(ArmDisconnected):
            server.close()
    print(
        f"ft-node stopped: {server.ticks} ticks, {server.node.loop_overruns} overruns, "
        f"max lateness {server.max_lateness * 1e3:.1f} ms, {server.bad_messages} bad messages",
        flush=True,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
