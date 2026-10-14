"""ft-view: watch the robot (ft-sim's scene, or ft-node's status for the real arm).

Modes: ``native`` (MuJoCo window), ``web`` (browser, for headless machines), ``snapshot``
(PNG) and ``record`` (MP4).

This module must not import mujoco (or anything that does) at import time: offscreen modes
choose the GL backend through ``MUJOCO_GL``, which mujoco reads once, on first import.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from ..config import ConfigError, add_config_args, config_from_args

MODES = ("native", "web", "snapshot", "record")
SOURCES = ("auto", "scene", "status")

NO_DISPLAY_HELP = """\
ft-view: --mode native needs a display, but neither DISPLAY nor WAYLAND_DISPLAY is set.
  On a headless machine, run the web viewer and tunnel its port from your laptop:
      ft-view --mode web                              # here
      ssh -L {port}:localhost:{port} <this-host>      # on the laptop; open http://localhost:{port}
  or run ft-view --mode native on the laptop itself (it only needs the ZMQ ports), or record:
      ft-view --mode record --out run.mp4 --duration 20"""


def have_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1280x720") from None
    if w < 16 or h < 16 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError("width and height must be even and at least 16")
    return w, h


def _on_sigterm(_signum, _frame) -> None:
    """SIGTERM unwinds like Ctrl-C, so renderers, video files and servers are closed properly.
    Later SIGTERMs are ignored so they cannot interrupt that cleanup."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ft-view",
        description="View the robot: ft-sim's full scene, or (real arm) ft-node's joint state.",
    )
    add_config_args(parser)
    parser.add_argument("--mode", choices=MODES, default="native", help="default: native")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="auto",
        help="scene: ft-sim; status: ft-node (robot only, works for the real arm); "
        "auto (default): with sim.enabled, ft-sim if it answers within ~1 s (and simulates this "
        "config's robot.host), else status; with a real-robot config always status",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="S",
        help="exit after S seconds (record: video length)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output file (snapshot, record)")
    parser.add_argument(
        "--size",
        type=_size,
        default=(1280, 720),
        metavar="WxH",
        help="image size for snapshot/record (default 1280x720)",
    )
    parser.add_argument(
        "--camera",
        default=None,
        metavar="AZ,EL,DIST,X,Y,Z",
        help="free camera: azimuth and elevation [deg], distance [m] and look-at point [m]",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="address the web viewer binds (default 127.0.0.1; 0.0.0.0 exposes it on the LAN)",
    )
    parser.add_argument("--web-port", type=int, default=None, help="override viewer.web_port")
    parser.add_argument(
        "--wait",
        type=float,
        default=10.0,
        metavar="S",
        help="how long to wait for ft-sim (--source scene) or a first state (snapshot, record)",
    )
    parser.add_argument("--no-workspace", action="store_true", help="hide the workspace box")
    parser.add_argument(
        "--device",
        choices=["gamepad", "keyboard", "script", "webpad"],
        default=None,
        help="web: the operator's input device, whose controls the panel shows expanded; "
        "webpad embeds the web joystick (default: input.device)",
    )
    args = parser.parse_args(argv)
    if args.mode in ("snapshot", "record") and args.out is None:
        parser.error(f"--mode {args.mode} needs --out")
    if args.mode == "record" and not (args.duration and args.duration > 0):
        parser.error("--mode record needs --duration S (S > 0)")
    if args.duration is not None and args.duration < 0:
        parser.error("--duration must not be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        print(f"ft-view: bad config: {e}", file=sys.stderr)
        return 2
    web_port = args.web_port or cfg.viewer.web_port

    if args.mode == "native" and not have_display():
        print(NO_DISPLAY_HELP.format(port=web_port), file=sys.stderr)
        return 2
    if args.mode in ("snapshot", "record") and not have_display():
        # Headless offscreen rendering. Must happen before mujoco is imported anywhere.
        os.environ.setdefault("MUJOCO_GL", "egl")

    signal.signal(signal.SIGTERM, _on_sigterm)

    from ..protocol import ProtocolError
    from .overlay import CameraSpec
    from .source import SourceError, open_source

    try:
        camera = CameraSpec.parse(args.camera) if args.camera else CameraSpec()
    except ValueError as e:
        print(f"ft-view: --camera: {e}", file=sys.stderr)
        return 2
    workspace = None if args.no_workspace else (cfg.workspace.lower, cfg.workspace.upper)

    try:
        src = open_source(cfg, args.source, args.wait)
    except (SourceError, ProtocolError) as e:
        print(f"ft-view: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    print(f"ft-view: {args.mode} view of {src.describe()}", flush=True)
    try:
        if args.mode == "native":
            from .native import run_native

            return run_native(src, cfg.viewer.fps, camera, workspace, args.duration)
        if args.mode == "web":
            from .web import run_web

            return run_web(
                src,
                args.web_host,
                web_port,
                cfg.viewer.fps,
                camera,
                workspace,
                args.duration,
                device=args.device or cfg.input.device,
                webpad_port=cfg.input.webpad_port,
            )
        from .record import record, snapshot

        if args.mode == "snapshot":
            return snapshot(src, args.out, args.size, camera, workspace, args.wait)
        return record(
            src, args.out, args.size, camera, args.duration, cfg.viewer.fps, workspace, args.wait
        )
    except KeyboardInterrupt:
        return 0
    finally:
        src.close()


if __name__ == "__main__":
    sys.exit(main())
