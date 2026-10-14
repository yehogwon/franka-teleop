"""Web joystick ("webpad"): a gamepad drawn in a browser page that drives an SDL virtual pad.

::

    browser page --WebSocket (JSON pad state, ~20 Hz)--> WebPadServer (thread)
        --newest state--> WebPadSource.read() (ft-joystick's loop, main thread)
        --> SDL virtual GameController --> GamepadReader --> GamepadMapper --> TeleopCommand

From the SDL device on, everything is the code path of a physical pad: the same reader, mapping
(deadzone, expo, rising-edge buttons, LB re-arm after a reconnect), client loop and command
stream, with ``source="gamepad"``. ft-node cannot tell the two apart.

What a browser needs on top, chosen to behave like a wireless pad:

- One page holds the pad at a time (the "controller"); other pages are spectators until they
  take it over. The virtual pad is plugged in only while a controller page is connected:
  closing it unplugs the pad, and a takeover re-plugs it, so the mapper again demands an LB
  release before the deadman counts.
- A controller page that stops sending for :data:`STALE_S` (frozen tab, network stall) reads
  as a pad with every button and stick released.
- The page only accepts WebSocket connections from pages this server served (same origin), on a
  host that is ``localhost`` or an IP address, so another site open in the browser cannot drive
  the robot (cross-site WebSocket hijacking, DNS rebinding).

ft-node accepts the commands as a gamepad's, so the "webpad" entry of teleop.allowed_sources is
enforced by ft-joystick (it refuses to start otherwise; in sim it is allowed automatically).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import itertools
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from importlib import resources
from urllib.parse import urlsplit

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from ..config import InputConfig
from .gamepad import GamepadReader, GamepadSource
from .mapping import BUTTONS, STICK_AXES, TRIGGER_AXES, Intent
from .sdl_virtual import VirtualGamepad

log = logging.getLogger(__name__)

STALE_S = 0.25
"""A controller page that has sent nothing for this long reads as fully released. Pages send
their state on every change and at least every 50 ms."""
MAX_MESSAGE_BYTES = 4096
PAGE_FILE = "webpad.html"


@dataclass(frozen=True)
class PadState:
    """Normalized pad state as a page reports it. Missing names read as 0 / released."""

    axes: dict[str, float] = field(default_factory=dict)
    buttons: dict[str, bool] = field(default_factory=dict)


NEUTRAL = PadState()


def _quiet_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"{__name__}.{name}")
    logger.setLevel(logging.ERROR)
    return logger


def parse_state(msg: dict) -> PadState:
    """Validate a page's ``{"axes": {...}, "buttons": {...}}``. Sticks are clamped to [-1, 1],
    triggers to [0, 1]; unknown names are ignored. Raises ValueError on wrong types or
    non-finite values (the caller then treats the pad as released)."""
    axes_in, buttons_in = msg.get("axes", {}), msg.get("buttons", {})
    if not isinstance(axes_in, dict) or not isinstance(buttons_in, dict):
        raise ValueError("axes and buttons must be objects")
    axes = {}
    for name in STICK_AXES + TRIGGER_AXES:
        if name not in axes_in:
            continue
        v = axes_in[name]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ValueError(f"axis {name!r} must be a finite number, got {v!r}")
        lo = 0.0 if name in TRIGGER_AXES else -1.0
        axes[name] = min(1.0, max(lo, float(v)))
    buttons = {}
    for name in BUTTONS:
        if name not in buttons_in:
            continue
        v = buttons_in[name]
        if not isinstance(v, bool):
            raise ValueError(f"button {name!r} must be true or false, got {v!r}")
        buttons[name] = v
    return PadState(axes, buttons)


def _host_ok(host: str) -> bool:
    """Only literal addresses and localhost: a DNS name could be rebound to this machine."""
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def same_origin(request: Request) -> bool:
    """The WebSocket comes from a page served by this server (Origin matches Host)."""
    host, origin = request.headers.get("Host"), request.headers.get("Origin")
    if not host or not origin:
        return False
    o = urlsplit(origin)
    return o.scheme in ("http", "https") and o.netloc == host and _host_ok(o.hostname or "")


@dataclass(frozen=True)
class Snapshot:
    """What the pad looks like to the reading thread."""

    generation: int
    """Incremented whenever a page takes the pad (a new page means a re-plugged pad)."""
    held: bool
    """A controller page is connected."""
    state: PadState
    last_rx: float
    """Clock time of the controller's newest state message."""
    pages: int


class WebPadServer:
    """Serves the pad page and its WebSocket from a daemon thread with its own event loop.

    Thread-safe: the reading thread calls :meth:`snapshot`, :meth:`rumble` and :meth:`close`.
    """

    def __init__(
        self, host: str, port: int, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.host = host
        self.port = port
        self.clock = clock
        self._page = resources.files(__package__).joinpath(PAGE_FILE).read_bytes()
        self._lock = threading.Lock()
        self._pages: dict[int, ServerConnection] = {}
        self._ids = itertools.count(1)
        self._controller: int | None = None
        self._generation = 0
        self._state = NEUTRAL
        self._last_rx = -math.inf
        self.bad_messages = 0
        self.gaps: list[float] = []
        """Gaps longer than STALE_S between a controller's messages [s] (the pad was released)."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    # -- lifecycle -----------------------------------------------------------------------------

    def start(self, timeout: float = 5.0) -> None:
        """Bind and serve. Raises OSError if the port is taken (no fallback port: the viewer
        embeds the pad at the configured one)."""
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        if not ready.wait(timeout):
            raise TimeoutError("web joystick server did not start")
        if self._error is not None:
            raise self._error

    def _run(self, ready: threading.Event) -> None:
        try:
            asyncio.run(self._serve(ready))
        except BaseException as e:  # reported to start() if it happens while starting
            self._error = e
            ready.set()

    async def _serve(self, ready: threading.Event) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._http,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
            logger=_quiet_logger("server"),  # no line per page load in the operator's terminal
        ) as server:
            self.port = server.sockets[0].getsockname()[1]  # the bound one, also for port 0
            ready.set()
            await self._stop.wait()

    def close(self) -> None:
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # loop already stopped
                loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(5.0)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0", "::1", "::") else self.host
        return f"http://{host}:{self.port}/"

    # -- HTTP ----------------------------------------------------------------------------------

    def _http(self, connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.partition("?")[0]
        if path == "/ws":
            if not same_origin(request):
                return connection.respond(HTTPStatus.FORBIDDEN, "cross-origin connection refused\n")
            return None  # continue with the WebSocket handshake
        if path in ("/", "/index.html"):
            headers = Headers(
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(self._page))),
                    ("Cache-Control", "no-store"),
                    ("Origin-Agent-Cluster", "?1"),
                ]
            )
            return Response(HTTPStatus.OK, "OK", headers, self._page)
        if path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ok\n")
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    # -- WebSocket -----------------------------------------------------------------------------

    async def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            page = next(self._ids)
            self._pages[page] = ws
            if self._controller is None:
                self._take(page)
        await self._broadcast_roles()
        try:
            async for raw in ws:
                if await self._on_message(page, raw):
                    await self._broadcast_roles()
        except websockets.ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._pages.pop(page, None)
                if self._controller == page:
                    self._controller = None  # unplugged; another page must take it explicitly
                    self._state = NEUTRAL
            await self._broadcast_roles()

    def _take(self, page: int) -> None:
        """Make ``page`` the controller (lock held): a fresh pad, all released."""
        self._controller = page
        self._generation += 1
        self._state = NEUTRAL
        self._last_rx = self.clock()

    async def _on_message(self, page: int, raw: str | bytes) -> bool:
        """Handle one message. True if roles changed."""
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError("message must be an object")
            kind = msg.get("type")
            state = parse_state(msg) if kind == "state" else None
        except ValueError as e:  # includes json.JSONDecodeError
            self.bad_messages += 1
            if self.bad_messages == 1:
                log.warning("web joystick: ignoring a bad message (%s)", e)
            with self._lock:
                if self._controller == page:  # fail safe: a garbled pad is a released pad
                    self._state, self._last_rx = NEUTRAL, self.clock()
            return False
        with self._lock:
            if kind == "state" and self._controller == page:
                now = self.clock()
                if now - self._last_rx > STALE_S:
                    self.gaps.append(now - self._last_rx)
                self._state, self._last_rx = state, now
            elif kind == "take" and self._controller != page:
                self._take(page)
                return True
        return False

    async def _broadcast_roles(self) -> None:
        with self._lock:
            pages, controller = dict(self._pages), self._controller
        held = controller is not None
        for page, ws in pages.items():
            role = "controller" if page == controller else "spectator"
            msg = {"type": "role", "role": role, "pages": len(pages), "held": held}
            with contextlib.suppress(websockets.ConnectionClosed):
                await ws.send(json.dumps(msg))

    # -- reading thread ------------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                generation=self._generation,
                held=self._controller is not None,
                state=self._state,
                last_rx=self._last_rx,
                pages=len(self._pages),
            )

    def rumble(self, low: float, high: float, duration_ms: int) -> bool:
        """Ask the controller page to vibrate / flash. False if no page holds the pad."""
        with self._lock:
            ws = self._pages.get(self._controller) if self._controller is not None else None
        loop = self._loop
        if ws is None or loop is None:
            return False
        msg = json.dumps({"type": "rumble", "low": low, "high": high, "ms": int(duration_ms)})
        with contextlib.suppress(RuntimeError):
            asyncio.run_coroutine_threadsafe(self._send(ws, msg), loop)
            return True
        return False

    @staticmethod
    async def _send(ws: ServerConnection, msg: str) -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg)


class WebPadSource(GamepadSource):
    """Input source: a :class:`WebPadServer` feeding an SDL virtual pad that a regular
    :class:`GamepadSource` reads. Reports itself as ``"gamepad"``, like a physical pad."""

    name = "gamepad"
    device = "webpad"

    def __init__(
        self,
        cfg: InputConfig | None = None,
        *,
        server: WebPadServer | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        cfg = cfg if cfg is not None else InputConfig()
        self.clock = clock
        self.server = server or WebPadServer(cfg.webpad_host, cfg.webpad_port, clock=clock)
        if self.server._thread is None:
            self.server.start()
        try:
            # Attach first and open the reader at the virtual pad's index, so the reader locks
            # onto it and never adopts a physical pad plugged into the same machine. The reader's
            # own connect/disconnect messages describe the virtual device; this source reports
            # page connections instead.
            self.pad = VirtualGamepad()
            reader = GamepadReader(self.pad.index, logger=_quiet_logger("reader"))
            if not reader.connected:
                raise RuntimeError("the SDL virtual pad did not show up")
        except BaseException:
            self.server.close()
            raise
        super().__init__(cfg, reader=reader)
        # Plugged in only while a page holds it. Every unplug is followed by at least one read
        # before the next plug-in: if the reader saw both in one event batch it would take the
        # re-plugged pad for a different one and ignore it, as it should for a physical pad.
        self.pad.detach()
        self.reader.read()
        self._generation = 0
        self._snap = self.server.snapshot()
        self.stale = False
        log.info("web joystick: open %s (one page at a time holds the pad)", self.server.url)

    def read(self) -> Intent:
        snap = self._snap = self.server.snapshot()
        if not snap.held:
            self.pad.detach()
            self.stale = False
        elif snap.generation != self._generation and self.pad.attached:
            # Another page took the pad: unplug it for this tick, plug it in at the next one.
            self.pad.detach()
        else:
            self._generation = snap.generation
            if not self.pad.attached:
                self.pad.attach()
            self.stale = self.clock() - snap.last_rx > STALE_S
            self._apply(NEUTRAL if self.stale else snap.state)
        return super().read()

    def _apply(self, state: PadState) -> None:
        for name in STICK_AXES + TRIGGER_AXES:
            self.pad.set_axis(name, state.axes.get(name, 0.0))
        for name in BUTTONS:
            self.pad.set_button(name, state.buttons.get(name, False))

    def rumble(self, low: float, high: float, duration_ms: int) -> bool:
        return self.server.rumble(low, high, duration_ms)

    def status(self) -> str:
        snap = self._snap
        if not snap.held:
            if snap.pages:
                return f"web pad: free, take it on the page ({self.server.url})"
            return f"web pad: no page connected, open {self.server.url}"
        if self.stale:
            return "web pad: page not responding, pad released"
        return super().status() or f"web pad: {self.server.url}"

    def close(self) -> None:
        try:
            super().close()
            self.pad.detach()
        finally:
            self.server.close()
