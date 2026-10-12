"""ft-sim: franky-sim (MuJoCo behind the libfranka wire protocol) plus the streams viewers use.

Threads:

- main: the 1 kHz physics loop (franky-sim command processing, ``mj_step``, robot state over
  UDP). The SceneState publisher runs inside it as a post-step callback and owns the PUB socket.
- scene-rpc: answers SceneRequest with the scene MJCF (REP socket).
- stats: prints loop timing every few seconds (``--verbose``).

The libfranka client expects a robot state every millisecond; anything that stalls the main
thread shows up on the client as a communication constraint violation. So the loop does no I/O
besides non-blocking sends, and all printing happens in other threads.
"""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import threading
import time
from collections.abc import Callable

import msgspec
import numpy as np
import zmq

from ..config import Config, ConfigError, add_config_args, config_from_args
from ..protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    SceneInfo,
    SceneRequest,
    SceneState,
    decode,
    encode,
)
from ..transport import context, publisher, send

READY_PREFIX = "ft-sim ready:"
STATS_PREFIX = "ft-sim stats:"
SUMMARY_PREFIX = "ft-sim summary:"
STATS_INTERVAL_S = 5.0
MAX_LAG_S = 0.02
"""If the loop falls this far behind schedule, it resynchronizes instead of catching up in a
burst of back-to-back ticks."""

Stats = tuple[float, float, float]
"""(mean, p99, max) [s]"""


class TickSeries:
    """Durations recorded once per tick: a ring of recent values (for periodic reports) and a
    histogram over the whole run (for the exit summary). ``record`` is O(1) and runs in the
    physics loop; everything else runs in other threads and tolerates racing with it."""

    RING = 1 << 14
    BIN_S = 10e-6
    BINS = 5000  # 50 ms; the last bin collects everything longer

    def __init__(self) -> None:
        self.ring = np.zeros(self.RING)
        self.hist = np.zeros(self.BINS, dtype=np.int64)
        self.n = 0
        self.total = 0.0
        self.max = 0.0

    def record(self, dt: float) -> None:
        n = self.n
        self.ring[n & (self.RING - 1)] = dt
        b = int(dt * (1.0 / self.BIN_S))
        self.hist[b if b < self.BINS else self.BINS - 1] += 1
        self.total += dt
        if dt > self.max:
            self.max = dt
        self.n = n + 1

    def recent(self, since: int) -> np.ndarray:
        """Values recorded after the ``since``-th tick (at most RING of them)."""
        n = self.n
        k = min(n - since, self.RING)
        if k <= 0:
            return np.zeros(0)
        idx = np.arange(n - k, n) & (self.RING - 1)
        return self.ring[idx]

    def summary(self) -> Stats:
        """(mean, p99, max) over the whole run [s]; p99 is a 10 µs histogram bin's upper edge."""
        n = int(self.hist.sum())
        if n == 0:
            return 0.0, 0.0, 0.0
        p99_bin = int(np.searchsorted(np.cumsum(self.hist), 0.99 * n))
        return self.total / max(self.n, 1), (p99_bin + 1) * self.BIN_S, self.max


def format_stats(period: Stats, busy: Stats, rtf: float, n: int) -> str:
    """One machine-parseable line of key=value pairs, times in ms. ``period`` is the time
    between tick starts (1 ms when healthy), ``busy`` the time a tick spends working."""
    ms = [1e3 * x for x in (*period, *busy)]
    return (
        f"ticks={n} tick_mean_ms={ms[0]:.3f} tick_p99_ms={ms[1]:.3f} tick_max_ms={ms[2]:.3f} "
        f"busy_mean_ms={ms[3]:.3f} busy_p99_ms={ms[4]:.3f} busy_max_ms={ms[5]:.3f} rtf={rtf:.3f}"
    )


def mean_p99_max(x: np.ndarray) -> Stats:
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    return float(x.mean()), float(np.percentile(x, 99)), float(x.max())


class ScenePublisher:
    """Post-step callback: publishes the full qpos every ``every`` physics steps.

    Runs inside franky-sim's loop, so it must stay cheap and never block: a counter, one qpos
    copy, a non-blocking send. The PUB socket is only ever used from this (the loop's) thread.
    """

    def __init__(self, sock: zmq.Socket, nq: int, every: int):
        self._sock = sock
        self._nq = nq
        self._every = max(1, every)
        self._steps = 0
        self.seq = 0

    def __call__(self, sim) -> None:
        self._steps += 1
        if self._steps < self._every:
            return
        self._steps = 0
        self.seq += 1
        data = sim.data
        send(
            self._sock,
            SceneState(seq=self.seq, t_sim=data.time, qpos=data.qpos.tobytes(), nq=self._nq),
        )


class SceneRpc(threading.Thread):
    """REP server for the scene: a SceneRequest gets the (pre-encoded) SceneInfo of the
    running scene; anything else (several frames, undecodable, another protocol version) gets
    an error reply, a msgpack map ``{"v": PROTOCOL_VERSION, "error": "<why>"}``.

    REP must answer every request exactly once or the socket wedges, so every received request
    is answered, and socket errors are reported and survived: the thread only ends on
    :meth:`stop` (or when the ZMQ context is terminated).
    """

    POLL_MS = 100

    def __init__(self, sock: zmq.Socket, info: SceneInfo):
        super().__init__(name="scene-rpc", daemon=True)
        self._sock = sock  # created by the main thread; owned by this thread from start() on
        self._reply = encode(info)
        self._stop_event = threading.Event()
        self.served = 0
        self.rejected = 0

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self._serve_one()
                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        break
                    print(f"ft-sim: scene RPC: {e}; continuing", file=sys.stderr)
                    self._stop_event.wait(self.POLL_MS / 1000)  # no busy loop on a bad socket
        finally:
            self._sock.close(linger=0)

    def _serve_one(self) -> None:
        if not self._sock.poll(self.POLL_MS, zmq.POLLIN):
            return
        frames = self._sock.recv_multipart()
        self._sock.send(self.reply_to(frames))

    def reply_to(self, frames: list[bytes]) -> bytes:
        """The reply to a request of ``frames``: the SceneInfo, or an error reply."""
        if len(frames) != 1:
            error = f"expected a one-frame SceneRequest, got {len(frames)} frames"
        else:
            try:
                decode(frames[0], SceneRequest)
            except ProtocolError as e:
                error = str(e)
            else:
                self.served += 1
                return self._reply
        self.rejected += 1
        print(f"ft-sim: bad scene request ({error}); sent an error reply", file=sys.stderr)
        return msgspec.msgpack.encode({"v": PROTOCOL_VERSION, "error": error})


class StatsReporter(threading.Thread):
    def __init__(self, period: TickSeries, busy: TickSeries, dt_sim: float, interval: float):
        super().__init__(name="stats", daemon=True)
        self._period, self._busy, self._dt_sim = period, busy, dt_sim
        self._interval = interval
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)

    def run(self) -> None:
        since_p, since_b, t_prev = self._period.n, self._busy.n, time.monotonic()
        while not self._stop_event.wait(self._interval):
            n_p, n_b, now = self._period.n, self._busy.n, time.monotonic()
            rtf = (n_b - since_b) * self._dt_sim / (now - t_prev)
            line = format_stats(
                mean_p99_max(self._period.recent(since_p)),
                mean_p99_max(self._busy.recent(since_b)),
                rtf,
                n_b - since_b,
            )
            print(f"{STATS_PREFIX} {line}", flush=True)
            since_p, since_b, t_prev = n_p, n_b, now


class SimLoop:
    """Runs ``step`` at 1 kHz × realtime_factor against absolute deadlines (so sleep overshoot
    does not accumulate into a slower-than-real-time simulation)."""

    def __init__(self, step: Callable[[], None], dt_sim: float, realtime_factor: float):
        self._step = step
        self.dt_sim = dt_sim
        self.period_s = dt_sim / realtime_factor
        self.running = True  # set here, not in run(): a signal may stop the loop before it runs
        self.period = TickSeries()
        self.busy = TickSeries()
        self.t_start = self.t_end = 0.0

    def run(self) -> None:
        perf, sleep = time.perf_counter, time.sleep
        period, busy, period_s = self.period, self.busy, self.period_s
        deadline = t_prev = self.t_start = perf()
        first = True
        while self.running:
            t0 = perf()
            self._step()
            t1 = perf()
            if first:
                first = False
            else:
                period.record(t0 - t_prev)
            busy.record(t1 - t0)
            t_prev = t0
            deadline += period_s
            delay = deadline - perf()
            if delay > 0.0:
                sleep(delay)
            elif delay < -MAX_LAG_S:
                deadline = perf()
        self.t_end = perf()

    def stop(self) -> None:
        self.running = False

    def summary(self) -> str:
        wall = max(self.t_end - self.t_start, 1e-9)
        rtf = self.busy.n * self.dt_sim / wall
        return format_stats(self.period.summary(), self.busy.summary(), rtf, self.busy.n)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ft-sim",
        description="Simulated Franka arm: franky-sim (libfranka protocol on robot.host) "
        "plus the scene stream and scene RPC used by ft-view.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--scene", choices=["empty", "tabletop"], default=None, help="override sim.scene"
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        metavar="F",
        help="simulated seconds per wall-clock second (default 1.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=f"print loop timing every {STATS_INTERVAL_S:g} s",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.realtime_factor > 0:
        print("ft-sim: --realtime-factor must be positive", file=sys.stderr)
        return 2
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        print(f"ft-sim: bad config: {e}", file=sys.stderr)
        return 2
    if args.scene:
        cfg.sim.scene = args.scene
    if not cfg.sim.enabled:
        print(
            f"ft-sim: refusing to start: {args.config} has sim.enabled = false, so robot.host "
            f"({cfg.robot.host}) is a real robot. Use a simulation config (configs/sim.toml).",
            file=sys.stderr,
        )
        return 2
    return run(cfg, args.realtime_factor, args.verbose)


def run(cfg: Config, realtime_factor: float = 1.0, verbose: bool = False) -> int:
    # Heavy imports here, so that `ft-sim --help` and config errors are instant.
    from franky_sim import SimulationServer

    from .scene import ROBOT_PREFIX, make_simulator, portable_xml

    loop: SimLoop | None = None
    interrupted = False

    def on_signal(_signum, _frame):
        nonlocal interrupted
        if loop is not None:
            loop.stop()
        elif not interrupted:  # still starting up: unwind, once (not again during cleanup)
            interrupted = True
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    sim = server = pub = rpc = reporter = None
    try:
        sim, _robot = make_simulator(cfg.sim.scene, cfg.sim.initial_q)
        xml = portable_xml(sim)  # mutates the ET tree temporarily: compute once, before start

        try:
            pub = publisher(cfg.net.bind(cfg.net.scene_port))
            rep = context().socket(zmq.REP)
            rep.setsockopt(zmq.LINGER, 0)
            try:
                rep.bind(cfg.net.bind(cfg.net.scene_rpc_port))
            except zmq.ZMQError:
                rep.close()
                raise
        except zmq.ZMQError as e:
            print(
                f"ft-sim: cannot bind the scene ports {cfg.net.scene_port}/"
                f"{cfg.net.scene_rpc_port} on {cfg.net.bind_host}: {e}. "
                "Is another ft-sim running with this config?",
                file=sys.stderr,
            )
            return 1

        server = SimulationServer(sim, hostname_candidates=[cfg.robot.host])
        try:
            server.init()  # binds robot.host:1337/1338, then starts the simulation
        except OSError as e:
            rep.close()
            print(
                f"ft-sim: cannot serve the robot on {cfg.robot.host} (ports 1337/1338): {e}. "
                "Another ft-sim (or franky-sim) may already be using that address, or it is "
                "not a local address of this machine.",
                file=sys.stderr,
            )
            return 1

        nq = int(sim.model.nq)
        every = max(1, round(1000.0 / cfg.viewer.fps))
        dt_sim = float(sim.model.opt.timestep)
        publish = ScenePublisher(pub, nq, every)
        sim.register_post_step_callback(publish)
        info = SceneInfo(xml=xml, nq=nq, robot_prefix=ROBOT_PREFIX, robot_host=cfg.robot.host)
        rpc = SceneRpc(rep, info)
        rpc.start()

        loop = SimLoop(lambda: server.run_once(False), dt_sim, realtime_factor)
        # Keep stalls out of the loop: warm numpy's percentile path (its first call takes
        # ~10 ms), move the startup heap out of the GC's reach, and let other threads hold the
        # GIL for at most 0.5 ms at a time.
        mean_p99_max(np.zeros(8))
        gc.collect()
        gc.freeze()
        sys.setswitchinterval(5e-4)
        if verbose:
            reporter = StatsReporter(loop.period, loop.busy, dt_sim, STATS_INTERVAL_S)
            reporter.start()
        rate = 1.0 / (every * dt_sim)
        print(
            f"{READY_PREFIX} robot host {cfg.robot.host} | scene {cfg.sim.scene} (nq {nq}) | "
            f"state {cfg.net.bind(cfg.net.scene_port)} @ {rate:.1f} Hz | "
            f"rpc {cfg.net.bind(cfg.net.scene_rpc_port)} | realtime factor {realtime_factor:g}",
            flush=True,
        )
        loop.run()
    except KeyboardInterrupt:
        pass  # signal during startup
    finally:
        if reporter is not None:
            reporter.stop()
        if rpc is not None:
            rpc.stop()
        if server is not None:
            server.cleanup()
        if sim is not None and sim.is_initialized and not sim.is_cleaned_up:
            sim.cleanup()
        if pub is not None:
            pub.close(linger=0)
    if loop is not None and loop.busy.n:
        print(f"{SUMMARY_PREFIX} {loop.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
