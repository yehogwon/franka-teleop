"""ft-check-rt: can this laptop run libfranka's 1 kHz control loop?

Checks the kernel (PREEMPT_RT), the realtime rlimits libfranka needs, an actual SCHED_FIFO switch,
the CPU frequency governor, the wake-up jitter of a 1 kHz loop and, with a robot host, the route
and round-trip time to the arm.

The jitter test is a Python loop sleeping to absolute 1 ms deadlines. It only approximates
libfranka's C++ loop (which is paced by the robot's UDP state packets and has far less overhead
per tick), so treat it as a lower bound on how well the machine holds 1 kHz: a box that fails
here will not do better under libfranka.

Exit code 0 when no check FAILs, 1 otherwise. ``--json`` prints the results as one JSON object.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import ConfigError, add_config_args, config_from_args
from .report import Report, Result, fail, passed, warn

FRANKA_RT_DOCS = "https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html"
FRANKA_TROUBLESHOOTING = "https://frankarobotics.github.io/docs/troubleshooting.html"

RTPRIO_NEEDED = 99
"""libfranka switches its control thread to sched_get_priority_max(SCHED_FIFO), i.e. 99."""
MEMLOCK_MIN_KIB = 102400
"""Franka's recommended limits.conf memlock value (100 MiB)."""
CAP_SYS_NICE = 23

# Wake-up lateness of the 1 kHz loop [us]. A tick later than one period (1000 us) is a missed
# control cycle; libfranka stops the robot after a few of those (communication_constraints).
JITTER_PASS_P99_US = 100.0
JITTER_PASS_MAX_US = 500.0
JITTER_FAIL_P99_US = 250.0
JITTER_FAIL_MAX_US = 1000.0

PING_COUNT = 10
PING_WARN_AVG_MS = 0.5

LIMITS_HINT = (
    """\
let your user raise realtime priority (then log out and back in; `ulimit -r` must print 99):
  sudo groupadd -f realtime && sudo usermod -aG realtime $USER
  and add to /etc/security/limits.conf:
    @realtime soft rtprio 99
    @realtime soft priority 99
    @realtime soft memlock 102400
    @realtime hard rtprio 99
    @realtime hard priority 99
    @realtime hard memlock 102400
see """
    + FRANKA_RT_DOCS
    + "#allowing-real-time-permissions"
)

NETWORK_HINT = """\
check the Ethernet cable and that the robot is powered on, and give the laptop's wired interface
a static IP in the robot's subnet (e.g. 172.16.0.1/24 for a robot at 172.16.0.2)"""


# -- kernel / limits / scheduler ---------------------------------------------------------------


def check_kernel(realtime_flag: str | None, uname_version: str, release: str) -> Result:
    """``realtime_flag``: contents of /sys/kernel/realtime (None if absent)."""
    rt = (realtime_flag or "").strip() == "1" or "PREEMPT_RT" in uname_version
    data = {"release": release, "version": uname_version, "sys_kernel_realtime": realtime_flag}
    if rt:
        return passed("kernel", f"PREEMPT_RT kernel {release}", **data)
    model = next(
        (t for t in uname_version.split() if t.startswith("PREEMPT")), "no PREEMPT_RT in uname -v"
    )
    return fail(
        "kernel",
        f"{release} is not a realtime kernel ({model}; /sys/kernel/realtime "
        f"{'absent' if realtime_flag is None else realtime_flag.strip()})",
        "install a PREEMPT_RT kernel and boot it; libfranka (robot.realtime = enforce) refuses to\n"
        "control the robot without one:\n"
        "  Ubuntu 22.04/24.04 (Ubuntu Pro):\n"
        "    sudo pro attach <token> && sudo pro enable realtime-kernel\n"
        "  otherwise build a PREEMPT_RT kernel: " + FRANKA_RT_DOCS + "\n"
        "then `uname -v` shows PREEMPT_RT (or /sys/kernel/realtime is 1)",
        **data,
    )


def _fmt_limit(v: int) -> str:
    return "unlimited" if v == resource.RLIM_INFINITY else str(v)


def check_rtprio(soft: int, hard: int, euid: int, cap_sys_nice: bool) -> Result:
    data = {"soft": _fmt_limit(soft), "hard": _fmt_limit(hard), "euid": euid}
    if euid == 0:
        return passed("rtprio", "running as root: any realtime priority allowed", **data)
    if cap_sys_nice:
        return passed("rtprio", "CAP_SYS_NICE: any realtime priority allowed", **data)
    if soft == resource.RLIM_INFINITY or soft >= RTPRIO_NEEDED:
        return passed("rtprio", f"RLIMIT_RTPRIO {_fmt_limit(soft)} >= {RTPRIO_NEEDED}", **data)
    hint = LIMITS_HINT
    if hard == resource.RLIM_INFINITY or hard >= RTPRIO_NEEDED:
        hint = f"the hard limit allows it: `ulimit -r {RTPRIO_NEEDED}` in this shell, or\n" + hint
    return fail(
        "rtprio",
        f"RLIMIT_RTPRIO is {_fmt_limit(soft)} (hard {_fmt_limit(hard)}); libfranka asks for "
        f"SCHED_FIFO priority {RTPRIO_NEEDED}",
        hint,
        **data,
    )


def check_memlock(soft: int, hard: int) -> Result:
    data = {"soft": _fmt_limit(soft), "hard": _fmt_limit(hard)}
    if soft == resource.RLIM_INFINITY or soft >= MEMLOCK_MIN_KIB * 1024:
        size = "unlimited" if soft == resource.RLIM_INFINITY else f"{soft // 1024**2} MiB"
        return passed("memlock", f"RLIMIT_MEMLOCK {size}", **data)
    return warn(
        "memlock",
        f"RLIMIT_MEMLOCK is {soft // 1024} KiB (< {MEMLOCK_MIN_KIB} KiB recommended by Franka)",
        LIMITS_HINT,
        **data,
    )


def has_cap_sys_nice(proc_status: str) -> bool:
    """CAP_SYS_NICE in the effective capability set (from /proc/self/status)."""
    m = re.search(r"^CapEff:\s*([0-9a-fA-F]+)", proc_status, re.M)
    return bool(m) and bool(int(m.group(1), 16) & (1 << CAP_SYS_NICE))


_SCHED_FIFO_CHILD = """
import os, sys
prio = os.sched_get_priority_max(os.SCHED_FIFO)
try:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
except OSError as e:
    print(prio, e)
    sys.exit(1)
print(prio, "ok")
"""


def try_sched_fifo() -> tuple[bool, int, str]:
    """Switch a short-lived child process to SCHED_FIFO at the maximum priority.
    Returns (ok, priority, message)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _SCHED_FIFO_CHILD],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, RTPRIO_NEEDED, str(e)
    prio_s, _, msg = proc.stdout.strip().partition(" ")
    prio = int(prio_s) if prio_s.isdigit() else RTPRIO_NEEDED
    return proc.returncode == 0, prio, msg or proc.stderr.strip()


def check_sched_fifo(ok: bool, priority: int, message: str) -> Result:
    if ok:
        return passed("sched_fifo", f"a child process switched to SCHED_FIFO priority {priority}")
    return fail(
        "sched_fifo",
        f"sched_setscheduler(SCHED_FIFO, {priority}) failed: {message}",
        "libfranka does exactly this for its control thread; fix the rtprio check above (and log\n"
        "in again). If rtprio passes, a container or cgroup without a realtime budget\n"
        "(CONFIG_RT_GROUP_SCHED, cpu.rt_runtime_us) can also refuse it",
        error=message,
    )


# -- CPU frequency -----------------------------------------------------------------------------


def read_governors() -> list[str]:
    govs = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")):
        text = _read(path)
        if text:
            govs.append(text.strip())
    return govs


def check_governors(governors: Sequence[str]) -> Result:
    counts = dict(Counter(governors))
    if not governors:
        return warn(
            "cpu_governor",
            "no cpufreq interface found (VM, or frequency scaling controlled by the BIOS?)",
            "if this is the laptop: set the BIOS/UEFI power profile to maximum performance",
            governors=counts,
        )
    if set(governors) == {"performance"}:
        return passed(
            "cpu_governor", f"all {len(governors)} CPUs use 'performance'", governors=counts
        )
    summary = ", ".join(f"{n} CPUs '{g}'" for g, n in sorted(counts.items()))
    return warn(
        "cpu_governor",
        f"{summary}: frequency scaling adds wake-up latency to the 1 kHz loop",
        "sudo cpupower frequency-set -g performance\n"
        "(or: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)\n"
        "this lasts until reboot; set it again before each session or persist it, e.g.\n"
        'a systemd unit or cpufrequtils with GOVERNOR="performance" in /etc/default/cpufrequtils',
        governors=counts,
    )


# -- 1 kHz jitter ------------------------------------------------------------------------------


def measure_lateness(seconds: float, period_s: float = 0.001) -> tuple[list[float], int]:
    """Run a loop that sleeps to absolute deadlines every ``period_s``.

    Returns (wake-up lateness of each tick [us], ticks skipped because a wake-up came more than
    a full period late). Phase is kept, like a loop paced by the robot's 1 kHz packets."""
    period = round(period_s * 1e9)
    clock = time.monotonic_ns
    lateness: list[float] = []
    missed = 0
    deadline = clock() + period
    end = deadline + round(seconds * 1e9)
    while deadline < end:
        now = clock()
        if deadline > now:
            time.sleep((deadline - now) / 1e9)
        woke = clock()
        lateness.append((woke - deadline) / 1e3)
        deadline += period
        while deadline <= woke:
            deadline += period
            missed += 1
    return lateness, missed


def lateness_stats(lateness: Sequence[float], missed: int = 0) -> dict[str, float]:
    if not lateness:
        return {"n": 0, "missed": missed, "p50_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    s = sorted(lateness)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, round(p / 100.0 * (len(s) - 1)))]

    return {
        "n": len(s),
        "missed": missed,
        "p50_us": pct(50),
        "p99_us": pct(99),
        "max_us": s[-1],
    }


def jitter_child(seconds: float, fifo: bool) -> None:
    """Entry point of the jitter child process: prints one JSON line with the stats."""
    policy = "SCHED_OTHER"
    if fifo:
        try:
            prio = os.sched_get_priority_max(os.SCHED_FIFO)
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
            policy = f"SCHED_FIFO {prio}"
        except OSError:
            pass
    lateness, missed = measure_lateness(seconds)
    print(json.dumps({"policy": policy, **lateness_stats(lateness, missed)}))


def run_jitter(seconds: float, fifo: bool) -> dict:
    code = (
        "from franka_teleop.tools.check_rt import jitter_child; "
        f"jitter_child({float(seconds)!r}, {bool(fifo)})"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=seconds + 30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr else "no output")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def check_jitter(stats: dict) -> Result:
    p50, p99, mx = stats["p50_us"], stats["p99_us"], stats["max_us"]
    missed, policy = stats.get("missed", 0), stats.get("policy", "?")
    detail = (
        f"{policy}, {stats['n']} ticks: wake-up lateness p50 {p50:.0f} us, p99 {p99:.0f} us, "
        f"max {mx:.0f} us, {missed} missed cycles"
    )
    hint = (
        "a Python loop only approximates libfranka's C++ loop, but a box that misses deadlines\n"
        "here will miss them under libfranka too (the robot then stops with\n"
        "communication_constraints_violation): boot the PREEMPT_RT kernel, set the 'performance'\n"
        "governor, close heavy programs (browser, GPU load), disable power saving in the BIOS\n"
        "see " + FRANKA_TROUBLESHOOTING + "#troubleshooting-realtime-control-loops"
    )
    if missed > 0 or mx > JITTER_FAIL_MAX_US or p99 > JITTER_FAIL_P99_US:
        return fail("jitter", detail, hint, **stats)
    if mx > JITTER_PASS_MAX_US or p99 > JITTER_PASS_P99_US:
        return warn("jitter", detail, hint, **stats)
    return passed("jitter", detail, **stats)


# -- network to the robot ----------------------------------------------------------------------


@dataclass
class Route:
    dev: str = ""
    via: str = ""
    src: str = ""
    error: str = ""


def parse_ip_route(output: str) -> Route:
    """Parse `ip route get <host>`, e.g. ``172.16.0.2 dev enp3s0 src 172.16.0.1 uid 1000``."""
    tokens = output.split()
    route = Route()
    for key, value in itertools.pairwise(tokens):
        if key in ("dev", "via", "src") and not getattr(route, key):
            setattr(route, key, value)
    if not route.dev:
        route.error = output.strip() or "no route"
    return route


def ip_route_get(host: str) -> Route:
    if shutil.which("ip") is None:
        return Route(error="`ip` not found (install iproute2)")
    try:
        proc = subprocess.run(
            ["ip", "route", "get", host], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Route(error=str(e))
    if proc.returncode != 0:
        return Route(error=(proc.stderr or proc.stdout).strip())
    return parse_ip_route(proc.stdout)


def is_wireless(dev: str, sys_class_net: str = "/sys/class/net") -> bool:
    base = Path(sys_class_net) / dev
    return (base / "wireless").exists() or (base / "phy80211").exists()


def link_speed_mbps(dev: str, sys_class_net: str = "/sys/class/net") -> int | None:
    text = _read(Path(sys_class_net) / dev / "speed")
    try:
        speed = int(text) if text else None
    except ValueError:
        return None
    return speed if speed and speed > 0 else None


def check_route(host: str, route: Route, wireless: bool, speed_mbps: int | None) -> Result:
    data = {"dev": route.dev, "via": route.via, "src": route.src, "speed_mbps": speed_mbps}
    if route.error:
        return fail("route", f"no route to {host}: {route.error}", NETWORK_HINT, **data)
    if route.dev == "lo":
        return passed("route", f"{host} is local (loopback: simulator)", **data)
    speed = f", {speed_mbps} Mb/s" if speed_mbps else ""
    if wireless:
        return warn(
            "route",
            f"{host} is reached over {route.dev}, a wireless interface",
            "connect the robot by Ethernet cable; Wi-Fi latency and loss break the 1 kHz loop",
            **data,
        )
    if route.via:
        return warn(
            "route",
            f"{host} is routed through gateway {route.via} on {route.dev}",
            "connect the laptop directly to the robot, without a router in between, and give its\n"
            "wired interface a static IP in the robot's subnet (e.g. 172.16.0.1/24 for 172.16.0.2)",
            **data,
        )
    return passed("route", f"{host} via {route.dev} (src {route.src}{speed}), direct link", **data)


@dataclass
class PingStats:
    transmitted: int = 0
    received: int = 0
    min_ms: float | None = None
    avg_ms: float | None = None
    max_ms: float | None = None

    @property
    def loss(self) -> float:
        return 1.0 - self.received / self.transmitted if self.transmitted else 1.0


def parse_ping(output: str) -> PingStats:
    """Parse the summary of iputils/busybox ping."""
    stats = PingStats()
    m = re.search(r"(\d+) packets transmitted, (\d+) (?:packets )?received", output)
    if m:
        stats.transmitted, stats.received = int(m.group(1)), int(m.group(2))
    m = re.search(r"min/avg/max(?:/m?dev)? = ([\d.]+)/([\d.]+)/([\d.]+)", output)
    if m:
        stats.min_ms, stats.avg_ms, stats.max_ms = map(float, m.groups())
    return stats


def run_ping(host: str, count: int = PING_COUNT) -> tuple[str, int]:
    """Unprivileged ping: 0.2 s is the smallest interval a normal user may use."""
    if shutil.which("ping") is None:
        return "", -1
    try:
        proc = subprocess.run(
            ["ping", "-n", "-c", str(count), "-i", "0.2", "-W", "1", host],
            capture_output=True,
            text=True,
            timeout=count * 0.2 + 15,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e), -1
    return proc.stdout + proc.stderr, proc.returncode


def check_ping(host: str, output: str, returncode: int) -> Result:
    if returncode == -1 and not output:
        return warn("ping", "`ping` not found, RTT not measured", "install iputils-ping")
    s = parse_ping(output)
    data = {"transmitted": s.transmitted, "received": s.received, "avg_ms": s.avg_ms}
    if s.received == 0 or s.avg_ms is None:
        if s.transmitted:
            reason = f"no reply to {s.transmitted} pings"
        else:
            reason = output.strip().splitlines()[-1] if output.strip() else f"exit {returncode}"
        return fail(
            "ping",
            f"{host} does not answer: {reason}",
            NETWORK_HINT,
            **data,
        )
    detail = (
        f"{host}: avg {s.avg_ms:.3f} ms (min {s.min_ms:.3f}, max {s.max_ms:.3f}), "
        f"{s.loss:.0%} loss over {s.transmitted} pings"
    )
    hint = (
        "use a direct cable (no switch, no Wi-Fi, no USB/Thunderbolt dock if avoidable) and\n"
        "disable NIC power saving/interrupt coalescing (e.g. sudo ethtool -C <dev> rx-usecs 0)\n"
        "thorough test as root: sudo ping " + host + " -i 0.001 -D -c 10000 -s 1200\n"
        "see " + FRANKA_TROUBLESHOOTING + "#network-ping-test"
    )
    if s.avg_ms > PING_WARN_AVG_MS:
        return warn("ping", detail + f" (> {PING_WARN_AVG_MS} ms)", hint, **data)
    if s.received < s.transmitted:
        return warn("ping", detail, hint, **data)
    return passed("ping", detail, **data)


# -- main --------------------------------------------------------------------------------------


def _read(path: str | os.PathLike) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def run_checks(report: Report, robot_host: str | None, jitter_seconds: float) -> None:
    uname = os.uname()
    report.add(check_kernel(_read("/sys/kernel/realtime"), uname.version, uname.release))
    report.add(
        check_rtprio(
            *resource.getrlimit(resource.RLIMIT_RTPRIO),
            euid=os.geteuid(),
            cap_sys_nice=has_cap_sys_nice(_read("/proc/self/status") or ""),
        )
    )
    report.add(check_memlock(*resource.getrlimit(resource.RLIMIT_MEMLOCK)))
    fifo = report.add(check_sched_fifo(*try_sched_fifo()))
    report.add(check_governors(read_governors()))

    if jitter_seconds > 0:
        try:
            stats = run_jitter(jitter_seconds, fifo=fifo.status == "PASS")
            report.add(check_jitter(stats))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            report.add(fail("jitter", f"the 1 kHz test loop did not run: {e}"))
    else:
        report.note("jitter: skipped (--jitter-seconds 0)")

    if robot_host is None:
        report.note("robot network: skipped (sim.enabled config; pass --robot-host to check it)")
        return
    route = ip_route_get(robot_host)
    report.add(
        check_route(
            robot_host,
            route,
            wireless=bool(route.dev) and is_wireless(route.dev),
            speed_mbps=link_speed_mbps(route.dev) if route.dev else None,
        )
    )
    report.add(check_ping(robot_host, *run_ping(robot_host)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft-check-rt",
        description="Check that this machine can run libfranka's 1 kHz realtime control loop.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--robot-host",
        default=None,
        help="robot to check the network path to (default: robot.host when sim.enabled is false)",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=float,
        default=3.0,
        help="duration of the 1 kHz loop jitter test, 0 to skip (default: 3)",
    )
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        parser.error(str(e))
    robot_host = args.robot_host or (None if cfg.sim.enabled else cfg.robot.host)

    report = Report("ft-check-rt", as_json=args.json)
    try:
        run_checks(report, robot_host, args.jitter_seconds)
    except Exception as e:  # still print the report (and valid JSON) on an unexpected error
        report.add(fail("internal", f"{type(e).__name__}: {e}"))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
