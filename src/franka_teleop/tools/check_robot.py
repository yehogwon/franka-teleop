"""ft-check-robot: read-only pre-flight check of the arm. Never moves the robot or the gripper.

Connects with ``franky.Robot`` directly (FrankyArm configures the robot) and reports the
connection, robot mode, active errors, joint positions against ``robot.joint_lower/upper``, the
end-effector position against the ``[workspace]`` box, the state rate and the gripper.

``franky.Robot``'s constructor itself sends SetCollisionBehavior with scalar default thresholds.
This tool passes the config's lowest thresholds, so the check never leaves the robot less
sensitive than the config asks; ft-node sets the per-joint values when it starts.

FAIL is something to fix before a session (network, versions, realtime setup, joints outside
the limits); WARN is something the operator should look at (robot mode, errors, pose). Exit code
0 when no check FAILs, 1 otherwise. A robot server version mismatch prints how to switch the
franky wheel with uv.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import franky
import numpy as np

from ..arm.franky_arm import active_errors
from ..config import Config, ConfigError, add_config_args, config_from_args, repo_configs_dir
from .report import Report, Result, fail, passed, warn

FRANKY_INDEX_ROOT = "https://timschneider42.github.io/franky/whl/by-robot-server-version/"
FRANKY_FAQ = (
    "https://timschneider42.github.io/franky/getting-started/faq.html"
    "#i-am-seeing-server-version-mismatch-errors-what-now"
)
FRANKA_COMPAT = "https://frankarobotics.github.io/docs/doc/libfranka/docs/compatibility_matrix.html"

# franky publishes one wheel index per robot server version; each carries
# franky-control 2.0.0+libfranka.<v> for CPython 3.12 (checked 2026-10). Value: bundled libfranka.
FRANKY_WHEEL_INDEXES = {
    3: "0.7.1",
    4: "0.8.0",
    5: "0.9.2",
    6: "0.12.1",
    7: "0.13.3",
    8: "0.14.2",
    9: "0.17.0",
    10: "0.21.2",
}
# Minimum robot system version per robot server version (libfranka compatibility matrix).
# System 5.x is the Franka Research 3; 4.x and older the Franka Emika Panda.
ROBOT_SYSTEM_VERSION = {
    3: "3.0.0",
    4: "4.0.0",
    5: "4.2.1",
    6: "5.2.0",
    7: "5.5.0",
    8: "5.7.0",
    9: "5.7.2",
    10: "5.9.0",
}
SIM_SERVER_VERSION = 10
"""franky-sim emulates an FR3 speaking robot server version 10."""

JOINT_WARN_MARGIN = 0.1
"""rad: WARN when a joint is closer than this to robot.joint_lower/upper."""
STATE_READS = 200
STATE_RATE_WARN_HZ = 900.0
"""The robot sends its state at 1 kHz."""

NETWORK_HINT = """\
check the Ethernet cable, that the laptop's wired interface has a static IP in the robot's
subnet (e.g. 172.16.0.1/24 for a robot at 172.16.0.2), and in Desk that the joints are unlocked
(brakes open) and FCI is activated. `ft-check-rt --robot-host HOST` checks route and ping."""


# -- versions ----------------------------------------------------------------------------------


def parse_versions(message: str) -> tuple[int | None, int | None]:
    """(robot server version, library version) from an IncompatibleVersionException message.

    libfranka >= 0.15: "Incompatible library version (server version: 9, library version: 10)"
    older:             "Incompatible library version. Server version: 4 Library version: 5"
    """
    server = re.search(r"server version:?\s*(\d+)", message, re.I)
    library = re.search(r"library version:?\s*(\d+)", message, re.I)
    return (
        int(server.group(1)) if server else None,
        int(library.group(1)) if library else None,
    )


def installed_franky() -> tuple[str, str | None, int | None]:
    """(franky-control version, bundled libfranka version, robot server version it speaks)."""
    try:
        version = importlib.metadata.version("franky-control")
    except importlib.metadata.PackageNotFoundError:
        version = "?"
    m = re.search(r"libfranka\.(\d+\.\d+\.\d+)", version)
    libfranka = m.group(1) if m else None
    if libfranka is None:  # PyPI wheel: no local version label, read the bundled library's name
        libs = Path(franky.__file__).resolve().parents[1] / "franky_control.libs"
        for lib in sorted(libs.glob("libfranka*.so.*")):
            m = re.search(r"\.so\.(\d+\.\d+\.\d+)$", lib.name)
            if m:
                libfranka = m.group(1)
    server = next((s for s, lf in FRANKY_WHEEL_INDEXES.items() if lf == libfranka), None)
    return version, libfranka, server


def pyproject_path() -> Path:
    return repo_configs_dir().parent / "pyproject.toml"


def uv_index_snippet(server: int) -> str:
    return (
        "[[tool.uv.index]]\n"
        f'name = "franky-server-{server}"\n'
        f'url = "{FRANKY_INDEX_ROOT}{server}/"\n'
        "explicit = true\n"
        "\n"
        "[tool.uv.sources]\n"
        f'franky-control = {{ index = "franky-server-{server}" }}'
    )


def version_mismatch_detail(server: int | None, library: int | None) -> str:
    franky_version, libfranka, _ = installed_franky()
    robot = "the robot's server version is unknown"
    if server is not None:
        robot = f"the robot speaks robot server version {server}"
        if server in ROBOT_SYSTEM_VERSION:
            robot += f" (robot system >= {ROBOT_SYSTEM_VERSION[server]})"
    installed = f"franky-control {franky_version}"
    if libfranka:
        installed += f" (libfranka {libfranka})"
    return f"{robot}; the installed {installed} speaks {library if library is not None else '?'}"


def version_guidance(server: int | None) -> str:
    """How to switch franky to the wheel for the robot's server version (multi-line hint)."""
    pyproject = pyproject_path()
    sim_note = (
        f"Note: franky-sim emulates server version {SIM_SERVER_VERSION} (FR3) and needs the\n"
        "default PyPI wheel. Rehearse in sim before switching (or in a second checkout): with\n"
        "another wheel, ft-sim and the sim tests fail with 'ProtocolException: incorrect object\n"
        "size'. To switch back, remove both sections and run the same uv command."
    )
    if server is None:
        return (
            "find the robot's system version in Desk, look up its server version in\n"
            f"{FRANKA_COMPAT}\n"
            f"and use the matching index from {FRANKY_INDEX_ROOT}\n"
            f"as described in {FRANKY_FAQ}\n" + sim_note
        )
    snippet = uv_index_snippet(server).splitlines()
    snippet = ["", *(f"  {line}" if line else "" for line in snippet), ""]
    command = f"cd {pyproject.parent} && uv lock --upgrade-package franky-control && uv sync"
    if server in FRANKY_WHEEL_INDEXES:
        lines = [
            f"install the franky wheel built for server version {server} "
            f"(libfranka {FRANKY_WHEEL_INDEXES[server]}); add to {pyproject}:",
            *snippet,
            f"then: {command}",
            "The franky-control==2.0.0 pin stays: uv resolves it to "
            f"2.0.0+libfranka.{FRANKY_WHEEL_INDEXES[server]} from that",
            "index (no --prerelease flag needed). Then run ft-check-robot again.",
        ]
    else:
        known = f"{min(FRANKY_WHEEL_INDEXES)}-{max(FRANKY_WHEEL_INDEXES)}"
        lines = [
            f"no franky wheel index for server version {server} existed when this tool was",
            f"written (only {known}). If {FRANKY_INDEX_ROOT} lists {server} now,",
            f"add to {pyproject}:",
            *snippet,
            f"then: {command}",
            f"Otherwise build franky against a libfranka that speaks {server} ({FRANKA_COMPAT}),",
            "or update the robot's system image.",
        ]
    return "\n".join([*lines, sim_note])


# -- connection --------------------------------------------------------------------------------


def make_robot(cfg: Config, realtime: franky.RealtimeConfig) -> franky.Robot:
    return franky.Robot(
        cfg.robot.host,
        realtime_config=realtime,
        default_torque_threshold=min(cfg.robot.collision_torque_thresholds),
        default_force_threshold=min(cfg.robot.collision_force_thresholds),
    )


def connection_failure(host: str, e: Exception) -> Result:
    msg = str(e)
    if isinstance(e, franky.IncompatibleVersionException):
        server, library = parse_versions(msg)
        return fail(
            "server_version",
            version_mismatch_detail(server, library),
            version_guidance(server),
            robot_server_version=server,
            franky_server_version=library,
            error=msg,
        )
    if isinstance(e, franky.ProtocolException):
        return fail(
            "connection",
            f"protocol error talking to {host}: {msg}",
            "franky and the robot (or simulator) disagree on the wire format: a franky wheel\n"
            f"for another robot server version? franky-sim needs the default wheel (version "
            f"{SIM_SERVER_VERSION}).",
            error=msg,
        )
    if isinstance(e, franky.NetworkException):
        if "refused" in msg.lower():
            detail = f"{host} refused the connection: {msg}"
            hint = (
                "the host answers but FCI does not listen: in Desk unlock the joints and activate\n"
                "FCI (for a simulator: is ft-sim running on this address?)\n" + NETWORK_HINT
            )
        else:
            detail = f"cannot reach {host}: {msg}"
            hint = NETWORK_HINT
        return fail("connection", detail, hint, error=msg)
    return fail("connection", f"{type(e).__name__}: {msg}", NETWORK_HINT, error=msg)


def connect(cfg: Config, report: Report) -> franky.Robot | None:
    host = cfg.robot.host
    enforce = cfg.robot.realtime == "enforce"
    realtime = franky.RealtimeConfig.Enforce if enforce else franky.RealtimeConfig.Ignore
    try:
        robot = make_robot(cfg, realtime)
    except franky.RealtimeException as e:
        report.add(
            fail(
                "realtime",
                f"robot.realtime = enforce, but libfranka refused this machine: {e}",
                "run ft-check-rt for the kernel/rtprio fixes; for a read-only look on a dev box\n"
                "pass --realtime ignore",
                error=str(e),
            )
        )
        report.note("continuing the read-only checks with realtime = ignore")
        try:
            robot = make_robot(cfg, franky.RealtimeConfig.Ignore)
        except franky.Exception as e2:
            report.add(connection_failure(host, e2))
            return None
    except franky.Exception as e:
        report.add(connection_failure(host, e))
        return None
    else:
        if enforce:
            report.add(passed("realtime", "robot.realtime = enforce: libfranka accepted it"))
        elif not cfg.sim.enabled:
            report.add(
                warn(
                    "realtime",
                    "robot.realtime = ignore against a real robot: control deadlines are not "
                    "guaranteed",
                    "use robot.realtime = enforce on a PREEMPT_RT kernel (see ft-check-rt); drop\n"
                    "--realtime ignore / FT_REALTIME=ignore",
                )
            )
        else:
            report.note("realtime: robot.realtime = ignore (simulator)")

    franky_version, libfranka, server = installed_franky()
    versions = f"franky-control {franky_version}"
    if libfranka:
        versions += f", libfranka {libfranka}"
    if server is not None:
        versions += f", robot server version {server}"
    report.add(
        passed(
            "connection",
            f"connected to {host} ({versions})",
            host=host,
            franky_control=franky_version,
            libfranka=libfranka,
            server_version=server,
        )
    )
    return robot


# -- state checks ------------------------------------------------------------------------------

MODE_HINTS = {
    "Move": ("a motion is running", "wait for it to end; is another FCI program connected?"),
    "Guiding": ("hand-guiding is active", "release the guiding buttons on the arm"),
    "Reflex": (
        "the robot stopped on an error (see errors)",
        "clear it with error recovery (Desk, or franky's recover_from_errors) before teleop",
    ),
    "UserStopped": (
        "the user stop / external activation device is engaged",
        "release it before teleop; the arm cannot move while it is pressed",
    ),
    "AutomaticErrorRecovery": ("error recovery is running", "wait a moment and check again"),
    "Other": (
        "not ready for FCI commands",
        "in Desk: unlock the joints (brakes) and activate FCI",
    ),
}


def check_mode(mode: str) -> Result:
    if mode == "Idle":
        return passed("robot_mode", "Idle (ready for commands)", mode=mode)
    what, hint = MODE_HINTS.get(mode, ("unexpected mode", "check the robot in Desk"))
    return warn("robot_mode", f"{mode}: {what}", hint, mode=mode)


def check_errors(current: Sequence[str], last_motion: Sequence[str] = ()) -> Result:
    data = {"current": list(current), "last_motion": list(last_motion)}
    previous = f" (last motion ended with: {', '.join(last_motion)})" if last_motion else ""
    if not current:
        return passed("errors", "no active errors" + previous, **data)
    return warn(
        "errors",
        f"{len(current)} active: {', '.join(current)}" + previous,
        "run error recovery before teleop (Desk, or franky's recover_from_errors); if an error\n"
        "comes back immediately, fix its cause first (e.g. a joint outside its limits)",
        **data,
    )


def _vec(v: Sequence[float], fmt: str = "+.3f") -> str:
    return "[" + ", ".join(format(float(x), fmt) for x in v) + "]"


def check_joints(
    q: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    warn_margin: float = JOINT_WARN_MARGIN,
) -> Result:
    margins = [min(qi - lo, hi - qi) for qi, lo, hi in zip(q, lower, upper, strict=True)]
    data = {"q": [float(x) for x in q], "margin": margins}
    qs = f"q = {_vec(q)}"
    outside = [i for i, m in enumerate(margins) if m < 0]
    near = [i for i, m in enumerate(margins) if 0 <= m < warn_margin]

    def side(i: int) -> str:
        return "lower" if q[i] - lower[i] < upper[i] - q[i] else "upper"

    def joints(idx: list[int]) -> str:
        return ", ".join(f"joint {i + 1} {abs(margins[i]):.3f} rad {side(i)}" for i in idx)

    if outside:
        return fail(
            "joints",
            f"outside robot.joint_lower/upper: {joints(outside)}; {qs}",
            "hand-guide the arm back inside the limits (guiding mode). The config limits are the\n"
            "intersection of the FR3 and Panda limits (configs/*.toml), stricter than either arm",
            **data,
        )
    if near:
        return warn(
            "joints",
            f"within {warn_margin} rad of a limit: {joints(near)}; {qs}",
            "hand-guide the arm away from the limit; motion towards it is slowed and then blocked",
            **data,
        )
    k = int(np.argmin(margins))
    return passed(
        "joints",
        f"all {len(q)} inside limits, closest joint {k + 1} at {margins[k]:.2f} rad; {qs}",
        **data,
    )


def check_workspace(p: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> Result:
    data = {"ee": [float(x) for x in p], "lower": list(lower), "upper": list(upper)}
    pos = f"EE at {_vec(p)} m"
    excess = []  # (distance outside, description)
    for axis, x, lo, hi in zip("xyz", p, lower, upper, strict=True):
        if x < lo:
            excess.append((lo - x, f"{axis} {x:.3f} < {lo}"))
        elif x > hi:
            excess.append((x - hi, f"{axis} {x:.3f} > {hi}"))
    if excess:
        d, _ = max(excess)
        where = "; ".join(e for _, e in excess)
        return warn(
            "workspace",
            f"{pos} is {d * 100:.1f} cm outside the [workspace] box ({where})",
            "ft-node will only allow motion back towards the box: steer it inside, or hand-guide\n"
            "the arm there first. If the box is wrong for this table, fix [workspace] (config)",
            **data,
        )
    walls = [(x - lo, f"{a} lower") for a, x, lo in zip("xyz", p, lower, strict=True)]
    walls += [(hi - x, f"{a} upper") for a, x, hi in zip("xyz", p, upper, strict=True)]
    d, wall = min(walls)
    return passed(
        "workspace", f"{pos} inside the box, {d * 100:.1f} cm from the {wall} wall", **data
    )


def measure_state_rate(robot: franky.Robot, reads: int = STATE_READS) -> dict[str, float]:
    """Read the state ``reads`` times; each read waits for the next 1 kHz state packet."""
    wall, stamps = [], []
    for _ in range(reads):
        s = robot.state
        wall.append(time.perf_counter())
        stamps.append(s.time.to_sec())
    span = wall[-1] - wall[0]
    fresh = sum(b > a for a, b in itertools.pairwise(stamps))
    return {
        "reads": reads,
        "rate_hz": (reads - 1) / span if span > 0 else 0.0,
        "fresh": fresh,
        "max_gap_ms": max(b - a for a, b in itertools.pairwise(wall)) * 1e3,
    }


def check_state_rate(stats: dict[str, float], sim: bool) -> Result:
    detail = (
        f"{stats['rate_hz']:.0f} Hz over {stats['reads']} reads "
        f"({stats['fresh']} new states, max gap {stats['max_gap_ms']:.1f} ms)"
    )
    if stats["rate_hz"] >= STATE_RATE_WARN_HZ and stats["fresh"] >= stats["reads"] - 1:
        return passed("state_rate", detail, **stats)
    hint = (
        "franky-sim paces itself with sleep() on a non-RT kernel; this is expected in simulation"
        if sim
        else "the robot sends its state at 1 kHz; fewer states mean delayed or lost UDP packets:\n"
        "use a direct cable, no Wi-Fi, and see ft-check-rt --robot-host"
    )
    return warn("state_rate", detail + f" (< {STATE_RATE_WARN_HZ:.0f} Hz expected)", hint, **stats)


def check_gripper(host: str) -> Result:
    try:
        gripper = franky.Gripper(host)
        width, max_width = float(gripper.width), float(gripper.max_width)
        grasped = bool(gripper.is_grasped)
    except franky.Exception as e:
        return fail(
            "gripper",
            f"robot.gripper = true but the gripper does not answer: {e}",
            "check that the Franka Hand is attached and initialized in Desk, or set\n"
            "robot.gripper = false in the config if the arm has no hand",
            error=str(e),
        )
    data = {"width": width, "max_width": max_width, "grasped": grasped}
    detail = f"connected, width {width * 1e3:.1f} mm"
    if max_width <= 0:
        return warn(
            "gripper",
            detail + ", but max width is 0: the hand is not homed",
            "home the gripper in Desk (this tool never moves it)",
            **data,
        )
    holding = ", holding an object" if grasped else ""
    return passed("gripper", f"{detail} of {max_width * 1e3:.1f} mm{holding}", **data)


# -- main --------------------------------------------------------------------------------------


def run_checks(cfg: Config, report: Report) -> None:
    robot = connect(cfg, report)
    if robot is None:
        return
    try:
        s = robot.state
        report.add(check_mode(s.robot_mode.name))
        errors = active_errors(s.current_errors)
        report.add(check_errors(errors, active_errors(s.last_motion_errors)))
        report.add(check_joints(list(s.q), cfg.robot.joint_lower, cfg.robot.joint_upper))
        ee = np.asarray(s.O_T_EE.matrix, dtype=float)[:3, 3]
        report.add(check_workspace(ee, cfg.workspace.lower, cfg.workspace.upper))
        report.add(check_state_rate(measure_state_rate(robot), sim=cfg.sim.enabled))
    except franky.Exception as e:
        report.add(fail("state", f"reading the robot state failed: {e}", NETWORK_HINT))
        return
    if cfg.robot.gripper:
        report.add(check_gripper(cfg.robot.host))
    else:
        report.note("gripper: robot.gripper = false, not checked")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft-check-robot",
        description="Read-only check of the robot (never moves it): connection, server version, "
        "mode, errors, joints, workspace, state rate, gripper.",
    )
    add_config_args(parser)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        parser.error(str(e))

    report = Report("ft-check-robot", as_json=args.json)
    report.note(
        f"config {args.config}: robot.host {cfg.robot.host} "
        f"({'simulator' if cfg.sim.enabled else 'real robot'})"
    )
    try:
        run_checks(cfg, report)
    except Exception as e:  # still print the report (and valid JSON) on an unexpected error
        report.add(fail("internal", f"{type(e).__name__}: {e}"))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
