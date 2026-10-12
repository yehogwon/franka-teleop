"""Typed configuration shared by every process.

A deployment is described by one TOML file. ``configs/sim.toml`` and ``configs/real.toml``
must differ only in the keys listed in :data:`PARITY_ALLOWED_DIFF`; a test enforces this so
that a session rehearsed in simulation runs unchanged against the real robot.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import msgspec

# Joint limits [rad]. The robot type is not known in advance, so the default is the
# intersection of both arms' limits, which is valid for either.
FR3_JOINT_LOWER = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
FR3_JOINT_UPPER = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)
PANDA_JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
PANDA_JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
JOINT_LOWER = tuple(max(a, b) for a, b in zip(FR3_JOINT_LOWER, PANDA_JOINT_LOWER, strict=True))
JOINT_UPPER = tuple(min(a, b) for a, b in zip(FR3_JOINT_UPPER, PANDA_JOINT_UPPER, strict=True))

# The usual Franka "ready" pose.
READY_Q = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)

# Dotted keys allowed to differ between configs/sim.toml and configs/real.toml.
PARITY_ALLOWED_DIFF = frozenset({"robot.host", "sim.enabled"})

# Environment variable that overrides robot.realtime, e.g. FT_REALTIME=ignore on a dev box
# without a PREEMPT_RT kernel. The config files themselves always say "enforce".
REALTIME_ENV = "FT_REALTIME"


class _Base(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    pass


class RobotConfig(_Base):
    host: str = "127.0.0.1"
    """FCI address of the robot, or the loopback address franky-sim binds to."""
    realtime: Literal["enforce", "ignore"] = "enforce"
    dynamics_factor: float = 0.2
    """franky relative dynamics factor in (0, 1]: scales velocity/acceleration/jerk limits."""
    gripper: bool = True
    gripper_speed: float = 0.05
    gripper_force: float = 20.0
    collision_torque_thresholds: list[float] = msgspec.field(
        default_factory=lambda: [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    )
    collision_force_thresholds: list[float] = msgspec.field(
        default_factory=lambda: [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    )
    home_q: list[float] = msgspec.field(default_factory=lambda: list(READY_Q))
    joint_lower: list[float] = msgspec.field(default_factory=lambda: list(JOINT_LOWER))
    joint_upper: list[float] = msgspec.field(default_factory=lambda: list(JOINT_UPPER))


class SimConfig(_Base):
    enabled: bool = True
    """True: ft-teleop starts franky-sim on robot.host. False: robot.host is a real robot."""
    scene: Literal["empty", "tabletop"] = "tabletop"
    initial_q: list[float] = msgspec.field(default_factory=lambda: list(READY_Q))


class NetConfig(_Base):
    """Publishers bind on bind_host; subscribers connect to connect_host."""

    bind_host: str = "127.0.0.1"
    connect_host: str = "127.0.0.1"
    cmd_port: int = 5555
    """ft-joystick PUB (binds) -> ft-node SUB: TeleopCommand."""
    status_port: int = 5556
    """ft-node PUB (binds) -> ft-joystick/ft-view SUB: NodeStatus."""
    scene_port: int = 5557
    """ft-sim PUB (binds) -> ft-view SUB: SceneState (full MuJoCo qpos)."""
    scene_rpc_port: int = 5558
    """ft-sim REP (binds) <- ft-view REQ: SceneInfo (MJCF of the running scene)."""

    def bind(self, port: int) -> str:
        return f"tcp://{self.bind_host}:{port}"

    def connect(self, port: int) -> str:
        return f"tcp://{self.connect_host}:{port}"


class TeleopConfig(_Base):
    rate_hz: float = 100.0
    cmd_timeout_ms: float = 150.0
    """A command older than this is stale: the node commands zero velocity."""
    hold_ms: int = 100
    """franky holds each velocity target this long; the robot stops by itself if the node dies."""
    max_linear_speed: float = 0.15
    max_angular_speed: float = 0.6
    speed_levels: list[float] = msgspec.field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    initial_speed_level: int = 1
    default_frame: Literal["base", "tool"] = "base"
    operator_yaw_deg: float = 0.0
    """Rotation of the operator's stick frame about base z. 0: operator stands behind the robot
    looking along +x. 180: operator faces the robot."""
    allowed_sources: list[str] = msgspec.field(default_factory=lambda: ["gamepad"])
    """Input sources allowed to move the robot. When sim.enabled, keyboard and script are added."""


class WorkspaceConfig(_Base):
    lower: list[float] = msgspec.field(default_factory=lambda: [0.25, -0.40, 0.03])
    upper: list[float] = msgspec.field(default_factory=lambda: [0.70, 0.40, 0.70])
    """Axis-aligned box (robot base frame, metres) the end effector may not leave."""
    slow_zone: float = 0.05
    """Outward velocity is ramped to zero over this distance from a wall."""


class JointGuardConfig(_Base):
    margin: float = 0.2
    """Joint velocity towards a limit is ramped to zero over this distance [rad]..."""
    clearance: float = 0.03
    """...ending this far before the limit [rad], to absorb tracking lag."""
    max_joint_speed: float = 1.0
    """Twist is scaled down when the predicted joint speed exceeds this [rad/s]."""
    damping: float = 0.05
    """Damping of the pseudo-inverse used to predict joint velocities."""
    sigma_slow: float = 0.06
    """Singularity guard: below this smallest singular value of the weighted Jacobian, motion
    that decreases it is slowed..."""
    sigma_stop: float = 0.035
    """...and stopped at this value. Cartesian tracking breaks down near 0.025 (elbow straight)."""
    rot_weight: float = 0.2
    """Weight [m] of the rotational Jacobian rows, so sigma has units of metres per radian."""
    probe_rad: float = 0.002
    """Joint-space step [rad] probing whether the twist lowers sigma. Kept small: near a
    singularity predicted joint speeds are large, and a long horizon would step across it."""


class InputConfig(_Base):
    device: Literal["gamepad", "keyboard", "script"] = "gamepad"
    rate_hz: float = 100.0
    gamepad_index: int = 0
    deadzone: float = 0.12
    expo: float = 0.5
    """Blend between linear (0) and cubic (1) stick response."""
    invert: list[str] = msgspec.field(default_factory=list)
    """Intent axes to negate, from: lin_x lin_y lin_z ang_x ang_y ang_z."""


class ViewerConfig(_Base):
    fps: float = 60.0
    web_port: int = 8080


class Config(_Base):
    robot: RobotConfig = msgspec.field(default_factory=RobotConfig)
    sim: SimConfig = msgspec.field(default_factory=SimConfig)
    net: NetConfig = msgspec.field(default_factory=NetConfig)
    teleop: TeleopConfig = msgspec.field(default_factory=TeleopConfig)
    workspace: WorkspaceConfig = msgspec.field(default_factory=WorkspaceConfig)
    joint_guard: JointGuardConfig = msgspec.field(default_factory=JointGuardConfig)
    input: InputConfig = msgspec.field(default_factory=InputConfig)
    viewer: ViewerConfig = msgspec.field(default_factory=ViewerConfig)

    def validate(self) -> Config:
        _check(len(self.robot.home_q) == 7, "robot.home_q needs 7 values")
        _check(len(self.robot.joint_lower) == 7, "robot.joint_lower needs 7 values")
        _check(len(self.robot.joint_upper) == 7, "robot.joint_upper needs 7 values")
        _check(
            all(
                lo < hi
                for lo, hi in zip(self.robot.joint_lower, self.robot.joint_upper, strict=True)
            ),
            "robot.joint_lower must be below robot.joint_upper",
        )
        _check(0.0 < self.robot.dynamics_factor <= 1.0, "robot.dynamics_factor must be in (0, 1]")
        _check(len(self.robot.collision_torque_thresholds) == 7, "need 7 torque thresholds")
        _check(len(self.robot.collision_force_thresholds) == 6, "need 6 force thresholds")
        _check(len(self.sim.initial_q) == 7, "sim.initial_q needs 7 values")
        _check(len(self.workspace.lower) == 3 and len(self.workspace.upper) == 3, "workspace is 3D")
        jg = self.joint_guard
        _check(0 < jg.sigma_stop < jg.sigma_slow, "need 0 < joint_guard.sigma_stop < sigma_slow")
        _check(jg.margin > 0 and jg.clearance >= 0, "joint_guard.margin/clearance invalid")
        _check(0 < jg.probe_rad <= 0.05, "joint_guard.probe_rad must be in (0, 0.05]")
        _check(
            all(lo < hi for lo, hi in zip(self.workspace.lower, self.workspace.upper, strict=True)),
            "workspace.lower must be below workspace.upper",
        )
        _check(self.teleop.rate_hz > 0 and self.input.rate_hz > 0, "rates must be positive")
        _check(len(self.teleop.speed_levels) > 0, "teleop.speed_levels is empty")
        _check(
            all(0.0 < s <= 1.0 for s in self.teleop.speed_levels),
            "teleop.speed_levels must be in (0, 1]",
        )
        _check(
            0 <= self.teleop.initial_speed_level < len(self.teleop.speed_levels),
            "teleop.initial_speed_level out of range",
        )
        _check(
            self.teleop.hold_ms >= 1000.0 / self.teleop.rate_hz,
            "teleop.hold_ms must cover at least one node tick",
        )
        bad = set(self.input.invert) - {"lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"}
        _check(not bad, f"unknown input.invert axes: {sorted(bad)}")
        return self

    def allowed_sources(self) -> set[str]:
        sources = set(self.teleop.allowed_sources)
        if self.sim.enabled:
            sources |= {"keyboard", "script"}
        return sources


class ConfigError(ValueError):
    pass


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ConfigError(msg)


def load_config(path: str | os.PathLike | None = None, **overrides: Any) -> Config:
    """Load a TOML config (defaults if ``path`` is None) and apply dotted-key overrides.

    ``overrides`` use ``__`` for nesting, e.g. ``load_config(p, robot__host="127.0.0.2")``.
    The ``FT_REALTIME`` environment variable overrides ``robot.realtime``.
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    if os.environ.get(REALTIME_ENV):
        _set_dotted(data, "robot.realtime", os.environ[REALTIME_ENV])
    for key, value in overrides.items():
        _set_dotted(data, key.replace("__", "."), value)
    try:
        cfg = msgspec.convert(data, Config)
    except msgspec.ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e
    return cfg.validate()


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def flatten(cfg: Config) -> dict[str, Any]:
    """Config as {dotted.key: value}, for diffs and logging."""
    out: dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            out[prefix] = obj

    walk("", msgspec.to_builtins(cfg))
    return out


def dump_toml(cfg: Config) -> str:
    """Serialize a config to TOML (round-trips through load_config)."""

    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        raise TypeError(f"cannot write {v!r} to TOML")

    lines: list[str] = []
    for section, fields in msgspec.to_builtins(cfg).items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {value(v)}" for k, v in fields.items()]
        lines.append("")
    return "\n".join(lines)


def write_config(cfg: Config, path: str | os.PathLike) -> Path:
    """Write the effective config, e.g. for child processes started by a launcher or test."""
    path = Path(path)
    path.write_text(dump_toml(cfg))
    return path


def config_diff(a: Config, b: Config) -> set[str]:
    fa, fb = flatten(a), flatten(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


def repo_configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def add_config_args(parser) -> None:
    """Standard --config/--realtime/--connect-host flags for every ft-* command."""
    parser.add_argument(
        "--config",
        "-c",
        default=os.environ.get("FT_CONFIG", str(repo_configs_dir() / "sim.toml")),
        help="TOML config file (default: $FT_CONFIG or configs/sim.toml)",
    )
    parser.add_argument(
        "--realtime",
        choices=["enforce", "ignore"],
        default=None,
        help=f"override robot.realtime (also ${REALTIME_ENV}); 'ignore' only on a non-RT dev box",
    )
    parser.add_argument("--connect-host", default=None, help="override net.connect_host")


def config_from_args(args) -> Config:
    overrides: dict[str, Any] = {}
    if getattr(args, "realtime", None):
        overrides["robot__realtime"] = args.realtime
    if getattr(args, "connect_host", None):
        overrides["net__connect_host"] = args.connect_host
    return load_config(args.config, **overrides)
