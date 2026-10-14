"""Messages exchanged between the ft-* processes (msgpack via msgspec, over ZMQ).

Topology::

    ft-joystick --TeleopCommand--> ft-node --franky/libfranka--> robot | franky-sim
    ft-joystick, ft-view <--NodeStatus-- ft-node
    ft-view <--SceneState-- ft-sim        ft-view --SceneRequest/SceneInfo--> ft-sim

Every message carries ``v`` (protocol version). Decoders reject other versions so that
mismatched checkouts fail loudly instead of misinterpreting fields.

Discrete actions (home, gripper toggle, ...) are sent as monotonically increasing counters
rather than one-shot events. Command streams are lossy (ZMQ CONFLATE keeps only the newest
message), and a counter survives dropped messages: the node acts once per increment.
"""

from __future__ import annotations

import math

import msgspec

PROTOCOL_VERSION = 1

# Names of the discrete-action counters in TeleopCommand, in a stable order.
ACTIONS = ("gripper", "home", "stop", "recover", "frame", "speed_up", "speed_down")

# Node modes reported in NodeStatus.mode.
MODES = ("idle", "teleop", "homing", "error", "locked")


class ProtocolError(ValueError):
    pass


class TeleopCommand(msgspec.Struct, kw_only=True, frozen=True):
    """Operator intent, sent by ft-joystick at input.rate_hz (also when idle, as a heartbeat).

    ``linear``/``angular`` are normalized to [-1, 1] and expressed in the *operator* frame, which
    follows the robot base convention for an operator standing behind the robot looking along +x:
    linear = (forward, left, up), angular = (roll about forward, pitch about left, yaw about up).
    The node applies speed scaling, operator yaw, base/tool frame and safety limits.
    """

    v: int = PROTOCOL_VERSION
    session: int
    """Random id per client process. A new session must release the deadman before moving."""
    seq: int
    t_send: float
    """Client wall-clock time [s] (time.time()); only comparable to the node's clock on one host."""
    source: str
    """"gamepad" | "keyboard" | "script". The web joystick (ft-joystick --device webpad) drives an
    SDL virtual pad and reports "gamepad", like a physical pad."""
    deadman: bool
    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper: int = 0
    home: int = 0
    stop: int = 0
    recover: int = 0
    frame: int = 0
    speed_up: int = 0
    speed_down: int = 0

    def counters(self) -> dict[str, int]:
        return {a: getattr(self, a) for a in ACTIONS}


class NodeStatus(msgspec.Struct, kw_only=True, frozen=True):
    """Node state, published by ft-node every tick."""

    v: int = PROTOCOL_VERSION
    seq: int
    t: float
    """Node wall-clock time [s]."""
    mode: str
    message: str = ""
    """Last notable event or error, human readable."""
    sim: bool
    q: tuple[float, ...]
    dq: tuple[float, ...]
    ee_pos: tuple[float, float, float]
    ee_quat: tuple[float, float, float, float]
    """End-effector orientation, quaternion (x, y, z, w), robot base frame."""
    gripper_width: float = math.nan
    gripper_closed: bool = False
    """Desired gripper state (last toggle)."""
    speed_level: int
    speed_scale: float
    frame: str
    twist: tuple[float, ...] = (0.0,) * 6
    """Last commanded twist (vx, vy, vz, wx, wy, wz) after limits, base frame."""
    limiters: tuple[str, ...] = ()
    """Limiters that reduced the last twist: "workspace", "floor", "singularity", "joint"."""
    session: int = 0
    """Session currently in control (0: none)."""
    last_cmd_seq: int = -1
    """Echo of the newest command's seq; lets clients compute round-trip time."""
    cmd_age_ms: float = math.inf
    robot_errors: str = ""
    loop_overruns: int = 0


class SceneState(msgspec.Struct, kw_only=True, frozen=True):
    """Full MuJoCo state of the simulated scene, published by ft-sim at viewer.fps."""

    v: int = PROTOCOL_VERSION
    seq: int
    t_sim: float
    """Simulation time [s]."""
    qpos: bytes
    """float64 array of length nq, little endian (numpy .tobytes())."""
    nq: int


class SceneRequest(msgspec.Struct, kw_only=True, frozen=True):
    v: int = PROTOCOL_VERSION


class SceneInfo(msgspec.Struct, kw_only=True, frozen=True):
    """Reply to SceneRequest: the MJCF of the running scene, so viewers render the same model."""

    v: int = PROTOCOL_VERSION
    xml: str
    """MJCF with <compiler meshdir> set to MESHDIR_PLACEHOLDER; see sim.scene.resolve_meshdir."""
    nq: int
    robot_prefix: str
    """Name prefix of the robot's joints/bodies in the scene, e.g. "robot0_"."""
    robot_host: str = ""
    """robot.host of ft-sim's config: the address it serves the simulated arm on. Viewers
    compare it with their own config's robot.host, so they never show a simulation that belongs
    to another config. "" means unknown (an ft-sim from before this field) and is accepted."""


_encoder = msgspec.msgpack.Encoder()
_decoders: dict[type, msgspec.msgpack.Decoder] = {}


def encode(msg: msgspec.Struct) -> bytes:
    return _encoder.encode(msg)


def decode[T](data: bytes, typ: type[T]) -> T:
    dec = _decoders.get(typ)
    if dec is None:
        dec = _decoders[typ] = msgspec.msgpack.Decoder(typ)
    try:
        msg = dec.decode(data)
    except msgspec.DecodeError as e:
        raise ProtocolError(f"cannot decode {typ.__name__}: {e}") from e
    if msg.v != PROTOCOL_VERSION:
        raise ProtocolError(
            f"{typ.__name__} has protocol version {msg.v}, expected {PROTOCOL_VERSION}; "
            "update both ends to the same checkout"
        )
    return msg
