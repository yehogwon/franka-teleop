"""Where a viewer gets its model and state from.

- :class:`SceneSource`: ft-sim. The model comes from its scene RPC (the MJCF of the running
  scene, cubes included) and the full ``qpos`` from its SceneState stream.
- :class:`StatusSource`: ft-node. A robot-only model; the arm joints and fingers come from
  NodeStatus. This is how the real arm is viewed.

Both subscribe to NodeStatus (for overlays) and are polled without blocking from the viewer's
own loop: ``poll()`` drains the sockets and updates :attr:`Source.qpos` in place.
"""

from __future__ import annotations

import math
import sys
import time

import mujoco
import numpy as np
import zmq

from ..config import Config
from ..protocol import NodeStatus, SceneInfo, SceneRequest, SceneState, decode, encode
from ..sim.scene import load_model, robot_qpos_addrs, scene_xml
from ..transport import context, recv_latest, subscriber

SCENE_PROBE_TIMEOUT_S = 1.0
DEFAULT_FINGER_POS = 0.02  # franky-sim's initial hand width is 4 cm


class SourceError(RuntimeError):
    pass


class Source:
    """Model + newest state. Subclasses implement ``_poll_state``."""

    kind = ""

    def __init__(self, cfg: Config, model: mujoco.MjModel):
        self.cfg = cfg
        self.model = model
        self.qpos = model.qpos0.copy()
        """Newest qpos (length model.nq). Updated in place by poll()."""
        self.has_state = False
        """True once a state has been received."""
        self.status: NodeStatus | None = None
        """Newest NodeStatus, if ft-node is running."""
        self.t_state = math.nan
        """time.monotonic() of the newest state."""
        self._status_sub = subscriber(cfg.net.connect(cfg.net.status_port))

    def describe(self) -> str:
        raise NotImplementedError

    def poll(self) -> bool:
        """Drain the sockets without blocking. True if qpos changed."""
        status, _ = recv_latest(self._status_sub, NodeStatus)
        if status is not None:
            self.status = status
        changed = self._poll_state(status)
        if changed:
            self.has_state = True
            self.t_state = time.monotonic()
        return changed

    def wait_for_state(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self.has_state and time.monotonic() < deadline:
            self.poll()
            if not self.has_state:
                time.sleep(0.01)
        return self.has_state

    def _poll_state(self, status: NodeStatus | None) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        self._status_sub.close()


class SceneSource(Source):
    """Full scene from ft-sim."""

    kind = "scene"

    def __init__(self, cfg: Config, info: SceneInfo):
        model = load_model(info.xml)
        if model.nq != info.nq:
            raise SourceError(f"scene model has nq={model.nq} but ft-sim reports {info.nq}")
        super().__init__(cfg, model)
        self.info = info
        self.arm_addr, self.finger_addr = robot_qpos_addrs(model, info.robot_prefix)
        self._scene_sub = subscriber(cfg.net.connect(cfg.net.scene_port))
        self._warned_nq = False

    def describe(self) -> str:
        return f"ft-sim scene ({self.cfg.net.connect(self.cfg.net.scene_port)}, nq {self.model.nq})"

    def _poll_state(self, status: NodeStatus | None) -> bool:
        msg, _ = recv_latest(self._scene_sub, SceneState)
        if msg is None:
            return False
        if msg.nq != self.model.nq or len(msg.qpos) != 8 * self.model.nq:
            if not self._warned_nq:
                print(
                    f"ft-view: SceneState has nq={msg.nq}, the model {self.model.nq}; "
                    "ft-sim was restarted with another scene? Restart ft-view.",
                    file=sys.stderr,
                )
                self._warned_nq = True
            return False
        self.qpos[:] = np.frombuffer(msg.qpos, dtype="<f8")
        return True

    def close(self) -> None:
        self._scene_sub.close()
        super().close()


class StatusSource(Source):
    """Robot-only view from ft-node's NodeStatus (works for the real arm)."""

    kind = "status"

    def __init__(self, cfg: Config):
        super().__init__(cfg, load_model(scene_xml("empty")))
        self.arm_addr, self.finger_addr = robot_qpos_addrs(self.model)
        # Until ft-node reports, show the configured start pose (has_state stays False).
        self.qpos[self.arm_addr] = cfg.sim.initial_q
        self.qpos[self.finger_addr] = DEFAULT_FINGER_POS

    def describe(self) -> str:
        return f"ft-node status ({self.cfg.net.connect(self.cfg.net.status_port)}, robot only)"

    def _poll_state(self, status: NodeStatus | None) -> bool:
        if status is None or len(status.q) != len(self.arm_addr):
            return False
        self.qpos[self.arm_addr] = status.q
        if math.isfinite(status.gripper_width):
            self.qpos[self.finger_addr] = status.gripper_width / 2.0
        return True


def request_scene_info(cfg: Config, timeout: float) -> SceneInfo | None:
    """Ask ft-sim for the scene. None if it does not answer within ``timeout``.

    A fresh REQ socket per attempt: REQ is a strict send/recv state machine, so after a
    timeout the old socket cannot send again.
    """
    sock = context().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(cfg.net.connect(cfg.net.scene_rpc_port))
    try:
        sock.send(encode(SceneRequest()))
        if not sock.poll(int(timeout * 1000), zmq.POLLIN):
            return None
        return decode(sock.recv(), SceneInfo)
    finally:
        sock.close()


def open_source(cfg: Config, kind: str = "auto", wait: float = 10.0) -> Source:
    """``auto``: ft-sim if it answers within ~1 s, else ft-node status (real robot).
    ``scene``: wait up to ``wait`` seconds for ft-sim. ``status``: ft-node status."""
    if kind == "status":
        return StatusSource(cfg)
    if kind == "auto":
        info = request_scene_info(cfg, SCENE_PROBE_TIMEOUT_S)
        return SceneSource(cfg, info) if info is not None else StatusSource(cfg)
    if kind != "scene":
        raise ValueError(f"unknown source {kind!r}")
    deadline = time.monotonic() + wait
    while True:
        info = request_scene_info(cfg, min(SCENE_PROBE_TIMEOUT_S, max(wait, 0.1)))
        if info is not None:
            return SceneSource(cfg, info)
        if time.monotonic() >= deadline:
            raise SourceError(
                f"ft-sim did not answer on {cfg.net.connect(cfg.net.scene_rpc_port)} within "
                f"{wait:g} s. Is ft-sim running with this config?"
            )
