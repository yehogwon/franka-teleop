"""Kinematic stand-in for unit tests. It mimics the franky semantics the node relies on:
velocity targets are held for hold_ms and then ramp to zero, motion types cannot change while
moving, errors latch until recover(). There is no dynamics and no forward kinematics: the
end-effector pose is integrated from the twist, joints from a fixed Jacobian."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from ..config import READY_Q
from ..geometry import exp_so3
from .base import ArmError, ArmState

# A fixed, well-conditioned Jacobian: joint i mostly drives twist component i, joint 7 adds yaw.
FAKE_JACOBIAN = np.hstack([np.eye(6), np.array([[0, 0, 0, 0, 0, 1.0]]).T])
READY_EE = np.array([[1.0, 0, 0, 0.307], [0, -1.0, 0, 0], [0, 0, -1.0, 0.487], [0, 0, 0, 1]])


class FakeArm:
    def __init__(
        self,
        clock: Callable[[], float],
        q0: Sequence[float] = READY_Q,
        O_T_EE: np.ndarray = READY_EE,
        accel: float = 2.0,
        joint_motion_s: float = 0.5,
        joint_stop_s: float = 0.2,
    ):
        self.clock = clock
        self.q = np.array(q0, dtype=float)
        self.T = np.array(O_T_EE, dtype=float)
        self.v = np.zeros(6)
        self.accel = accel
        self.joint_motion_s = joint_motion_s
        self.joint_stop_s = joint_stop_s
        self._joint_stop_until: float | None = None
        self._t = clock()
        self._target = np.zeros(6)
        self._hold_until = -math.inf
        self._mode: str | None = None  # "velocity" | "joint" | None
        self._joint_goal: np.ndarray | None = None
        self._joint_start: tuple[float, np.ndarray] | None = None
        self.errors: tuple[str, ...] = ()
        self.gripper_closed = False
        self.gripper_width = 0.08
        self.calls: list[tuple[str, object]] = []
        self.recover_ok = True

    # -- simulation ----------------------------------------------------------------------------

    def _advance(self) -> None:
        now = self.clock()
        dt, self._t = max(0.0, now - self._t), now
        if dt == 0.0:
            return
        if self._mode == "velocity":
            target = self._target if now <= self._hold_until else np.zeros(6)
            dv = target - self.v
            step = self.accel * dt
            n = np.linalg.norm(dv)
            self.v = target if n <= step else self.v + dv * step / n
            self.T[:3, 3] += self.v[:3] * dt
            self.T[:3, :3] = exp_so3(self.v[3:] * dt) @ self.T[:3, :3]
            self.q += np.linalg.pinv(FAKE_JACOBIAN) @ self.v * dt
            if now > self._hold_until and not self.v.any():
                self._mode = None
        elif self._mode == "joint":
            if self._joint_stop_until is not None:
                # A joint stop motion decelerates for a while; the motion type stays "joint".
                if now >= self._joint_stop_until:
                    self._mode, self._joint_stop_until = None, None
            else:
                t0, q0 = self._joint_start
                s = min(1.0, (now - t0) / self.joint_motion_s)
                self.q = q0 + s * (self._joint_goal - q0)
                if s >= 1.0:
                    self._mode = None
        self.gripper_width = 0.0 if self.gripper_closed else 0.08

    # -- Arm protocol --------------------------------------------------------------------------

    def read(self) -> ArmState:
        self._advance()
        return ArmState(
            t=self._t,
            q=self.q.copy(),
            dq=np.linalg.pinv(FAKE_JACOBIAN) @ self.v,
            O_T_EE=self.T.copy(),
            jacobian=FAKE_JACOBIAN.copy(),
            errors=self.errors,
            robot_mode="Reflex" if self.errors else ("Move" if self._mode else "Idle"),
            in_control=self._mode is not None,
            gripper_width=self.gripper_width,
        )

    def jacobian_at(self, q: np.ndarray) -> np.ndarray:
        return FAKE_JACOBIAN.copy()

    def send_twist(self, twist: np.ndarray, hold_ms: int) -> None:
        self._advance()
        self.calls.append(("send_twist", np.array(twist, dtype=float)))
        if self.errors:
            raise ArmError(f"robot has errors: {self.errors}")
        if self._mode == "joint":
            raise ArmError("motion type change while moving")
        self._mode = "velocity"
        self._target = np.array(twist, dtype=float)
        self._hold_until = self._t + hold_ms / 1000.0

    def halt(self) -> None:
        self._advance()
        self.calls.append(("halt", None))
        if self._mode == "velocity":
            self._target = np.zeros(6)
        elif self._mode == "joint" and self._joint_stop_until is None:
            self._joint_stop_until = self._t + self.joint_stop_s

    def stop(self) -> None:
        self._advance()
        self.calls.append(("stop", None))
        self._mode = None
        self.v[:] = 0.0

    def move_joints(self, q: Sequence[float]) -> None:
        self._advance()
        self.calls.append(("move_joints", np.array(q, dtype=float)))
        if self.errors:
            raise ArmError(f"robot has errors: {self.errors}")
        if self._mode == "velocity":
            raise ArmError("motion type change while moving")
        self._mode = "joint"
        self._joint_stop_until = None
        self._joint_goal = np.array(q, dtype=float)
        self._joint_start = (self._t, self.q.copy())

    def motion_finished(self) -> bool:
        self._advance()
        if self.errors:
            raise ArmError(f"motion aborted: {self.errors}")
        return self._mode is None

    def recover(self) -> bool:
        self.calls.append(("recover", None))
        if self.recover_ok:
            self.errors = ()
        return self.recover_ok

    def set_gripper(self, closed: bool) -> None:
        self.calls.append(("set_gripper", closed))
        self.gripper_closed = closed

    def close(self) -> None:
        self.calls.append(("close", None))

    # -- test hooks ----------------------------------------------------------------------------

    def inject_error(self, name: str = "cartesian_reflex") -> None:
        """Simulate a reflex: the motion aborts and errors latch until recover()."""
        self.errors = (name,)
        self._mode = None
        self.v[:] = 0.0
