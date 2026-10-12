"""The interface the teleop node drives. FrankyArm talks libfranka (real arm or franky-sim);
FakeArm is a kinematic stand-in for unit tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ArmState:
    t: float
    """Robot time [s]."""
    q: np.ndarray
    dq: np.ndarray
    O_T_EE: np.ndarray
    """4x4 end-effector pose in the robot base frame."""
    jacobian: np.ndarray
    """6x7 zero (base-frame) Jacobian of the end effector."""
    errors: tuple[str, ...] = ()
    """Names of active robot errors (reflexes etc.)."""
    robot_mode: str = "Idle"
    in_control: bool = False
    """A motion is currently executing."""
    gripper_width: float = math.nan
    gripper_busy: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def ee_pos(self) -> np.ndarray:
        return self.O_T_EE[:3, 3]

    @property
    def ee_rot(self) -> np.ndarray:
        return self.O_T_EE[:3, :3]


class ArmError(RuntimeError):
    """The robot rejected or aborted a motion (reflex, limit violation, ...). Needs recovery."""


class ArmDisconnected(RuntimeError):
    """The connection to the robot is gone. Not recoverable in-process."""


class Arm(Protocol):
    def read(self) -> ArmState:
        """Current state. One libfranka state read (~1 ms)."""
        ...

    def jacobian_at(self, q: np.ndarray) -> np.ndarray:
        """6x7 zero Jacobian of the end effector at configuration q (the robot's own model)."""
        ...

    def keypoints_at(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """(position, 3x7 translational Jacobian) of the elbow/wrist joint frames and the flange
        at configuration q, for keeping the arm's links above the table."""
        ...

    def send_twist(self, twist: np.ndarray, hold_ms: int) -> None:
        """Stream a base-frame twist (vx, vy, vz, wx, wy, wz). Preempts the previous velocity
        target; the robot holds it for hold_ms and then decelerates to rest by itself."""
        ...

    def halt(self) -> None:
        """Decelerate the current motion to rest within the dynamics limits (non-blocking).
        Uses a stop motion of the running motion's type, so it is legal mid-motion."""
        ...

    def stop(self) -> None:
        """Abort the current motion immediately. Swallows the expected 'preempted' error.
        Prefer halt(); this is for shutdown."""
        ...

    def move_joints(self, q: Sequence[float]) -> None:
        """Start an asynchronous joint motion. The robot must not be in a velocity motion."""
        ...

    def motion_finished(self) -> bool:
        """True when no motion is executing. Raises ArmError if the last motion failed."""
        ...

    def recover(self) -> bool:
        """Automatic error recovery. True on success."""
        ...

    def set_gripper(self, closed: bool) -> None:
        """Request gripper closed (grasp) or open. Non-blocking; the latest request wins."""
        ...

    def close(self) -> None: ...
