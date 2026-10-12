"""Arm implementation over franky (libfranka). The same class drives the real arm and
franky-sim; only the host differs."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence

import franky
import numpy as np

from ..config import RobotConfig
from .base import ArmDisconnected, ArmError, ArmState

log = logging.getLogger(__name__)

# Names of the boolean fields of franky.Errors. `bool(errors)` is always True, so errors must be
# checked field by field.
ERROR_NAMES = tuple(
    n
    for n in dir(franky.Errors())
    if not n.startswith("_") and isinstance(getattr(franky.Errors(), n), bool)
)


def active_errors(errors: franky.Errors) -> tuple[str, ...]:
    return tuple(n for n in ERROR_NAMES if getattr(errors, n))


def _is_preemption(e: BaseException) -> bool:
    return "preempted" in str(e).lower()


def call[T](fn: Callable[[], T]) -> T:
    """Run a franky call and map its exceptions onto ArmError / ArmDisconnected."""
    try:
        return fn()
    except franky.NetworkException as e:
        raise ArmDisconnected(str(e)) from e
    except franky.InvalidMotionTypeException as e:
        raise ArmError(f"motion type change while moving: {e}") from e
    except franky.Exception as e:
        raise ArmError(str(e)) from e


class FrankyArm:
    def __init__(self, cfg: RobotConfig):
        realtime = (
            franky.RealtimeConfig.Enforce
            if cfg.realtime == "enforce"
            else franky.RealtimeConfig.Ignore
        )
        self.cfg = cfg
        self.robot = franky.Robot(cfg.host, realtime_config=realtime)
        self.robot.relative_dynamics_factor = cfg.dynamics_factor
        call(
            lambda: self.robot.set_collision_behavior(
                cfg.collision_torque_thresholds, cfg.collision_force_thresholds
            )
        )
        self._gripper: _GripperWorker | None = None
        if cfg.gripper:
            self._gripper = _GripperWorker(
                franky.Gripper(cfg.host), speed=cfg.gripper_speed, force=cfg.gripper_force
            )
            self._gripper.start()

    # -- state -------------------------------------------------------------------------------

    def read(self) -> ArmState:
        s = call(lambda: self.robot.state)
        jac = np.asarray(self.robot.model.zero_jacobian(franky.Frame.EndEffector, s), dtype=float)
        g = self._gripper
        return ArmState(
            t=s.time.to_sec(),
            q=np.asarray(s.q, dtype=float),
            dq=np.asarray(s.dq, dtype=float),
            O_T_EE=np.asarray(s.O_T_EE.matrix, dtype=float),
            jacobian=jac,
            errors=active_errors(s.current_errors),
            robot_mode=s.robot_mode.name,
            in_control=bool(self.robot.is_in_control),
            gripper_width=g.width if g else math.nan,
            gripper_busy=g.busy if g else False,
            extra={"gripper_error": g.last_error if g else ""},
        )

    # -- motion ------------------------------------------------------------------------------

    def send_twist(self, twist: np.ndarray, hold_ms: int) -> None:
        twist = np.asarray(twist, dtype=float)
        motion = franky.CartesianVelocityMotion(
            franky.RobotVelocity(franky.Twist(twist[:3], twist[3:])),
            duration=franky.Duration(int(hold_ms)),
        )
        call(lambda: self.robot.move(motion, asynchronous=True))

    def stop(self) -> None:
        call(self.robot.stop)
        try:
            call(lambda: self.robot.join_motion(timeout=1.0))
        except ArmError as e:
            if not _is_preemption(e.__cause__ or e):
                raise

    def move_joints(self, q: Sequence[float]) -> None:
        motion = franky.JointMotion(franky.JointState(list(map(float, q))))
        call(lambda: self.robot.move(motion, asynchronous=True))

    def motion_finished(self) -> bool:
        try:
            return bool(call(self.robot.poll_motion))
        except ArmError as e:
            if _is_preemption(e.__cause__ or e):
                return True
            raise

    def recover(self) -> bool:
        return bool(call(self.robot.recover_from_errors))

    # -- gripper -----------------------------------------------------------------------------

    def set_gripper(self, closed: bool) -> None:
        if self._gripper is not None:
            self._gripper.request(closed)

    def close(self) -> None:
        try:
            if self.robot.is_in_control:
                self.stop()
        except (ArmError, ArmDisconnected) as e:
            log.warning("stop on close failed: %s", e)
        if self._gripper is not None:
            self._gripper.shutdown()


class _GripperWorker(threading.Thread):
    """Owns the franky.Gripper. Gripper calls block for up to seconds, so they never run on the
    node's control thread; the newest request wins."""

    POLL_S = 0.1

    def __init__(self, gripper: franky.Gripper, speed: float, force: float):
        super().__init__(name="gripper", daemon=True)
        self._gripper = gripper
        self._speed = speed
        self._force = force
        self._cond = threading.Condition()
        self._pending: bool | None = None
        self._stopped = False
        self.width = math.nan
        self.busy = False
        self.last_error = ""

    def request(self, closed: bool) -> None:
        with self._cond:
            self._pending = closed
            self.busy = True
            self._cond.notify()

    def shutdown(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self.join(timeout=5.0)

    def run(self) -> None:
        while True:
            with self._cond:
                if self._pending is None and not self._stopped:
                    self._cond.wait(self.POLL_S)
                if self._stopped:
                    return
                closed, self._pending = self._pending, None
            try:
                if closed is True:
                    # Wide epsilons: succeed whether or not an object is between the fingers.
                    self._gripper.grasp(
                        0.0, self._speed, self._force, epsilon_inner=1.0, epsilon_outer=1.0
                    )
                elif closed is False:
                    self._gripper.move(self._gripper.max_width, self._speed)
                self.width = float(self._gripper.width)
                self.last_error = ""
            except franky.Exception as e:
                self.last_error = str(e)
                log.warning("gripper: %s", e)
            finally:
                with self._cond:
                    self.busy = self._pending is not None
