"""Arm implementation over franky (libfranka). The same class drives the real arm and
franky-sim; only the host differs."""

from __future__ import annotations

import logging
import math
import threading
import time
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
    except Exception as e:
        # franky also raises classes outside franky.Exception (e.g. a plain RuntimeError when
        # its motion planner fails, GripperException, MotionReuseException). The node must see
        # them as arm errors rather than crash.
        raise ArmError(f"{type(e).__name__}: {e}") from e


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
        self._motion_kind: str | None = None  # "velocity" | "joint": type of the last motion
        self._F_T_EE = self._EE_T_K = None
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
        self._F_T_EE, self._EE_T_K = s.F_T_EE, s.EE_T_K
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

    def jacobian_at(self, q: np.ndarray) -> np.ndarray:
        if self._F_T_EE is None:
            self.read()
        jac = self.robot.model.zero_jacobian(
            franky.Frame.EndEffector, np.asarray(q, dtype=float), self._F_T_EE, self._EE_T_K
        )
        return np.asarray(jac, dtype=float)

    KEYPOINT_FRAMES = ("Joint4", "Joint5", "Joint6", "Joint7", "Flange")

    def keypoints_at(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._F_T_EE is None:
            self.read()
        q = np.asarray(q, dtype=float)
        model, out = self.robot.model, []
        for name in self.KEYPOINT_FRAMES:
            frame = getattr(franky.Frame, name)
            T = np.asarray(model.pose(frame, q, self._F_T_EE, self._EE_T_K).matrix, dtype=float)
            J = np.asarray(model.zero_jacobian(frame, q, self._F_T_EE, self._EE_T_K), dtype=float)
            out.append((T[:3, 3], J[:3]))
        return out

    # -- motion ------------------------------------------------------------------------------

    def send_twist(self, twist: np.ndarray, hold_ms: int) -> None:
        twist = np.asarray(twist, dtype=float)
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            raise ArmError(f"refusing to send non-finite or malformed twist {twist}")
        motion = franky.CartesianVelocityMotion(
            franky.RobotVelocity(franky.Twist(twist[:3], twist[3:])),
            duration=franky.Duration(int(hold_ms)),
        )
        call(lambda: self.robot.move(motion, asynchronous=True))
        self._motion_kind = "velocity"

    def halt(self) -> None:
        if not self.robot.is_in_control:
            return
        if self._motion_kind == "velocity":
            motion = franky.CartesianVelocityStopMotion()
        elif self._motion_kind == "joint":
            motion = franky.JointStopMotion()
        else:
            return
        try:
            call(lambda: self.robot.move(motion, asynchronous=True))
        except ArmError as e:
            # The motion may have finished between the check and the call.
            if self.robot.is_in_control:
                raise
            log.debug("halt after motion end: %s", e)

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
        self._motion_kind = "joint"

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

    def close(self, timeout_s: float = 3.0) -> None:
        """Release the robot. A running motion is first halted and allowed to come to rest;
        only if that fails is it aborted with stop()."""
        try:
            if self.robot.is_in_control:
                self.halt()
                deadline = time.monotonic() + timeout_s
                while self.robot.is_in_control and time.monotonic() < deadline:
                    time.sleep(0.01)
            if self.robot.is_in_control:
                log.warning("motion still running after halt; stopping it")
                self.stop()
        except (ArmError, ArmDisconnected) as e:
            log.warning("halt/stop on close failed: %s", e)
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
                if closed is not None:
                    self.last_error = ""  # an error stays visible until a command succeeds
                self.width = float(self._gripper.width)
            except Exception as e:  # keep serving: a dead worker would drop every later toggle
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("gripper: %s", self.last_error)
            finally:
                with self._cond:
                    self.busy = self._pending is not None
