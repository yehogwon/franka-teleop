"""The teleop state machine. One `tick()` per control period: take the newest operator command,
read the arm once, decide, command the arm, and report a NodeStatus.

Safety rules (all enforced here, independent of the input device):

- Motion and gripper commands need the deadman held in a *fresh* command (younger than
  teleop.cmd_timeout_ms). A stale command stream is treated as a released deadman.
- After connecting, error recovery, unlocking or homing, the deadman must be seen released once
  before anything moves again ("armed"), so a stuck button or a deflected stick cannot start
  motion by surprise.
- Releasing the deadman, a stale stream, or the stop button decelerates the arm with a stop
  motion (dynamics-limited), never by cutting the control loop.
- franky additionally holds each velocity target for only teleop.hold_ms, so the arm comes to rest
  by itself even if this process dies.
- Robot errors (reflexes) latch the "error" mode until the operator presses recover.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from ..arm.base import Arm, ArmError, ArmState
from ..config import Config
from ..geometry import quat_from_mat
from ..protocol import ACTIONS, NodeStatus, TeleopCommand
from .limits import (
    hand_keypoints,
    limit_floor,
    limit_joints,
    limit_singularity,
    limit_workspace,
    predict_joint_velocity,
    shape_twist,
)

IDLE, TELEOP, HOMING, ERROR, LOCKED = "idle", "teleop", "homing", "error", "locked"

# A stream that was stale for longer than this must see a deadman release before motion resumes.
# Shorter dropouts (a few lost packets) resume while the operator keeps holding the deadman.
REARM_AFTER_STALE_MS = 1000.0
STALE_MESSAGE = "operator stream stale"


class TeleopNode:
    def __init__(self, arm: Arm, cfg: Config, clock: Callable[[], float] = time.monotonic):
        self.arm = arm
        self.cfg = cfg
        self.clock = clock
        self.sim = cfg.sim.enabled
        self.allowed_sources = cfg.allowed_sources()
        self.mode = IDLE
        self.message = "waiting for operator"
        self.speed_level = cfg.teleop.initial_speed_level
        self.frame = cfg.teleop.default_frame
        self.gripper_closed: bool | None = None
        self.gripper_error = ""
        self.session = 0
        self.counters: dict[str, int] = {}
        self.last_seq = -1
        self.last_cmd: TeleopCommand | None = None
        self.last_cmd_time = -math.inf
        self.armed = False
        self.moving = False  # a velocity motion we started may still be running
        self.sent_nonzero = False  # the last twist sent was non-zero
        self.homing_started = False
        self.joint_motion = False  # a joint (or joint stop) motion may still be running
        self.twist = np.zeros(6)
        self.limiters: tuple[str, ...] = ()
        self.status_seq = 0
        self.loop_overruns = 0
        self.state: ArmState | None = None

    # -- command intake --------------------------------------------------------------------------

    def _accept(self, cmd: TeleopCommand, now: float) -> set[str]:
        """Validate a command; return the discrete actions it triggers."""
        if cmd.source not in self.allowed_sources:
            self.message = f"ignored command from source {cmd.source!r} (not allowed here)"
            return set()
        if not all(math.isfinite(x) for x in (*cmd.linear, *cmd.angular)):
            # Never let NaN/inf reach the limiters (NaN comparisons are all False) or franky.
            # The command is dropped, so the stream goes stale and the arm halts.
            self.message = "ignored command with non-finite intent"
            return set()
        counters = cmd.counters()
        if cmd.session != self.session:
            # New client: adopt its counters as the baseline and require a deadman release.
            self.session, self.counters, self.last_seq = cmd.session, counters, cmd.seq
            self.armed = False
            self.message = f"operator session {cmd.session} connected"
            self.last_cmd, self.last_cmd_time = cmd, now
            return set()
        if cmd.seq <= self.last_seq:
            return set()  # duplicate or reordered
        events = {a for a in ACTIONS if counters[a] > self.counters.get(a, 0)}
        self.counters, self.last_seq = counters, cmd.seq
        self.last_cmd, self.last_cmd_time = cmd, now
        return events

    def cmd_age_ms(self, now: float) -> float:
        return (now - self.last_cmd_time) * 1000.0

    # -- main step -------------------------------------------------------------------------------

    def tick(self, cmd: TeleopCommand | None) -> NodeStatus:
        now = self.clock()
        events = self._accept(cmd, now) if cmd is not None else set()
        try:
            self.state = s = self.arm.read()
            age = self.cmd_age_ms(now)
            fresh = self.last_cmd is not None and age <= self.cfg.teleop.cmd_timeout_ms
            deadman = fresh and self.last_cmd.deadman
            if age > REARM_AFTER_STALE_MS:
                self.armed = False
            if fresh and self.message == STALE_MESSAGE:
                self.message = "operator stream resumed"
            if not s.in_control:
                self.moving = self.sent_nonzero = self.joint_motion = False

            gripper_error = s.extra.get("gripper_error", "")
            if gripper_error and gripper_error != self.gripper_error:
                self.message = f"gripper error: {gripper_error}"
            self.gripper_error = gripper_error
            if (s.errors or s.robot_mode == "Reflex") and self.mode != ERROR:
                self._enter_error(f"robot error: {', '.join(s.errors) or s.robot_mode}")
            self._handle_events(events if fresh else set(), deadman, s)
            # Arm on an observed release; after the events, so that a release sent together
            # with "recover" counts.
            if fresh and not self.last_cmd.deadman and self.mode not in (ERROR, LOCKED):
                self.armed = True

            if self.mode == HOMING:
                self._step_homing(deadman, s)
            elif self.mode in (IDLE, TELEOP):
                self._step_teleop(deadman, fresh, s)
            else:
                self.twist, self.limiters = np.zeros(6), ()
        except ArmError as e:
            self._enter_error(f"arm error: {e}")
        return self._status(now)

    def _handle_events(self, events: set[str], deadman: bool, s: ArmState) -> None:
        if "stop" in events and self.mode != LOCKED:
            self._halt()
            if self.mode != ERROR:
                self.mode, self.message = LOCKED, "locked by operator (press recover to unlock)"
            self.armed = False
        if "recover" in events and "stop" not in events:  # both at once: stop wins
            if self.mode == ERROR:
                ok = self.arm.recover()
                errors = self.arm.read().errors if ok else ("recovery failed",)
                if ok and not errors:
                    self.mode, self.message = IDLE, "recovered; release the deadman to continue"
                else:
                    self.message = f"recovery failed: {', '.join(errors)}"
                self.armed = False
            elif self.mode == LOCKED:
                self.mode, self.message = IDLE, "unlocked; release the deadman to continue"
                self.armed = False
        if "speed_up" in events:
            self.speed_level = min(self.speed_level + 1, len(self.cfg.teleop.speed_levels) - 1)
        if "speed_down" in events:
            self.speed_level = max(self.speed_level - 1, 0)
        if "frame" in events:
            if self.sent_nonzero:
                # Switching frames mid-motion can reverse the direction the stick commands.
                self.message = "release the sticks to switch frames"
            else:
                self.frame = "tool" if self.frame == "base" else "base"
                self.message = f"{self.frame} frame"
        may_act = deadman and self.armed and self.mode in (IDLE, TELEOP)
        if "gripper" in events:
            if may_act and self.cfg.robot.gripper:
                closed = self.gripper_closed
                if closed is None:
                    closed = s.gripper_width < 0.5 * 0.08
                self.gripper_closed = not closed
                self.arm.set_gripper(self.gripper_closed)
                self.message = "gripper closing" if self.gripper_closed else "gripper opening"
            elif not may_act:
                self.message = "hold the deadman to use the gripper"
        if "home" in events:
            if may_act:
                self.mode, self.homing_started = HOMING, False
                self.message = "homing (keep holding the deadman)"
            else:
                self.message = "hold the deadman to home"

    def _step_teleop(self, deadman: bool, fresh: bool, s: ArmState) -> None:
        if not (deadman and self.armed):
            if self.moving:
                self._halt()
            self.mode, self.twist, self.limiters = IDLE, np.zeros(6), ()
            if not fresh and self.last_cmd is not None:
                self.message = STALE_MESSAGE
            return
        if self.joint_motion:
            # A joint motion (e.g. an aborted homing's stop motion) is still decelerating, and
            # franky cannot switch motion types mid-motion: wait for it to end.
            self.mode, self.twist, self.limiters = IDLE, np.zeros(6), ()
            self.message = "waiting for the arm to stop"
            return
        cmd = self.last_cmd
        tc = self.cfg.teleop
        scale = tc.speed_levels[self.speed_level]
        twist = shape_twist(cmd.linear, cmd.angular, scale, self.frame, s.ee_rot, tc)
        twist, ws_limited = limit_workspace(
            twist, s.ee_pos, self.cfg.workspace, tc.max_linear_speed
        )
        ws = self.cfg.workspace
        keypoints = [(p, J, ws.link_clearance) for p, J in self.arm.keypoints_at(s.q)]
        keypoints += [(p, J, ws.hand_clearance) for p, J in hand_keypoints(s.O_T_EE, s.jacobian)]
        dq = predict_joint_velocity(twist, s.jacobian, self.cfg.joint_guard.damping)
        twist, floor_limited = limit_floor(
            twist, dq, keypoints, ws.floor, ws.slow_zone, tc.max_linear_speed
        )
        twist, sing_limited = limit_singularity(
            twist, s.q, s.jacobian, self.arm.jacobian_at, self.cfg.joint_guard
        )
        twist, joint_flags = limit_joints(
            twist,
            s.q,
            s.jacobian,
            self.cfg.robot.joint_lower,
            self.cfg.robot.joint_upper,
            self.cfg.joint_guard,
            dq_measured=s.dq,
        )
        flags = tuple(
            name
            for name, on in (
                ("workspace", ws_limited),
                ("floor", floor_limited),
                ("singularity", sing_limited),
            )
            if on
        )
        self.limiters = flags + tuple(f for f in joint_flags if f not in flags)
        self.twist = twist
        self.mode = TELEOP
        nonzero = bool(np.any(np.abs(twist) > 1e-6))
        # Send non-zero targets every tick; send zero once to brake, then let the hold expire.
        if nonzero or self.sent_nonzero:
            self.arm.send_twist(twist, self.cfg.teleop.hold_ms)
            self.moving, self.sent_nonzero = True, nonzero

    def _step_homing(self, deadman: bool, s: ArmState) -> None:
        self.twist, self.limiters = np.zeros(6), ()
        if not (deadman and self.armed):
            self._halt()
            self.mode, self.message, self.armed = IDLE, "homing aborted: deadman released", False
            return
        if not self.homing_started:
            if s.in_control:
                if self.sent_nonzero:
                    self.arm.halt()
                    self.sent_nonzero = False
                return  # wait for the velocity motion to come to rest
            self.arm.move_joints(self.cfg.robot.home_q)
            self.homing_started, self.moving, self.joint_motion = True, False, True
            return
        if self.arm.motion_finished():
            self.mode, self.message, self.armed = IDLE, "home reached; release the deadman", False

    def _halt(self) -> None:
        self.arm.halt()
        self.moving = self.sent_nonzero = False
        self.twist = np.zeros(6)

    def _enter_error(self, message: str) -> None:
        self.mode, self.message, self.armed = ERROR, message, False
        self.moving = self.sent_nonzero = False
        self.twist, self.limiters = np.zeros(6), ()

    # -- reporting -------------------------------------------------------------------------------

    def _status(self, now: float) -> NodeStatus:
        s = self.state
        self.status_seq += 1
        if s is None:
            q = dq = (math.nan,) * 7
            pos, quat, width, errors = (math.nan,) * 3, (math.nan,) * 4, math.nan, ""
        else:
            q, dq = tuple(map(float, s.q)), tuple(map(float, s.dq))
            pos = tuple(map(float, s.ee_pos))
            quat = tuple(map(float, quat_from_mat(s.ee_rot)))
            width, errors = float(s.gripper_width), ", ".join(s.errors)
        closed = self.gripper_closed
        if closed is None:
            closed = bool(width < 0.04) if not math.isnan(width) else False
        return NodeStatus(
            seq=self.status_seq,
            t=time.time(),
            mode=self.mode,
            message=self.message,
            sim=self.sim,
            q=q,
            dq=dq,
            ee_pos=pos,
            ee_quat=quat,
            gripper_width=width,
            gripper_closed=closed,
            speed_level=self.speed_level,
            speed_scale=self.cfg.teleop.speed_levels[self.speed_level],
            frame=self.frame,
            twist=tuple(map(float, self.twist)),
            limiters=self.limiters,
            session=self.session,
            last_cmd_seq=self.last_seq,
            cmd_age_ms=self.cmd_age_ms(now),
            robot_errors=errors,
            loop_overruns=self.loop_overruns,
        )

    def shutdown(self, timeout_s: float = 3.0) -> None:
        """Bring the arm to rest and release it. Whatever is running (teleop, homing, or a stop
        motion that is still braking) is halted and waited for, so close() never has to abort
        a motion mid-deceleration."""
        try:
            if self.arm.read().in_control:
                self.arm.halt()
                deadline = self.clock() + timeout_s
                while not self.arm.motion_finished() and self.clock() < deadline:
                    time.sleep(0.01)
        except ArmError:
            pass
        finally:
            self.arm.close()
