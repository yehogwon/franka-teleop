"""Pure functions turning operator intent into a safe base-frame twist.

Pipeline (TeleopNode applies them in this order):
  shape_twist        normalized intent -> twist [m/s, rad/s]: speed level, operator yaw, frame
  limit_workspace    outward EE velocity ramps to zero at the walls of the workspace box
  limit_singularity  motion towards a kinematic singularity (e.g. the elbow straightening at the
                     edge of reach) slows and stops before Cartesian tracking breaks down
  limit_joints       scale the twist so no joint approaches its limit or moves too fast, using
                     the Jacobian to predict joint velocities
All limiters scale or clip only what moves *towards* a boundary, so the operator can always back
out of a limited configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..config import JointGuardConfig, TeleopConfig, WorkspaceConfig
from ..geometry import rot_z


def shape_twist(
    linear: Sequence[float],
    angular: Sequence[float],
    speed_scale: float,
    frame: str,
    ee_rot: np.ndarray,
    cfg: TeleopConfig,
) -> np.ndarray:
    """Normalized operator intent -> base-frame twist (vx, vy, vz, wx, wy, wz).

    In "base" mode the operator frame is the robot base frame rotated by operator_yaw_deg about z.
    In "tool" mode (forward, left, up) are the end effector's (+x, +y, +z) axes, +z being the
    approach direction out of the fingers, and rotations are about the tool axes.
    """
    lin = np.clip(np.asarray(linear, dtype=float), -1.0, 1.0) * cfg.max_linear_speed * speed_scale
    ang = np.clip(np.asarray(angular, dtype=float), -1.0, 1.0) * cfg.max_angular_speed * speed_scale
    R = rot_z(np.deg2rad(cfg.operator_yaw_deg)) if frame == "base" else np.asarray(ee_rot)
    return np.concatenate([R @ lin, R @ ang])


def limit_workspace(
    twist: np.ndarray, ee_pos: np.ndarray, cfg: WorkspaceConfig, max_linear_speed: float
) -> tuple[np.ndarray, bool]:
    """Clamp outward linear velocity to gain * distance-to-wall, gain = max speed / slow_zone.

    Inward motion is never restricted, and outside the box outward motion is blocked without
    pulling the arm back (no motion the operator did not ask for).
    """
    out = np.array(twist, dtype=float)
    gain = max_linear_speed / cfg.slow_zone
    limited = False
    for i in range(3):
        v = out[i]
        if v > 0:
            cap = gain * max(0.0, cfg.upper[i] - ee_pos[i])
            if v > cap:
                out[i], limited = cap, True
        elif v < 0:
            cap = gain * max(0.0, ee_pos[i] - cfg.lower[i])
            if -v > cap:
                out[i], limited = -cap, True
    return out, limited


def predict_joint_velocity(twist: np.ndarray, jacobian: np.ndarray, damping: float) -> np.ndarray:
    """Damped least-squares joint velocity that realizes `twist`."""
    J = np.asarray(jacobian, dtype=float)
    JJt = J @ J.T + (damping**2) * np.eye(J.shape[0])
    return J.T @ np.linalg.solve(JJt, np.asarray(twist, dtype=float))


def min_singular_value(jacobian: np.ndarray, rot_weight: float) -> float:
    """Smallest singular value of the Jacobian with rotational rows weighted by rot_weight [m]."""
    W = np.diag([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    return float(np.linalg.svd(W @ np.asarray(jacobian), compute_uv=False)[-1])


def limit_singularity(
    twist: np.ndarray,
    q: np.ndarray,
    jacobian: np.ndarray,
    jacobian_at: Callable[[np.ndarray], np.ndarray],
    cfg: JointGuardConfig,
) -> tuple[np.ndarray, bool]:
    """Scale the twist to zero between sigma_slow and sigma_stop, but only if it heads into the
    singularity: sigma is probed a small joint-space step (probe_rad) along the predicted joint
    velocity, i.e. its directional derivative."""
    sigma = min_singular_value(jacobian, cfg.rot_weight)
    if sigma >= cfg.sigma_slow or not np.any(twist):
        return np.asarray(twist, dtype=float), False
    dq = predict_joint_velocity(twist, jacobian, cfg.damping)
    peak = float(np.max(np.abs(dq)))
    if peak < 1e-12:
        return np.asarray(twist, dtype=float), False
    step = dq * (cfg.probe_rad / peak)
    sigma_next = min_singular_value(jacobian_at(np.asarray(q) + step), cfg.rot_weight)
    if sigma_next >= sigma:
        return np.asarray(twist, dtype=float), False
    scale = float(np.clip((sigma - cfg.sigma_stop) / (cfg.sigma_slow - cfg.sigma_stop), 0.0, 1.0))
    return np.asarray(twist, dtype=float) * scale, True


def limit_joints(
    twist: np.ndarray,
    q: np.ndarray,
    jacobian: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    cfg: JointGuardConfig,
    dq_measured: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Uniformly scale the twist so that

    - a joint moving towards a limit does so at most
      max_joint_speed * (distance - clearance) / margin, i.e. it slows down over the last
      `margin` radians and stops `clearance` before the limit;
    - no joint exceeds max_joint_speed.

    The joint speed towards a limit is the larger of the prediction (damped pseudo-inverse) and
    the measured speed, because the robot's own inverse kinematics may move a joint faster than
    the prediction. Uniform scaling keeps the direction the operator asked for; twists that move
    joints away from their limits are unaffected.
    """
    dq = predict_joint_velocity(twist, jacobian, cfg.damping)
    scale, flags = 1.0, []
    lower, upper = np.asarray(lower), np.asarray(upper)
    for i, v in enumerate(dq):
        if abs(v) < 1e-9:
            continue
        speed = abs(v)
        if dq_measured is not None and np.sign(dq_measured[i]) == np.sign(v):
            speed = max(speed, abs(dq_measured[i]))
        dist = (upper[i] - q[i]) if v > 0 else (q[i] - lower[i])
        allowed = cfg.max_joint_speed * float(np.clip((dist - cfg.clearance) / cfg.margin, 0, 1))
        if speed > allowed:
            scale = min(scale, allowed / speed)
            if allowed < cfg.max_joint_speed and "joint" not in flags:
                flags.append("joint")
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > cfg.max_joint_speed:
        s = cfg.max_joint_speed / peak
        if s < scale:
            scale = s
        if "singularity" not in flags:
            flags.append("singularity")
    if scale >= 1.0:
        return np.array(twist, dtype=float), ()
    return np.asarray(twist, dtype=float) * scale, tuple(flags)
