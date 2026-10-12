"""Things every viewer draws the same way: the default camera, the workspace box and the
NodeStatus summary. MuJoCo-side helpers work on an ``MjvScene`` (the passive viewer's
``user_scn`` or an offscreen renderer's scene)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from ..protocol import NodeStatus

WORKSPACE_RGBA = np.array([0.15, 0.75, 1.0, 1.0], dtype=np.float32)
WORKSPACE_FILL_RGBA = np.array([0.15, 0.75, 1.0, 0.07], dtype=np.float32)


@dataclass(frozen=True)
class CameraSpec:
    """A MuJoCo free camera: degrees, metres."""

    azimuth: float = 150.0  # in front of the robot, to its right (-y side)
    elevation: float = -20.0
    distance: float = 1.7
    lookat: tuple[float, float, float] = (0.35, 0.0, 0.38)

    @classmethod
    def parse(cls, text: str) -> CameraSpec:
        """'azimuth,elevation,distance,x,y,z'"""
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 6:
            raise ValueError("expected 6 comma-separated numbers: azimuth,elevation,distance,x,y,z")
        return cls(parts[0], parts[1], parts[2], (parts[3], parts[4], parts[5]))

    def apply(self, cam: mujoco.MjvCamera) -> None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        cam.distance = self.distance
        cam.lookat[:] = self.lookat

    def position(self) -> np.ndarray:
        """Eye position (MuJoCo convention: the camera looks along azimuth/elevation)."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return np.asarray(self.lookat) - self.distance * forward


def box_edges(lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """The 12 edges of an axis-aligned box, shape (12, 2, 3)."""
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)])
    edges = [(i, i | (1 << k)) for i in range(8) for k in range(3) if not i & (1 << k)]
    return np.array([[corners[a], corners[b]] for a, b in edges])


def add_workspace(scn: mujoco.MjvScene, lower: Sequence[float], upper: Sequence[float]) -> int:
    """Append the workspace box (12 lines and a faint fill) to ``scn``. Returns #geoms added."""
    added = 0
    eye9 = np.eye(3).ravel()
    for a, b in box_edges(lower, upper):
        if scn.ngeom >= scn.maxgeom:
            return added
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), a, eye9, WORKSPACE_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, a, b)
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1
        added += 1
    if scn.ngeom < scn.maxgeom:
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_BOX, (hi - lo) / 2, (hi + lo) / 2, eye9, WORKSPACE_FILL_RGBA
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1
        added += 1
    return added


def status_lines(status: NodeStatus | None) -> list[tuple[str, str]]:
    """(label, value) rows summarizing ft-node's state."""
    if status is None:
        return [("node", "no status (ft-node not running?)")]
    width = status.gripper_width
    gripper = "n/a" if not math.isfinite(width) else f"{1000 * width:.1f} mm"
    gripper += " (closing)" if status.gripper_closed else ""
    rows = [
        ("mode", status.mode + (" [sim]" if status.sim else "")),
        ("speed", f"level {status.speed_level + 1} ({100 * status.speed_scale:.0f} %)"),
        ("frame", status.frame),
        ("ee [m]", " ".join(f"{x:+.3f}" for x in status.ee_pos)),
        ("gripper", gripper),
    ]
    if status.limiters:
        rows.append(("limits", ", ".join(status.limiters)))
    if status.message:
        rows.append(("note", status.message))
    return rows
