"""ft-view --mode web: viser server, mesh handles posed like MuJoCo's geoms."""

from __future__ import annotations

import time
import urllib.request

import mujoco
import numpy as np
import pytest

from franka_teleop.config import READY_Q
from franka_teleop.protocol import NodeStatus
from franka_teleop.sim.scene import load_model, robot_qpos_addrs, scene_xml
from franka_teleop.viz.overlay import STALE_AFTER_S, CameraSpec
from franka_teleop.viz.source import Source
from franka_teleop.viz.web import WebViewer

from ..conftest import free_port, make_test_config
from ..sim.harness import Proc, script


class ManualSource(Source):
    """A source whose qpos the test sets."""

    kind = "manual"

    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self._dirty = False

    def describe(self) -> str:
        return "test"

    def set(self, qpos: np.ndarray) -> None:
        self.qpos[:] = qpos
        self._dirty = True

    def _poll_state(self, status) -> bool:
        dirty, self._dirty = self._dirty, False
        return dirty


def http_get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode(errors="replace")


@pytest.fixture
def viewer():
    cfg = make_test_config()
    src = ManualSource(cfg, load_model(scene_xml("tabletop")))
    v = WebViewer(
        src, "127.0.0.1", free_port(), CameraSpec(), (cfg.workspace.lower, cfg.workspace.upper)
    )
    yield v
    v.close()
    src.close()


def test_serves_the_viewer_page(viewer):
    status, body = http_get(f"http://127.0.0.1:{viewer.port}/")
    assert status == 200 and "<html" in body.lower()


def test_mesh_poses_equal_mujoco_geom_poses_after_an_update(viewer):
    model = viewer.model
    arm, fingers = robot_qpos_addrs(model)
    qpos = model.qpos0.copy()
    qpos[arm] = np.add(READY_Q, [0.4, 0.3, -0.2, 0.3, 0.5, -0.4, 0.6])
    qpos[fingers] = 0.035
    # Move and tilt the red cube (free joint: xyz, then a wxyz quaternion).
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint")
    adr = model.jnt_qposadr[jid]
    qpos[adr : adr + 7] = [0.4, 0.1, 0.2, np.cos(0.3), np.sin(0.3), 0.0, 0.0]
    viewer.src.set(qpos)
    assert viewer.update()

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    assert len(viewer.handles) == len(viewer.geom_ids) > 30
    for h, g in zip(viewer.handles, viewer.geom_ids, strict=True):
        assert np.allclose(h.position, data.geom_xpos[g], atol=1e-6)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, np.asarray(h.wxyz, dtype=float))
        assert np.allclose(mat, data.geom_xmat[g], atol=1e-6)
    # Nothing new: no update.
    assert not viewer.update()


def test_panel_marks_stale_state_and_status(viewer):
    """The panel must not say "live" forever: state and NodeStatus go stale after ~1 s."""
    src = viewer.src
    assert "waiting for state" in viewer._source_text()
    src.set(src.qpos)
    viewer.update()
    assert viewer._source_text().endswith("— live")
    src.t_state -= STALE_AFTER_S + 1.0
    assert "**STALE: no update for 2 s**" in viewer._source_text()
    assert "no status (ft-node not running?)" in viewer._status_text()
    src.status = NodeStatus(
        seq=0,
        t=0.0,
        mode="teleop",
        sim=True,
        q=(0.0,) * 7,
        dq=(0.0,) * 7,
        ee_pos=(0.3, 0.0, 0.5),
        ee_quat=(1.0, 0.0, 0.0, 0.0),
        speed_level=1,
        speed_scale=0.5,
        frame="base",
    )
    src.t_status = time.monotonic()
    text = viewer._status_text()
    assert "STALE" not in text and "| mode | teleop [sim] |" in text
    src.t_status -= STALE_AFTER_S + 1.0
    assert "| node | STALE: no status for 2 s (last known below) |" in viewer._status_text()
    src.set(src.qpos)  # a new state: live again
    viewer.update()
    assert viewer._source_text().endswith("— live")


@pytest.mark.sim
@pytest.mark.e2e
def test_ft_view_web_runs_against_ft_sim(viz_sim, viz_config_path):
    port = free_port()
    p = Proc(
        [
            script("ft-view"),
            "--config",
            viz_config_path,
            "--mode",
            "web",
            "--web-port",
            str(port),
            "--duration",
            "4",
        ]
    )
    try:
        line = p.wait_for_line("ft-view: web viewer on", 60.0)
        assert f"http://127.0.0.1:{port}" in line and "ft-sim scene" in line
        assert http_get(f"http://127.0.0.1:{port}/")[0] == 200
        assert p.wait(30.0) == 0, p.output
    finally:
        p.kill()
    assert "Traceback" not in p.output
