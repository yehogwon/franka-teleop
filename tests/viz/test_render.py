"""ft-view --mode snapshot / record: offscreen MuJoCo rendering (EGL when headless)."""

from __future__ import annotations

import math
import os
import subprocess

import imageio.v3 as iio
import imageio_ffmpeg
import mujoco
import numpy as np
import pytest

from franka_teleop.sim.scene import ROBOT_PREFIX, load_model
from franka_teleop.viz.overlay import CameraSpec

from ..sim.harness import request_scene_info, script

pytestmark = [pytest.mark.sim, pytest.mark.e2e, pytest.mark.display]


def headless_env() -> dict[str, str]:
    """No display and no GL preference: ft-view must pick EGL by itself."""
    drop = {"DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM"}
    return {k: v for k, v in os.environ.items() if k not in drop}


def ft_view(config: str, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [script("ft-view"), "--config", config, *args],
        env=headless_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def project(spec: CameraSpec, fovy_deg: float, size: tuple[int, int], p: np.ndarray):
    """Pixel (u, v) of a world point seen by a MuJoCo free camera."""
    az, el = math.radians(spec.azimuth), math.radians(spec.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    right = np.cross(fwd, up)
    d = p - spec.position()
    w, h = size
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return w / 2 + f * (d @ right) / (d @ fwd), h / 2 - f * (d @ up) / (d @ fwd)


def robot_pixels(img: np.ndarray) -> np.ndarray:
    """The FR3 is white/grey; the ground is blue, the sky black, the workspace box cyan."""
    rgb = img[..., :3].astype(int)
    return (rgb.min(-1) > 120) & (rgb.max(-1) - rgb.min(-1) < 30)


def test_snapshot_shows_the_robot(viz_sim, viz_config_path, tmp_path):
    out = tmp_path / "snap.png"
    size = (480, 320)
    r = ft_view(viz_config_path, "--mode", "snapshot", "--out", str(out), "--size", "480x320")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Traceback" not in r.stderr  # renderer closed explicitly (no EGL error at exit)
    img = iio.imread(out)
    assert img.shape[:2] == (size[1], size[0])
    assert img[..., :3].std() > 20  # not blank

    mask = robot_pixels(img)
    assert mask.mean() > 0.005
    # The robot is where MuJoCo's kinematics put it: white pixels around the projected hand.
    model = load_model(request_scene_info(viz_sim.cfg).xml)
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_PREFIX + "fr3_joint1")
    data.qpos[model.jnt_qposadr[jid] : model.jnt_qposadr[jid] + 7] = viz_sim.cfg.sim.initial_q
    mujoco.mj_kinematics(model, data)
    for body in ("hand", "fr3_link1"):
        u, v = project(
            CameraSpec(), model.vis.global_.fovy, size, data.body(ROBOT_PREFIX + body).xpos
        )
        u, v = round(u), round(v)
        assert 0 <= u < size[0] and 0 <= v < size[1], (body, u, v)
        window = mask[max(v - 15, 0) : v + 15, max(u - 15, 0) : u + 15]
        assert window.sum() > 20, (body, u, v)
    # Blank corners: the empty half of the image has no robot.
    assert mask[:, : size[0] // 6].sum() < 10


def test_record_writes_a_playable_mp4(viz_sim, viz_config_path, tmp_path):
    out = tmp_path / "run.mp4"
    duration, fps = 1.5, viz_sim.cfg.viewer.fps
    r = ft_view(
        viz_config_path,
        "--mode",
        "record",
        "--out",
        str(out),
        "--duration",
        str(duration),
        "--size",
        "320x240",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Traceback" not in r.stderr
    n_frames, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    expected = round(duration * fps)
    assert abs(n_frames - expected) <= 0.1 * expected, (n_frames, expected)
    assert secs == pytest.approx(duration, rel=0.1)
    first = iio.imread(out, index=0)
    assert first.shape == (240, 320, 3)
    assert robot_pixels(first).mean() > 0.005


def test_snapshot_without_a_state_fails_clearly(tmp_path, config_file):
    from ..conftest import make_test_config

    path = config_file(make_test_config())  # nobody publishes on these ports
    r = ft_view(
        str(path), "--mode", "snapshot", "--source", "status", "--wait", "0.5", "--out", "x.png"
    )
    assert r.returncode == 1
    assert "no state from ft-node status" in r.stderr
