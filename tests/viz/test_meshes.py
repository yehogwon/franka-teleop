"""Geometry for non-MuJoCo renderers, and the shared camera/overlay helpers."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from franka_teleop.config import READY_Q
from franka_teleop.protocol import NodeStatus
from franka_teleop.sim.scene import CUBES, load_model, scene_xml
from franka_teleop.viz import meshes
from franka_teleop.viz.overlay import CameraSpec, add_workspace, box_edges, status_lines


@pytest.fixture(scope="module")
def model():
    return load_model(scene_xml("tabletop"))


def signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@pytest.mark.parametrize(
    ("make", "volume"),
    [
        (lambda: meshes.box(np.array([0.1, 0.2, 0.3])), 8 * 0.1 * 0.2 * 0.3),
        (lambda: meshes.sphere(0.5), 4 / 3 * math.pi * 0.5**3),
        (lambda: meshes.cylinder(0.2, 0.4), math.pi * 0.2**2 * 0.8),
        (lambda: meshes.capsule(0.2, 0.4), math.pi * 0.2**2 * 0.8 + 4 / 3 * math.pi * 0.2**3),
    ],
    ids=["box", "sphere", "cylinder", "capsule"],
)
def test_primitives_are_closed_and_wound_outward(make, volume):
    v, f = make()
    assert f.min() >= 0 and f.max() < len(v)
    # Positive signed volume close to the analytic one: closed, counter-clockwise from outside.
    assert signed_volume(v, f) == pytest.approx(volume, rel=0.03)


def test_visual_geoms_are_the_ones_mujoco_shows(model):
    found, planes = meshes.model_meshes(model)
    ids = {m.geom_id for m in found}
    assert all(model.geom_group[g] < 3 for g in ids)
    # No collision geometry (franky-sim puts it in group 3)...
    assert not any(model.geom_group[g] == 3 for g in ids)
    # ...but the cubes are both visual and collision (group 0), and must be drawn.
    names = {m.name for m in found}
    assert {f"{name}_geom" for name, *_ in CUBES} <= names
    n_visual_meshes = sum(
        model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2
        for g in range(model.ngeom)
    )
    assert len(found) == n_visual_meshes + len(CUBES)
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in planes] == ["ground"]


def test_colors_follow_mujoco_material_rules(model):
    found, _ = meshes.model_meshes(model)
    by_name = {m.name: m for m in found}
    for name, _pos, rgba in CUBES:  # explicit geom rgba
        assert np.allclose(by_name[f"{name}_geom"].rgba, [float(x) for x in rgba.split()])
    # FR3 links use materials ("white", "black", ...) with the default geom rgba.
    link_colors = {tuple(np.round(m.rgba, 3)) for m in found if "link" in m.name}
    assert (1.0, 1.0, 1.0, 1.0) in link_colors and (0.2, 0.2, 0.2, 1.0) in link_colors


def test_mesh_vertices_are_in_the_geom_frame(model):
    """MuJoCo's own geom bounding boxes (geom frame) enclose exactly our vertices."""
    found, _ = meshes.model_meshes(model)
    for m in found:
        if model.geom_type[m.geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        center, half = model.geom_aabb[m.geom_id][:3], model.geom_aabb[m.geom_id][3:]
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        assert np.allclose(lo, center - half, atol=1e-4), m.name
        assert np.allclose(hi, center + half, atol=1e-4), m.name
        assert m.faces.max() < len(m.vertices)


def test_geom_poses_match_mujoco(model):
    data = mujoco.MjData(model)
    data.qpos[:7] = READY_Q
    mujoco.mj_kinematics(model, data)
    ids = np.arange(model.ngeom)
    pos, quat = meshes.geom_poses(data, ids)
    for g in ids:
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat[g])
        assert np.allclose(mat, data.geom_xmat[g], atol=1e-9)
    assert np.allclose(pos, data.geom_xpos)


def test_box_edges():
    edges = box_edges([0, 0, 0], [1, 2, 3])
    assert edges.shape == (12, 2, 3)
    lengths = sorted(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).round(9))
    assert lengths == [1.0] * 4 + [2.0] * 4 + [3.0] * 4


def test_workspace_overlay_is_appended_to_a_mujoco_scene(model):
    scn = mujoco.MjvScene(model, maxgeom=100)
    scn.ngeom = 5
    assert add_workspace(scn, [0.2, -0.4, 0.0], [0.8, 0.4, 0.7]) == 13
    assert scn.ngeom == 18
    assert scn.geoms[17].type == mujoco.mjtGeom.mjGEOM_BOX


def test_camera_spec_matches_mujoco_free_camera(model):
    """The web viewer places its camera with CameraSpec.position(); it must be MuJoCo's."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spec = CameraSpec(azimuth=120.0, elevation=-30.0, distance=2.0, lookat=(0.3, 0.1, 0.2))
    cam = mujoco.MjvCamera()
    spec.apply(cam)
    scn = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn
    )
    eye = (np.array(scn.camera[0].pos) + np.array(scn.camera[1].pos)) / 2
    assert np.allclose(eye, spec.position(), atol=1e-6)
    assert CameraSpec.parse("120,-30,2,0.3,0.1,0.2") == spec
    with pytest.raises(ValueError):
        CameraSpec.parse("1,2,3")


def test_status_lines():
    assert status_lines(None)[0][0] == "node"
    status = NodeStatus(
        seq=1,
        t=0.0,
        mode="teleop",
        sim=False,
        q=READY_Q,
        dq=(0.0,) * 7,
        ee_pos=(0.3, 0.0, 0.5),
        ee_quat=(0.0, 0.0, 0.0, 1.0),
        speed_level=1,
        speed_scale=0.5,
        frame="tool",
        limiters=("workspace",),
    )
    rows = dict(status_lines(status))
    assert rows["mode"] == "teleop" and rows["frame"] == "tool"
    assert rows["gripper"] == "n/a" and rows["limits"] == "workspace"
    assert rows["ee [m]"] == "+0.300 +0.000 +0.500"
