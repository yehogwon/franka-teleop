import mujoco
import numpy as np

from franka_teleop.config import READY_Q
from franka_teleop.sim.scene import (
    CUBES,
    MESHDIR_PLACEHOLDER,
    load_model,
    make_simulator,
    portable_xml,
    robot_qpos_addrs,
    scene_xml,
)


def test_tabletop_scene_loads_with_robot_and_cubes():
    sim, _ = make_simulator("tabletop", READY_Q)
    xml = portable_xml(sim)
    assert MESHDIR_PLACEHOLDER in xml
    model = load_model(xml)
    arm, fingers = robot_qpos_addrs(model)
    assert len(arm) == 7 and len(fingers) == 2
    assert model.nq == 9 + 7 * len(CUBES)
    for name, *_ in CUBES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def test_empty_scene_is_robot_only_and_xml_is_reproducible():
    xml = scene_xml("empty")
    assert xml == scene_xml("empty")
    model = load_model(xml)
    assert model.nq == 9


def test_viewer_model_matches_physics_model_kinematics():
    """The viewer's model (from portable XML) puts the hand where the physics model does."""
    sim, _ = make_simulator("tabletop", READY_Q)
    sim.start()
    try:
        physics_model, physics_data = sim.model, sim.data
        viewer_model = load_model(portable_xml(sim))
        viewer_data = mujoco.MjData(viewer_model)
        viewer_data.qpos[:] = physics_data.qpos
        mujoco.mj_kinematics(viewer_model, viewer_data)
        mujoco.mj_kinematics(physics_model, physics_data)
        assert np.allclose(viewer_data.xpos, physics_data.xpos, atol=1e-9)
    finally:
        sim.cleanup()
