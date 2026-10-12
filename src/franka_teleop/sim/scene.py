"""MuJoCo scenes on top of franky-sim's FR3 model. Shared by ft-sim (physics) and ft-view
(rendering), so both always load the same MJCF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import franky_sim
import mujoco
import numpy as np
from franky_sim.mujoco_simulator import FrankaMujocoRobot, MujocoSimulator

ROBOT_PREFIX = "robot0_"
MESHDIR_PLACEHOLDER = "@FRANKY_SIM_MESHES@"
ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")

# Cubes for the "tabletop" scene: (name, position, rgba). 5 cm, graspable by the Franka Hand.
CUBES = (
    ("cube_red", (0.50, 0.00, 0.025), "0.85 0.2 0.2 1"),
    ("cube_green", (0.45, 0.18, 0.025), "0.2 0.7 0.3 1"),
    ("cube_blue", (0.55, -0.18, 0.025), "0.2 0.35 0.85 1"),
)
CUBE_HALF = 0.025


def meshes_dir() -> Path:
    return Path(franky_sim.__file__).parent / "assets" / "meshes"


def make_simulator(
    scene: str = "tabletop", initial_q: Sequence[float] | None = None
) -> tuple[MujocoSimulator, FrankaMujocoRobot]:
    """An initialized (not yet started) franky-sim MuJoCo simulator with one FR3 + hand."""
    sim = MujocoSimulator(enable_visualization=False)
    sim.init()
    kwargs = {} if initial_q is None else {"initial_q": tuple(float(x) for x in initial_q)}
    robot = sim.add_robot(**kwargs)
    add_objects(sim.worldbody, scene)
    return sim, robot


def add_objects(worldbody: ET.Element, scene: str) -> None:
    if scene == "empty":
        return
    if scene != "tabletop":
        raise ValueError(f"unknown scene {scene!r} (expected 'empty' or 'tabletop')")
    for name, pos, rgba in CUBES:
        body = ET.SubElement(worldbody, "body", name=name, pos=" ".join(map(str, pos)))
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_geom",
            type="box",
            size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
            rgba=rgba,
            mass="0.1",
            friction="1.5 0.01 0.001",
            condim="4",
        )


def portable_xml(sim: MujocoSimulator) -> str:
    """The scene MJCF with the (machine-specific) mesh directory replaced by a placeholder."""
    root = sim.scene
    compiler = root.find("compiler")
    original = compiler.get("meshdir")
    compiler.set("meshdir", MESHDIR_PLACEHOLDER)
    try:
        return ET.tostring(root, encoding="unicode")
    finally:
        compiler.set("meshdir", original)


def resolve_meshdir(xml: str) -> str:
    return xml.replace(MESHDIR_PLACEHOLDER, str(meshes_dir()))


def scene_xml(scene: str = "tabletop") -> str:
    """Portable MJCF of a scene, without starting physics (used by viewers of the real arm)."""
    sim, _ = make_simulator(scene)
    return portable_xml(sim)


def load_model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(resolve_meshdir(xml))


def robot_qpos_addrs(
    model: mujoco.MjModel, prefix: str = ROBOT_PREFIX
) -> tuple[np.ndarray, np.ndarray]:
    """qpos addresses of the 7 arm joints and the 2 finger joints."""

    def addr(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if jid < 0:
            raise KeyError(f"joint {prefix + name} not in model")
        return int(model.jnt_qposadr[jid])

    return np.array([addr(j) for j in ARM_JOINTS]), np.array([addr(j) for j in FINGER_JOINTS])
