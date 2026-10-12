"""Triangle meshes for the visual geoms of an MjModel (for renderers other than MuJoCo's).

Visual geoms are those MuJoCo itself shows by default: groups 0-2. franky-sim's FR3 puts its
visual meshes in group 2 (contype 0) and its collision meshes and fingertip pads in group 3;
the scene's cubes and ground are group 0 (visual and collision at once).

Vertices are expressed in the geom frame, so a mesh is placed with ``geom_xpos``/``geom_xmat``.
For mesh geoms that holds because the compiler re-centres each mesh on its inertial frame and
folds that offset into the geom pose (``mesh_pos``/``mesh_quat`` only record it).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

VISIBLE_GROUPS = 3
"""Geom groups < this are drawn (MuJoCo's default MjvOption.geomgroup)."""

_DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class GeomMesh:
    geom_id: int
    name: str
    vertices: np.ndarray
    """(V, 3) float32, geom frame."""
    faces: np.ndarray
    """(F, 3) uint32, counter-clockwise seen from outside."""
    rgba: np.ndarray
    """(4,) float in [0, 1]."""


def geom_rgba(model: mujoco.MjModel, g: int) -> np.ndarray:
    """MuJoCo's rule: an explicit geom rgba beats the material; the default rgba does not."""
    rgba = model.geom_rgba[g]
    matid = model.geom_matid[g]
    if matid >= 0 and np.allclose(rgba, _DEFAULT_GEOM_RGBA):
        rgba = model.mat_rgba[matid]
    return np.array(rgba, dtype=float)


def visual_geoms(model: mujoco.MjModel) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_group[g] < VISIBLE_GROUPS]


def geom_name(model: mujoco.MjModel, g: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if name:
        return name
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "world"
    return f"{body}_geom{g}"


def model_meshes(model: mujoco.MjModel) -> tuple[list[GeomMesh], list[int]]:
    """Triangle meshes of all visual geoms, and the ids of visual planes (drawn as grids).
    Geom types without a mesh representation here (height fields, SDFs) are skipped."""
    meshes, planes = [], []
    for g in visual_geoms(model):
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            planes.append(g)
            continue
        tri = geom_triangles(model, g)
        if tri is not None:
            meshes.append(GeomMesh(g, geom_name(model, g), *tri, geom_rgba(model, g)))
    return meshes, planes


def geom_triangles(model: mujoco.MjModel, g: int) -> tuple[np.ndarray, np.ndarray] | None:
    size = model.geom_size[g]
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        v, f = mesh_triangles(model, int(model.geom_dataid[g]))
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        v, f = box(size)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        v, f = sphere(size[0])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        v, f = sphere(1.0)
        v = v * size
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        v, f = cylinder(size[0], size[1])
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        v, f = capsule(size[0], size[1])
    else:
        return None
    return v.astype(np.float32), f.astype(np.uint32)


def mesh_triangles(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    """A mesh asset, with a vertex per distinct (position, normal) pair, so that a renderer that
    averages face normals per vertex reproduces the mesh's sharp edges instead of smoothing
    them away."""
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = model.mesh_vert[va : va + vn]
    faces = model.mesh_face[fa : fa + fn]  # indices local to this mesh
    if model.mesh_normalnum[mesh_id] == 0:
        return verts.copy(), faces.copy()
    normal_ids = model.mesh_facenormal[fa : fa + fn]
    pairs = np.stack([faces.ravel(), normal_ids.ravel()], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return verts[uniq[:, 0]], inverse.reshape(-1, 3)


# -- primitives (geom frame: centred, axis along z) --------------------------------------------


def box(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """24 vertices (4 per side, for crisp edges), 12 triangles."""
    verts, faces = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1.0
            v = np.cross(n, u)
            c = n * half
            u, v = u * half, v * half
            base = len(verts)
            verts += [c - u - v, c + u - v, c + u + v, c - u + v]
            faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(verts), np.array(faces)


def lathe(profile: np.ndarray, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a (r, z) profile, ordered bottom to top, about the z axis."""
    theta = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    r, z = profile[:, :1], profile[:, 1:]
    verts = np.stack(
        [r * np.cos(theta), r * np.sin(theta), np.broadcast_to(z, (len(profile), segments))],
        axis=-1,
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(len(profile) - 1), np.arange(segments), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, b = i * segments + j, i * segments + (j + 1) % segments
    c, d = b + segments, a + segments
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return verts, faces


def sphere(radius: float, rings: int = 16, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    return lathe(np.stack([radius * np.cos(phi), radius * np.sin(phi)], 1), segments)


def capsule(radius: float, half_length: float, rings: int = 8, segments: int = 32):
    phi = np.linspace(0.0, np.pi / 2, rings + 1)
    bottom = np.stack([radius * np.sin(phi), -half_length - radius * np.cos(phi)], 1)
    top = np.stack([radius * np.cos(phi), half_length + radius * np.sin(phi)], 1)
    return lathe(np.concatenate([bottom, top]), segments)


def cylinder(radius: float, half_length: float, segments: int = 32):
    side_v, side_f = lathe(np.array([[radius, -half_length], [radius, half_length]]), segments)
    verts, faces = [side_v], [side_f]
    n = len(side_v)
    theta = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], 1)
    k = np.arange(segments)
    for z, up in ((-half_length, False), (half_length, True)):
        cap = np.concatenate([[[0.0, 0.0, z]], np.column_stack([ring, np.full(segments, z)])])
        centre, rim, rim_next = n, n + 1 + k, n + 1 + (k + 1) % segments
        tri = np.stack([np.full(segments, centre), rim, rim_next], 1)
        faces.append(tri if up else tri[:, ::-1])
        verts.append(cap)
        n += len(cap)
    return np.concatenate(verts), np.concatenate(faces)


# -- poses ---------------------------------------------------------------------------------------


def geom_poses(data: mujoco.MjData, geom_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World positions (N, 3) and wxyz quaternions (N, 4) of geoms (after mj_kinematics)."""
    pos = data.geom_xpos[geom_ids].copy()
    mats = data.geom_xmat[geom_ids]
    quats = np.empty((len(geom_ids), 4))
    for k in range(len(geom_ids)):
        mujoco.mju_mat2Quat(quats[k], mats[k])
    return pos, quats
