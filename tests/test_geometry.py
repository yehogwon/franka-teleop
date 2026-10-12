import numpy as np

from franka_teleop.geometry import exp_so3, log_so3, mat_from_quat, quat_from_mat, rot_z


def random_rot(rng):
    q = rng.normal(size=4)
    return mat_from_quat(q / np.linalg.norm(q))


def test_quat_roundtrip_and_convention():
    rng = np.random.default_rng(0)
    for _ in range(500):
        R = random_rot(rng)
        assert np.allclose(mat_from_quat(quat_from_mat(R)), R, atol=1e-9)
    # franky's ready pose: 180 deg about x  ->  (x, y, z, w) = (1, 0, 0, 0)
    assert np.allclose(quat_from_mat(np.diag([1.0, -1.0, -1.0])), [1, 0, 0, 0])
    assert np.allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0])


def test_exp_log_roundtrip_including_near_pi():
    rng = np.random.default_rng(1)
    for _ in range(500):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        assert np.allclose(log_so3(exp_so3(w)), w, atol=1e-7)
    for axis in np.eye(3):
        R = exp_so3(axis * np.pi)
        assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-6)
    assert np.allclose(log_so3(np.eye(3)), 0)
