import msgspec
import pytest

from franka_teleop.config import (
    FR3_JOINT_LOWER,
    FR3_JOINT_UPPER,
    JOINT_LOWER,
    JOINT_UPPER,
    PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER,
    PARITY_ALLOWED_DIFF,
    Config,
    ConfigError,
    config_diff,
    dump_toml,
    flatten,
    load_config,
    write_config,
)

from .conftest import CONFIGS


def test_sim_and_real_differ_only_in_allowed_keys():
    sim = load_config(CONFIGS / "sim.toml")
    real = load_config(CONFIGS / "real.toml")
    assert config_diff(sim, real) == PARITY_ALLOWED_DIFF
    assert sim.sim.enabled and not real.sim.enabled
    assert sim.robot.realtime == real.robot.realtime == "enforce"


def _dotted_keys(obj, prefix=""):
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        yield from _dotted_keys(v, key) if isinstance(v, dict) else [key]


def test_config_files_spell_out_every_key():
    """Every setting is visible in the files, so nothing silently falls back to code defaults."""
    import tomllib

    for name in ("sim.toml", "real.toml"):
        raw = tomllib.loads((CONFIGS / name).read_text())
        file_keys = set(flatten(msgspec.convert(raw, Config)).keys())
        written = set(_dotted_keys(raw))
        assert written == file_keys, f"{name} missing {sorted(file_keys - written)}"


def test_joint_limit_defaults_are_the_fr3_panda_intersection():
    for lo, fr3, panda in zip(JOINT_LOWER, FR3_JOINT_LOWER, PANDA_JOINT_LOWER, strict=True):
        assert lo == max(fr3, panda)
    for hi, fr3, panda in zip(JOINT_UPPER, FR3_JOINT_UPPER, PANDA_JOINT_UPPER, strict=True):
        assert hi == min(fr3, panda)
    cfg = load_config(CONFIGS / "sim.toml")
    assert tuple(cfg.robot.joint_lower) == JOINT_LOWER
    assert tuple(cfg.robot.joint_upper) == JOINT_UPPER
    for q, lo, hi in zip(cfg.robot.home_q, JOINT_LOWER, JOINT_UPPER, strict=True):
        assert lo < q < hi


def test_overrides_and_realtime_env(monkeypatch):
    cfg = load_config(CONFIGS / "real.toml", robot__host="10.0.0.5", net__cmd_port=6000)
    assert cfg.robot.host == "10.0.0.5" and cfg.net.cmd_port == 6000
    monkeypatch.setenv("FT_REALTIME", "ignore")
    assert load_config(CONFIGS / "real.toml").robot.realtime == "ignore"


def test_allowed_sources_extend_only_in_sim():
    assert load_config(CONFIGS / "real.toml").allowed_sources() == {"gamepad"}
    assert load_config(CONFIGS / "sim.toml").allowed_sources() == {"gamepad", "keyboard", "script"}


@pytest.mark.parametrize(
    "override",
    [
        {"robot__dynamics_factor": 0.0},
        {"robot__dynamics_factor": 1.5},
        {"robot__home_q": [0.0] * 6},
        {"workspace__lower": [1.0, 0.0, 0.0]},
        {"teleop__speed_levels": []},
        {"teleop__speed_levels": [0.5, 2.0]},
        {"teleop__initial_speed_level": 9},
        {"teleop__hold_ms": 5},
        {"input__invert": ["lin_q"]},
        {"robot__realtime": "sometimes"},
        {"robot__bogus_key": 1},
    ],
)
def test_invalid_configs_are_rejected(override):
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "sim.toml", **override)


def test_dump_toml_roundtrips(tmp_path):
    cfg = load_config(CONFIGS / "sim.toml", robot__host="127.3.4.5", input__invert=["ang_y"])
    path = write_config(cfg, tmp_path / "effective.toml")
    assert load_config(path) == cfg
    assert dump_toml(cfg) == path.read_text()
