"""ft-view --mode native: MuJoCo's passive viewer. Run under `xvfb-run -a` on a headless box."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from ..conftest import make_test_config
from ..sim.harness import script


def ft_view(config: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [script("ft-view"), "--config", config, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.sim
@pytest.mark.e2e
@pytest.mark.display
@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display (xvfb-run -a)")
def test_native_viewer_runs_and_exits(viz_sim, viz_config_path):
    t0 = time.monotonic()
    r = ft_view(viz_config_path, "--mode", "native", "--duration", "2.5", env=dict(os.environ))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ft-sim scene" in r.stdout
    assert "Traceback" not in r.stderr
    assert time.monotonic() - t0 < 60


def test_native_without_a_display_explains_the_alternatives(config_file):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    cfg = make_test_config()
    t0 = time.monotonic()
    r = ft_view(str(config_file(cfg)), "--mode", "native", env=env)
    assert r.returncode == 2
    assert "--mode web" in r.stderr
    assert f"ssh -L {cfg.viewer.web_port}:localhost:{cfg.viewer.web_port}" in r.stderr
    assert time.monotonic() - t0 < 10  # fails fast, before loading any model
