from __future__ import annotations

import pytest

from ..conftest import make_test_config
from ..sim.harness import FtSim


@pytest.fixture(scope="session")
def viz_sim(tmp_path_factory):
    """One ft-sim (tabletop, ready pose) shared by the viewer tests. Tests must not move it."""
    sim = FtSim(make_test_config(), tmp_path_factory.mktemp("viz_sim") / "config.toml")
    try:
        sim.wait_ready()
        yield sim
    finally:
        sim.kill()


@pytest.fixture(scope="session")
def viz_config_path(viz_sim):
    return viz_sim.args[viz_sim.args.index("--config") + 1]
