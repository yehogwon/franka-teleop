"""Test helper: run franky-sim (tabletop scene, ready pose) on argv[1] until SIGTERM.

Integration tests run the simulator in its own process, as in deployment, so its 1 kHz Python
loop does not compete for the GIL with the code under test."""

import signal
import sys

from franky_sim import SimulationServer

from franka_teleop.config import READY_Q
from franka_teleop.sim.scene import make_simulator


def main() -> None:
    sim, _ = make_simulator("tabletop", READY_Q)
    server = SimulationServer(sim, hostname_candidates=[sys.argv[1]])
    server.init()

    def stop(*_):
        server.running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("ready", flush=True)
    server.run_forever()
    server.cleanup()
    sim.cleanup()


if __name__ == "__main__":
    main()
