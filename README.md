# franka-teleop

Gamepad teleoperation for Franka arms (FR3 / Panda), built so that a session rehearsed in
simulation runs unchanged on the real robot.

- **Simulation**: [franky-sim](https://github.com/TimSchneider42/franky-sim) runs MuJoCo behind the
  real libfranka network protocol, so the teleop node talks to it through
  [franky](https://github.com/TimSchneider42/franky) exactly as it will talk to the arm.
- **Teleop**: `ft-joystick` streams operator intent to `ft-node`, which applies deadman, watchdog,
  workspace and joint-limit guards and drives the arm with Cartesian velocity motions.
- **Deployment**: everything runs on one Linux laptop wired to the arm. `configs/sim.toml` and
  `configs/real.toml` differ only in `robot.host` and `sim.enabled` (enforced by a test).

> Work in progress — full usage and deployment guide to follow.

## Setup

```bash
uv sync
uv run pytest
```
