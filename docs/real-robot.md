# Real robot

Everything runs on one Linux laptop wired straight to the arm: `ft-node` (talking libfranka
through franky), the joystick and the viewer. The session is the one you rehearse in
[simulation](simulation.md); `configs/real.toml` differs from `configs/sim.toml` only in
`robot.host` and `sim.enabled`.

> Not yet run on hardware: everything here was verified against franky-sim, which speaks the
> robot's protocol but simulates no reflexes. The preflight tools below exist to catch what only
> the real setup can show. The software deadman is **not** an emergency stop: keep the arm's
> hardware stop within reach whenever it is powered.

## With a real gamepad

Do it in this order.

1. **Install.**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is not installed
   git clone git@github.com:yehogwon/franka-teleop.git && cd franka-teleop
   uv sync                                              # uv fetches its own Python 3.12
   ```

2. **Realtime laptop.** libfranka needs realtime scheduling.
   - Install a PREEMPT_RT kernel. On Ubuntu the simplest is Ubuntu Pro's real-time kernel
     (`sudo pro enable realtime-kernel`); otherwise build one as in Franka's libfranka
     installation guide.
   - Allow your user realtime priority and locked memory:

     ```bash
     sudo groupadd realtime && sudo usermod -aG realtime $USER
     printf '@realtime - rtprio 99\n@realtime - memlock unlimited\n' | sudo tee -a /etc/security/limits.conf
     ```

     then log out and in again.
   - Set the CPU governor to `performance`.
   - Check: `uv run ft-check-rt`. Every check must pass; it prints the fix for each failure.

   Never run the robot with `FT_REALTIME=ignore`: without realtime scheduling the robot's
   communication reflexes interrupt motion.

3. **Network.** Plug the laptop's built-in Ethernet port straight into the arm (FR3: the
   robot's base; Panda: the control box). Give the laptop a static IP in the robot's subnet,
   e.g. `172.16.0.1/24` for a robot at `172.16.0.2`.

4. **Desk.** Open Desk (`https://<robot-ip>`), unlock the joints and activate FCI.

5. **Config** (`configs/real.toml`):
   - `robot.host`: the robot's IP.
   - `workspace.lower` / `upper` (the box the TCP may not leave) and `workspace.floor` (the
     table height): measure them in the robot's base frame. The defaults are a guess.
   - `teleop.operator_yaw_deg = 180` if you face the robot instead of standing behind it.
   - `robot.home_q` if the default home pose doesn't suit your setup.

   Make the same changes in `configs/sim.toml`, except `robot.host`: a test checks that the two
   files differ only in `robot.host` and `sim.enabled`.

6. **Gripper.** After power-on the Franka Hand must be homed once, in Desk or with
   `uv run python -c "import franky; franky.Gripper('<robot-ip>').homing()"`, before its widths
   are valid. An unhomed hand shows up as a gripper error in the status line.

7. **Gamepad.** Plug it in (a cable avoids wireless dropouts) and check every axis with
   `uv run ft-check-joystick`: the left stick pushed away from you must read `fwd +1.00`. Fix a
   reversed axis with `input.invert`. See [input-devices.md](input-devices.md#gamepad).

8. **Read-only check.**

   ```bash
   uv run ft-check-robot -c configs/real.toml
   ```

   It connects without moving the arm and reports the robot mode, errors, joint margins, whether
   the end effector is inside the workspace, the state rate and the gripper. If the robot's
   libfranka *server version* doesn't match the installed franky, it prints the exact
   `pyproject.toml` change ([below](#robot-type-and-server-version)).

9. **Rehearse** on the laptop, without any realtime override: `uv run ft-teleop`.

10. **Go.**

    ```bash
    uv run ft-teleop -c configs/real.toml
    ```

    It runs the read-only check first (skip with `--skip-preflight`), then the node, the viewer
    and the gamepad. For the first sessions:
    - press d-pad ↓ to the lowest speed level (25 %);
    - consider lowering `robot.dynamics_factor` (e.g. 0.1) for gentler acceleration;
    - try small motions in each direction before going near the table.

    Releasing LB stops the arm; B stops and locks it (START unlocks, and recovers from a robot
    error).

## With a virtual joystick

The web pad, keyboard and scripts are accepted only in simulation unless you allow them. Add
them to `teleop.allowed_sources` in **both** config files (the parity test compares them; in sim
they are allowed anyway):

```toml
[teleop]
allowed_sources = ["gamepad", "webpad"]    # and/or "keyboard", "script"
```

Then, after the same checklist (steps 1–6, 8, 9):

```bash
uv run ft-teleop -c configs/real.toml --input webpad      # gamepad in the web viewer
uv run ft-teleop -c configs/real.toml --input keyboard
uv run ft-teleop -c configs/real.toml --input script --script my_motion.toml
```

What to know before you do:

- **Web pad.** Its commands are a gamepad's, through the same reader and mapping; only the
  browser around it differs ([input-devices.md](input-devices.md#web-joystick-webpad)). The
  viewer draws the arm from ft-node's joint states. `input.webpad_host` stays `127.0.0.1`
  unless you want to drive from a tablet: with `0.0.0.0` anyone on the network who opens the page
  can take the pad, so do that only on an isolated network.
- **Keyboard.** The deadman is a toggle, not a hold: it stays on until you press space again
  (motion still stops when you let go of the keys). A tap moves for 0.6 s.
- **Script.** Nobody holds the deadman: the script does. Run it in simulation first, at the same
  speed level, and keep the hardware stop in hand.

`ft-joystick` refuses `--device webpad` on a robot config unless `"webpad"` is allowed (the node
can't tell the web pad from a physical one); for keyboard and script the node itself ignores the
commands of a source that isn't allowed.

## Robot type and server version

The robot type doesn't matter to the code: franky reads the kinematic model from the robot, and
the default joint limits (the intersection of FR3 and Panda limits) suit both arms. What matters
is the robot's **libfranka server version**. The pinned `franky-control==2.0.0` from PyPI speaks
version 10 (libfranka 0.21.2), which recent FR3 system images use and which franky-sim emulates.
For other versions franky publishes wheels per server version (3 → libfranka 0.7.1, 4 → 0.8.0,
5 → 0.9.2, 6 → 0.12.1, 7 → 0.13.3, 8 → 0.14.2, 9 → 0.17.0, 10 → 0.21.2). On a mismatch,
`ft-check-robot` prints a snippet like:

```toml
[[tool.uv.index]]
name = "franky-server-9"
url = "https://timschneider42.github.io/franky/whl/by-robot-server-version/9/"
explicit = true

[tool.uv.sources]
franky-control = { index = "franky-server-9" }
```

followed by `uv lock --upgrade-package franky-control && uv sync`. A non-default wheel can no
longer talk to franky-sim (protocol version 10). Rehearse in sim first, or keep a second
checkout.
