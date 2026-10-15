# Simulated robot

The simulator is [franky-sim](https://github.com/TimSchneider42/franky-sim): a MuJoCo FR3 on a
table with three cubes, served on a loopback address with the **real libfranka network
protocol**. `ft-node` drives it through franky exactly as it drives the arm, so everything you
rehearse here (joystick, limits, deadman, gripper, homing) runs the same code as on the robot.
What it does not simulate: reflexes and other robot errors (its error recovery always succeeds).

All commands use `configs/sim.toml` by default. It differs from `configs/real.toml` only in
`robot.host` (127.0.0.1) and `sim.enabled`.

## Realtime

The config says `robot.realtime = "enforce"`, as for the robot. On a machine without a
PREEMPT_RT kernel and realtime permissions, `ft-node` then stops with *unable to set realtime
scheduling*. For simulation on such a machine, override it for the run:

```bash
FT_REALTIME=ignore uv run ft-teleop …     # or: uv run ft-teleop --realtime ignore …
```

On the deployment laptop, rehearse *without* the override, so the rehearsal also checks its
realtime setup (`uv run ft-check-rt`).

## With a real gamepad

Plug the pad into the machine that runs the simulation:

```bash
uv run ft-check-joystick --once     # is it detected?
uv run ft-teleop                    # ft-sim + ft-node + viewer + gamepad
```

Hold LB and move the sticks ([controls](input-devices.md#gamepad)). This is the rehearsal for the
real robot: same pad, same processes, same config apart from the robot's address.

## With a virtual joystick

No pad needed:

```bash
uv run ft-teleop --input webpad     # a gamepad in the web viewer (http://localhost:8080)
uv run ft-teleop --input keyboard   # type in this terminal
uv run ft-teleop --input script --script configs/scripts/demo.toml
```

The web pad is a virtual gamepad that reaches the node through the same reader, mapping and
command stream as a physical one; on a laptop it rehearses the gamepad path without the pad. See
[input-devices.md](input-devices.md) for each device.

## Viewers

`ft-teleop --viewer auto` (the default) opens the web viewer for `--input webpad`, otherwise a
native MuJoCo window when `$DISPLAY` is set, otherwise none. Choose explicitly with
`--viewer native | web | none`, or run a viewer on its own:

```bash
uv run ft-view                                   # native window
uv run ft-view --mode web                        # browser, http://localhost:8080
uv run ft-view --mode record --out run.mp4 --duration 10   # offscreen (EGL), no display needed
uv run ft-view --mode snapshot --out frame.png
```

In simulation the viewer draws ft-sim's full scene. If it stops receiving updates for more than
1 s, it marks the view **STALE**.

## On a headless server over SSH

The simulation runs on the server; you watch (and drive) from your machine's browser.

**Web pad** (nothing to install on your machine):

```bash
# server
FT_REALTIME=ignore uv run ft-teleop --input webpad
# your machine
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 server    # open http://localhost:8080
```

**Keyboard** (type in the SSH terminal, watch in the browser):

```bash
ssh -L 8080:localhost:8080 server
FT_REALTIME=ignore uv run ft-teleop --input keyboard --viewer web
```

**Your real gamepad, plugged into your laptop.** `ft-joystick` runs on the laptop (which needs a
checkout of this repo and `uv sync`); the simulator, node and viewer run on the server. SSH
carries the command port to the server (`-R`) and the status port back (`-L`):

```bash
# server, one process per terminal (or append &)
FT_REALTIME=ignore uv run ft-sim
FT_REALTIME=ignore uv run ft-node
uv run ft-view --mode web

# laptop
ssh -L 8080:localhost:8080 -L 5556:localhost:5556 -R 5555:localhost:5555 server
uv run ft-joystick                  # in a second laptop terminal; open http://localhost:8080
```

Both ends use the same config (same ports). The node judges a command's freshness by when it
arrives, not by the laptop's clock, so the two machines need no clock sync; a tunnel that stalls
for more than 150 ms simply stops the arm. `ft-teleop`'s orderly shutdown doesn't span two
machines: stop the joystick first, then the node (Ctrl-C each).

## Running the processes yourself

`ft-teleop` starts everything in order, gives every process the same effective config, and shuts
the whole session down if any of them dies. The same processes by hand:

```bash
uv run ft-sim           # franky-sim on robot.host
uv run ft-node          # connects to the robot (here: the simulator)
uv run ft-view          # native window; --mode web | record | snapshot
uv run ft-joystick      # --device gamepad | webpad | keyboard | script
```
