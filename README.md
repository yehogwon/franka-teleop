# franka-teleop

Gamepad teleoperation for Franka arms (FR3 or Panda). A session rehearsed in simulation runs
unchanged on the real robot.

The simulator is [franky-sim](https://github.com/TimSchneider42/franky-sim), a MuJoCo server that
speaks the **real libfranka network protocol**. The teleop node drives it through
[franky](https://github.com/TimSchneider42/franky) exactly as it drives the arm, so moving to
hardware changes one line of config: the robot's IP address.

```
                      operator laptop (Linux, PREEMPT_RT, wired to the arm)
 ┌─────────────┐  TeleopCommand   ┌──────────┐  franky / libfranka   ┌────────────────────────┐
 │ ft-joystick │ ───────────────▶ │ ft-node  │ ────────────────────▶ │ Franka arm             │
 │ gamepad /   │  100 Hz, ZMQ     │ deadman, │   Cartesian velocity  │   or                   │
 │ keyboard /  │ ◀─────────────── │ watchdog,│   motions, 1 kHz      │ ft-sim (franky-sim on  │
 │ script      │    NodeStatus    │ limiters │                       │ 127.0.0.1, MuJoCo)     │
 └─────────────┘                  └────┬─────┘                       └───────────┬────────────┘
                                       │ NodeStatus                   SceneState │ (sim only)
                                       ▼                                         ▼
                                 ┌───────────────────────────────────────────────────┐
                                 │ ft-view: native window, web viewer, video/snapshot │
                                 └───────────────────────────────────────────────────┘
```

`configs/sim.toml` and `configs/real.toml` differ **only** in `robot.host` and `sim.enabled`. A
test enforces this.

## Setup

```bash
uv sync          # managed CPython 3.12 (the system Python is never used)
uv run pytest -m "not slow"
```

Linux x86_64 only, because franky ships wheels only for that platform.

## Quick start (simulation)

```bash
uv run ft-check-joystick --once          # is the gamepad detected?
uv run ft-teleop                         # ft-sim + ft-node + viewer + gamepad (configs/sim.toml)
uv run ft-teleop --input keyboard        # no gamepad: drive from the terminal
uv run ft-teleop --input script --script configs/scripts/demo.toml
```

On a machine **without a PREEMPT_RT kernel** (a dev box), libfranka cannot get realtime
scheduling. Run with `FT_REALTIME=ignore` (or `--realtime ignore`). The config files always say
`enforce`, so a rehearsal on the deployment laptop also exercises its realtime setup.

`ft-teleop` starts everything in the right order, passes every child the same effective config,
and if any process dies it shuts the whole session down (joystick → node → viewer → sim). You
can also run the processes yourself:

```bash
uv run ft-sim                     # franky-sim on robot.host
uv run ft-node                    # connects to the robot (or sim)
uv run ft-view                    # native window; --mode web | snapshot | record
uv run ft-joystick                # --device gamepad | keyboard | script
```

### Headless machine / over SSH

```bash
# on the server
FT_REALTIME=ignore uv run ft-teleop --input keyboard --viewer web
# on your machine
ssh -L 8080:localhost:8080 server        # then open http://localhost:8080
```

`ft-view --mode record --out run.mp4 --duration 10` and `--mode snapshot` render offscreen
(EGL) without any display. Keyboard input works in an SSH terminal.

## Controls

Hold **LB** (the deadman) to move. Releasing it stops the arm. Nothing moves, not even the
gripper, unless the deadman is held.

| Gamepad | Action |
|---|---|
| LB (hold) | deadman |
| left stick | forward/back, left/right |
| RT / LT | up / down |
| right stick ←→ | yaw |
| right stick ↑↓ | pitch |
| d-pad ←→ | roll |
| d-pad ↑↓ | speed level up / down (25 / 50 / 75 / 100 %) |
| A | gripper close / open (needs deadman) |
| X | toggle base / tool frame |
| Y | go home (keep holding the deadman; releasing it aborts) |
| B | stop and lock |
| START | recover from a robot error / unlock |

Directions are given for an operator standing **behind the robot, looking along +x**. If you
face the robot, set `teleop.operator_yaw_deg = 180`. In tool frame, "up" moves along the approach
direction (out of the fingers). Before the first real run, use `ft-check-joystick` to check the
directions on your pad.

Keyboard (`--input keyboard`): `space` toggles the deadman (terminals have no key-up events),
`w/s a/d r/f` translate, `j/l i/k u/o` yaw, pitch and roll, `g` gripper, `h` home, `b` stop,
`c` recover, `t` frame, `]`/`[` speed, `q` quit. Keyboard and script input are accepted only
when `sim.enabled` is true, unless you add them to `teleop.allowed_sources`.

## Safety design

The node enforces these rules itself, whatever the input device. All of them are covered by
tests (`tests/node`).

- **Deadman and freshness.** The node moves the arm only while the newest command is younger
  than `teleop.cmd_timeout_ms` (150 ms) *and* has the deadman held. A stale stream counts as a
  released deadman.
- **Arming.** After a new joystick session, error recovery, unlocking, homing or a stream outage
  longer than 1 s, the deadman must be seen *released* once before anything moves. A stuck
  button or a stick already pushed can't start motion.
- **Smooth stops.** Releasing the deadman, a stale stream or the stop button sends a franky
  *stop motion* (dynamics-limited deceleration). The control loop is never just cut.
- **Robot-side watchdog.** Each velocity target is held by franky for at most `teleop.hold_ms`
  (100 ms), so the arm stops by itself even if `ft-node` dies.
- **Limiters.** Each one restricts only motion *towards* a boundary, so you can always back out:
  - *workspace box*: outward velocity of the TCP ramps to zero at the walls;
  - *floor guard*: the box only constrains the TCP, and a hand tilted past horizontal puts the
    wrist below it. The elbow and wrist joint frames, the flange and the hand-body corners are
    kept above `workspace.floor` plus a clearance, using the robot's kinematic model;
  - *singularity guard*: motion that lowers the smallest singular value of the (rotation-weighted)
    Jacobian slows and stops before Cartesian tracking breaks down at the edge of reach. It uses
    the robot's own kinematic model;
  - *joint-limit guard*: stops `clearance` short of each limit, using the larger of the
    predicted and the measured joint speed;
  - default joint limits are the **intersection of FR3 and Panda limits**, valid for either arm.
- **Errors** (reflexes) latch until START. Error recovery is franky's automatic recovery.
- **One controller.** `ft-joystick` binds the command port, so a second joystick fails to start.
- **No orphans.** When the session stops, through any of Ctrl-C, SIGTERM, a closed terminal
  (SIGHUP) or a child crashing, `ft-teleop` shuts down in order and halts the arm before
  disconnecting. If the launcher itself is killed, the kernel sends its children SIGTERM.
- **Sanity checks on the config.** A config with `sim.enabled` must point at a loopback
  address, so keyboard and script input can never reach a real arm by accident.
  `teleop.hold_ms` is capped at 250 ms. Commands with non-finite values are dropped.

Measured against franky-sim (`tests/node/test_node_sim.py`, sim in a separate process):

| Event | Arm at rest after |
|---|---|
| deadman released | 68 ms |
| joystick process dies | 229 ms (150 ms timeout + braking) |
| `ft-node` dies | 158 ms (100 ms robot-side hold + braking) |

| Guard | Result |
|---|---|
| workspace walls (4 directions, full speed) | 0 mm overshoot |
| singularity (reaching out and up) | σ ≥ 0.035, tracking error ≤ 5 mm/s |
| wrist joint driven into its limit | stops ≥ 26 mrad short |
| hand rolled over 15 cm above the table | wrist and hand stay clear of it |

A randomized operator soak (`tests/node/test_soak.py`; random sticks, deadman, speed, frame,
gripper, home, stop/recover) checks every status message: no robot errors, the workspace is
never left, no joint limit is crossed, and σ stays above the stop threshold. The soak is how the
wrist-on-table gap was found. The soak and two independent code reviews also found the other
failure modes listed above; each now has a regression test.

The software deadman is **not** an emergency stop. Keep the Franka external enabling/emergency
stop device in hand whenever the real arm is powered.

## Deploying to the real robot

Everything runs on the laptop. Do it in this order:

1. **Realtime laptop.** Install a PREEMPT_RT kernel, add yourself to a `realtime` group with
   `rtprio 99` and `memlock unlimited`, and set the CPU governor to `performance`. Then check:
   `uv run ft-check-rt`. Every check must pass; it prints the fix for each failure.
2. **Network.** Plug the laptop's built-in Ethernet port straight into the arm (FR3: the
   robot's base; Panda: the control box). Give the laptop a static IP in the robot's subnet,
   e.g. `172.16.0.1/24` for a robot at `172.16.0.2`.
3. **Desk.** Unlock the brakes and activate FCI.
4. **Config.** In `configs/real.toml` set `robot.host`. Measure your table and set `workspace`
   (and `robot.home_q` if needed). Mirror any change in `configs/sim.toml`: the parity test
   allows only `robot.host` and `sim.enabled` to differ.
5. **Read-only check.** `uv run ft-check-robot -c configs/real.toml` connects without moving the
   arm. It reports the robot mode, errors, joint margins, whether the end effector is inside the
   workspace, the state rate and the gripper. If the robot's *server version* doesn't match the
   installed franky, it prints the exact `pyproject.toml` change (see below).
6. **Rehearse on the laptop.** `uv run ft-teleop` (sim config), with the same kernel, gamepad
   and processes as the real run.
7. **Go.** `uv run ft-teleop -c configs/real.toml` runs `ft-check-robot` first, then the node,
   viewer and gamepad. Start at the lowest speed level (d-pad ↓). For the first sessions,
   consider lowering `robot.dynamics_factor`.

### Robot type and server version

The robot type doesn't matter to the code: franky reads the kinematic model from the robot, and
the default joint limits suit both arms. What matters is the robot's **libfranka server
version**. The pinned `franky-control==2.0.0` from PyPI speaks version 10 (libfranka 0.21.2),
which recent FR3 system images use and which franky-sim emulates. For other versions franky
publishes wheels per server version (3 → libfranka 0.7.1, 4 → 0.8.0, 5 → 0.9.2, 6 → 0.12.1,
7 → 0.13.3, 8 → 0.14.2, 9 → 0.17.0, 10 → 0.21.2). On a mismatch, `ft-check-robot` prints a
snippet like:

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

## Configuration

Every key is written out, with comments, in `configs/sim.toml` / `configs/real.toml`. The ones
you are most likely to change:

| Key | Meaning |
|---|---|
| `robot.host` | FCI address of the arm (sim: loopback address franky-sim binds) |
| `robot.dynamics_factor` | franky's scale on acceleration/jerk limits (default 0.2). Operator speed is capped by `teleop.max_*_speed` |
| `teleop.max_linear_speed`, `max_angular_speed`, `speed_levels` | operator speed at full stick |
| `teleop.operator_yaw_deg` | where the operator stands |
| `workspace.lower` / `upper` | TCP box in the base frame [m] |
| `workspace.floor`, `link_clearance`, `hand_clearance` | table height and how far the wrist and hand stay above it |
| `joint_guard.*` | joint-limit and singularity guard tuning |
| `input.deadzone`, `expo`, `invert` | stick feel and axis inversion |
| `net.*_port` | ZMQ ports (publishers bind on `bind_host`, subscribers connect to `connect_host`) |

| Port | Publisher → subscriber | Message |
|---|---|---|
| 5555 | ft-joystick → ft-node | `TeleopCommand` |
| 5556 | ft-node → ft-joystick, ft-view | `NodeStatus` |
| 5557 | ft-sim → ft-view | `SceneState` |
| 5558 | ft-sim ⇄ ft-view (REQ/REP) | `SceneInfo` |
| 8080 | ft-view `--mode web` | HTTP |

Messages are msgpack, defined in `src/franka_teleop/protocol.py`. Every message is versioned,
and discrete actions are sent as counters, so a lost packet never loses or repeats a button
press.

## Development

```bash
uv run pytest -m "not slow"                    # ~3 min; starts franky-sim processes
xvfb-run -a uv run pytest                      # includes the native-viewer test
FT_SOAK_SECONDS=300 uv run pytest -s tests/node/test_soak.py
uv run ruff check src tests && uv run ruff format --check src tests
```

| Marker | Meaning |
|---|---|
| `sim` | needs franky-sim |
| `e2e` | spawns the installed `ft-*` console scripts |
| `display` | needs X or EGL |
| `slow` | soak |

Gamepad tests attach an SDL *virtual* controller, so the real pygame input path runs headless.
The real-robot code path of `ft-teleop` (preflight → node → joystick) is tested end to end
against franky-sim on a loopback address.

## Known limitations

- **No simulated reflexes.** franky-sim does not simulate reflexes or other robot errors, and its
  error recovery always succeeds. The error, latch and recovery logic is tested only against a
  kinematic fake arm that raises them.
- **Homing ignores the workspace box.** Homing moves in joint space. Keep holding the deadman
  and watch the arm.
- **The floor guard models only the Franka Hand.** It uses the Franka Hand's outline
  (`HAND_POINTS` in `node/limits.py`) and assumes a flat table at `workspace.floor`. It knows
  nothing about other obstacles. For other tools, change `HAND_POINTS`.
- **Singularity thresholds come from the simulator.** `sigma_slow`/`sigma_stop` were tuned where
  franky-sim's Cartesian tracking breaks down. Check them on the real arm at low speed.
- **Not tested on real hardware.** A physical gamepad's trigger ranges and rumble, a PREEMPT_RT
  laptop and the real arm can only be checked on the deployment hardware. The preflight tools
  (`ft-check-rt`, `ft-check-joystick`, `ft-check-robot`) exist for exactly that.
