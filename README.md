# franka-teleop

Gamepad teleoperation for Franka arms (FR3 or Panda). A session rehearsed in simulation runs
unchanged on the real robot.

## How it works

```
                      operator laptop (Linux, PREEMPT_RT, wired to the arm)
 ┌─────────────┐  TeleopCommand   ┌──────────┐  franky / libfranka   ┌────────────────────────┐
 │ ft-joystick │ ───────────────▶ │ ft-node  │ ────────────────────▶ │ Franka arm             │
 │ gamepad /   │  100 Hz, ZMQ     │ deadman, │   Cartesian velocity  │   or                   │
 │ web pad /   │ ◀─────────────── │ watchdog,│   motions, 1 kHz      │ ft-sim (franky-sim on  │
 │ keyboard /  │    NodeStatus    │ limiters │                       │ 127.0.0.1, MuJoCo)     │
 │ script      │                  │          │                       │                        │
 └─────────────┘                  └────┬─────┘                       └───────────┬────────────┘
                                       │ NodeStatus                   SceneState │ (sim only)
                                       ▼                                         ▼
                                 ┌───────────────────────────────────────────────────┐
                                 │ ft-view: native window, web viewer, video/snapshot │
                                 └───────────────────────────────────────────────────┘
```

- **`ft-joystick`** reads the operator: a physical gamepad, or a virtual joystick (a gamepad
  drawn in the browser, the keyboard or a script). It sends `TeleopCommand`s at 100 Hz over ZMQ.
- **`ft-node`** turns them into Cartesian velocity motions through
  [franky](https://github.com/TimSchneider42/franky) (libfranka). It enforces the deadman, a
  watchdog and the workspace, floor, singularity and joint-limit guards.
- **The robot** is the real arm or [franky-sim](https://github.com/TimSchneider42/franky-sim), a
  MuJoCo server that speaks the **real libfranka network protocol**. The node drives both the same
  way, so moving to hardware changes the robot's address: `configs/sim.toml` and
  `configs/real.toml` differ only in `robot.host` and `sim.enabled` (a test enforces this).
- **`ft-view`** shows the arm in a native window, in the browser (for headless machines) or as a
  video. **`ft-teleop`** starts the whole session in order and shuts it down safely.

## Setup

```bash
uv sync          # managed CPython 3.12 (the system Python is never used)
uv run pytest -m "not slow"
```

Linux x86_64 only, because franky ships wheels only for that platform.

## Use cases

|  | Real gamepad | Virtual joystick (web pad, keyboard, script) |
|---|---|---|
| **Simulated robot** | `uv run ft-teleop` | `uv run ft-teleop --input webpad` |
| **Real robot** | `uv run ft-teleop -c configs/real.toml` | `… -c configs/real.toml --input webpad` (opt-in) |

On a machine without a realtime kernel (e.g. a dev server), add `FT_REALTIME=ignore` for
simulation; never for the real robot.

### Simulated robot, real gamepad

Plug the pad into the machine running the simulation and rehearse the real session:

```bash
uv run ft-check-joystick --once     # is the pad detected?
uv run ft-teleop                    # ft-sim + ft-node + viewer + gamepad
```

The pad can also stay on your laptop while the simulation runs on a server (SSH tunnels):
[simulation.md](docs/simulation.md#on-a-headless-server-over-ssh).

### Simulated robot, virtual joystick

No pad needed. The web pad is a virtual gamepad, read by the same code as a physical one:

```bash
uv run ft-teleop --input webpad     # gamepad in the web viewer: hold LB (or space) + drag a stick
uv run ft-teleop --input keyboard   # type in this terminal
uv run ft-teleop --input script --script configs/scripts/demo.toml
```

Over SSH: `ssh -L 8080:localhost:8080 -L 8081:localhost:8081 server`, then open
`http://localhost:8080`. Details: [simulation.md](docs/simulation.md).

### Real robot, real gamepad

The deployment: a PREEMPT_RT laptop wired to the arm, FCI active in Desk, `robot.host` and the
workspace set in `configs/real.toml`. Then:

```bash
uv run ft-check-rt                              # realtime setup
uv run ft-check-robot -c configs/real.toml      # read-only: connects, never moves
uv run ft-teleop -c configs/real.toml           # preflight, node, viewer, gamepad
```

Follow the full checklist in [real-robot.md](docs/real-robot.md) the first time.

### Real robot, virtual joystick

Allowed only after you add the device to `teleop.allowed_sources` (in both config files):

```bash
uv run ft-teleop -c configs/real.toml --input webpad
```

What each device means on the real arm:
[real-robot.md](docs/real-robot.md#with-a-virtual-joystick).

## Controls

Hold **LB** (the deadman) to move; releasing it stops the arm. Left stick: forward/back and
left/right; RT/LT: up/down; right stick: yaw and pitch; d-pad: roll and speed level; A gripper,
X base/tool frame, Y home, B stop, START recover. Every device and key:
[input-devices.md](docs/input-devices.md). The web viewer lists the controls in its panel.

## Safety

The node enforces these rules itself, whatever the input device:

- the arm moves only while a fresh command (under 150 ms old) holds the deadman;
- a new joystick must release the deadman once before anything moves;
- stops are smooth, dynamics-limited decelerations;
- franky holds each velocity target for at most 100 ms, so the arm stops by itself if `ft-node`
  dies;
- limiters keep the hand inside a workspace box, the wrist and hand above the table, and the arm
  away from singularities and joint limits.

Details and measurements: [safety.md](docs/safety.md).

The software deadman is **not** an emergency stop. Keep the arm's hardware stop within reach
whenever it is powered.

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
| `input.webpad_host`, `webpad_port` | where the web joystick is served (default 127.0.0.1:8081) |
| `net.*_port` | ZMQ ports (publishers bind on `bind_host`, subscribers connect to `connect_host`) |

| Port | Publisher → subscriber | Message |
|---|---|---|
| 5555 | ft-joystick → ft-node | `TeleopCommand` |
| 5556 | ft-node → ft-joystick, ft-view | `NodeStatus` |
| 5557 | ft-sim → ft-view | `SceneState` |
| 5558 | ft-sim ⇄ ft-view (REQ/REP) | `SceneInfo` |
| 8080 | ft-view `--mode web` | HTTP |
| 8081 | ft-joystick `--device webpad` | HTTP + WebSocket (pad state, JSON) |

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
| `browser` | drives the web pages in Chromium (`uv run playwright install chromium`; skipped without it) |

Gamepad tests attach an SDL *virtual* controller, so the real pygame input path runs headless.
The web joystick tests drive the pad page in Chromium, inside the web viewer, against
franky-sim.
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
- **franky's planner can occasionally reject a motion.** It can fail numerically ("Motion planner
  failed, code -110"), which so far has been seen only at near-zero speeds while a limiter ramped
  the target down. The node retries a single rejection transparently. Three within 2 s halt the
  arm and require a deadman release. If the robot reports an error at the same time, the node
  latches the error mode.
- **The web joystick depends on the browser.** Tested in Chromium only; closing the tab stops the
  arm once the browser has torn the page down (up to 0.5 s here). See
  [input-devices.md](docs/input-devices.md#web-joystick-webpad).
- **Not tested on real hardware.** A physical gamepad's trigger ranges and rumble, a PREEMPT_RT
  laptop and the real arm can only be checked on the deployment hardware. The preflight tools
  (`ft-check-rt`, `ft-check-joystick`, `ft-check-robot`) exist for exactly that.
