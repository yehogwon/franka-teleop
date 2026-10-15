# Input devices: real and virtual joysticks

`ft-joystick` turns operator input into `TeleopCommand`s for `ft-node` at 100 Hz. It has four
input devices (`ft-teleop --input …`, `ft-joystick --device …`):

| Device | What it is | Moves the robot by default in |
|---|---|---|
| `gamepad` | a physical Xbox / PlayStation-style pad, plugged into the machine running `ft-joystick` | sim and real |
| `webpad` | a gamepad drawn in a browser page, driving an SDL *virtual* pad | sim only |
| `keyboard` | the terminal that runs `ft-teleop` / `ft-joystick` | sim only |
| `script` | timed steps from a TOML file | sim only |

"Sim only" means the device is accepted automatically when `sim.enabled = true`. To use it on the
real robot, add it to `teleop.allowed_sources` (see
[real-robot.md](real-robot.md#with-a-virtual-joystick)).

Whatever the device, the node applies the same rules ([safety.md](safety.md)): nothing moves
unless the **deadman** is held in a fresh command, and after a joystick connects (or the robot
recovers, unlocks or finishes homing) the deadman must be seen *released* once before it counts.

## Gamepad

Any pad SDL has a GameController mapping for (Xbox, PlayStation, Switch Pro, most Logitech)
reads in the same Xbox-style layout. Other pads fall back to a raw Xbox-like layout with a
warning.

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
direction (out of the fingers).

Check a pad before using it:

```bash
uv run ft-check-joystick --once    # list the pads SDL sees
uv run ft-check-joystick           # live view of every axis and button of input.gamepad_index
```

The live view prints the raw axes and buttons and the resulting intent: push the left stick away
from you and it should read `fwd +1.00`. If an axis is reversed, list it in `input.invert`
(e.g. `["ang_y"]`). `input.deadzone` and `input.expo` set the stick
feel. A cable avoids wireless dropouts.

If the pad drops out, the reader waits for *that* pad (matched by its identity, not its slot)
and ignores any other pad. When it comes back, LB must be released once before it counts again.

## Web joystick (`webpad`)

```bash
uv run ft-teleop --input webpad     # opens the web viewer with the pad in its side panel
```

`ft-joystick --device webpad` serves a gamepad page on `input.webpad_port` (8081). The web viewer
(8080) embeds it in its panel; `http://localhost:8081` is the pad on its own, e.g. on a tablet.
It has the gamepad's layout and labels: hold **LB** while moving a stick. A stick without LB does
nothing, as on the real pad; the page then outlines LB in red and says so.

- **Mouse:** click the pad and hold the **space bar** as LB while dragging a stick. The space bar
  counts only while the pad has keyboard focus (holding it before the click works too).
- **Touch screen:** hold LB with one finger and a stick with another.

It talks to the robot exactly as a physical pad does. The page's state drives an SDL *virtual*
gamepad inside `ft-joystick`, which the unchanged gamepad reader and mapping read (deadzone,
expo, rising-edge buttons, LB re-arm), and the commands carry `source = "gamepad"`. On top of
that, the page behaves like a wireless pad:

- One page holds the pad; other pages watch and can **Take control**, which re-plugs the pad.
- Closing the page unplugs the pad. A page that sends nothing for 250 ms (frozen tab, network
  stall) reads as fully released, so the node halts.
- Losing focus or hiding the page releases every button and stick.
- Only pages served by `ft-joystick` itself, on `localhost` or an IP address, may connect, so
  another website open in the same browser cannot drive the robot.

Because the node cannot tell it from a physical pad, `ft-joystick` itself refuses
`--device webpad` unless `"webpad"` is allowed (automatic in sim).

Limits: tested in Chromium only. Closing the tab stops the arm once the browser has torn the page
down (0.35–0.5 s for the viewer page with software rendering on the headless test box, ~25 ms for
the pad on its own page). The pad asks for its own browser process (`Origin-Agent-Cluster`) so
the 3D view can't delay its heartbeat; a browser that ignores this may make the embedded pad
stop and go while it renders slowly. The pad on its own page avoids that.

## Keyboard

```bash
uv run ft-teleop --input keyboard
```

Keyboard input reads the **terminal** that runs `ft-teleop` (or `ft-joystick`), so it works over
plain SSH. The viewers only display: the browser viewer uses W/A/S/D/Q/E for its camera and the
native MuJoCo window has its own key bindings, so keep the terminal focused while you type.

| Key | Action |
|---|---|
| `space` | deadman on / off (a toggle: terminals report no key releases) |
| `w` / `s`, `a` / `d`, `r` / `f` | forward / back, left / right, up / down |
| `j` / `l`, `i` / `k`, `u` / `o` | yaw, pitch, roll |
| `g`, `h`, `b`, `c`, `t` | gripper, home, stop, recover, frame |
| `]` / `[` | speed level up / down |
| `q` | quit |

A key counts as held for 0.6 s after a press and 0.12 s after each autorepeat, so a tap moves
briefly and the arm stops about 0.1 s after you let go. Any space press while the deadman is on
turns it off at once, and holding space leaves it off.

## Script

```bash
uv run ft-teleop --input script --script configs/scripts/demo.toml
```

A script is a list of timed steps; `ft-joystick` plays it as if an operator held the controls:

```toml
description = "2 s forward"

[[step]]
duration = 2.0              # seconds
deadman = true              # default true
linear = [0.5, 0.0, 0.0]    # (forward, left, up), each in [-1, 1]
angular = [0.0, 0.0, 0.0]   # (roll, pitch, yaw), each in [-1, 1]
press = ["gripper"]         # actions fired once at the start of the step
```

A 0.3 s step with the deadman released is added before and after, so a script always starts
armed and ends stopped. `configs/scripts/smoke.toml` moves a few centimetres forward;
`configs/scripts/demo.toml` drives a square, closes and opens the gripper and yaws a little.

## Controls in the web viewer

The web viewer lists the keyboard and gamepad controls in its panel, with the device the session
was started with expanded.
