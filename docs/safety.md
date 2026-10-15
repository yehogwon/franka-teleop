# Safety design

The **deadman** is a control the operator must keep holding for the arm to move (named after
the railway "dead man's handle": if the driver lets go, the train stops). Here it is LB on the
gamepad and the web pad, and a toggle on the keyboard; see [input-devices.md](input-devices.md).

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
- **Web joystick only in sim by default.** The node can't tell the web pad from a physical one
  (that is the point), so `ft-joystick` enforces it: `--device webpad` refuses to start unless
  `"webpad"` is allowed, which `sim.enabled` does automatically. On the robot, add it to
  `teleop.allowed_sources` deliberately. Like the node's own source check, this guards against
  misconfiguration, not against a hostile client.
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
gripper, home, stop/recover) checks every status message: no robot errors, the TCP stays
within 1 mm of the workspace box (the test allows 2 mm), no joint limit is crossed, and σ stays
above the stop threshold. The soak is how the
wrist-on-table gap was found. The soak and two independent code reviews also found the other
failure modes listed above; each now has a regression test.

The software deadman is **not** an emergency stop. Keep the Franka external enabling/emergency
stop device in hand whenever the real arm is powered.
