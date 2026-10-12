"""``--mode native``: MuJoCo's passive viewer (needs a display)."""

from __future__ import annotations

import threading
import time

import mujoco
import mujoco.viewer

from .overlay import CameraSpec, add_workspace, status_lines
from .source import Source

STATUS_PERIOD_S = 0.1


def run_native(
    src: Source,
    fps: float,
    camera: CameraSpec,
    workspace=None,
    duration: float | None = None,
) -> int:
    model = src.model
    data = mujoco.MjData(model)
    src.poll()
    data.qpos[:] = src.qpos
    mujoco.mj_forward(model, data)
    period = 1.0 / fps
    t_end = None if duration is None else time.monotonic() + duration
    before = set(threading.enumerate())
    v = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    ui_threads = [t for t in threading.enumerate() if t not in before]
    try:
        with v.lock():
            camera.apply(v.cam)
            if workspace is not None:
                v.user_scn.ngeom = 0
                add_workspace(v.user_scn, *workspace)
        t_status = 0.0
        next_frame = time.monotonic()
        while v.is_running() and (t_end is None or time.monotonic() < t_end):
            if src.poll():
                with v.lock():
                    data.qpos[:] = src.qpos
                    mujoco.mj_forward(model, data)
            now = time.monotonic()
            if now - t_status > STATUS_PERIOD_S:
                t_status = now
                v.set_texts(_overlay_text(src))
            v.sync()
            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()
    finally:
        v.close()
        # The viewer's UI thread tears down its GL context after close() returns. Interpreter
        # exit runs glfw.terminate (atexit); racing that teardown segfaults, so wait for it.
        for t in ui_threads:
            t.join(timeout=5.0)
    return 0


def _overlay_text(src: Source):
    source = src.kind + ("" if src.has_state else " (waiting for state)")
    rows = [("source", source), *status_lines(src.status)]
    labels = "\n".join(k for k, _ in rows)
    values = "\n".join(v for _, v in rows)
    return (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values)
