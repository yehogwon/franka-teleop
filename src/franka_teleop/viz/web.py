"""``--mode web``: a viser (three.js) viewer in the browser, for headless machines.

The MjModel's visual geoms are uploaded once as meshes; each frame only their poses are sent,
in one atomic batch so a browser never draws half an update. viser skips poses that did not
change, so static geoms cost nothing after the first frame.
"""

from __future__ import annotations

import html
import sys
import time

import mujoco
import numpy as np
import viser

from ..input.controls import DEVICES, controls_markdown
from .meshes import geom_poses, geom_rgba, model_meshes
from .overlay import CameraSpec, box_edges, state_text, status_lines
from .source import Source

STATUS_PERIOD_S = 0.2
GROUND_RGB = (205, 214, 224)
WORKSPACE_RGB = (38, 190, 255)
WEBPAD_HINT = (
    "**Joystick below** (`ft-joystick --device webpad`): hold **LB**, or the space bar after "
    "clicking the pad, while moving a stick. Keys pressed on the 3D view move the camera."
)
WEBPAD_HEIGHT_PX = 400

# Loader for the joystick frame. The pad is served by ft-joystick on its own port, so the frame
# cannot use a relative URL: this srcdoc page (same origin as the viewer, so it can read the
# viewer's address) builds http://<viewer host>:<webpad port>/, waits until ft-joystick answers
# (it starts after the viewer) and then navigates there.
WEBPAD_LOADER = """<!doctype html><meta charset="utf-8">
<style>body{margin:0;padding:8px;font:12px/1.4 system-ui,sans-serif;color:#66727f}
code{font-size:11px;word-break:break-all}</style>
<div id="m">Connecting to the web joystick…</div>
<script>
const port = __PORT__;
const loc = parent.location;
const host = loc.hostname.includes(":") ? "[" + loc.hostname + "]" : loc.hostname;
const url = "http://" + host + ":" + port + "/";
const viewerPort = loc.port || "80";
let tries = 0;
async function probe() {
  try {
    await fetch(url + "healthz", { mode: "no-cors", cache: "no-store" });
    location.replace(url);
  } catch (e) {
    if (++tries >= 3) {
      document.getElementById("m").innerHTML =
        "Waiting for the web joystick at <code>" + url + "</code> " +
        "(<code>ft-joystick --device webpad</code>, or <code>ft-teleop --input webpad</code>)." +
        "<br><br>Over SSH, forward its port as well:<br><code>ssh -L " + viewerPort +
        ":localhost:" + viewerPort + " -L " + port + ":localhost:" + port +
        " &lt;server&gt;</code>";
    }
    setTimeout(probe, 1000);
  }
}
probe();
</script>"""


def webpad_frame(port: int) -> str:
    """HTML for viser's add_html: an iframe that loads the web joystick on ``port``."""
    doc = WEBPAD_LOADER.replace("__PORT__", str(int(port)))
    return (
        f'<iframe title="web joystick" srcdoc="{html.escape(doc, quote=True)}" '
        f'style="width:100%;height:{WEBPAD_HEIGHT_PX}px;border:0;display:block"></iframe>'
    )


DISPLAY_ONLY_HINT = (
    "**Display only:** keys pressed here (W/A/S/D/Q/E) move this camera, not the robot."
)


def _rgb(rgba: np.ndarray) -> tuple[int, int, int]:
    return tuple(round(255 * float(np.clip(c, 0.0, 1.0))) for c in rgba[:3])


class WebViewer:
    def __init__(
        self,
        src: Source,
        host: str,
        port: int,
        camera: CameraSpec,
        workspace=None,
        device: str | None = None,
        webpad_port: int | None = None,
    ):
        """``device``: the operator's input device; its controls are shown expanded, the
        others collapsed. ``"webpad"`` embeds the web joystick (served on ``webpad_port``)."""
        if device == "webpad" and webpad_port is None:
            raise ValueError("device 'webpad' needs webpad_port")
        self.src = src
        self.model = src.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = src.qpos
        mujoco.mj_kinematics(self.model, self.data)

        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        scene = self.server.scene
        scene.set_up_direction("+z")
        self.server.initial_camera.position = camera.position()
        self.server.initial_camera.look_at = camera.lookat
        self.server.initial_camera.fov = np.deg2rad(self.model.vis.global_.fovy)  # as MuJoCo

        meshes, planes = model_meshes(self.model)
        self.geom_ids = np.array([m.geom_id for m in meshes], dtype=int)
        pos, quat = geom_poses(self.data, self.geom_ids)
        self.handles: list[viser.MeshHandle] = []
        for m, p, q in zip(meshes, pos, quat, strict=True):
            alpha = float(m.rgba[3])
            self.handles.append(
                scene.add_mesh_simple(
                    f"/scene/{m.name}",
                    m.vertices,
                    m.faces,
                    color=_rgb(m.rgba),
                    opacity=None if alpha >= 1.0 else alpha,
                    side="double",  # a few faces of the FR3 meshes are wound inward
                    position=p,
                    wxyz=q,
                )
            )
        for g in planes:
            self._add_ground(g)
        if workspace is not None:
            scene.add_line_segments(
                "/workspace", box_edges(*workspace), WORKSPACE_RGB, thickness=0.004
            )

        gui = self.server.gui
        self.webpad_html = None
        if device == "webpad":
            with gui.add_folder("joystick"):
                self.webpad_html = gui.add_html(webpad_frame(webpad_port))
        with gui.add_folder("ft-view"):
            hint = WEBPAD_HINT if device == "webpad" else DISPLAY_ONLY_HINT
            self._hint_md = gui.add_markdown(hint)
            self._source_md = gui.add_markdown(self._source_text())
            self._status_md = gui.add_markdown(self._status_text())
        self.controls_md = {}
        shown = "gamepad" if device == "webpad" else device  # the web pad is a gamepad
        for d in sorted(DEVICES, key=lambda d: d != shown):
            text = controls_markdown("webpad" if device == "webpad" and d == "gamepad" else d)
            with gui.add_folder(f"controls: {d}", expand_by_default=d == shown):
                self.controls_md[d] = gui.add_markdown(text)
        self._t_status = time.monotonic()

    def _add_ground(self, g: int) -> None:
        half = self.model.geom_size[g][:2]
        p, q = geom_poses(self.data, np.array([g]))
        rgba = geom_rgba(self.model, g)
        self.server.scene.add_grid(
            "/ground",
            width=2 * float(half[0]) or 10.0,
            height=2 * float(half[1]) or 10.0,
            infinite_grid=not half.all(),
            cell_size=0.1,  # matches the MuJoCo scene's checker squares
            section_size=0.5,
            cell_color=(150, 160, 175),
            section_color=(110, 120, 135),
            plane_color=GROUND_RGB if rgba[:3].min() > 0.95 else _rgb(rgba),
            plane_opacity=0.9,
            fade_distance=8.0,
            position=p[0],
            wxyz=q[0],
        )

    @property
    def port(self) -> int:
        return self.server.get_port()

    def update(self, force: bool = False) -> bool:
        """Poll the source; on new state push all geom poses in one batch. True if updated."""
        changed = self.src.poll()
        if changed or force:
            self.data.qpos[:] = self.src.qpos
            mujoco.mj_kinematics(self.model, self.data)
            pos, quat = geom_poses(self.data, self.geom_ids)
            with self.server.atomic():
                for h, p, q in zip(self.handles, pos, quat, strict=True):
                    h.position = p
                    h.wxyz = q
        now = time.monotonic()
        if now - self._t_status > STATUS_PERIOD_S:
            self._t_status = now
            self._source_md.content = self._source_text()
            self._status_md.content = self._status_text()
        return changed

    def _source_text(self, now: float | None = None) -> str:
        state = state_text(self.src.has_state, self.src.state_age(now))
        if state.startswith("STALE"):
            state = f"**{state}**"
        return f"**source** {self.src.describe()} — {state}"

    def _status_text(self, now: float | None = None) -> str:
        rows = status_lines(self.src.status, self.src.status_age(now))
        return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)

    def run(self, fps: float, duration: float | None = None) -> None:
        period = 1.0 / fps
        t_end = None if duration is None else time.monotonic() + duration
        next_frame = time.monotonic()
        while t_end is None or time.monotonic() < t_end:
            self.update()
            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()

    def close(self) -> None:
        self.server.stop()


def run_web(
    src: Source,
    host: str,
    port: int,
    fps: float,
    camera: CameraSpec,
    workspace=None,
    duration: float | None = None,
    device: str | None = None,
    webpad_port: int | None = None,
) -> int:
    viewer = WebViewer(src, host, port, camera, workspace, device, webpad_port)
    try:
        actual = viewer.port
        if actual != port:
            print(f"ft-view: port {port} is busy, serving on {actual} instead", file=sys.stderr)
        print(f"ft-view: web viewer on http://{host}:{actual} ({src.describe()})", flush=True)
        if host in ("127.0.0.1", "localhost"):
            forwards = f"-L {actual}:localhost:{actual}"
            if device == "webpad":  # the embedded joystick is served on its own port
                forwards += f" -L {webpad_port}:localhost:{webpad_port}"
            print(
                f"ft-view: from another machine: ssh {forwards} <this-host>, "
                f"then open http://localhost:{actual}",
                flush=True,
            )
        viewer.run(fps, duration)
    finally:
        viewer.close()
    return 0
