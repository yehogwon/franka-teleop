"""Offscreen rendering: ``--mode snapshot`` (PNG) and ``--mode record`` (MP4).

Headless, this needs ``MUJOCO_GL=egl`` set before mujoco is first imported; ft-view's CLI does
that. The renderer is closed explicitly: left to ``__del__`` at interpreter shutdown, EGL
raises a traceback.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from .overlay import CameraSpec, add_workspace
from .source import Source


class OffscreenRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        camera: CameraSpec,
        workspace: tuple[Sequence[float], Sequence[float]] | None = None,
    ):
        # The offscreen framebuffer is sized from the model; make room for the requested size.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        camera.apply(self.camera)
        self.workspace = workspace

    def render(self, qpos: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        if self.workspace is not None:
            add_workspace(self.renderer.scene, *self.workspace)
        return self.renderer.render(out=out)

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> OffscreenRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _wait_for_state(src: Source, wait: float) -> bool:
    if src.wait_for_state(wait):
        return True
    print(f"ft-view: no state from {src.describe()} within {wait:g} s", file=sys.stderr)
    return False


def snapshot(
    src: Source,
    out: Path,
    size: tuple[int, int],
    camera: CameraSpec,
    workspace=None,
    wait: float = 10.0,
) -> int:
    import imageio.v3 as iio

    if not _wait_for_state(src, wait):
        return 1
    with OffscreenRenderer(src.model, *size, camera, workspace) as r:
        img = r.render(src.qpos)
    iio.imwrite(out, img)
    print(f"ft-view: wrote {out} ({size[0]}x{size[1]}) from {src.describe()}")
    return 0


def record(
    src: Source,
    out: Path,
    size: tuple[int, int],
    camera: CameraSpec,
    duration: float,
    fps: float,
    workspace=None,
    wait: float = 10.0,
) -> int:
    """Exactly round(duration * fps) frames, each rendered at its scheduled wall-clock time
    from the newest state (late frames are rendered late, never skipped), so the video's length
    and frame count are deterministic."""
    import imageio.v2 as iio

    if not _wait_for_state(src, wait):
        return 1
    n_frames = max(1, round(duration * fps))
    writer = iio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        quality=8,
        ffmpeg_log_level="error",
    )
    late = 0
    try:
        with OffscreenRenderer(src.model, *size, camera, workspace) as r:
            frame = np.empty((size[1], size[0], 3), dtype=np.uint8)
            t0 = time.monotonic()
            for k in range(n_frames):
                delay = t0 + k / fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0 / fps:
                    late += 1
                src.poll()
                writer.append_data(r.render(src.qpos, out=frame))
    finally:
        writer.close()  # also on Ctrl-C: the frames so far form a playable file
    msg = f"ft-view: wrote {out} ({n_frames} frames, {size[0]}x{size[1]} @ {fps:g} fps)"
    print(msg + (f", {late} rendered late" if late else ""))
    return 0
