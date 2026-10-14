"""The web joystick in a real browser (Chromium via Playwright), against a simulated arm.

``ft-teleop --input webpad`` starts ft-sim, ft-node, the web viewer and ft-joystick serving the
pad; the tests use the pad the way an operator does: inside the viewer's panel, with the mouse
on a stick and the space bar as LB.

Needs Playwright's Chromium (``uv run playwright install chromium``); skipped without it.
"""

from __future__ import annotations

import glob
import os
import signal
import subprocess
import time
import urllib.request

import pytest

from ..conftest import free_loopback_host, make_test_config
from .test_launcher import BIN, Launcher, StatusRecorder

pytestmark = [pytest.mark.e2e, pytest.mark.sim, pytest.mark.browser]

PAD_FRAME = 'iframe[title="web joystick"]'


def chromium_executable(p) -> str | None:
    """Playwright's own Chromium, else the newest one in its cache (any build drives these
    pages), else None."""
    exe = p.chromium.executable_path
    if os.path.exists(exe):
        return exe
    cache = os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")
    found = sorted(glob.glob(cache), key=lambda s: int(s.split("chromium-")[1].split("/")[0]))
    return found[-1] if found else None


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        exe = chromium_executable(p)
        if exe is None:
            pytest.skip("no Chromium for Playwright: uv run playwright install chromium")
        b = p.chromium.launch(
            executable_path=exe, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"]
        )
        yield b
        b.close()


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            assert time.monotonic() < deadline, f"{url} did not come up"
            time.sleep(0.2)


@pytest.fixture
def session(config_file):
    """A running ``ft-teleop --input webpad`` sim session (web viewer chosen automatically)."""
    cfg = make_test_config(free_loopback_host())
    run = Launcher(config_file(cfg), "--input", "webpad", "--viewer", "auto")
    rec = StatusRecorder(cfg)
    rec.start()
    try:
        wait_http(f"http://127.0.0.1:{cfg.input.webpad_port}/healthz", 90)
        wait_http(f"http://127.0.0.1:{cfg.viewer.web_port}/", 30)
        yield cfg, rec
    finally:
        rec.stop()
        run.proc.send_signal(signal.SIGTERM)
        rc = run.wait(60)
    assert rc == 0, f"ft-teleop exited {rc}:\n{run.stderr}"
    assert "Traceback" not in run.stderr, run.stderr
    assert run.leftovers() == {}


def open_viewer(browser, cfg):
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(f"http://127.0.0.1:{cfg.viewer.web_port}/")
    pad = page.frame_locator(PAD_FRAME)
    pad.locator("#msg").filter(has_text="You hold the pad").wait_for(timeout=30_000)
    return page, pad


def stick_centre(pad, stick: str = "ls") -> tuple[float, float, float]:
    box = pad.locator(f'circle[data-ctl="{stick}"]').bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, box["height"] / 2


def between(statuses, t0, t1):
    return [s for s in statuses if t0 <= s.t <= t1]


def test_embedded_pad_drives_the_arm_like_a_gamepad(session, browser):
    cfg, rec = session
    page, pad = open_viewer(browser, cfg)
    cx, cy, r = stick_centre(pad)

    # A stick without LB does nothing (the deadman).
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx, cy - r, steps=4)
    t0 = time.time()
    time.sleep(0.8)
    t1 = time.time()
    idle = between(rec.statuses, t0, t1)
    assert idle and all(s.mode == "idle" for s in idle)

    # LB (space bar) + stick up: forward, steadily, at the speed level's full stick speed.
    page.keyboard.down("Space")
    t0 = time.time()
    time.sleep(1.5)
    t1 = time.time()
    page.keyboard.up("Space")  # LB released, stick still pushed
    time.sleep(0.8)
    t2 = time.time()
    page.mouse.up()

    moving = between(rec.statuses, t0 + 0.3, t1)
    assert moving and all(s.mode == "teleop" for s in moving), {s.mode for s in moving}
    speed = cfg.teleop.max_linear_speed * cfg.teleop.speed_levels[cfg.teleop.initial_speed_level]
    assert max(s.twist[0] for s in moving) == pytest.approx(speed, rel=0.01)
    held = between(rec.statuses, t0, t1 + 0.1)
    assert held[-1].ee_pos[0] - held[0].ee_pos[0] > 0.05

    # LB released: the node stops commanding motion at once (then brakes), stick or not.
    stopped = [s for s in rec.statuses if s.t > t1 and s.mode == "idle"]
    assert stopped and stopped[0].t - t1 < 0.2, stopped[:1]
    after = between(rec.statuses, t1 + 0.4, t2)
    assert after and all(s.mode == "idle" and s.twist == (0.0,) * 6 for s in after)
    page.close()


def test_closing_the_page_stops_the_arm(session, browser):
    cfg, rec = session
    page, pad = open_viewer(browser, cfg)
    cx, cy, r = stick_centre(pad)
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.keyboard.down("Space")
    page.mouse.move(cx, cy - r, steps=4)
    deadline = time.monotonic() + 5
    while not (rec.statuses and rec.statuses[-1].mode == "teleop"):
        assert time.monotonic() < deadline, "never started moving"
        time.sleep(0.05)
    t_close = time.time()
    # The operator closes the tab with LB and the stick held (unload handlers run). The pad
    # releases on pagehide; the time measured here is dominated by the browser tearing down the
    # software-rendered 3D view first (0.35-0.5 s on this headless box; a standalone pad page
    # releases in ~25 ms).
    page.close(run_before_unload=True)
    time.sleep(1.5)
    stopped = [s for s in rec.statuses if s.t > t_close and s.mode == "idle"]
    assert stopped, "arm still in teleop after the page closed"
    assert stopped[0].t - t_close < 1.0


def test_second_page_takes_over_with_a_fresh_pad(session, browser):
    cfg, rec = session
    viewer, pad_a = open_viewer(browser, cfg)
    other = browser.new_page(viewport={"width": 700, "height": 500})
    other.goto(f"http://127.0.0.1:{cfg.input.webpad_port}/")  # the pad on its own, e.g. a phone
    other.locator("#msg").filter(has_text="Another page holds the pad").wait_for(timeout=10_000)
    assert "2 pages open" in pad_a.locator("#msg").inner_text()

    other.locator("#take").click()
    other.locator("#msg").filter(has_text="You hold the pad").wait_for(timeout=5_000)
    pad_a.locator("#msg").filter(has_text="Another page holds the pad").wait_for(timeout=5_000)

    cx, cy, r = stick_centre(other)
    other.mouse.move(cx, cy)
    other.mouse.down()
    other.keyboard.down("Space")
    other.mouse.move(cx, cy - r, steps=4)
    t0 = time.time()
    time.sleep(1.0)
    t1 = time.time()
    other.keyboard.up("Space")
    other.mouse.up()
    moving = between(rec.statuses, t0 + 0.3, t1)
    assert moving and all(s.mode == "teleop" for s in moving)
    other.close()
    viewer.close()


def test_viewer_explains_how_to_reach_a_missing_pad(config_file, browser):
    """No ft-joystick running: the panel says where the pad should be and how to forward it."""
    cfg = make_test_config(free_loopback_host())
    proc = subprocess.Popen(
        [str(BIN / "ft-view"), "--config", str(config_file(cfg)), "--mode", "web",
         "--source", "status", "--device", "webpad"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        wait_http(f"http://127.0.0.1:{cfg.viewer.web_port}/", 30)
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{cfg.viewer.web_port}/")
        # Playwright's frame_locator does not find an about:srcdoc frame; its handle does.
        frame = page.wait_for_selector(PAD_FRAME, timeout=15_000).content_frame()
        deadline = time.monotonic() + 15
        while "ssh -L" not in (text := frame.locator("#m").inner_text()):
            assert time.monotonic() < deadline, text
            time.sleep(0.2)
        port, web = cfg.input.webpad_port, cfg.viewer.web_port
        assert f"http://127.0.0.1:{port}/" in text
        assert f"ssh -L {web}:localhost:{web} -L {port}:localhost:{port}" in text
        page.close()
    finally:
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    assert f"-L {cfg.input.webpad_port}:localhost:{cfg.input.webpad_port}" in out
