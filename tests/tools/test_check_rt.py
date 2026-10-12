"""ft-check-rt: pure check helpers with injected inputs, and the real run on this machine."""

import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from franka_teleop.tools import check_rt as rt

from ..conftest import make_test_config

INF = resource.RLIM_INFINITY


def box_is_rt() -> bool:
    try:
        flag = Path("/sys/kernel/realtime").read_text().strip() == "1"
    except OSError:
        flag = False
    return flag or "PREEMPT_RT" in os.uname().version


non_rt_box = pytest.mark.skipif(box_is_rt(), reason="this machine runs a PREEMPT_RT kernel")


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    monkeypatch.delenv("FT_REALTIME", raising=False)
    monkeypatch.delenv("FT_CONFIG", raising=False)


# -- kernel / limits ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "version"),
    [
        ("1\n", "#1 SMP PREEMPT_DYNAMIC Fri Mar 6 07:58:08 UTC 2026"),
        (None, "#8-Ubuntu SMP PREEMPT_RT Mon Jun 2 12:00:00 UTC 2025"),
    ],
)
def test_kernel_rt_detected(flag, version):
    r = rt.check_kernel(flag, version, "6.8.1-1004-realtime")
    assert r.status == "PASS" and "6.8.1-1004-realtime" in r.detail


@pytest.mark.parametrize("flag", [None, "0\n"])
def test_kernel_not_rt_fails_with_hint(flag):
    r = rt.check_kernel(flag, "#106-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar 6", "6.8.0-106-generic")
    assert r.status == "FAIL"
    assert "PREEMPT_DYNAMIC" in r.detail
    assert "pro enable realtime-kernel" in r.hint and rt.FRANKA_RT_DOCS in r.hint


@pytest.mark.parametrize(
    ("soft", "hard", "euid", "cap", "status"),
    [
        (99, 99, 1000, False, "PASS"),
        (INF, INF, 1000, False, "PASS"),
        (0, 0, 0, False, "PASS"),  # root
        (0, 0, 1000, True, "PASS"),  # CAP_SYS_NICE
        (0, 0, 1000, False, "FAIL"),
        (50, 50, 1000, False, "FAIL"),
    ],
)
def test_rtprio(soft, hard, euid, cap, status):
    assert rt.check_rtprio(soft, hard, euid, cap).status == status


def test_rtprio_hints():
    r = rt.check_rtprio(0, 0, 1000, False)
    assert "@realtime hard rtprio 99" in r.hint and "usermod -aG realtime" in r.hint
    assert not r.hint.startswith("the hard limit allows it")
    assert rt.check_rtprio(0, 99, 1000, False).hint.startswith("the hard limit allows it")


def test_memlock():
    assert rt.check_memlock(INF, INF).status == "PASS"
    assert rt.check_memlock(100 * 1024**2, INF).status == "PASS"
    small = rt.check_memlock(64 * 1024, 64 * 1024)
    assert small.status == "WARN" and "64 KiB" in small.detail and "memlock" in small.hint


def test_cap_sys_nice_parsing():
    assert rt.has_cap_sys_nice("Name:\tpython\nCapEff:\t0000000000800000\n")
    assert rt.has_cap_sys_nice("CapEff:\t000001ffffffffff\n")
    assert not rt.has_cap_sys_nice("CapEff:\t0000000000000000\n")
    assert not rt.has_cap_sys_nice("")


def test_sched_fifo_result():
    assert rt.check_sched_fifo(True, 99, "ok").status == "PASS"
    r = rt.check_sched_fifo(False, 99, "[Errno 1] Operation not permitted")
    assert r.status == "FAIL" and "Operation not permitted" in r.detail and "rtprio" in r.hint


# -- governor / jitter -------------------------------------------------------------------------


def test_governors():
    assert rt.check_governors(["performance"] * 8).status == "PASS"
    mixed = rt.check_governors(["performance", "powersave", "powersave"])
    assert mixed.status == "WARN" and "2 CPUs 'powersave'" in mixed.detail
    assert "cpupower frequency-set -g performance" in mixed.hint
    assert rt.check_governors(["schedutil"] * 4).status == "WARN"
    assert rt.check_governors([]).status == "WARN"


def stats(p50=50.0, p99=80.0, mx=300.0, missed=0):
    return {"policy": "SCHED_FIFO 99", "n": 3000, "missed": missed, "p50_us": p50,
            "p99_us": p99, "max_us": mx}  # fmt: skip


@pytest.mark.parametrize(
    ("kw", "status"),
    [
        ({}, "PASS"),
        ({"p99": 150.0}, "WARN"),
        ({"mx": 800.0}, "WARN"),
        ({"p99": 400.0}, "FAIL"),
        ({"mx": 1500.0}, "FAIL"),
        ({"missed": 1}, "FAIL"),
    ],
)
def test_jitter_thresholds(kw, status):
    r = rt.check_jitter(stats(**kw))
    assert r.status == status
    assert "SCHED_FIFO 99" in r.detail and "p99" in r.detail
    if status != "PASS":
        assert "approximates libfranka" in r.hint


def test_lateness_stats():
    s = rt.lateness_stats([float(i) for i in range(1, 101)], missed=2)
    assert s["n"] == 100 and s["missed"] == 2 and s["max_us"] == 100.0
    assert 49 <= s["p50_us"] <= 51 and 98 <= s["p99_us"] <= 100
    assert rt.lateness_stats([])["n"] == 0


def test_measure_lateness_runs_at_1khz():
    lateness, missed = rt.measure_lateness(0.05)
    assert 45 <= len(lateness) + missed <= 51
    assert all(x >= 0 for x in lateness)


# -- network -----------------------------------------------------------------------------------

IPUTILS_OK = """\
PING 172.16.0.2 (172.16.0.2) 56(84) bytes of data.
64 bytes from 172.16.0.2: icmp_seq=1 ttl=64 time=0.212 ms
64 bytes from 172.16.0.2: icmp_seq=2 ttl=64 time=0.198 ms

--- 172.16.0.2 ping statistics ---
10 packets transmitted, 10 received, 0% packet loss, time 1834ms
rtt min/avg/max/mdev = 0.181/0.205/0.245/0.019 ms
"""
IPUTILS_SLOW_LOSSY = """\
--- 172.16.0.2 ping statistics ---
10 packets transmitted, 8 received, 20% packet loss, time 1834ms
rtt min/avg/max/mdev = 0.400/0.912/2.100/0.300 ms
"""
IPUTILS_LOSSY = """\
10 packets transmitted, 9 received, 10% packet loss, time 1834ms
rtt min/avg/max/mdev = 0.181/0.205/0.245/0.019 ms
"""
IPUTILS_UNREACHABLE = """\
PING 172.16.0.2 (172.16.0.2) 56(84) bytes of data.
From 172.16.0.1 icmp_seq=1 Destination Host Unreachable

--- 172.16.0.2 ping statistics ---
10 packets transmitted, 0 received, +10 errors, 100% packet loss, time 1840ms
"""
BUSYBOX_OK = """\
--- 172.16.0.2 ping statistics ---
5 packets transmitted, 5 packets received, 0% packet loss
round-trip min/avg/max = 0.101/0.150/0.230 ms
"""


def test_parse_ping():
    s = rt.parse_ping(IPUTILS_OK)
    assert (s.transmitted, s.received) == (10, 10) and s.loss == 0.0
    assert (s.min_ms, s.avg_ms, s.max_ms) == (0.181, 0.205, 0.245)
    b = rt.parse_ping(BUSYBOX_OK)
    assert (b.transmitted, b.received, b.avg_ms) == (5, 5, 0.150)
    u = rt.parse_ping(IPUTILS_UNREACHABLE)
    assert (u.transmitted, u.received, u.avg_ms) == (10, 0, None) and u.loss == 1.0


def test_check_ping():
    assert rt.check_ping("172.16.0.2", IPUTILS_OK, 0).status == "PASS"
    slow = rt.check_ping("172.16.0.2", IPUTILS_SLOW_LOSSY, 1)
    assert slow.status == "WARN" and "0.912" in slow.detail and "> 0.5 ms" in slow.detail
    assert "network-ping-test" in slow.hint
    lossy = rt.check_ping("172.16.0.2", IPUTILS_LOSSY, 0)
    assert lossy.status == "WARN" and "10% loss" in lossy.detail
    down = rt.check_ping("172.16.0.2", IPUTILS_UNREACHABLE, 1)
    assert down.status == "FAIL" and "no reply to 10 pings" in down.detail
    assert "static IP" in down.hint and "cable" in down.hint
    unknown = rt.check_ping("robot", "ping: robot: Name or service not known\n", 2)
    assert unknown.status == "FAIL" and "Name or service not known" in unknown.detail
    assert rt.check_ping("172.16.0.2", "", -1).status == "WARN"  # no ping binary


def test_parse_ip_route():
    direct = rt.parse_ip_route("172.16.0.2 dev enp3s0 src 172.16.0.1 uid 1000 \n    cache \n")
    assert (direct.dev, direct.via, direct.src, direct.error) == ("enp3s0", "", "172.16.0.1", "")
    gw = rt.parse_ip_route("8.8.8.8 via 10.2.0.1 dev eth0 src 10.2.9.13 uid 10003\n    cache\n")
    assert (gw.dev, gw.via) == ("eth0", "10.2.0.1")
    lo = rt.parse_ip_route("local 127.0.0.1 dev lo src 127.0.0.1 uid 10003\n    cache <local>")
    assert lo.dev == "lo"
    assert rt.parse_ip_route("").error


def test_check_route():
    direct = rt.Route(dev="enp3s0", src="172.16.0.1")
    r = rt.check_route("172.16.0.2", direct, wireless=False, speed_mbps=1000)
    assert r.status == "PASS" and "enp3s0" in r.detail and "1000 Mb/s" in r.detail
    wifi = rt.check_route("172.16.0.2", rt.Route(dev="wlp2s0"), wireless=True, speed_mbps=None)
    assert wifi.status == "WARN" and "wireless" in wifi.detail and "Ethernet" in wifi.hint
    gw = rt.check_route("172.16.0.2", rt.Route(dev="eth0", via="10.0.0.1"), False, None)
    assert gw.status == "WARN" and "gateway 10.0.0.1" in gw.detail
    lo = rt.check_route("127.0.0.1", rt.Route(dev="lo"), False, None)
    assert lo.status == "PASS" and "loopback" in lo.detail
    none = rt.check_route("172.16.0.2", rt.Route(error="Network is unreachable"), False, None)
    assert none.status == "FAIL" and "static IP" in none.hint


def test_sysfs_helpers(tmp_path):
    (tmp_path / "wlp2s0" / "wireless").mkdir(parents=True)
    (tmp_path / "enp3s0").mkdir()
    (tmp_path / "enp3s0" / "speed").write_text("1000\n")
    (tmp_path / "wlp2s0" / "speed").write_text("-1\n")
    assert rt.is_wireless("wlp2s0", str(tmp_path)) and not rt.is_wireless("enp3s0", str(tmp_path))
    assert rt.link_speed_mbps("enp3s0", str(tmp_path)) == 1000
    assert rt.link_speed_mbps("wlp2s0", str(tmp_path)) is None
    assert rt.link_speed_mbps("missing", str(tmp_path)) is None


# -- the real thing on this machine ------------------------------------------------------------


@non_rt_box
def test_this_box_fails_the_kernel_check(capsys):
    code = rt.main(["--jitter-seconds", "0.2"])
    out = capsys.readouterr().out
    assert code == 1
    kernel = next(line for line in out.splitlines() if " kernel: " in line)
    assert kernel.startswith("FAIL  kernel: ") and "not a realtime kernel" in kernel
    assert "fix: install a PREEMPT_RT kernel" in out and rt.FRANKA_RT_DOCS in out
    for name in ("rtprio", "memlock", "sched_fifo", "cpu_governor", "jitter"):
        assert f" {name}: " in out
    assert "robot network: skipped" in out  # default config is configs/sim.toml
    assert out.rstrip().splitlines()[-1].startswith("ft-check-rt: NOT READY")


@non_rt_box
@pytest.mark.e2e
def test_console_script_json():
    exe = Path(sys.executable).with_name("ft-check-rt")
    proc = subprocess.run(
        [str(exe), "--jitter-seconds", "0.2", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),  # FT_CONFIG/FT_REALTIME already removed by the fixture
    )
    assert proc.returncode == 1, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["tool"] == "ft-check-rt" and doc["ok"] is False and doc["counts"]["FAIL"] >= 1
    by_name = {r["name"]: r for r in doc["results"]}
    kernel = by_name["kernel"]
    assert kernel["status"] == "FAIL" and "PREEMPT_RT" in kernel["hint"]
    assert kernel["data"]["release"] == os.uname().release
    assert by_name["jitter"]["data"]["n"] > 0
    if by_name["rtprio"]["status"] == "FAIL":
        assert by_name["sched_fifo"]["status"] == "FAIL"
        assert by_name["jitter"]["data"]["policy"] == "SCHED_OTHER"


@pytest.mark.skipif(not (shutil.which("ping") and shutil.which("ip")), reason="needs ping and ip")
def test_robot_host_defaults_to_config_when_not_simulated(config_file, capsys):
    cfg = make_test_config("127.0.0.1", sim__enabled=False)
    code = rt.main(["--config", str(config_file(cfg)), "--jitter-seconds", "0", "--json"])
    doc = json.loads(capsys.readouterr().out)
    by_name = {r["name"]: r for r in doc["results"]}
    assert by_name["route"]["status"] == "PASS" and by_name["route"]["data"]["dev"] == "lo"
    assert by_name["ping"]["status"] == "PASS" and by_name["ping"]["data"]["received"] > 0
    assert "jitter" not in by_name
    assert code == (0 if doc["ok"] else 1)
